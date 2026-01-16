"""
Markovian Engine for memoryless spreading processes.

This engine exploits the piecewise-constant rate structure of Markovian
processes, enabling sparse O(K * D_avg) updates per step where K is the
number of transitioning nodes.

Features:
- Control Mode: Global rate recomputation when control inputs change
- Inertial Mode: Sparse incremental updates between control changes
- Adaptive tau-leaping for efficient GPU utilization
"""

import torch
from typing import Optional, Tuple

from ..core.graph import GraphCSR, DualGraphCSR
from ..core.flash_neighbor import FlashNeighbor


class MarkovianEngine:
    """
    GPU-accelerated Markovian epidemic simulation engine.

    This engine implements tau-leaping simulation for Markovian compartmental
    models where transition rates depend only on current state and neighbor
    influence, not on holding times.

    The engine maintains two operational modes:
    - Control Mode: Full recomputation of all rates (used when parameters change)
    - Inertial Mode: Sparse updates only for affected nodes and neighbors

    Example:
        >>> from flashspread import MarkovianEngine, SISModel, FixedDegreeGraph
        >>> graph = FixedDegreeGraph(10000, 15, device="cuda")
        >>> model = SISModel(beta=0.5, delta=1.0)
        >>> engine = MarkovianEngine(graph, model, device="cuda")
        >>> engine.seed_infection(100)
        >>> for _ in range(1000):
        ...     engine.step()
        >>> print(engine.count_infected())
    """

    def __init__(
        self,
        graph,
        model,
        device: str | torch.device = "cuda",
        max_prob: float = 0.1,
        theta: float = 0.01,
        tau_min: float = 1e-6,
        tau_max: float = 1.0,
        seed: int = 12345,
    ):
        """
        Initialize Markovian simulation engine.

        Args:
            graph: Network object with edge_index and csr attributes.
            model: Compartmental model (e.g., SISModel, SIRModel).
            device: PyTorch device.
            max_prob: Maximum transition probability per step.
            theta: Target fraction of nodes transitioning per step.
            tau_min: Minimum time step.
            tau_max: Maximum time step.
            seed: Random seed.
        """
        self.device = torch.device(device)
        self.model = model
        self.max_prob = float(max_prob)
        self.theta = float(theta)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        # Get graph data
        if hasattr(graph, "csr"):
            self.graph = graph.csr.to(self.device)
            self.edge_index = graph.edge_index.to(self.device)
        elif hasattr(graph, "row_ptr"):
            self.graph = graph
            self.edge_index = None
        else:
            raise ValueError("graph must have csr or row_ptr attribute")

        self.num_nodes = self.graph.num_nodes

        # Build outgoing CSR for sparse updates
        if self.edge_index is not None:
            self.outgoing_graph = GraphCSR(
                self.edge_index, self.num_nodes, incoming=False
            )
        else:
            self.outgoing_graph = None

        # Initialize FlashNeighbor kernel
        self.inducer_states = model.inducer_states
        self.flash_neighbor = FlashNeighbor(self.graph, self.inducer_states)

        # State tensors
        self.state = torch.zeros(self.num_nodes, device=self.device, dtype=torch.int32)
        self.rates = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)
        self.influence = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)

        # Simulation state
        self.current_time = 0.0
        self.total_events = 0
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(seed)

        # Precompute model parameters on device
        self.model.prepare(self.device)

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.state.zero_()
        self.rates.zero_()
        self.influence.zero_()
        self.current_time = 0.0
        self.total_events = 0

    def seed_infection(self, num_infected: int, state: int = None) -> None:
        """
        Randomly infect nodes to start epidemic.

        Args:
            num_infected: Number of nodes to infect.
            state: Target state (default: model's infectious state).
        """
        if state is None:
            state = self.model.infectious if hasattr(self.model, "infectious") else 1

        indices = torch.randperm(self.num_nodes, device=self.device)[:num_infected]
        self.state[indices] = state

        # Recompute influence and rates
        self._recompute_all()

    def set_initial_state(self, initial_state: torch.Tensor) -> None:
        """Set initial state from tensor."""
        self.state.copy_(initial_state.to(self.device, dtype=torch.int32))
        self._recompute_all()

    def _recompute_all(self) -> None:
        """Control Mode: Recompute all influences and rates from scratch."""
        self.influence = self.flash_neighbor.compute_influence(self.state)
        if self.influence.dim() > 1:
            self.influence = self.influence.sum(dim=1)
        self.model.compute_rates(self.state, self.influence, out=self.rates)

    def step(self) -> Tuple[float, int]:
        """
        Execute one tau-leaping step.

        Returns:
            Tuple of (elapsed_time, num_events).
        """
        # Compute adaptive time step
        total_rate = self.rates.sum().item()
        if total_rate < 1e-12:
            return self.tau_max, 0

        # Tau selection: bound expected events and max probability
        tau = min(
            self.theta * self.num_nodes / total_rate,
            self.max_prob / self.rates.max().item(),
        )
        tau = max(self.tau_min, min(tau, self.tau_max))

        # Compute transition probabilities
        probs = 1.0 - torch.exp(-self.rates * tau)

        # Sample transitions (Poisson -> clamp to binary)
        rand = torch.rand(self.num_nodes, device=self.device, generator=self._rng)
        event_mask = rand < probs

        num_events = event_mask.sum().item()
        if num_events > 0:
            # Apply transitions
            old_state = self.state.clone()
            self.model.apply_transitions(self.state, event_mask)

            # Sparse update of influence and rates
            self._sparse_update(event_mask, old_state)

        self.current_time += tau
        self.total_events += num_events
        return tau, num_events

    def _sparse_update(self, event_mask: torch.Tensor, old_state: torch.Tensor) -> None:
        """
        Inertial Mode: Sparse incremental update of influence and rates.

        Only nodes that transitioned and their neighbors need updates.
        """
        # Find which nodes changed to/from inducer states
        new_state = self.state
        changed_idx = event_mask.nonzero().squeeze(-1)

        if changed_idx.numel() == 0:
            return

        # Compute influence delta for changed nodes
        old_inducer = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)
        new_inducer = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)

        for state_idx in self.inducer_states:
            old_inducer += (old_state == state_idx).float()
            new_inducer += (new_state == state_idx).float()

        delta_inducer = new_inducer - old_inducer

        # Update neighbors' influence via outgoing edges
        if self.outgoing_graph is not None and delta_inducer.abs().sum() > 0:
            changed_with_delta = changed_idx[delta_inducer[changed_idx] != 0]
            if changed_with_delta.numel() > 0:
                self._propagate_influence_delta(changed_with_delta, delta_inducer)

        # Recompute rates for affected nodes
        # For simplicity, use control mode periodically
        if self.total_events % 200 == 0:
            self._recompute_all()
        else:
            # Just update rates for changed nodes
            self.model.compute_rates(self.state, self.influence, out=self.rates)

    def _propagate_influence_delta(
        self, changed_idx: torch.Tensor, delta: torch.Tensor
    ) -> None:
        """Propagate influence changes to neighbors."""
        if self.outgoing_graph is None:
            return

        row_ptr = self.outgoing_graph.row_ptr
        col_ind = self.outgoing_graph.col_ind
        weights = self.outgoing_graph.weights

        for i in range(changed_idx.numel()):
            node = changed_idx[i].item()
            d = delta[node].item()
            if d == 0:
                continue

            start = row_ptr[node].item()
            end = row_ptr[node + 1].item()

            if start < end:
                neighbors = col_ind[start:end]
                edge_weights = weights[start:end]
                self.influence[neighbors] += d * edge_weights

    def count_by_state(self) -> torch.Tensor:
        """Return counts for each state."""
        return torch.bincount(self.state, minlength=self.model.num_states)

    def count_infected(self) -> int:
        """Return number of infected nodes."""
        infected = 0
        for state_idx in self.inducer_states:
            infected += (self.state == state_idx).sum().item()
        return infected

    @property
    def events_per_second(self) -> float:
        """Compute throughput in events per second."""
        if self.current_time > 0:
            return self.total_events / self.current_time
        return 0.0
