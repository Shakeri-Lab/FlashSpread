"""
Renewal Engine for non-Markovian (age-dependent) spreading processes.

This engine handles renewal processes where transition hazards depend on
holding times (ages). Since all ages advance continuously, sparse updates
are inapplicable and dense O(N) synchronous stepping is required.

Features:
- Age tracking per node with renewal reset on transition
- Adaptive Bernoulli tau-leaping for tunable fidelity
- erfcx-based numerically stable hazard evaluation
- CUDA Graph support for maximum throughput
"""

import torch
from typing import Tuple, Optional

from ..core.graph import GraphCSR
from ..core.flash_neighbor import FlashNeighbor, FlashNeighborInfectivity, reference_influence


class RenewalEngine:
    """
    GPU-accelerated non-Markovian epidemic simulation engine.

    This engine implements adaptive Bernoulli tau-leaping for renewal
    processes where transition hazards are age-dependent. The key difference
    from Markovian simulation is that all node hazards change at every time
    step (since ages advance), requiring dense updates.

    Example:
        >>> from flashspread import RenewalEngine, SEIRModel, FixedDegreeGraph
        >>> graph = FixedDegreeGraph(10000, 15, device="cuda")
        >>> model = SEIRModel(beta=0.3, mean_ei=5.0, median_ei=4.0,
        ...                   mean_ir=3.9, median_ir=1.5)
        >>> engine = RenewalEngine(graph, model, device="cuda")
        >>> engine.seed_infection(100, state=1)  # Start with Exposed
        >>> while engine.current_time < 50.0:
        ...     engine.step()
        >>> print(engine.count_by_state())
    """

    def __init__(
        self,
        graph,
        model,
        device: str | torch.device = "cuda",
        epsilon: float = 0.03,
        tau_max: float = 1.0,
        seed: int = 12345,
        bf16_weights: bool = False,
    ):
        """
        Initialize Renewal simulation engine.

        Args:
            graph: Network object with edge_index and csr attributes.
            model: Non-Markovian compartmental model with age-dependent hazards.
            device: PyTorch device.
            epsilon: Accuracy parameter for adaptive step selection.
                    Bounds maximum transition probability per step.
            tau_max: Maximum time step (caps step size when rates are low).
            seed: Random seed.
            bf16_weights: If True, downcast edge weights to bfloat16 to
                         reduce memory traffic in FlashNeighbor.
        """
        self.device = torch.device(device)
        self.model = model
        self.epsilon = float(epsilon)
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

        # Optionally downcast weights to bf16 for reduced memory traffic
        if bf16_weights and hasattr(self.graph, 'to_bf16_weights'):
            self.graph = self.graph.to_bf16_weights()

        # Initialize FlashNeighbor kernel (or CPU fallback)
        self.inducer_states = model.inducer_states
        self._cpu_fallback = (self.device.type != "cuda")
        if not self._cpu_fallback:
            self.flash_neighbor = FlashNeighbor(self.graph, self.inducer_states)
        else:
            self.flash_neighbor = None  # use reference_influence on CPU

        # State tensors
        self.state = torch.zeros(self.num_nodes, device=self.device, dtype=torch.int32)
        self.age = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)
        self.rates = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)
        self.pressure = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)

        # Working buffers
        self.event_prob = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)
        self.event_mask = torch.zeros(self.num_nodes, device=self.device, dtype=torch.bool)
        self.next_state = torch.zeros(self.num_nodes, device=self.device, dtype=torch.int32)
        self.rand_buffer = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)

        # Scalar tensors for CUDA Graph compatibility
        self.tau = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.epsilon_t = torch.tensor(self.epsilon, device=self.device, dtype=torch.float32)
        self.tau_max_t = torch.tensor(self.tau_max, device=self.device, dtype=torch.float32)
        self.min_rate_t = torch.tensor(1e-9, device=self.device, dtype=torch.float32)

        # RNG state using simple counter-based approach for reproducibility
        self.seed_counter = (
            torch.arange(self.num_nodes, device=self.device, dtype=torch.int64)
            + int(seed) * 1000003
        )
        self.inv_uint32 = torch.tensor(1.0 / (2.0**32), device=self.device, dtype=torch.float32)

        # Simulation state
        self.current_time = 0.0
        self.total_steps = 0

        # Prepare model parameters on device
        if hasattr(self.model, "prepare"):
            self.model.prepare(self.device)

    def reset(self) -> None:
        """Reset all simulation state for clean re-use (e.g., RL episodes)."""
        self.state.zero_()
        self.age.zero_()
        self.rates.zero_()
        self.pressure.zero_()
        self.event_prob.zero_()
        self.event_mask.zero_()
        self.next_state.zero_()
        self.rand_buffer.zero_()
        self.current_time = 0.0
        self.total_steps = 0

    def seed_infection(self, num_infected: int, state: int = None) -> None:
        """
        Randomly seed initial infections.

        Args:
            num_infected: Number of nodes to infect.
            state: Target state (default: first non-susceptible state).
        """
        if state is None:
            state = getattr(self.model, 'exposed', 1)

        indices = torch.randperm(self.num_nodes, device=self.device)[:num_infected]
        self.state[indices] = state
        self.age[indices] = 0.0  # Fresh entry into state

    def set_initial_state(
        self, initial_state: torch.Tensor, initial_age: torch.Tensor = None
    ) -> None:
        """
        Set initial state and optionally ages.

        Args:
            initial_state: [N] tensor of states.
            initial_age: [N] tensor of ages (default: all zeros).
        """
        self.state.copy_(initial_state.to(self.device, dtype=torch.int32))
        if initial_age is not None:
            self.age.copy_(initial_age.to(self.device, dtype=torch.float32))
        else:
            self.age.zero_()

    def _xorshift32(self, x: torch.Tensor) -> torch.Tensor:
        """XorShift32 PRNG for fast uniform random generation."""
        x = x ^ ((x << 13) & 0xFFFFFFFF)
        x = x ^ (x >> 17)
        x = x ^ ((x << 5) & 0xFFFFFFFF)
        return x & 0xFFFFFFFF

    def _rand_uniform(self, out: torch.Tensor) -> torch.Tensor:
        """Generate uniform random numbers using counter-based RNG."""
        self.seed_counter.copy_(self._xorshift32(self.seed_counter))
        out.copy_(self.seed_counter.to(dtype=torch.float32) * self.inv_uint32)
        out.clamp_(min=1e-12, max=1.0 - 1e-7)
        return out

    def _step_impl(self) -> torch.Tensor:
        """
        Internal step implementation for CUDA Graph compatibility.

        Returns tau as a tensor (not scalar) for graph capture.
        """
        # Step 1: Compute pressure (influence from infectious neighbors)
        if self._cpu_fallback:
            pressure = reference_influence(
                self.edge_index, self.num_nodes, self.state,
                self.inducer_states,
                weights=self.graph.weights,
            )
        else:
            pressure = self.flash_neighbor.compute_influence(self.state)
        if pressure.dim() > 1:
            pressure = pressure.sum(dim=1)
        self.pressure.copy_(pressure)

        # Step 2: Compute hazard rates
        self.model.compute_rates(self.age, self.state, self.pressure, out=self.rates)

        # Step 3: Adaptive step selection
        max_rate = self.rates.max()
        tau_candidate = self.epsilon_t / (max_rate + 1e-12)
        tau = torch.minimum(tau_candidate, self.tau_max_t)
        tau = torch.where(max_rate < self.min_rate_t, self.tau_max_t, tau)
        self.tau.copy_(tau)

        # Step 4: Compute transition probabilities (Bernoulli)
        # p = 1 - exp(-lambda * tau)
        self.event_prob.copy_(self.rates)
        self.event_prob.mul_(self.tau)
        self.event_prob.neg_().exp_()  # exp(-rate * tau)
        self.event_prob.neg_().add_(1.0)  # 1 - exp(...)

        # Step 5: Sample Bernoulli transitions
        self._rand_uniform(self.rand_buffer)
        torch.lt(self.rand_buffer, self.event_prob, out=self.event_mask)

        # Step 6: Apply transitions and renewal reset
        self.age.add_(self.tau)  # All ages advance
        self.model.apply_transitions(self.state, self.event_mask, out=self.next_state)

        # Reset age to 0 for nodes that transitioned (renewal property)
        changed = self.next_state != self.state
        self.age.masked_fill_(changed, 0.0)
        self.state.copy_(self.next_state)

        return self.tau

    def step(self) -> Tuple[float, torch.Tensor]:
        """
        Execute one adaptive tau-leaping step.

        Returns:
            Tuple of (elapsed_time, current_state).
        """
        tau = float(self._step_impl().item())
        self.current_time += tau
        self.total_steps += 1
        return tau, self.state

    def simulate_until(self, target_time: float) -> None:
        """
        Simulate until target time is reached.

        Args:
            target_time: Simulation end time.
        """
        while self.current_time < target_time:
            self.step()

    def count_by_state(self) -> torch.Tensor:
        """Return counts for each state."""
        return torch.bincount(self.state, minlength=self.model.num_states)

    def count_infected(self) -> int:
        """Return number of nodes in inducer states."""
        count = 0
        for state_idx in self.inducer_states:
            count += (self.state == state_idx).sum().item()
        return count


class RenewalEngineCUDAGraph(RenewalEngine):
    """
    CUDA Graph optimized version of Renewal Engine.

    This class captures the step() operation as a CUDA Graph for
    reduced kernel launch overhead. Multiple steps are batched into
    a single graph replay.

    Note: CUDA Graphs require static tensor shapes and control flow.
    The sparse hazard evaluation mode is disabled in favor of dense
    evaluation which is more compatible with graph capture.
    """

    def __init__(self, *args, steps_per_launch: int = 50, **kwargs):
        """
        Initialize CUDA Graph engine.

        Args:
            *args: Arguments passed to RenewalEngine.
            steps_per_launch: Number of steps per CUDA Graph replay.
            **kwargs: Keyword arguments passed to RenewalEngine.
        """
        super().__init__(*args, **kwargs)

        if self.device.type != "cuda":
            raise RuntimeError("RenewalEngineCUDAGraph requires CUDA device")

        # Disable sparse hazard mode for CUDA Graph compatibility
        if hasattr(self.model, "sparse_hazard"):
            self.model.sparse_hazard = False

        self.steps_per_launch = int(steps_per_launch)
        self.step_time_accumulator = torch.zeros(1, device=self.device, dtype=torch.float32)

        # Capture the graph
        self.graph_exec = None
        self._capture_graph()

    def _static_step(self) -> None:
        """Single step for CUDA Graph capture (no return value)."""
        tau = self._step_impl()
        self.step_time_accumulator.add_(tau)

    def _capture_graph(self) -> None:
        """Capture CUDA Graph of multiple steps."""
        # Warmup runs
        for _ in range(3):
            self._static_step()
        torch.cuda.synchronize()

        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(self.steps_per_launch):
                self._static_step()

        self.graph_exec = g

    def step(self) -> Tuple[float, torch.Tensor]:
        """
        Execute steps_per_launch steps via CUDA Graph replay.

        Returns:
            Tuple of (total_elapsed_time, current_state).
        """
        self.step_time_accumulator.zero_()
        self.graph_exec.replay()

        elapsed = float(self.step_time_accumulator.item())
        self.current_time += elapsed
        self.total_steps += self.steps_per_launch

        return elapsed, self.state


class RenewalEngineNonMarkov(RenewalEngine):
    """
    Renewal engine with non-Markovian edge-based transmission.

    Implements the source-node compromise: instead of binary inducer checks,
    precomputes an infectivity[N] buffer encoding beta * h_IR(age[j]) for
    infectious nodes. FlashNeighborInfectivity then accumulates weighted
    infectivity from neighbors, enabling age-dependent transmission rates
    at O(N) cost instead of O(E) per-edge ages.

    The infectivity profile reuses the I->R lognormal hazard parameters,
    matching the assumption that infectiousness tracks viral load.

    Example:
        >>> from flashspread import SEIRModel, FixedDegreeGraph
        >>> from flashspread.engines.renewal import RenewalEngineNonMarkov
        >>> graph = FixedDegreeGraph(10000, 15, device="cuda")
        >>> model = SEIRModel(beta=0.3)
        >>> engine = RenewalEngineNonMarkov(graph, model, device="cuda")
        >>> engine.seed_infection(100, state=1)
        >>> dt, state = engine.step()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Infectivity buffer: infectivity[j] = beta * h_IR(age[j]) if I, else 0
        self.infectivity = torch.zeros(
            self.num_nodes, device=self.device, dtype=torch.float32
        )

        # Use infectivity-weighted kernel instead of state-based kernel
        self.flash_neighbor_inf = FlashNeighborInfectivity(self.graph)

    def _step_impl(self) -> torch.Tensor:
        """
        Step with non-Markovian edge transmission.

        Data flow:
        1. Infectivity pre-pass: compute infectivity[j] for all nodes
        2. FlashNeighborInfectivity: pressure[i] = sum_j w_ji * infectivity[j]
        3. compute_rates_nonmarkov: S rate = pressure, E/I rates = hazard(age)
        4-6. Adaptive tau, Bernoulli, transitions (same as base class)
        """
        # Step 0 (NEW): Infectivity pre-pass
        self.model.compute_infectivity(self.age, self.state, out=self.infectivity)

        # Step 1: FlashNeighbor with infectivity
        pressure = self.flash_neighbor_inf.compute_influence(self.infectivity)
        if pressure.dim() > 1:
            pressure = pressure.sum(dim=1)
        self.pressure.copy_(pressure)

        # Step 2: Compute rates (S rate = pressure directly)
        self.model.compute_rates_nonmarkov(
            self.age, self.state, self.pressure, out=self.rates
        )

        # Steps 3-6: identical to base class
        # Step 3: Adaptive step selection
        max_rate = self.rates.max()
        tau_candidate = self.epsilon_t / (max_rate + 1e-12)
        tau = torch.minimum(tau_candidate, self.tau_max_t)
        tau = torch.where(max_rate < self.min_rate_t, self.tau_max_t, tau)
        self.tau.copy_(tau)

        # Step 4: Bernoulli probability p = 1 - exp(-rate * tau)
        self.event_prob.copy_(self.rates)
        self.event_prob.mul_(self.tau)
        self.event_prob.neg_().exp_()
        self.event_prob.neg_().add_(1.0)

        # Step 5: Sample Bernoulli transitions
        self._rand_uniform(self.rand_buffer)
        torch.lt(self.rand_buffer, self.event_prob, out=self.event_mask)

        # Step 6: Apply transitions and renewal reset
        self.age.add_(self.tau)
        self.model.apply_transitions(self.state, self.event_mask, out=self.next_state)

        changed = self.next_state != self.state
        self.age.masked_fill_(changed, 0.0)
        self.state.copy_(self.next_state)

        return self.tau


class RenewalEngineNonMarkovCUDAGraph(RenewalEngineNonMarkov):
    """
    CUDA Graph optimized version of non-Markovian edge engine.

    Captures the full step (infectivity pre-pass + FlashNeighborInfectivity
    + rates + Bernoulli) as a CUDA Graph for reduced kernel launch overhead.
    """

    def __init__(self, *args, steps_per_launch: int = 50, **kwargs):
        super().__init__(*args, **kwargs)

        if self.device.type != "cuda":
            raise RuntimeError("RenewalEngineNonMarkovCUDAGraph requires CUDA device")

        # Disable sparse hazard mode for CUDA Graph compatibility
        if hasattr(self.model, "sparse_hazard"):
            self.model.sparse_hazard = False

        self.steps_per_launch = int(steps_per_launch)
        self.step_time_accumulator = torch.zeros(
            1, device=self.device, dtype=torch.float32
        )

        self.graph_exec = None
        self._capture_graph()

    def _static_step(self) -> None:
        """Single step for CUDA Graph capture."""
        tau = self._step_impl()
        self.step_time_accumulator.add_(tau)

    def _capture_graph(self) -> None:
        """Capture CUDA Graph of multiple steps."""
        for _ in range(3):
            self._static_step()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(self.steps_per_launch):
                self._static_step()

        self.graph_exec = g

    def step(self) -> Tuple[float, torch.Tensor]:
        """Execute steps_per_launch steps via CUDA Graph replay."""
        self.step_time_accumulator.zero_()
        self.graph_exec.replay()

        elapsed = float(self.step_time_accumulator.item())
        self.current_time += elapsed
        self.total_steps += self.steps_per_launch

        return elapsed, self.state
