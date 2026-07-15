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

import math
import operator
from typing import Tuple

import torch

from ..core.graph import as_csr
from ..core.host_rng import (
    INITIAL_CONDITION_DOMAIN,
    _fill_splitmix_counter_,
    _splitmix_uniform_,
    normalize_seed,
    offset_seed,
    project_seed,
)
from ..core.reference import (
    reference_influence_csr,
    reference_influence_infectivity_csr,
)
from ..utils import (
    validate_compartment,
    validate_initial_tensors,
    validate_model_contract,
    validate_population_count,
    validate_fp32_control,
)


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
        _, model_inducers = validate_model_contract(
            model,
            markovian=False,
            methods=("compute_rates", "apply_transitions"),
        )
        self.epsilon = validate_fp32_control(
            "epsilon", epsilon, positive=True
        )
        self.tau_max = validate_fp32_control(
            "tau_max", tau_max, positive=True
        )
        self._seed = normalize_seed(seed)
        if not isinstance(bf16_weights, bool):
            raise TypeError("bf16_weights must be a bool")

        # Validate step-control parameters. epsilon <= 0 makes the adaptive
        # tau collapse to 0, freezing current_time so any `while
        # current_time < tf` loop hangs silently.
        if not math.isfinite(self.epsilon) or not (self.epsilon > 0.0):
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not math.isfinite(self.tau_max) or not (self.tau_max > 0.0):
            raise ValueError(f"tau_max must be > 0, got {self.tau_max}")

        # One runtime graph representation for every engine.  In particular,
        # the CPU path consumes this CSR directly instead of pairing a sorted
        # weight array with an unrelated COO edge order.
        self.graph = as_csr(graph, self.device)

        self.num_nodes = self.graph.num_nodes
        if self.num_nodes <= 0:
            raise ValueError("RenewalEngine requires a graph with at least one node")

        # Optionally downcast weights to bf16 for reduced memory traffic
        if bf16_weights and hasattr(self.graph, 'to_bf16_weights'):
            self.graph = self.graph.to_bf16_weights()
        # Keep later CUDA contexts pinned to the graph's physical device.
        # An abstract request such as ``cuda`` must not be re-resolved against
        # a different ambient device after construction.
        self.device = self.graph.row_ptr.device
        self._graph_signature = self.graph._mutation_signature()

        # Initialize FlashNeighbor kernel (or CPU fallback)
        self.inducer_states = model_inducers
        self._cpu_fallback = (self.device.type != "cuda")
        if not self._cpu_fallback:
            from ..core.flash_neighbor import FlashNeighbor

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
        self.rand_buffer = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float64)

        # Scalar tensors for CUDA Graph compatibility
        self.tau = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.epsilon_t = torch.tensor(self.epsilon, device=self.device, dtype=torch.float32)
        self.tau_max_t = torch.tensor(self.tau_max, device=self.device, dtype=torch.float32)

        # One 64-bit counter per node feeds a SplitMix round. This has the same
        # storage cost as the old int64-wrapped xorshift32 state but provides a
        # exact, equally weighted 52-bit midpoint uniform for rare events.
        self.seed_counter = self._initial_seed_counter(self._seed)

        # Dedicated generator for initial-condition sampling so that
        # seed_infection() is reproducible from the engine seed alone,
        # independent of the global torch RNG.
        self._init_gen = torch.Generator(device=self.device)
        self._init_gen.manual_seed(
            project_seed(self._seed, INITIAL_CONDITION_DOMAIN)
        )

        # Simulation state
        self.current_time = 0.0
        self.total_steps = 0

        # Prepare model parameters on device
        if hasattr(self.model, "prepare"):
            self.model.prepare(self.device)
        self._rate_evaluator = self.model.compute_rates

    def reseed(self, seed: int) -> None:
        """Start fresh private RNG streams without changing simulation state."""
        self._seed = normalize_seed(seed)
        self._fill_seed_counter(self._seed)
        self._init_gen.manual_seed(
            project_seed(self._seed, INITIAL_CONDITION_DOMAIN)
        )

    def reset(self, episode: int | None = None) -> None:
        """Reset all simulation state for clean re-use (e.g., RL episodes).

        Args:
            episode: If given, reseed the RNG streams with ``base_seed +
                episode`` so successive episodes draw *independent* (not
                identical) randomness. If None, reseed back to the base
                seed so the reset run reproduces the first run exactly.
        """
        eff_seed = offset_seed(
            self._seed, episode if episode is not None else 0, name="episode"
        )
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

        # Reseed both RNG streams; without this, RL episodes replay an
        # identical random stream (the step RNG counter and the initial
        # infection draw would otherwise be frozen from construction).
        self._fill_seed_counter(eff_seed)
        self._init_gen.manual_seed(
            project_seed(eff_seed, INITIAL_CONDITION_DOMAIN)
        )

    def seed_infection(self, num_infected: int, state: int = None) -> None:
        """
        Randomly seed initial infections.

        Args:
            num_infected: Number of nodes to infect.
            state: Target state (default: first non-susceptible state).
        """
        if state is None:
            state = getattr(self.model, 'exposed', 1)
        state = validate_compartment(state, self.model.num_states)
        num_infected = validate_population_count(num_infected, self.num_nodes)

        indices = torch.randperm(
            self.num_nodes, device=self.device, generator=self._init_gen
        )[:num_infected]
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
        state, age = validate_initial_tensors(
            initial_state,
            num_nodes=self.num_nodes,
            num_states=self.model.num_states,
            device=self.device,
            initial_age=initial_age,
        )
        self.state.copy_(state)
        if age is not None:
            self.age.copy_(age)
        else:
            self.age.zero_()

    def _initial_seed_counter(self, seed: int) -> torch.Tensor:
        """Construct distinct, seed-scrambled per-node 64-bit counters."""
        counter = torch.empty(
            self.num_nodes, device=self.device, dtype=torch.int64
        )
        return _fill_splitmix_counter_(counter, seed)

    def _fill_seed_counter(self, seed: int) -> None:
        """Reset counters in place without an N-element arange temporary."""
        _fill_splitmix_counter_(self.seed_counter, seed)

    def _rand_uniform(
        self,
        out: torch.Tensor,
        *,
        advance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate reproducible open 52-bit midpoint uniforms."""
        return _splitmix_uniform_(self.seed_counter, out, advance=advance)

    def _step_impl(self) -> torch.Tensor:
        """
        Internal step implementation for CUDA Graph compatibility.

        Returns tau as a tensor (not scalar) for graph capture.
        """
        self.graph._assert_unchanged(
            self._graph_signature, owner=type(self).__name__
        )
        self._compute_pressure_phase()
        return self._advance_from_pressure()

    def _compute_pressure_phase(self) -> None:
        """Compute dense infectious pressure into the persistent buffer."""
        # Step 1: Compute pressure (influence from infectious neighbors)
        if self._cpu_fallback:
            pressure = reference_influence_csr(
                self.graph, self.state, self.inducer_states
            )
        else:
            pressure = self.flash_neighbor.compute_influence(self.state)
        if pressure.dim() > 1:
            pressure = pressure.sum(dim=1)
        self.pressure.copy_(pressure)

    def _advance_from_pressure(self) -> torch.Tensor:
        """Shared adaptive-tau, sampling, and renewal transition tail."""
        self._evaluate_rates_phase()
        valid_step, safe_tau = self._select_tau_phase()
        self._transition_phase(valid_step, safe_tau)
        return self.tau

    def _evaluate_rates_phase(self) -> None:
        """Evaluate model exit rates from the current age, state, and pressure."""
        self._rate_evaluator(self.age, self.state, self.pressure, out=self.rates)

    def _select_tau_phase(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Select and validate tau, returning its validity and safe value."""
        # Step 3: Adaptive step selection
        min_rate, max_rate = torch.aminmax(self.rates)
        tau_candidate = self.epsilon_t / max_rate
        tau = torch.minimum(tau_candidate, self.tau_max_t)
        tau = torch.where(max_rate == 0.0, self.tau_max_t, tau)
        invalid_rates = (
            ~torch.isfinite(min_rate)
            | ~torch.isfinite(max_rate)
            | (min_rate < 0.0)
        )
        valid_step = (
            ~invalid_rates
            & torch.isfinite(tau)
            & (tau > 0.0)
        )
        zero = (min_rate - min_rate) * 0.0
        tau = torch.where(valid_step, tau, zero / zero)
        self.tau.copy_(tau)
        safe_tau = torch.where(valid_step, tau, 0.0)
        return valid_step, safe_tau

    def _transition_phase(
        self,
        valid_step: torch.Tensor,
        safe_tau: torch.Tensor,
    ) -> None:
        """Sample and apply one transactional Bernoulli renewal transition."""
        # Step 4: Compute transition probabilities (Bernoulli). ``-expm1``
        # preserves rare probabilities that ``1 - exp(-x)`` rounds to zero.
        self.event_prob.copy_(self.rates)
        self.event_prob.mul_(-safe_tau)
        self.event_prob.expm1_().neg_()

        # Step 5: Sample Bernoulli transitions
        self._rand_uniform(self.rand_buffer, advance=valid_step)
        torch.lt(self.rand_buffer, self.event_prob, out=self.event_mask)

        # Step 6: Apply transitions and renewal reset
        self.age.add_(safe_tau)  # All ages advance only for a valid step
        self.model.apply_transitions(self.state, self.event_mask, out=self.next_state)

        # Reset age to 0 for nodes that transitioned (renewal property)
        changed = self.next_state != self.state
        self.age.masked_fill_(changed, 0.0)
        self.state.copy_(self.next_state)

    def step(self) -> Tuple[float, torch.Tensor]:
        """
        Execute one adaptive tau-leaping step.

        Returns:
            Tuple of (elapsed_time, current_state).
        """
        tau = float(self._step_impl().item())
        if not math.isfinite(tau) or tau <= 0.0:
            raise FloatingPointError(
                "Renewal tau is non-finite or non-positive; check model "
                "parameters, edge weights, ages, and aggregate weighted degree"
            )
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

    def _capture_multistep_graph(self) -> None:
        """Capture a construction-time CUDA Graph and restore clean state.

        This private path runs only from ``_initialize_cuda_graph``, before a
        caller can set initial conditions. Resetting after capture avoids
        cloning every O(N) work buffer merely to preserve the known-empty
        construction state.
        """
        # Warmup (required for Triton JIT + CUDA Graph), then capture on the
        # engine's actual device rather than whichever CUDA device is ambient.
        with torch.cuda.device(self.device):
            for _ in range(3):
                self._static_step()
            torch.cuda.synchronize(self.device)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(self.steps_per_launch):
                    self._static_step()
            self.graph_exec = g

            self.reset()
            self.step_time_accumulator.zero_()

    def _initialize_cuda_graph(self, steps_per_launch: int) -> None:
        """Shared construction for captured reference-renewal variants."""
        if self.device.type != "cuda":
            raise RuntimeError("CUDA Graph execution requires a CUDA device")
        if isinstance(steps_per_launch, bool):
            raise TypeError("steps_per_launch must be an integer, not bool")
        try:
            self.steps_per_launch = operator.index(steps_per_launch)
        except TypeError as exc:
            raise TypeError("steps_per_launch must be an integer") from exc
        if self.steps_per_launch <= 0:
            raise ValueError(
                f"steps_per_launch must be positive, got {self.steps_per_launch}"
            )
        if self.steps_per_launch * self.tau_max > torch.finfo(torch.float64).max:
            raise ValueError(
                "steps_per_launch * tau_max must fit in the fp64 CUDA Graph "
                "elapsed-time accumulator"
            )
        if hasattr(self.model, "sparse_hazard"):
            self.model.sparse_hazard = False
        self.step_time_accumulator = torch.zeros(
            1, device=self.device, dtype=torch.float64
        )
        self.graph_exec = None
        self._capture_multistep_graph()

    def _static_step(self) -> None:
        self.step_time_accumulator.add_(self._step_impl())

    def _replay_cuda_graph(self) -> Tuple[float, torch.Tensor]:
        self.graph._assert_unchanged(
            self._graph_signature, owner=type(self).__name__
        )
        with torch.cuda.device(self.device):
            self.step_time_accumulator.zero_()
            self.graph_exec.replay()
            elapsed = float(self.step_time_accumulator.item())
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise FloatingPointError(
                "Renewal CUDA Graph elapsed time is non-finite or non-positive; "
                "check model parameters, edge weights, ages, and aggregate "
                "weighted degree"
            )
        self.current_time += elapsed
        self.total_steps += self.steps_per_launch
        return elapsed, self.state


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
        super().__init__(*args, **kwargs)
        self._initialize_cuda_graph(steps_per_launch)

    def step(self) -> Tuple[float, torch.Tensor]:
        """
        Execute steps_per_launch steps via CUDA Graph replay.

        Returns:
            Tuple of (total_elapsed_time, current_state).
        """
        return self._replay_cuda_graph()


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
        validate_model_contract(
            self.model,
            markovian=False,
            methods=(
                "compute_rates",
                "apply_transitions",
                "compute_infectivity",
                "compute_rates_nonmarkov",
            ),
        )

        # Infectivity buffer: infectivity[j] = beta * h_IR(age[j]) if I, else 0
        self.infectivity = torch.zeros(
            self.num_nodes, device=self.device, dtype=torch.float32
        )

        # Use infectivity-weighted kernel instead of state-based kernel.  The
        # same CSR gather has a PyTorch reference implementation on CPU.
        if self._cpu_fallback:
            self.flash_neighbor_inf = None
        else:
            from ..core.flash_neighbor import FlashNeighborInfectivity

            self.flash_neighbor_inf = FlashNeighborInfectivity(self.graph)
        self._rate_evaluator = self.model.compute_rates_nonmarkov

    def reset(self, episode: int | None = None) -> None:
        """Reset base renewal state and the age-dependent shedding buffer."""
        super().reset(episode=episode)
        self.infectivity.zero_()

    def _compute_pressure_phase(self) -> None:
        """Compute age-dependent shedding and its incoming pressure.

        Data flow:
        1. Infectivity pre-pass: compute infectivity[j] for all nodes
        2. FlashNeighborInfectivity: pressure[i] = sum_j w_ji * infectivity[j]
        The inherited step then runs ``compute_rates_nonmarkov`` and the same
        adaptive tau, Bernoulli, and transition hooks as the base engine.
        """
        self.model.compute_infectivity(self.age, self.state, out=self.infectivity)

        if self._cpu_fallback:
            pressure = reference_influence_infectivity_csr(
                self.graph, self.infectivity
            )
        else:
            pressure = self.flash_neighbor_inf.compute_influence(self.infectivity)
        if pressure.dim() > 1:
            pressure = pressure.sum(dim=1)
        self.pressure.copy_(pressure)


class RenewalEngineNonMarkovCUDAGraph(RenewalEngineNonMarkov):
    """
    CUDA Graph optimized version of non-Markovian edge engine.

    Captures the full step (infectivity pre-pass + FlashNeighborInfectivity
    + rates + Bernoulli) as a CUDA Graph for reduced kernel launch overhead.
    """

    def __init__(self, *args, steps_per_launch: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_cuda_graph(steps_per_launch)

    def step(self) -> Tuple[float, torch.Tensor]:
        """Execute steps_per_launch steps via CUDA Graph replay."""
        return self._replay_cuda_graph()
