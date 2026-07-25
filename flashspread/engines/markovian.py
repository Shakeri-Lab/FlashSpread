"""
Markovian Engine for memoryless spreading processes.

This engine exploits piecewise-constant Markovian influence. Every tau step
still samples N nodes, while changes from K transitioned nodes propagate over
their outgoing frontier on the GPU: O(N + sum frontier degrees).

The CPU path is a correctness reference. CUDA uses FlashNeighbor for dense
rebuilds and a Triton frontier kernel for weighted sparse propagation.
"""

import math
import operator
import warnings
from typing import Tuple

import torch

from ..core.graph import as_csr
from ..core.host_rng import (
    MARKOV_GENERATOR_DOMAIN,
    normalize_seed,
    offset_seed,
    project_seed,
    signed_int64,
)
from ..core.reference import reference_influence_csr
from ..utils import (
    sample_distinct_nodes,
    validate_compartment,
    validate_fp32_control,
    validate_initial_tensors,
    validate_model_contract,
    validate_population_count,
)


_MARKOV_NODE_BLOCK = 128
_MARKOV_REDUCTION_BLOCK = 1024
_MAX_CAPTURED_MARKOV_STEPS = 4096


def _reduction_level_sizes(num_values: int) -> tuple[int, ...]:
    """Fixed hierarchy sizes ending in one scalar partial."""
    sizes = [num_values]
    while sizes[-1] > 1:
        sizes.append(
            (sizes[-1] + _MARKOV_REDUCTION_BLOCK - 1)
            // _MARKOV_REDUCTION_BLOCK
        )
    return tuple(sizes)


class MarkovianEngine:
    """
    GPU-accelerated Markovian epidemic simulation engine.

    This engine implements tau-leaping simulation for Markovian compartmental
    models where transition rates depend only on current state and neighbor
    influence, not on holding times.

    Influence is initialized by a full incoming gather and maintained by sparse
    outgoing frontier updates, with an occasional rebuild to bound fp32 drift.

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
            tau_min: Preferred time-step floor. The ``max_prob`` safety bound
                overrides it when necessary.
            tau_max: Maximum time step.
            seed: Random seed.
        """
        self.device = torch.device(device)
        self.model = model
        _, model_inducers = validate_model_contract(
            model,
            markovian=True,
            methods=("prepare", "compute_rates", "apply_transitions"),
        )
        self.max_prob = validate_fp32_control(
            "max_prob", max_prob, positive=True
        )
        self.theta = validate_fp32_control("theta", theta, positive=True)
        self.tau_min = validate_fp32_control(
            "tau_min", tau_min, positive=True
        )
        self.tau_max = validate_fp32_control(
            "tau_max", tau_max, positive=True
        )
        if not math.isfinite(self.max_prob) or not (0.0 < self.max_prob < 1.0):
            raise ValueError(f"max_prob must be in (0, 1), got {self.max_prob}")
        if not math.isfinite(self.theta) or not (self.theta > 0.0):
            raise ValueError(f"theta must be positive, got {self.theta}")
        if (
            not math.isfinite(self.tau_min)
            or not math.isfinite(self.tau_max)
            or not (0.0 < self.tau_min <= self.tau_max)
        ):
            raise ValueError(
                "require 0 < tau_min <= tau_max, got "
                f"tau_min={self.tau_min}, tau_max={self.tau_max}"
            )
        if self.theta > 1.0:
            raise ValueError(f"theta is a target fraction and must be <= 1, got {self.theta}")
        validate_fp32_control(
            "-log1p(-max_prob)", -math.log1p(-self.max_prob), positive=True
        )

        # Generated undirected graphs explicitly advertise reciprocal rows.
        # Remember that provenance before normalizing the wrapper to GraphCSR:
        # the built-in frontier kernel can then reuse the same adjacency bytes
        # in the opposite traversal role instead of retaining a second CSR.
        # Reuse incoming rows as outgoing rows only for package-generated graphs
        # whose reciprocal-edge construction we control. Arbitrary duck-typed
        # ``symmetric=True`` metadata is not sufficient proof of that invariant.
        from ..core.network import _GeneratedGraph

        graph_is_symmetric = isinstance(graph, _GeneratedGraph) and graph.symmetric
        self.graph = as_csr(graph, self.device)
        # Bind the engine to actual CSR storage rather than retaining an
        # unindexed CUDA request that could follow a later ambient-device
        # switch in ``torch.cuda.device(self.device)``.
        self.device = self.graph.row_ptr.device

        self.num_nodes = self.graph.num_nodes
        if self.num_nodes <= 0:
            raise ValueError("MarkovianEngine requires a graph with at least one node")
        self._cpu_fallback = self.device.type != "cuda"

        # Built-in SIS/SIR models admit a fixed-shape Triton pipeline: their
        # rates and deterministic compartment transitions are fully described
        # by scalar parameters. Custom model protocols retain the generic
        # PyTorch path so no capability is lost.
        from ..config import supports_builtin_markovian

        self._builtin_gpu_kind = (
            None if self._cpu_fallback else supports_builtin_markovian(model)
        )
        if self._builtin_gpu_kind is not None:
            recovery_rate = (
                float(model.delta)
                if self._builtin_gpu_kind == "sis"
                else float(model.gamma)
            )
            validate_fp32_control("model.beta", model.beta, nonnegative=True)
            validate_fp32_control(
                "model recovery rate", recovery_rate, nonnegative=True
            )
            validate_fp32_control(
                "theta * num_nodes", self.theta * self.num_nodes, positive=True
            )

        # Sparse propagation needs the opposite orientation.  Build it from
        # canonical CSR so weights remain attached and no COO copy is retained.
        #
        # Aliasing incoming rows as outgoing rows requires more than structural
        # symmetry. Entry k of incoming row u is the edge ``col_ind[k] -> u``, so
        # reading that row as an outgoing row applies ``w(v -> u)`` where
        # ``w(u -> v)`` is required. Structural symmetry does not imply weight
        # symmetry, and a generated graph can acquire non-unit weights through
        # the public ``csr.weights`` compatibility setter while keeping its
        # ``_GeneratedGraph`` wrapper. Unit weights are the only case where the
        # two orientations are interchangeable, so restrict the alias to them and
        # let every weighted graph pay for a real transpose.
        if self._cpu_fallback:
            self.outgoing_graph = None
        elif (
            self._builtin_gpu_kind is not None
            and graph_is_symmetric
            and not self.graph.has_weights
        ):
            self.outgoing_graph = self.graph
        else:
            self.outgoing_graph = self.graph.transpose()
        self._shares_outgoing_csr = self.outgoing_graph is self.graph
        self._graph_signature = self.graph._mutation_signature()
        self._outgoing_graph_signature = (
            None
            if self.outgoing_graph is None or self.outgoing_graph is self.graph
            else self.outgoing_graph._mutation_signature()
        )

        # Initialize FlashNeighbor kernel (or CSR reference path on CPU).
        self.inducer_states = model_inducers
        if self._cpu_fallback:
            self.flash_neighbor = None
        else:
            from ..core.flash_neighbor import FlashNeighbor

            self.flash_neighbor = FlashNeighbor(self.graph, self.inducer_states)

        # State tensors
        self.state = torch.zeros(self.num_nodes, device=self.device, dtype=torch.int32)
        self.rates = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)
        # FlashNeighbor already owns its output buffer.  Reuse that allocation
        # for the common single-inducer case instead of retaining a second N
        # element influence array (4*N bytes at benchmark scale).
        self.influence = (
            self.flash_neighbor.out_buffer
            if self.flash_neighbor is not None and self.flash_neighbor.L == 1
            else torch.zeros(
                self.num_nodes, device=self.device, dtype=torch.float32
            )
        )
        # The built-in GPU kernel updates state safely in place and therefore
        # aliases this compatibility buffer instead of carrying another 4*N
        # bytes. Generic models still need a distinct transition output.
        self.next_state = (
            self.state
            if self._builtin_gpu_kind is not None
            else torch.zeros(self.num_nodes, device=self.device, dtype=torch.int32)
        )

        # Simulation state
        self.current_time = 0.0
        self.total_events = 0
        self.total_steps = 0
        self._steps_since_rebuild = 0
        self._frontier_rebuild_fraction = 0.20
        self.last_update_mode = "rebuild"
        self._seed = normalize_seed(seed)
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(project_seed(self._seed, MARKOV_GENERATOR_DOMAIN))

        if self._builtin_gpu_kind is not None:
            partial_sizes = _reduction_level_sizes(
                (self.num_nodes + _MARKOV_NODE_BLOCK - 1)
                // _MARKOV_NODE_BLOCK
            )
            self._rate_sum_levels = [
                torch.zeros(size, device=self.device, dtype=torch.float32)
                for size in partial_sizes
            ]
            self._rate_max_levels = [
                torch.zeros(size, device=self.device, dtype=torch.float32)
                for size in partial_sizes
            ]
            self._event_count_levels = [
                torch.zeros(size, device=self.device, dtype=torch.int64)
                for size in partial_sizes
            ]
            self._total_rate_device = self._rate_sum_levels[-1]
            self._max_rate_device = self._rate_max_levels[-1]
            self._tau_device = torch.tensor(
                self.tau_max, device=self.device, dtype=torch.float32
            )
            self._rng_seed_device = torch.tensor(
                [signed_int64(self._seed)], device=self.device, dtype=torch.int64
            )
            self._step_id_device = torch.zeros(
                1, device=self.device, dtype=torch.int64
            )
            self._event_count_device = self._event_count_levels[-1]
            self._probability_scale = -math.log1p(-self.max_prob)
            self._target_events = self.theta * self.num_nodes
            # Snapshot both rate parameters. Reading model.beta at every launch
            # made mutating it silently take effect -- bypassing the validation
            # above -- while a mutated recovery rate was silently ignored, so the
            # documented "parameters are copied at construction" contract held
            # for one parameter and not the other.
            self._recovery_rate = recovery_rate
            self._beta = float(model.beta)
            self._state_s = int(model.susceptible)
            self._state_i = int(model.infected)
        else:
            self._rate_sum_levels = None
            self._rate_max_levels = None
            self._event_count_levels = None
            self._total_rate_device = None
            self._max_rate_device = None
            self._tau_device = None
            self._rng_seed_device = None
            self._step_id_device = None
            self._event_count_device = None

        # Precompute model parameters on device
        self.model.prepare(self.device)

    def reseed(self, seed: int) -> None:
        """Reseed the private transition/initialization stream in place."""
        self._seed = normalize_seed(seed)
        self._rng.manual_seed(project_seed(self._seed, MARKOV_GENERATOR_DOMAIN))
        if self._rng_seed_device is not None:
            self._rng_seed_device.fill_(signed_int64(self._seed))
            self._step_id_device.zero_()

    def reset(self, episode: int | None = None) -> None:
        """Reset simulation to initial state.

        Args:
            episode: If given, reseed with a mixed derivation of
                ``(base_seed, episode)`` -- not their sum -- so
                successive RL episodes draw independent randomness;
                otherwise reset to the base seed (reproduces the first run).
        """
        eff_seed = offset_seed(
            self._seed, episode if episode is not None else 0, name="episode"
        )
        self.state.zero_()
        self.next_state.zero_()
        self.rates.zero_()
        self.influence.zero_()
        self.current_time = 0.0
        self.total_events = 0
        self.total_steps = 0
        self._steps_since_rebuild = 0
        self.last_update_mode = "rebuild"
        self._rng.manual_seed(project_seed(eff_seed, MARKOV_GENERATOR_DOMAIN))
        if self._rng_seed_device is not None:
            for buffer in (
                *self._rate_sum_levels,
                *self._rate_max_levels,
                *self._event_count_levels,
            ):
                buffer.zero_()
            self._tau_device.fill_(self.tau_max)
            self._rng_seed_device.fill_(signed_int64(eff_seed))
            self._step_id_device.zero_()
            self._event_count_device.zero_()

    def seed_infection(self, num_infected: int, state: int = None) -> None:
        """
        Randomly infect nodes to start epidemic.

        Args:
            num_infected: Number of nodes to infect.
            state: Target state (default: model's infectious state).
        """
        if state is None:
            # Models expose the inducer state as `infected` (SIS/SIR/SEIR);
            # `infectious` was a stale name that silently fell back to 1.
            state = getattr(self.model, "infected", 1)
        state = validate_compartment(state, self.model.num_states)
        num_infected = validate_population_count(num_infected, self.num_nodes)

        indices = sample_distinct_nodes(
                self.num_nodes,
                num_infected,
                device=self.device,
                generator=self._rng,
            )
        self.state[indices] = state

        # Recompute influence and rates
        self._recompute_all()

    def set_initial_state(self, initial_state: torch.Tensor) -> None:
        """Set initial state from tensor."""
        state, _ = validate_initial_tensors(
            initial_state,
            num_nodes=self.num_nodes,
            num_states=self.model.num_states,
            device=self.device,
        )
        self.state.copy_(state)
        self._recompute_all()

    def _recompute_all(self) -> None:
        """Control Mode: Recompute all influences and rates from scratch."""
        self._validate_graph_storage()
        if self._builtin_gpu_kind is not None:
            with torch.cuda.device(self.device):
                self._rebuild_influence()
                self._prepare_builtin_rates(reset_reductions=True)
            return
        self._rebuild_influence()
        self.model.compute_rates(self.state, self.influence, out=self.rates)

    def _rebuild_influence(self) -> None:
        """Rebuild incoming influence in the engine's persistent buffer."""
        if self._cpu_fallback:
            influence = reference_influence_csr(
                self.graph, self.state, self.inducer_states
            )
        else:
            influence = self.flash_neighbor.compute_influence(self.state)
        if influence.dim() > 1:
            influence = influence.sum(dim=1)
        if influence is not self.influence:
            self.influence.copy_(influence)

    def _prepare_builtin_rates(self, *, reset_reductions: bool = False) -> None:
        """Evaluate current built-in rates and their device-side reductions."""
        from ..core.flash_markovian import (
            _markov_rate_reduce_kernel,
            _markov_reduce_rate_partials_kernel,
        )

        # Every hierarchy level is overwritten, so rebuilding does not need a
        # scalar zeroing launch. Keep the keyword for the generic caller's
        # semantic clarity and backward-compatible private tests.
        _ = reset_reductions
        block_size = _MARKOV_NODE_BLOCK
        grid = ((self.num_nodes + block_size - 1) // block_size,)
        _markov_rate_reduce_kernel[grid](
            state_ptr=self.state,
            influence_ptr=self.influence,
            rates_ptr=self.rates,
            total_rate_ptr=self._rate_sum_levels[0],
            max_rate_ptr=self._rate_max_levels[0],
            beta=self._beta,
            recovery_rate=self._recovery_rate,
            N=self.num_nodes,
            STATE_S=self._state_s,
            STATE_I=self._state_i,
            BLOCK_SIZE=block_size,
        )
        for level in range(len(self._rate_sum_levels) - 1):
            input_sum = self._rate_sum_levels[level]
            input_max = self._rate_max_levels[level]
            output_sum = self._rate_sum_levels[level + 1]
            output_max = self._rate_max_levels[level + 1]
            _markov_reduce_rate_partials_kernel[(output_sum.numel(),)](
                input_sum_ptr=input_sum,
                input_max_ptr=input_max,
                output_sum_ptr=output_sum,
                output_max_ptr=output_max,
                N=input_sum.numel(),
                BLOCK_SIZE=_MARKOV_REDUCTION_BLOCK,
                num_warps=8,
            )

    def _reduce_builtin_event_counts(self) -> None:
        """Reduce fixed per-program event counters to one int64 scalar."""
        from ..core.flash_markovian import _markov_reduce_event_partials_kernel

        for level in range(len(self._event_count_levels) - 1):
            source = self._event_count_levels[level]
            target = self._event_count_levels[level + 1]
            _markov_reduce_event_partials_kernel[(target.numel(),)](
                input_ptr=source,
                output_ptr=target,
                N=source.numel(),
                BLOCK_SIZE=_MARKOV_REDUCTION_BLOCK,
                num_warps=8,
            )

    def step(self) -> Tuple[float, int]:
        """
        Execute one tau-leaping step.

        Returns:
            Tuple of (elapsed_time, num_events).
        """
        self._validate_graph_storage()
        if self._builtin_gpu_kind is not None:
            return self._step_builtin_gpu()

        # Compute adaptive time step. Custom model protocols are less constrained
        # than the built-ins, so reject negative rates explicitly rather than
        # letting cancellation masquerade as an absorbing state.
        min_rate_tensor, max_rate_tensor = torch.aminmax(self.rates)
        min_rate = min_rate_tensor.item()
        max_rate = max_rate_tensor.item()
        if not math.isfinite(min_rate) or not math.isfinite(max_rate):
            raise FloatingPointError(
                "Markovian rates are non-finite; check model parameters, edge "
                "weights, and aggregate weighted degree"
            )
        if min_rate < 0.0:
            raise ValueError(
                f"Markovian transition rates must be non-negative, got {min_rate}"
            )

        total_rate = self.rates.sum().item()
        if not math.isfinite(total_rate):
            raise FloatingPointError(
                "Markovian total rate is non-finite; check model parameters, "
                "edge weights, and aggregate weighted degree"
            )
        if total_rate <= 0.0:
            # No active reactions (absorbing state, e.g. all-R in SIR).
            # Still advance simulated time so external `while current_time
            # < tf` loops terminate; otherwise the engine silently hangs
            # after absorption.
            self.current_time += self.tau_max
            self.total_steps += 1
            return self.tau_max, 0

        # Tau selection: bound expected events and max probability
        probability_bound = -math.log1p(-self.max_prob) / max_rate
        tau = min(
            max(
                self.theta * self.num_nodes / total_rate,
                self.tau_min,
            ),
            probability_bound,
            self.tau_max,
        )
        validate_fp32_control("selected tau", tau, positive=True)

        # Compute transition probabilities
        probs = -torch.expm1(-self.rates * tau)

        # Sample transitions (Poisson -> clamp to binary)
        rand = torch.empty(
            self.num_nodes, device=self.device, dtype=torch.float64
        )
        rand.random_(0, 1 << 52, generator=self._rng)
        # Exact binary64 midpoints remove both endpoint atoms. Probabilities
        # below 2**-53 are conservatively treated as zero rather than inflated.
        rand.add_(0.5).mul_(2.0**-52)
        event_mask = rand < probs

        num_events = event_mask.sum().item()
        if num_events > 0:
            # Apply transitions: the model returns a fresh tensor when
            # out= is not passed, so we need to explicitly write back to
            # self.state. (Previously this return value was dropped, so
            # the engine silently made no progress on any compartmental
            # model; this is the fix.)
            changed_idx = event_mask.nonzero(as_tuple=False).squeeze(-1)
            old_changed_state = self.state[changed_idx].clone()
            self.model.apply_transitions(
                self.state, event_mask, out=self.next_state
            )
            self.state.copy_(self.next_state)

            # Sparse update of influence and rates
            self._sparse_update(changed_idx, old_changed_state)

        self.current_time += tau
        self.total_events += num_events
        self.total_steps += 1
        return tau, num_events

    def _step_builtin_gpu(self) -> Tuple[float, int]:
        """Fixed-shape SIS/SIR step with device reductions and fused frontier."""
        rebuild = self._steps_since_rebuild + 1 >= 256
        with torch.cuda.device(self.device):
            self._event_count_levels[0].zero_()
            self._launch_builtin_step(
                elapsed_ptr=self._tau_device,
                accumulate_time=False,
                accumulate_events=False,
                rebuild_influence=rebuild,
            )
            self._reduce_builtin_event_counts()
            tau = float(self._tau_device.item())
            num_events = int(self._event_count_device.item())

        # Bound atomic accumulation drift with a deterministic cadence. This
        # decision is host-known and adds no state-dependent synchronization.
        if rebuild:
            self._steps_since_rebuild = 0
            self.last_update_mode = "rebuild"
        else:
            self._steps_since_rebuild += 1
            self.last_update_mode = "frontier"

        if not math.isfinite(tau) or tau <= 0.0:
            raise FloatingPointError(
                "Markovian tau is non-finite or non-positive; check model "
                "parameters, edge weights, and aggregate weighted degree"
            )
        self.current_time += tau
        self.total_events += num_events
        self.total_steps += 1
        return tau, num_events

    def _launch_builtin_step(
        self,
        *,
        elapsed_ptr: torch.Tensor,
        accumulate_time: bool,
        accumulate_events: bool,
        rebuild_influence: bool,
    ) -> None:
        """Launch one fixed-shape built-in step without host synchronization."""
        from ..core.flash_markovian import (
            _markov_finalize_tau_kernel,
            _markov_transition_frontier_kernel,
        )

        _markov_finalize_tau_kernel[(1,)](
            total_rate_ptr=self._total_rate_device,
            max_rate_ptr=self._max_rate_device,
            tau_ptr=self._tau_device,
            elapsed_ptr=elapsed_ptr,
            step_id_ptr=self._step_id_device,
            target_events=self._target_events,
            probability_scale=self._probability_scale,
            tau_min=self.tau_min,
            tau_max=self.tau_max,
            ACCUMULATE_TIME=1 if accumulate_time else 0,
        )

        block_size = _MARKOV_NODE_BLOCK
        grid = ((self.num_nodes + block_size - 1) // block_size,)
        recovered = int(getattr(self.model, "recovered", self.model.susceptible))
        _markov_transition_frontier_kernel[grid](
            state_ptr=self.state,
            rates_ptr=self.rates,
            row_ptr_ptr=self.outgoing_graph.row_ptr,
            col_ind_ptr=self.outgoing_graph.col_ind,
            weights_ptr=self.outgoing_graph.weights_storage,
            influence_ptr=self.influence,
            tau_ptr=self._tau_device,
            rng_seed_ptr=self._rng_seed_device,
            step_id_ptr=self._step_id_device,
            event_count_ptr=self._event_count_levels[0],
            N=self.num_nodes,
            STATE_S=self._state_s,
            STATE_I=self._state_i,
            STATE_R=recovered,
            MODEL_SIS=1 if self._builtin_gpu_kind == "sis" else 0,
            HAS_WEIGHTS=self.outgoing_graph.has_weights,
            ACCUMULATE_EVENTS=1 if accumulate_events else 0,
            PROPAGATE_INFLUENCE=0 if rebuild_influence else 1,
            BLOCK_SIZE=block_size,
        )

        if rebuild_influence:
            self._rebuild_influence()

        # Leave rates/reduction scalars ready for the next step and make the
        # public ``rates`` tensor describe the newly updated state.
        self._prepare_builtin_rates()

    def _sparse_update(
        self,
        changed_idx: torch.Tensor,
        old_changed_state: torch.Tensor,
    ) -> None:
        """
        Inertial Mode: Sparse incremental update of influence and rates.

        Only nodes that transitioned and their neighbors need updates.
        """
        if changed_idx.numel() == 0:
            return

        if self._cpu_fallback:
            # Reference path: dense recomputation is both simpler and avoids
            # GPU-oriented ragged-scatter bookkeeping.
            self._recompute_all()
            self.last_update_mode = "rebuild"
            return

        # Compute influence deltas only for the K transitioned nodes.
        new_changed_state = self.state[changed_idx]
        old_inducer = torch.zeros_like(old_changed_state, dtype=torch.bool)
        new_inducer = torch.zeros_like(new_changed_state, dtype=torch.bool)
        for state_idx in self.inducer_states:
            old_inducer |= old_changed_state == state_idx
            new_inducer |= new_changed_state == state_idx

        delta_inducer = new_inducer.to(torch.float32) - old_inducer.to(torch.float32)

        active_delta = delta_inducer != 0.0
        frontier = changed_idx[active_delta]
        frontier_delta = delta_inducer[active_delta]

        # Sparse atomics cease to be attractive when a transitioned hub set
        # covers a material fraction of the graph. Use actual outgoing edge
        # work, not merely K, to choose the dense control path.
        rebuild_for_work = False
        if frontier.numel():
            degree = (
                self.outgoing_graph.row_ptr[frontier + 1]
                - self.outgoing_graph.row_ptr[frontier]
            )
            frontier_edges = int(degree.sum().item())
            rebuild_for_work = (
                frontier_edges
                >= self._frontier_rebuild_fraction * self.outgoing_graph.num_edges
            )

        if rebuild_for_work:
            self._recompute_all()
            self._steps_since_rebuild = 0
            self.last_update_mode = "rebuild"
            return

        if frontier.numel():
            self._propagate_influence_delta(frontier, frontier_delta)
        self.last_update_mode = "frontier"

        # Rates depend on transitioned nodes and affected neighbors. Refresh
        # them densely (cheap relative to graph traversal) and occasionally
        # rebuild influence to bound floating-point accumulation drift.
        self._steps_since_rebuild += 1
        if self._steps_since_rebuild >= 256:
            self._recompute_all()
            self._steps_since_rebuild = 0
            self.last_update_mode = "rebuild"
        else:
            self.model.compute_rates(self.state, self.influence, out=self.rates)

    def _propagate_influence_delta(
        self, changed_idx: torch.Tensor, delta: torch.Tensor
    ) -> None:
        """Propagate influence changes to neighbors."""
        from ..core.flash_markovian import propagate_frontier

        propagate_frontier(
            self.outgoing_graph, changed_idx, delta, self.influence
        )

    def _validate_graph_storage(self) -> None:
        """Protect cached transpose and captured pointer/specialization contracts."""
        self.graph._assert_unchanged(
            self._graph_signature, owner=type(self).__name__
        )
        if self._outgoing_graph_signature is not None:
            self.outgoing_graph._assert_unchanged(
                self._outgoing_graph_signature, owner=type(self).__name__
            )

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
    def events_per_simulated_time(self) -> float:
        """Number of realized transitions per unit of simulated time."""
        if self.current_time > 0:
            return self.total_events / self.current_time
        return 0.0

    @property
    def events_per_second(self) -> float:
        """Deprecated alias; this is not wall-clock throughput."""
        warnings.warn(
            "events_per_second is transitions per simulated time, not wall time; "
            "use events_per_simulated_time",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.events_per_simulated_time


class MarkovianEngineCUDAGraph(MarkovianEngine):
    """Batched fixed-shape CUDA Graph execution for built-in SIS/SIR models.

    One public :meth:`step` replays ``steps_per_launch`` internal tau-leaps.
    The graph performs a full incoming-CSR influence rebuild at least every
    256 internal steps and at every replay boundary, so incremental fp32
    atomics cannot drift without bound.
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
        steps_per_launch: int = 50,
    ):
        from ..config import supports_builtin_markovian

        if torch.device(device).type != "cuda":
            raise RuntimeError("CUDA Graph execution requires a CUDA device")
        if supports_builtin_markovian(model) is None:
            raise TypeError(
                "Markovian CUDA Graph execution supports the exact, unmodified "
                "built-in SISModel and SIRModel only; subclasses and models with "
                "shadowed rate/transition hooks must use the eager engine, whose "
                "generic path calls those hooks"
            )
        if isinstance(steps_per_launch, bool):
            raise TypeError("steps_per_launch must be an integer, not bool")
        try:
            steps_per_launch = operator.index(steps_per_launch)
        except TypeError as exc:
            raise TypeError("steps_per_launch must be an integer") from exc
        if steps_per_launch <= 0:
            raise ValueError(
                f"steps_per_launch must be positive, got {steps_per_launch}"
            )
        if steps_per_launch > _MAX_CAPTURED_MARKOV_STEPS:
            raise ValueError(
                "steps_per_launch must be <= "
                f"{_MAX_CAPTURED_MARKOV_STEPS} for Markov CUDA Graph capture"
            )
        if float(tau_max) * steps_per_launch > torch.finfo(torch.float64).max:
            raise ValueError(
                "steps_per_launch * tau_max must fit in the fp64 CUDA Graph "
                "elapsed-time accumulator"
            )

        super().__init__(
            graph,
            model,
            device=device,
            max_prob=max_prob,
            theta=theta,
            tau_min=tau_min,
            tau_max=tau_max,
            seed=seed,
        )
        self.steps_per_launch = steps_per_launch
        self.step_time_accumulator = torch.zeros(
            1, device=self.device, dtype=torch.float64
        )
        self.graph_exec = None
        self._capture_graph()

    def reset(self, episode: int | None = None) -> None:
        """Reset state and captured-graph summary buffers."""
        super().reset(episode=episode)
        if hasattr(self, "step_time_accumulator"):
            self.step_time_accumulator.zero_()

    def _static_step(self, *, rebuild_influence: bool) -> None:
        self._launch_builtin_step(
            elapsed_ptr=self.step_time_accumulator,
            accumulate_time=True,
            accumulate_events=True,
            rebuild_influence=rebuild_influence,
        )

    def _capture_graph(self) -> None:
        # Warm every captured specialization before recording.  The initial
        # all-susceptible state is canonical and reset below, so no O(N)
        # construction snapshot is needed.
        with torch.cuda.device(self.device):
            self._static_step(rebuild_influence=False)
            self._static_step(rebuild_influence=True)
            self._reduce_builtin_event_counts()
            torch.cuda.synchronize(self.device)

            graph_exec = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_exec):
                for index in range(self.steps_per_launch):
                    step_number = index + 1
                    rebuild = (
                        step_number % 256 == 0
                        or step_number == self.steps_per_launch
                    )
                    self._static_step(rebuild_influence=rebuild)
                self._reduce_builtin_event_counts()
            self.graph_exec = graph_exec

            self.reset()

    def step(self) -> Tuple[float, int]:
        """Replay ``steps_per_launch`` tau-leaps and return elapsed/events."""
        self._validate_graph_storage()
        with torch.cuda.device(self.device):
            self.step_time_accumulator.zero_()
            self._event_count_levels[0].zero_()
            self.graph_exec.replay()

            elapsed = float(self.step_time_accumulator.item())
            num_events = int(self._event_count_device.item())
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise FloatingPointError(
                "Markovian CUDA Graph elapsed time is non-finite or "
                "non-positive; check model parameters, edge weights, and "
                "aggregate weighted degree"
            )
        self.current_time += elapsed
        self.total_events += num_events
        self.total_steps += self.steps_per_launch
        self._steps_since_rebuild = 0
        self.last_update_mode = "rebuild"
        return elapsed, num_events
