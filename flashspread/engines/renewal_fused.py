"""
Fused Renewal Engine using a single Triton kernel per step.

This engine fuses CSR traversal, hazard computation, Bernoulli sampling,
and state transitions into a single kernel launch, eliminating intermediate
O(N) buffers from global memory. It uses the source-node compromise for
non-Markovian edge transmission.

Pipeline per step:
1. Infectivity pre-pass (PyTorch elementwise, CUDA Graph safe)
2. Fused FlashRenewal kernel (single Triton launch)
3. Tau reduction for next step (lightweight max + divide)
"""

import math

import torch
from typing import Tuple

try:
    import triton

    _TRITON_IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001 - keep import working without a GPU
    # Do not let a missing/broken Triton break `import flashspread`; capture
    # the real error so it can be surfaced (chained) if the user actually
    # constructs a GPU engine, instead of a vague "Triton is required".
    triton = None
    _TRITON_IMPORT_ERROR = _exc

from ..core.graph import GraphCSR
from ..core.flash_renewal_kernel import (
    _flash_renewal_fused_kernel,
    _flash_renewal_fused_wc_kernel,
    _pressure_merge_kernel,
    _flash_renewal_tail_kernel,
)
from ..models.hazards import lognormal_hazard_stable


# Auto-dispatch thresholds on D_max / D_mean. See the "Dispatch
# Strategy" paragraph in the paper's §5 for the empirical justification.
# Ratios below _DISPATCH_WARP_RATIO mean the degree distribution is
# essentially uniform -> keep the 1-thread-per-node kernel. Between
# _DISPATCH_WARP_RATIO and _DISPATCH_MERGE_RATIO the warp-per-node
# kernel pays off. Above _DISPATCH_MERGE_RATIO the intra-block load
# imbalance of the warp kernel dominates and edge-partitioned
# merge-based wins.
_DISPATCH_WARP_RATIO = 4.0
_DISPATCH_MERGE_RATIO = 50.0


def _auto_csr_strategy(row_ptr: torch.Tensor, num_nodes: int) -> str:
    """Pick a CSR traversal strategy from the graph's degree heterogeneity."""
    degrees = (row_ptr[1:] - row_ptr[:-1]).to(torch.float64)
    d_mean = float(degrees.mean().item())
    d_max = float(degrees.max().item())
    if d_mean < 1e-12:
        return "thread"
    ratio = d_max / d_mean
    if ratio >= _DISPATCH_MERGE_RATIO:
        return "merge"
    if ratio >= _DISPATCH_WARP_RATIO:
        return "warp"
    return "thread"


class RenewalEngineFused:
    """
    Fused Triton kernel renewal engine with non-Markovian edges.

    Combines the source-node compromise (infectivity pre-pass) with a
    fully fused Triton kernel that performs CSR traversal, hazard
    evaluation, Bernoulli sampling, and state transitions in a single
    kernel launch.

    Example:
        >>> from flashspread import SEIRModel, FixedDegreeGraph
        >>> from flashspread.engines.renewal_fused import RenewalEngineFused
        >>> graph = FixedDegreeGraph(10000, 15, device="cuda")
        >>> model = SEIRModel(beta=0.3)
        >>> engine = RenewalEngineFused(graph, model, device="cuda")
        >>> engine.seed_infection(100, state=model.exposed)
        >>> dt, state = engine.step()
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
        use_mixed_precision: bool = False,
        csr_strategy: str = "auto",
        nodes_per_block: int = 8,
        lanes_per_node: int = 32,
        edges_per_merge_block: int = 4096,
        # Deprecated: prefer csr_strategy="warp" over warp_collaborative=True.
        warp_collaborative: bool = False,
    ):
        if triton is None:
            raise RuntimeError(
                "Triton is required for RenewalEngineFused. Install the GPU "
                "extra with `pip install flashspread[gpu]` on a CUDA machine."
            ) from _TRITON_IMPORT_ERROR

        self.device = torch.device(device)
        self.model = model
        self.epsilon = float(epsilon)
        self.tau_max = float(tau_max)

        # Validate step-control parameters (epsilon<=0 -> tau collapses to
        # 0 and the simulation clock freezes).
        if not (self.epsilon > 0.0):
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not (self.tau_max > 0.0):
            raise ValueError(f"tau_max must be > 0, got {self.tau_max}")

        # CSR-traversal strategy controls how the kernel iterates the
        # incoming-edge list:
        #   "thread": 1 thread per node, scalar loop over neighbors.
        #             Fastest for uniform-degree graphs (regular, ER).
        #   "warp":   32 threads per node, chunked loop. Wins on
        #             moderately heavy-tailed distributions; hub
        #             traversal scales as O(D_max / 32) inside the
        #             node's warp.
        #   "merge":  Edge-partitioned; each block processes a fixed
        #             chunk of edges, recovers the source node via
        #             binary search in row_ptr, and atomic-adds to a
        #             pressure buffer. Delivers perfect load balance
        #             on extremely heavy tails (scale-free with hubs).
        #   "auto":   Choose at construction time from D_max / D_mean.
        self.nodes_per_block = int(nodes_per_block)
        self.lanes_per_node = int(lanes_per_node)
        self.edges_per_merge_block = int(edges_per_merge_block)

        # Get graph data
        if hasattr(graph, "csr"):
            self.graph = graph.csr.to(self.device)
        elif hasattr(graph, "row_ptr"):
            self.graph = graph
        else:
            raise ValueError("graph must have csr or row_ptr attribute")

        self.num_nodes = self.graph.num_nodes

        # Mixed-precision storage: state int8, age fp16, infectivity
        # bf16 (accumulator stays fp32 inside the kernel). Auto-enables
        # bf16 weights so the whole CSR traversal runs in reduced
        # precision; the kernel promotes every load to fp32/int32
        # before any arithmetic via the MIXED_PRECISION constexpr.
        self._use_mixed_precision = bool(use_mixed_precision)
        if self._use_mixed_precision:
            bf16_weights = True  # pull the existing weights-only path too

        if bf16_weights and hasattr(self.graph, 'to_bf16_weights'):
            self.graph = self.graph.to_bf16_weights()

        # Prepare model parameters on device
        if hasattr(self.model, "prepare"):
            self.model.prepare(self.device)

        # State tensors (read/write by fused kernel).
        # Dtypes: baseline uses int32/fp32 for state/age/infectivity;
        # mixed-precision mode (use_mixed_precision=True) drops these
        # to int8/fp16/bf16. The kernel's MIXED_PRECISION constexpr
        # path handles the promote-on-load / cast-on-store contract;
        # the accumulator (pressure) and all math intermediates are
        # always fp32.
        _state_dt  = torch.int8      if self._use_mixed_precision else torch.int32
        _age_dt    = torch.float16   if self._use_mixed_precision else torch.float32
        _inf_dt    = torch.bfloat16  if self._use_mixed_precision else torch.float32
        self.state = torch.zeros(self.num_nodes, device=self.device, dtype=_state_dt)
        self.age = torch.zeros(self.num_nodes, device=self.device, dtype=_age_dt)

        # Double-buffered: kernel writes to next_*, then we swap
        self.next_state = torch.zeros(self.num_nodes, device=self.device, dtype=_state_dt)
        self.next_age = torch.zeros(self.num_nodes, device=self.device, dtype=_age_dt)

        # Infectivity is now double-buffered and written inside the fused
        # kernel. The dense Python pre-pass only runs once, at bootstrap,
        # to populate `infectivity` from the user's initial seed.
        self.infectivity = torch.zeros(self.num_nodes, device=self.device, dtype=_inf_dt)
        self.next_infectivity = torch.zeros(self.num_nodes, device=self.device, dtype=_inf_dt)

        # Rates buffer (written by fused kernel for tau reduction)
        self.rates = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)

        # Resolve CSR strategy now that row_ptr is available.
        strategy = str(csr_strategy).lower()
        if strategy == "auto":
            strategy = _auto_csr_strategy(self.graph.row_ptr, self.num_nodes)
        if warp_collaborative and strategy == "thread":
            # Backwards-compatible: legacy kwarg forces warp-per-node.
            strategy = "warp"
        if strategy not in ("thread", "warp", "merge"):
            raise ValueError(
                f"csr_strategy must be one of 'auto'/'thread'/'warp'/'merge'; "
                f"got {csr_strategy!r}"
            )
        self.csr_strategy = strategy
        # Legacy flag (some benchmarks still read it).
        self.warp_collaborative = strategy == "warp"

        # Mixed-precision is wired into the thread-path fused kernel
        # AND into the merge-path (pressure atomic + tail kernel).
        # The warp-per-node kernel does NOT yet carry the
        # MIXED_PRECISION constexpr, so reject that combination
        # rather than silently produce garbage. Merge is safe because
        # the merge pressure scratch buffer stays fp32 regardless:
        # the atomic-add accumulation of hundreds of bf16 edge
        # contributions on a scale-free hub would otherwise absorb
        # small values and poison the Bernoulli draw.
        if self._use_mixed_precision and strategy == "warp":
            raise ValueError(
                "use_mixed_precision=True is not yet supported with "
                "csr_strategy='warp'. Use 'thread' or 'merge' (both "
                "are fully mixed-precision), or disable the flag."
            )

        if strategy == "merge":
            # The merge kernel needs a scratch pressure buffer and an
            # unrolled binary-search depth that covers N.
            import math as _math
            self._pressure_scratch = torch.zeros(
                self.num_nodes, device=self.device, dtype=torch.float32
            )
            # ceil(log2(N + 2)) guarantees the search terminates for all N.
            self._bsearch_iters = max(
                1, int(_math.ceil(_math.log2(max(self.num_nodes, 2) + 2)))
            )
            # Number of edges in the CSR. GraphCSR.col_ind has shape [E].
            self._num_edges = int(self.graph.col_ind.numel())
        else:
            self._pressure_scratch = None
            self._bsearch_iters = 0
            self._num_edges = 0

        # Tau: use previous step's value. First step uses tau_max.
        self.tau = torch.tensor([self.tau_max], device=self.device, dtype=torch.float32)
        self.epsilon_t = torch.tensor(self.epsilon, device=self.device, dtype=torch.float32)
        self.tau_max_t = torch.tensor(self.tau_max, device=self.device, dtype=torch.float32)

        # Active-node compaction buffers. Base engine never compacts, but
        # the fused kernel signature now takes these pointers so we
        # allocate tiny dummies that sit unread when USE_COMPACTION=0.
        # The CUDA-Graph subclass overrides these with real buffers when
        # use_active_compaction=True.
        self._active_nodes_dummy = torch.zeros(
            1, device=self.device, dtype=torch.int32
        )
        self._num_active_dummy = torch.tensor(
            [self.num_nodes], device=self.device, dtype=torch.int32
        )

        # RNG: step_id increments by 1 per step, used as seed perturbation.
        # Safe for 2^31 steps (>2 billion) without int32 overflow in kernel.
        self._base_seed = int(seed)
        self._rng_seed = int(seed)
        self._step_id = torch.zeros(1, device=self.device, dtype=torch.int64)

        # Dedicated generator for reproducible initial-condition sampling.
        self._init_gen = torch.Generator(device=self.device)
        self._init_gen.manual_seed(self._base_seed)

        # Simulation state
        self.current_time = 0.0
        self.total_steps = 0

        # SEIR state indices
        self._state_s = model.susceptible
        self._state_e = model.exposed
        self._state_i = model.infected
        self._state_r = model.recovered

        # Model parameters (on device)
        self._mu_ei = None
        self._sig_ei = None
        self._mu_ir = None
        self._sig_ir = None
        self._prepare_model_params()

        # Transmission mode is a kernel constexpr, baked in at first JIT.
        # Changing model.transmission_mode after engine construction will
        # not take effect until a new engine is created.
        self._transmission_age_dependent = (
            getattr(self.model, "transmission_mode", "constant") == "age_dependent"
        )
        self._beta = float(self.model.beta)

    def _prepare_model_params(self):
        """Extract lognormal parameters from model."""
        self._mu_ei = float(self.model._mu_ei.item())
        self._sig_ei = float(self.model._sig_ei.item())
        self._mu_ir = float(self.model._mu_ir.item())
        self._sig_ir = float(self.model._sig_ir.item())

    def reset(self, episode: int | None = None) -> None:
        """Reset all simulation state for clean re-use (e.g., RL episodes).

        Args:
            episode: If given, shift the effective RNG seed by ``episode``
                so successive episodes are statistically independent.
                Otherwise reset to the base seed (reproduces the first run).

        Note:
            For the CUDA-Graph subclass the per-step seed is baked into the
            captured graph, so changing ``episode`` only affects the
            initial-condition draw, not the in-graph step RNG. Reconstruct
            the engine if you need a fresh in-graph stream per episode.
        """
        self.state.zero_()
        self.age.zero_()
        self.next_state.zero_()
        self.next_age.zero_()
        self.infectivity.zero_()
        self.next_infectivity.zero_()
        self.rates.zero_()
        if self._pressure_scratch is not None:
            self._pressure_scratch.zero_()
        self.tau.fill_(self.tau_max)
        self._step_id.zero_()
        self.current_time = 0.0
        self.total_steps = 0

        eff_seed = self._base_seed + (int(episode) if episode is not None else 0)
        self._rng_seed = eff_seed
        self._init_gen.manual_seed(eff_seed)

    def seed_infection(self, num_infected: int, state: int = None) -> None:
        """Randomly seed initial infections."""
        if state is None:
            state = self._state_e  # default: Exposed for SEIR
        if not (0 <= num_infected <= self.num_nodes):
            raise ValueError(
                f"num_infected must be in [0, {self.num_nodes}], got {num_infected}"
            )
        indices = torch.randperm(
            self.num_nodes, device=self.device, generator=self._init_gen
        )[:num_infected]
        self.state[indices] = state
        self.age[indices] = 0.0
        # Bootstrap infectivity from the seeded state so the first kernel
        # launch reads a correct value; subsequent steps maintain it
        # in-kernel.
        self._infectivity_prepass()

    def _infectivity_prepass(self):
        """Compute infectivity based on model's transmission_mode."""
        i_mask = self.state == self._state_i

        if getattr(self.model, 'transmission_mode', 'constant') == 'age_dependent':
            hazard_all = lognormal_hazard_stable(
                torch.clamp(self.age, min=1e-10),
                self.model._mu_ir,
                self.model._sig_ir,
            )
            self.infectivity.copy_(
                torch.where(i_mask, self.model._beta_t * hazard_all, 0.0)
            )
        else:
            # Constant beta: matches original RenewalEngine semantics
            self.infectivity.copy_(
                torch.where(i_mask, self.model._beta_t, 0.0)
            )

    def _compute_tau(self):
        """Compute adaptive tau from rates written by fused kernel."""
        max_rate = self.rates.max()
        tau_candidate = self.epsilon_t / (max_rate + 1e-12)
        tau = torch.minimum(tau_candidate, self.tau_max_t)
        tau = torch.where(max_rate < 1e-9, self.tau_max_t, tau)
        self.tau.copy_(tau)

    def _step_impl(self):
        """Execute one fused step."""
        # The infectivity pre-pass no longer runs per step; the fused
        # kernel writes next_infectivity from new_state / new_age.

        # Increment step_id by 1 (CUDA Graph safe, no int32 overflow risk)
        self._step_id.add_(1)

        if self.csr_strategy == "merge":
            # Zero the pressure scratch, accumulate via atomics, then
            # run the tail kernel.
            self._pressure_scratch.zero_()

            merge_grid = lambda meta: (
                triton.cdiv(self._num_edges, meta["EDGES_PER_BLOCK"]),
            )
            _pressure_merge_kernel[merge_grid](
                row_ptr_ptr=self.graph.row_ptr,
                col_ind_ptr=self.graph.col_ind,
                weights_ptr=self.graph.weights,
                infectivity_ptr=self.infectivity,
                pressure_ptr=self._pressure_scratch,
                N=self.num_nodes,
                E=self._num_edges,
                EDGES_PER_BLOCK=self.edges_per_merge_block,
                BSEARCH_ITERS=self._bsearch_iters,
            )

            BLOCK_SIZE = 128
            tail_grid = lambda meta: (
                triton.cdiv(self.num_nodes, meta["BLOCK_SIZE"]),
            )
            _flash_renewal_tail_kernel[tail_grid](
                pressure_ptr=self._pressure_scratch,
                age_ptr=self.age,
                state_ptr=self.state,
                beta=self._beta,
                mu_ei=self._mu_ei,
                sig_ei=self._sig_ei,
                mu_ir=self._mu_ir,
                sig_ir=self._sig_ir,
                tau_ptr=self.tau,
                rng_seed=self._rng_seed,
                step_id_ptr=self._step_id,
                next_state_ptr=self.next_state,
                next_age_ptr=self.next_age,
                next_infectivity_ptr=self.next_infectivity,
                rates_ptr=self.rates,
                N=self.num_nodes,
                STATE_S=self._state_s,
                STATE_E=self._state_e,
                STATE_I=self._state_i,
                STATE_R=self._state_r,
                BLOCK_SIZE=BLOCK_SIZE,
                TRANSMISSION_AGE_DEPENDENT=1 if self._transmission_age_dependent else 0,
                MIXED_PRECISION=1 if self._use_mixed_precision else 0,
            )
        elif self.csr_strategy == "warp":
            grid = lambda meta: (
                triton.cdiv(self.num_nodes, meta["NODES_PER_BLOCK"]),
            )
            _flash_renewal_fused_wc_kernel[grid](
                row_ptr_ptr=self.graph.row_ptr,
                col_ind_ptr=self.graph.col_ind,
                weights_ptr=self.graph.weights,
                infectivity_ptr=self.infectivity,
                age_ptr=self.age,
                state_ptr=self.state,
                beta=self._beta,
                mu_ei=self._mu_ei,
                sig_ei=self._sig_ei,
                mu_ir=self._mu_ir,
                sig_ir=self._sig_ir,
                tau_ptr=self.tau,
                rng_seed=self._rng_seed,
                step_id_ptr=self._step_id,
                next_state_ptr=self.next_state,
                next_age_ptr=self.next_age,
                next_infectivity_ptr=self.next_infectivity,
                rates_ptr=self.rates,
                N=self.num_nodes,
                STATE_S=self._state_s,
                STATE_E=self._state_e,
                STATE_I=self._state_i,
                STATE_R=self._state_r,
                NODES_PER_BLOCK=self.nodes_per_block,
                LANES_PER_NODE=self.lanes_per_node,
                TRANSMISSION_AGE_DEPENDENT=1 if self._transmission_age_dependent else 0,
            )
        else:
            BLOCK_SIZE = 128
            grid = lambda meta: (triton.cdiv(self.num_nodes, meta["BLOCK_SIZE"]),)
            _flash_renewal_fused_kernel[grid](
                row_ptr_ptr=self.graph.row_ptr,
                col_ind_ptr=self.graph.col_ind,
                weights_ptr=self.graph.weights,
                infectivity_ptr=self.infectivity,
                age_ptr=self.age,
                state_ptr=self.state,
                beta=self._beta,
                mu_ei=self._mu_ei,
                sig_ei=self._sig_ei,
                mu_ir=self._mu_ir,
                sig_ir=self._sig_ir,
                tau_ptr=self.tau,
                rng_seed=self._rng_seed,
                step_id_ptr=self._step_id,
                next_state_ptr=self.next_state,
                next_age_ptr=self.next_age,
                next_infectivity_ptr=self.next_infectivity,
                rates_ptr=self.rates,
                active_nodes_ptr=self._active_nodes_dummy,
                num_active_ptr=self._num_active_dummy,
                N=self.num_nodes,
                STATE_S=self._state_s,
                STATE_E=self._state_e,
                STATE_I=self._state_i,
                STATE_R=self._state_r,
                BLOCK_SIZE=BLOCK_SIZE,
                TRANSMISSION_AGE_DEPENDENT=1 if self._transmission_age_dependent else 0,
                USE_COMPACTION=0,
                MIXED_PRECISION=1 if self._use_mixed_precision else 0,
            )

        # Swap buffers (pointer swap for eager — zero overhead)
        self.state, self.next_state = self.next_state, self.state
        self.age, self.next_age = self.next_age, self.age
        self.infectivity, self.next_infectivity = self.next_infectivity, self.infectivity

        # Step 3: Compute tau for next step from rates
        self._compute_tau()

    def step(self) -> Tuple[float, torch.Tensor]:
        """
        Execute one fused simulation step.

        Returns:
            Tuple of (elapsed_time, current_state).
        """
        tau_before = float(self.tau.item())
        self._step_impl()
        self.current_time += tau_before
        self.total_steps += 1
        return tau_before, self.state

    def simulate_until(self, target_time: float) -> None:
        """Simulate until target time is reached."""
        while self.current_time < target_time:
            self.step()

    def count_by_state(self) -> torch.Tensor:
        """Return counts for each state."""
        return torch.bincount(self.state, minlength=self.model.num_states)

    def count_infected(self) -> int:
        """Return number of nodes in inducer states."""
        return (self.state == self._state_i).sum().item()


class RenewalEngineFusedCUDAGraph(RenewalEngineFused):
    """
    CUDA Graph optimized version of fused renewal engine.

    Captures the full pipeline (infectivity pre-pass + fused Triton kernel
    + tau reduction) as a CUDA Graph for maximum throughput.

    State is snapshot/restored around graph capture so that the simulation
    starts exactly where the user expects (no "time travel" from warmup).
    """

    def __init__(
        self,
        *args,
        steps_per_launch: int = 50,
        use_active_compaction: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.device.type != "cuda":
            raise RuntimeError("RenewalEngineFusedCUDAGraph requires CUDA device")

        # Force even steps_per_launch for double-buffer unrolling
        self.steps_per_launch = int(steps_per_launch)
        if self.steps_per_launch % 2 != 0:
            self.steps_per_launch += 1

        self.step_time_accumulator = torch.zeros(
            1, device=self.device, dtype=torch.float32
        )

        # --------------------------------------------------------------
        # Active-node compaction (Fixed-Grid, Early-Exit pattern)
        # --------------------------------------------------------------
        # Goal: drop the fused kernel's per-step footprint from O(N)
        # blocks to O(|state != R|) blocks, recovering the back-end of
        # the epidemic where R dominates.
        #
        # Predicate is "state != R", NOT "state in E ∪ I": FlashSpread
        # uses a pull-based CSR gather, so S nodes must also evaluate
        # their incoming hazards to transition into E. Dropping S would
        # freeze the epidemic.
        #
        # Refresh cadence is one rebuild per engine.step() call, which
        # aligns exactly with the CUDA Graph replay window (one refresh
        # every steps_per_launch steps). Nodes that transition to R
        # within the replay remain in the list as harmless redundant
        # evaluations until the next refresh; this is mathematically
        # identical to the baseline.
        #
        # CUDA Graph compatibility: we keep the kernel launch grid
        # static at cdiv(N, BLOCK_SIZE) and the buffer pointers static
        # across refreshes. Refresh is an in-place copy into the
        # pre-allocated _active_nodes_buffer (outside the captured
        # graph). The kernel loads num_active and early-exits tail
        # blocks; no recapture, no dynamic shape.
        # --------------------------------------------------------------
        self._use_active_compaction = bool(use_active_compaction)
        if self._use_active_compaction and self.csr_strategy != "thread":
            # Only the thread-path fused kernel currently honours the
            # compaction constexpr. Fall back with an informative
            # message rather than silently ignoring the flag.
            raise ValueError(
                "use_active_compaction=True is currently only wired into "
                "csr_strategy='thread'. Pass csr_strategy='thread' to the "
                "engine or set use_active_compaction=False for this graph."
            )

        # Pre-allocated static buffers. We size the active-nodes
        # buffer to (N + BLOCK_SIZE) so that the fused kernel's last
        # thread block can safely issue a masked load at offsets
        # near N without touching unmapped memory; the tail positions
        # are initialised to 0 so any stray (mask-gated) read returns
        # a deterministic value rather than whatever the allocator
        # happened to leave there. BLOCK_SIZE=128 is the current fused
        # kernel block size; keep in sync if that ever changes.
        _SAFE_PAD = 128
        self._active_nodes_buffer = torch.zeros(
            self.num_nodes + _SAFE_PAD, device=self.device, dtype=torch.int32
        )
        self._active_nodes_buffer[: self.num_nodes] = torch.arange(
            self.num_nodes, device=self.device, dtype=torch.int32
        )
        self._num_active_device = torch.tensor(
            [self.num_nodes], device=self.device, dtype=torch.int32
        )
        # Expose the buffers so the fused-kernel call sites in the
        # inherited base class also see non-dummy pointers. This lets
        # the non-compacting path continue to pass them (with
        # USE_COMPACTION=0 the kernel never reads them).
        self._active_nodes_dummy = self._active_nodes_buffer
        self._num_active_dummy = self._num_active_device

        self.graph_exec = None
        self._capture_graph()

    def _refresh_active_set(self) -> None:
        """Rebuild the active-node list (state != R) into the static buffer.

        Triggers exactly one D2H sync (for .numel()) per CUDA Graph
        replay window; the actual index gather stays on-device via
        torch.nonzero. In-place copy preserves the underlying pointer
        so the captured kernel keeps using the same address.
        """
        if not self._use_active_compaction:
            return
        active_idx = torch.nonzero(
            self.state != self._state_r, as_tuple=False
        ).squeeze(-1).to(torch.int32)
        num = int(active_idx.numel())  # 1 D2H sync per refresh
        if num > 0:
            self._active_nodes_buffer[:num].copy_(active_idx)
        self._num_active_device.fill_(num)

    def _static_step_forward(self) -> None:
        """Step: state/age/infectivity -> next_* (+ copy back for CG)."""
        self.step_time_accumulator.add_(self.tau)
        # No more pre-pass inside the captured graph: the fused kernel
        # writes next_infectivity itself, and the initial infectivity
        # value is populated once in seed_infection() BEFORE capture.
        self._step_id.add_(1)

        if self.csr_strategy == "merge":
            self._pressure_scratch.zero_()

            merge_grid = lambda meta: (
                triton.cdiv(self._num_edges, meta["EDGES_PER_BLOCK"]),
            )
            _pressure_merge_kernel[merge_grid](
                row_ptr_ptr=self.graph.row_ptr,
                col_ind_ptr=self.graph.col_ind,
                weights_ptr=self.graph.weights,
                infectivity_ptr=self.infectivity,
                pressure_ptr=self._pressure_scratch,
                N=self.num_nodes,
                E=self._num_edges,
                EDGES_PER_BLOCK=self.edges_per_merge_block,
                BSEARCH_ITERS=self._bsearch_iters,
            )

            BLOCK_SIZE = 128
            tail_grid = lambda meta: (
                triton.cdiv(self.num_nodes, meta["BLOCK_SIZE"]),
            )
            _flash_renewal_tail_kernel[tail_grid](
                pressure_ptr=self._pressure_scratch,
                age_ptr=self.age,
                state_ptr=self.state,
                beta=self._beta,
                mu_ei=self._mu_ei,
                sig_ei=self._sig_ei,
                mu_ir=self._mu_ir,
                sig_ir=self._sig_ir,
                tau_ptr=self.tau,
                rng_seed=self._rng_seed,
                step_id_ptr=self._step_id,
                next_state_ptr=self.next_state,
                next_age_ptr=self.next_age,
                next_infectivity_ptr=self.next_infectivity,
                rates_ptr=self.rates,
                N=self.num_nodes,
                STATE_S=self._state_s,
                STATE_E=self._state_e,
                STATE_I=self._state_i,
                STATE_R=self._state_r,
                BLOCK_SIZE=BLOCK_SIZE,
                TRANSMISSION_AGE_DEPENDENT=1 if self._transmission_age_dependent else 0,
                MIXED_PRECISION=1 if self._use_mixed_precision else 0,
            )
        elif self.csr_strategy == "warp":
            grid = lambda meta: (
                triton.cdiv(self.num_nodes, meta["NODES_PER_BLOCK"]),
            )
            _flash_renewal_fused_wc_kernel[grid](
                row_ptr_ptr=self.graph.row_ptr,
                col_ind_ptr=self.graph.col_ind,
                weights_ptr=self.graph.weights,
                infectivity_ptr=self.infectivity,
                age_ptr=self.age,
                state_ptr=self.state,
                beta=self._beta,
                mu_ei=self._mu_ei,
                sig_ei=self._sig_ei,
                mu_ir=self._mu_ir,
                sig_ir=self._sig_ir,
                tau_ptr=self.tau,
                rng_seed=self._rng_seed,
                step_id_ptr=self._step_id,
                next_state_ptr=self.next_state,
                next_age_ptr=self.next_age,
                next_infectivity_ptr=self.next_infectivity,
                rates_ptr=self.rates,
                N=self.num_nodes,
                STATE_S=self._state_s,
                STATE_E=self._state_e,
                STATE_I=self._state_i,
                STATE_R=self._state_r,
                NODES_PER_BLOCK=self.nodes_per_block,
                LANES_PER_NODE=self.lanes_per_node,
                TRANSMISSION_AGE_DEPENDENT=1 if self._transmission_age_dependent else 0,
            )
        else:
            BLOCK_SIZE = 128
            grid = lambda meta: (triton.cdiv(self.num_nodes, meta["BLOCK_SIZE"]),)
            _flash_renewal_fused_kernel[grid](
                row_ptr_ptr=self.graph.row_ptr,
                col_ind_ptr=self.graph.col_ind,
                weights_ptr=self.graph.weights,
                infectivity_ptr=self.infectivity,
                age_ptr=self.age,
                state_ptr=self.state,
                beta=self._beta,
                mu_ei=self._mu_ei,
                sig_ei=self._sig_ei,
                mu_ir=self._mu_ir,
                sig_ir=self._sig_ir,
                tau_ptr=self.tau,
                rng_seed=self._rng_seed,
                step_id_ptr=self._step_id,
                next_state_ptr=self.next_state,
                next_age_ptr=self.next_age,
                next_infectivity_ptr=self.next_infectivity,
                rates_ptr=self.rates,
                active_nodes_ptr=self._active_nodes_buffer,
                num_active_ptr=self._num_active_device,
                N=self.num_nodes,
                STATE_S=self._state_s,
                STATE_E=self._state_e,
                STATE_I=self._state_i,
                STATE_R=self._state_r,
                BLOCK_SIZE=BLOCK_SIZE,
                TRANSMISSION_AGE_DEPENDENT=1 if self._transmission_age_dependent else 0,
                USE_COMPACTION=1 if self._use_active_compaction else 0,
                MIXED_PRECISION=1 if self._use_mixed_precision else 0,
            )

        # In CG mode: copy back (can't pointer-swap baked addresses)
        self.state.copy_(self.next_state)
        self.age.copy_(self.next_age)
        self.infectivity.copy_(self.next_infectivity)
        self._compute_tau()

    def _capture_graph(self) -> None:
        """Capture CUDA Graph with snapshot/restore to avoid state mutation."""
        # Snapshot all mutable tensor state
        tensor_snapshot = {}
        snapshot_names = [
            'state', 'age', 'next_state', 'next_age',
            'infectivity', 'next_infectivity',
            'rates', 'tau', '_step_id', 'step_time_accumulator',
        ]
        if self._pressure_scratch is not None:
            snapshot_names.append('_pressure_scratch')
        for name in snapshot_names:
            tensor_snapshot[name] = getattr(self, name).clone()
        saved_time = self.current_time
        saved_steps = self.total_steps

        # Warmup runs (required for Triton JIT + CUDA Graph)
        for _ in range(3):
            self._static_step_forward()
        torch.cuda.synchronize()

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(self.steps_per_launch):
                self._static_step_forward()

        self.graph_exec = g

        # Restore — simulation starts exactly where the user left it
        for name, snap in tensor_snapshot.items():
            getattr(self, name).copy_(snap)
        self.current_time = saved_time
        self.total_steps = saved_steps

    def step(self) -> Tuple[float, torch.Tensor]:
        """Execute steps_per_launch steps via CUDA Graph replay."""
        # Active-node compaction refresh: happens OUTSIDE the captured
        # graph. This is the only place we touch _active_nodes_buffer /
        # _num_active_device; the captured graph reads them at fixed
        # pointers via the USE_COMPACTION constexpr path.
        self._refresh_active_set()

        # Correctness contract for compaction: the fused kernel only
        # writes rate[i] for active i. Inactive (R) entries of the
        # rates buffer would otherwise retain STALE values from the
        # step at which that node last appeared in the active list
        # (when it was still E or I with nonzero rate). The tau
        # reduction `rates.max()` inside the captured graph would
        # then pick up those stale rates, shrink tau, and diverge
        # from the baseline trajectory. Zeroing rates once per replay
        # window (one fast memset, ~10us at N=1e6) pins inactive
        # positions to zero and matches the baseline's invariant
        # that rate[R] = 0 every step.
        if self._use_active_compaction:
            self.rates.zero_()

        self.step_time_accumulator.zero_()
        self.graph_exec.replay()

        elapsed = float(self.step_time_accumulator.item())
        self.current_time += elapsed
        self.total_steps += self.steps_per_launch

        return elapsed, self.state
