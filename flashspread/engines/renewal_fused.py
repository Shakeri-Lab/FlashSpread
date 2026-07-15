"""IO-aware renewal engine built from two globally ordered Triton phases.

The rate phase fuses susceptible-only CSR traversal with current-state hazard
evaluation. A reduction then chooses that same step's adaptive ``tau`` before a
shared transition kernel samples and renews ages. Age-dependent transmission
also writes next infectivity; constant transmission derives it from state.
This ordering enforces the epsilon contract while retaining one graph traversal
and no dense infectivity pre-pass per step.

Age-dependent infectivity is bootstrapped once; constant transmission carries
no per-node shedding buffer, and neither mode has a dense steady-state pre-pass.
Captured execution ping-pongs buffers A->B->A, so an even replay window needs
no full-array copy-back.
"""

import math
import operator
from typing import Tuple
import warnings

import torch

try:
    import triton

    _TRITON_IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001 - keep import working without a GPU
    # Do not let a missing/broken Triton break `import flashspread`; capture
    # the real error so it can be surfaced (chained) if the user actually
    # constructs a GPU engine, instead of a vague "Triton is required".
    triton = None
    _TRITON_IMPORT_ERROR = _exc

from ..core.graph import as_csr
from ..core.host_rng import (
    INITIAL_CONDITION_DOMAIN,
    normalize_seed,
    offset_seed,
    project_seed,
    signed_int64,
)
from ..core.flash_renewal_kernel import (
    _flash_renewal_rate_kernel,
    _flash_renewal_rate_wc_kernel,
    _pressure_merge_kernel,
    _flash_renewal_rate_from_pressure_kernel,
    _flash_renewal_finalize_tau_kernel,
    _flash_renewal_transition_kernel,
)
from ..models.hazards import lognormal_hazard_stable
from ..utils import (
    validate_compartment,
    validate_initial_tensors,
    validate_model_contract,
    validate_population_count,
    validate_fp32_control,
)


# Auto-dispatch starting thresholds on D_max / D_mean. These values predate
# the current globally ordered rate/reduction/transition pipeline and are
# heuristics, not current performance claims. The production acceptance matrix
# measures all three strategies so they can be re-tuned on each target GPU.
_DISPATCH_WARP_RATIO = 4.0
_DISPATCH_MERGE_RATIO = 50.0


def _auto_csr_strategy(row_ptr: torch.Tensor, num_nodes: int) -> str:
    """Pick a CSR traversal strategy from the graph's degree heterogeneity."""
    if num_nodes <= 0:
        raise ValueError("fused renewal engines require a graph with at least one node")
    # Keep the one construction-time temporary int32. Widening all N degrees
    # to fp64 added 8*N avoidable bytes on the largest supported graphs.
    degrees = row_ptr[1:] - row_ptr[:-1]
    d_mean = float(row_ptr[-1].item()) / num_nodes
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

    Combines source-node infectivity with a rate/traversal kernel and a shared
    transition kernel separated by the adaptive-tau reduction required for
    current-rate correctness.

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
        edges_per_merge_block: int = 1024,
        # Deprecated: prefer csr_strategy="warp" over warp_collaborative=True.
        warp_collaborative: bool = False,
    ):
        if triton is None:
            raise RuntimeError(
                "Triton is required for RenewalEngineFused. Install the GPU "
                "extra with `pip install flashspread[gpu]` on a CUDA machine."
            ) from _TRITON_IMPORT_ERROR
        for name, value in {
            "bf16_weights": bf16_weights,
            "use_mixed_precision": use_mixed_precision,
            "warp_collaborative": warp_collaborative,
        }.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        if warp_collaborative:
            warnings.warn(
                "warp_collaborative is deprecated; use csr_strategy='warp'",
                DeprecationWarning,
                stacklevel=2,
            )

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("RenewalEngineFused requires a CUDA device")
        self.model = model
        from ..config import supports_fused_renewal

        if not supports_fused_renewal(model):
            raise TypeError(
                "RenewalEngineFused requires the exact, unmodified built-in "
                "SEIRModel; custom models must use the reference backend"
            )
        _, model_inducers = validate_model_contract(
            model,
            markovian=False,
            methods=("prepare", "compute_rates", "apply_transitions"),
        )
        self.epsilon = validate_fp32_control(
            "epsilon", epsilon, positive=True
        )
        self.tau_max = validate_fp32_control(
            "tau_max", tau_max, positive=True
        )
        required_model = (
            "susceptible",
            "exposed",
            "infected",
            "recovered",
            "num_states",
            "beta",
            "prepare",
        )
        missing = [name for name in required_model if not hasattr(model, name)]
        if missing:
            raise TypeError(
                "the fused backend requires an ordered SEIR-style model; "
                f"missing {', '.join(missing)}"
            )
        raw_state_ids = (
            model.susceptible,
            model.exposed,
            model.infected,
            model.recovered,
        )
        if any(isinstance(value, bool) for value in raw_state_ids):
            raise TypeError("fused SEIR state indices must be integers")
        try:
            state_ids = tuple(operator.index(value) for value in raw_state_ids)
        except TypeError as exc:
            raise TypeError("fused SEIR state indices must be integers") from exc
        if len(set(state_ids)) != 4 or any(
            value < 0 or value >= int(model.num_states) for value in state_ids
        ):
            raise ValueError(
                "fused SEIR state indices must be distinct and lie within num_states"
            )
        if set(model_inducers) != {state_ids[2]}:
            raise ValueError(
                "the fused SEIR backend requires infected as its sole inducer state"
            )
        transmission_mode = getattr(model, "transmission_mode", "constant")
        if transmission_mode not in ("constant", "age_dependent"):
            raise ValueError(
                "transmission_mode must be 'constant' or 'age_dependent'"
            )
        raw_beta = model.beta
        if isinstance(raw_beta, torch.Tensor):
            if raw_beta.numel() != 1:
                raise TypeError("fused model beta must be a numeric scalar")
            raw_beta = raw_beta.item()
        # This mode is a kernel constexpr.  Freeze it at construction so
        # storage and every compiled specialization obey the same contract.
        self._transmission_age_dependent = transmission_mode == "age_dependent"
        self._beta = validate_fp32_control(
            "model.beta", raw_beta, nonnegative=True
        )

        # Validate step-control parameters (epsilon<=0 -> tau collapses to
        # 0 and the simulation clock freezes).
        if not math.isfinite(self.epsilon) or not (self.epsilon > 0.0):
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not math.isfinite(self.tau_max) or not (self.tau_max > 0.0):
            raise ValueError(f"tau_max must be > 0, got {self.tau_max}")

        # CSR-traversal strategy controls how the kernel iterates the
        # incoming-edge list:
        #   "thread": 1 thread per node, scalar loop over neighbors;
        #             intended for uniform-degree graphs (regular, ER).
        #   "warp":   32 threads per node, chunked loop; intended for
        #             moderately heavy-tailed distributions. Hub traversal
        #             scales as O(D_max / 32) inside the node's warp.
        #   "merge":  Edge-partitioned; each block processes a fixed
        #             chunk of edges, recovers the source node via
        #             binary search in row_ptr, and atomic-adds to a
        #             pressure buffer. Delivers perfect load balance
        #             on extremely heavy tails (scale-free with hubs).
        #   "auto":   Choose at construction time from D_max / D_mean.
        try:
            self.nodes_per_block = operator.index(nodes_per_block)
            self.lanes_per_node = operator.index(lanes_per_node)
            self.edges_per_merge_block = operator.index(edges_per_merge_block)
        except TypeError as exc:
            raise TypeError("CSR traversal sizes must be integers") from exc
        if any(
            isinstance(value, bool)
            for value in (nodes_per_block, lanes_per_node, edges_per_merge_block)
        ):
            raise TypeError("CSR traversal sizes must be integers, not bool")
        if self.nodes_per_block <= 0 or self.nodes_per_block & (self.nodes_per_block - 1):
            raise ValueError("nodes_per_block must be a positive power of two")
        if (
            self.lanes_per_node <= 0
            or self.lanes_per_node > 32
            or self.lanes_per_node & (self.lanes_per_node - 1)
        ):
            raise ValueError("lanes_per_node must be a power of two in [1, 32]")
        if self.nodes_per_block * self.lanes_per_node > 1024:
            raise ValueError("nodes_per_block * lanes_per_node must be <= 1024")
        if (
            self.edges_per_merge_block <= 0
            or self.edges_per_merge_block & (self.edges_per_merge_block - 1)
        ):
            raise ValueError("edges_per_merge_block must be a positive power of two")

        self.graph = as_csr(graph, self.device)
        # GraphCSR resolves an unindexed ``cuda`` request to concrete storage
        # (for example cuda:0). Retain that device so later context managers
        # cannot follow a changed process-wide current device.
        self.device = self.graph.row_ptr.device

        self.num_nodes = self.graph.num_nodes
        if self.num_nodes <= 0:
            raise ValueError("RenewalEngineFused requires a non-empty graph")

        # Mixed storage: state int8 and age-dependent infectivity bf16 while
        # age remains fp32 (accumulator/math also fp32). Storing age in fp16
        # can round age + tau back to the same value on high-rate hubs, freezing
        # renewal clocks while simulation time advances, so it is deliberately
        # excluded from the compact path. Mixed mode auto-enables
        # bf16 weights so the whole CSR traversal runs in reduced
        # precision; the kernel promotes every load to fp32/int32
        # before any arithmetic via the MIXED_PRECISION constexpr.
        self._use_mixed_precision = bool(use_mixed_precision)
        if self._use_mixed_precision:
            bf16_weights = True  # pull the existing weights-only path too

        if bf16_weights and hasattr(self.graph, 'to_bf16_weights'):
            self.graph = self.graph.to_bf16_weights()
        self._graph_signature = self.graph._mutation_signature()

        # Prepare model parameters on device
        if hasattr(self.model, "prepare"):
            self.model.prepare(self.device)

        # State tensors (read/write by fused kernel).
        # Dtypes: baseline uses int32/fp32 for state/age/infectivity;
        # mixed-storage mode drops state/infectivity to int8/bf16 and retains
        # fp32 age. The kernel's MIXED_PRECISION constexpr
        # path handles the promote-on-load / cast-on-store contract;
        # the accumulator (pressure) and all math intermediates are
        # always fp32.
        _state_dt = torch.int8 if self._use_mixed_precision else torch.int32
        _age_dt = torch.float32
        _inf_dt = torch.bfloat16 if self._use_mixed_precision else torch.float32
        self._inf_dtype = _inf_dt
        self.state = torch.zeros(self.num_nodes, device=self.device, dtype=_state_dt)
        self.age = torch.zeros(self.num_nodes, device=self.device, dtype=_age_dt)

        # Double-buffered: kernel writes to next_*, then we swap
        self.next_state = torch.zeros(self.num_nodes, device=self.device, dtype=_state_dt)
        self.next_age = torch.zeros(self.num_nodes, device=self.device, dtype=_age_dt)

        # Age-dependent shedding needs a gathered per-node payload.  Constant
        # transmission gathers source state and beta directly, so two scalar
        # dummies preserve one static kernel signature without carrying or
        # writing 2*N redundant infectivity values.
        infectivity_size = self.num_nodes if self._transmission_age_dependent else 1
        self._infectivity = torch.zeros(
            infectivity_size, device=self.device, dtype=_inf_dt
        )
        self._next_infectivity = torch.zeros_like(self._infectivity)

        # Rates are the sole O(N) intermediate. A global barrier between their
        # production and transition sampling is required for a correct adaptive
        # step; all traversal pressure stays in registers.
        self.rates = torch.zeros(self.num_nodes, device=self.device, dtype=torch.float32)

        # Resolve CSR strategy now that row_ptr is available.
        requested_strategy = str(csr_strategy).lower()
        if warp_collaborative and requested_strategy not in ("auto", "thread", "warp"):
            raise ValueError(
                "warp_collaborative=True conflicts with "
                f"csr_strategy={csr_strategy!r}"
            )
        if warp_collaborative:
            strategy = "warp"
        elif requested_strategy == "auto":
            strategy = _auto_csr_strategy(self.graph.row_ptr, self.num_nodes)
        else:
            strategy = requested_strategy
        if strategy not in ("thread", "warp", "merge"):
            raise ValueError(
                f"csr_strategy must be one of 'auto'/'thread'/'warp'/'merge'; "
                f"got {csr_strategy!r}"
            )
        self.csr_strategy = strategy
        # Legacy flag (some benchmarks still read it).
        self.warp_collaborative = strategy == "warp"

        # All traversal strategies promote compact state/age/infectivity and
        # weights before arithmetic. Thread/warp pressure stays in registers;
        # merge uses an explicit fp32 scratch array so hub atomics do not
        # absorb small bf16 contributions.

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

        # Current-step tau, computed from current-state rates before sampling.
        self.tau = torch.tensor([self.tau_max], device=self.device, dtype=torch.float32)
        self._max_rate = torch.zeros((), device=self.device, dtype=torch.float32)
        rate_nodes_per_program = (
            self.nodes_per_block if self.csr_strategy == "warp" else 128
        )
        self._max_rate_partials = torch.zeros(
            (self.num_nodes + rate_nodes_per_program - 1)
            // rate_nodes_per_program,
            device=self.device,
            dtype=torch.float32,
        )

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

        # RNG: a 64-bit base seed and step counter are mixed into each step key.
        self._base_seed = normalize_seed(seed)
        # Device-resident so CUDA Graph replay observes reset(episode=...)
        # without recapture. Passing a Python int bakes the seed into capture.
        self._rng_seed = torch.tensor(
            [signed_int64(self._base_seed)], device=self.device, dtype=torch.int64
        )
        self._step_id = torch.zeros(1, device=self.device, dtype=torch.int64)

        # Dedicated generator for reproducible initial-condition sampling.
        self._init_gen = torch.Generator(device=self.device)
        self._init_gen.manual_seed(
            project_seed(self._base_seed, INITIAL_CONDITION_DOMAIN)
        )

        # Simulation state
        self.current_time = 0.0
        self.total_steps = 0

        # SEIR state indices
        self._state_s, self._state_e, self._state_i, self._state_r = state_ids

        # Model parameters (on device)
        self._mu_ei = None
        self._sig_ei = None
        self._mu_ir = None
        self._sig_ir = None
        self._prepare_model_params()

    def _constant_infectivity_snapshot(self, state: torch.Tensor) -> torch.Tensor:
        """Materialize constant shedding only for diagnostic compatibility."""
        return torch.where(state == self._state_i, self._beta, 0.0).to(
            dtype=self._inf_dtype
        )

    @property
    def infectivity(self) -> torch.Tensor:
        """Current shedding values; constant-mode values are a derived snapshot."""
        if self._transmission_age_dependent:
            return self._infectivity
        return self._constant_infectivity_snapshot(self.state)

    @property
    def next_infectivity(self) -> torch.Tensor:
        """Alternate shedding values; constant-mode values are derived on access."""
        if self._transmission_age_dependent:
            return self._next_infectivity
        return self._constant_infectivity_snapshot(self.next_state)

    def reseed(self, seed: int) -> None:
        """Reseed device-visible transition and initial-condition streams."""
        self._base_seed = normalize_seed(seed)
        self._rng_seed.fill_(signed_int64(self._base_seed))
        self._step_id.zero_()
        self._init_gen.manual_seed(
            project_seed(self._base_seed, INITIAL_CONDITION_DOMAIN)
        )

    def _prepare_model_params(self):
        """Extract lognormal parameters from model."""
        names = ("_mu_ei", "_sig_ei", "_mu_ir", "_sig_ir")
        missing = [name for name in names if not hasattr(self.model, name)]
        if missing:
            raise TypeError(
                "the fused backend requires prepared log-normal parameters; "
                f"missing {', '.join(missing)}"
            )
        values = {}
        for name in names:
            raw = getattr(self.model, name)
            if isinstance(raw, torch.Tensor):
                if raw.numel() != 1:
                    raise TypeError(
                        f"prepared fused parameter {name} must be a scalar"
                    )
                raw = raw.item()
            values[name] = validate_fp32_control(
                f"model.{name}",
                raw,
                positive=name.startswith("_sig"),
            )
        self._mu_ei = values["_mu_ei"]
        self._sig_ei = values["_sig_ei"]
        self._mu_ir = values["_mu_ir"]
        self._sig_ir = values["_sig_ir"]

    def reset(self, episode: int | None = None) -> None:
        """Reset all simulation state for clean re-use (e.g., RL episodes).

        Args:
            episode: If given, shift the effective RNG seed by ``episode``
                so successive episodes are statistically independent.
                Otherwise reset to the base seed (reproduces the first run).

        """
        eff_seed = offset_seed(
            self._base_seed,
            episode if episode is not None else 0,
            name="episode",
        )
        self.state.zero_()
        self.age.zero_()
        self.next_state.zero_()
        self.next_age.zero_()
        self._infectivity.zero_()
        self._next_infectivity.zero_()
        self.rates.zero_()
        self._max_rate.zero_()
        self._max_rate_partials.zero_()
        if self._pressure_scratch is not None:
            self._pressure_scratch.zero_()
        self.tau.fill_(self.tau_max)
        self._step_id.zero_()
        self.current_time = 0.0
        self.total_steps = 0

        self._rng_seed.fill_(signed_int64(eff_seed))
        self._init_gen.manual_seed(
            project_seed(eff_seed, INITIAL_CONDITION_DOMAIN)
        )

    def seed_infection(self, num_infected: int, state: int = None) -> None:
        """Randomly seed initial infections."""
        if state is None:
            state = self._state_e  # default: Exposed for SEIR
        state = validate_compartment(state, self.model.num_states)
        num_infected = validate_population_count(num_infected, self.num_nodes)
        indices = torch.randperm(
            self.num_nodes, device=self.device, generator=self._init_gen
        )[:num_infected]
        self.state[indices] = state
        self.age[indices] = 0.0
        # Bootstrap infectivity from the seeded state so the first kernel
        # launch reads a correct value; subsequent steps maintain it
        # in-kernel.
        self._infectivity_prepass()

    def set_initial_state(
        self,
        initial_state: torch.Tensor,
        initial_age: torch.Tensor | None = None,
    ) -> None:
        """Initialize state/ages and bootstrap first-step infectivity."""
        state, age = validate_initial_tensors(
            initial_state,
            num_nodes=self.num_nodes,
            num_states=self.model.num_states,
            device=self.device,
            initial_age=initial_age,
        )
        self.state.copy_(state)
        if age is None:
            self.age.zero_()
        else:
            self.age.copy_(age)
        self._infectivity_prepass()
        # Keep both sides of the captured ping-pong coherent. In particular,
        # recovered nodes may be omitted by the compacted rate pass; a stale
        # infectious value in the alternate buffer would create phantom pressure.
        self.next_state.copy_(self.state)
        self.next_age.copy_(self.age)
        if self._transmission_age_dependent:
            self._next_infectivity.copy_(self._infectivity)

    def _infectivity_prepass(self):
        """Compute infectivity based on model's transmission_mode."""
        i_mask = self.state == self._state_i

        if self._transmission_age_dependent:
            hazard_all = lognormal_hazard_stable(
                torch.clamp(self.age, min=1e-10),
                self.model._mu_ir,
                self.model._sig_ir,
            )
            self._infectivity.copy_(
                torch.where(i_mask, self.model._beta_t * hazard_all, 0.0)
            )
        else:
            # The rate kernel gathers source state and applies beta directly.
            self._infectivity.zero_()

    def _compute_tau(self, *, elapsed_ptr: torch.Tensor | None = None) -> None:
        """Choose current tau and advance step metadata in two GPU nodes.

        A global max is the required ordering barrier after rate evaluation.
        Each rate program has already emitted one maximum, so this reduction
        reads only the compact partial array instead of rereading all public
        rates. One scalar Triton program then performs all remaining tau
        arithmetic, advances the device RNG step id, and optionally accumulates
        CUDA-Graph replay time.
        """
        torch.max(self._max_rate_partials, out=self._max_rate)
        _flash_renewal_finalize_tau_kernel[(1,)](
            max_rate_ptr=self._max_rate,
            tau_ptr=self.tau,
            # Compile-time dead when elapsed_ptr is omitted; reusing tau avoids
            # carrying another dummy tensor in the eager engine.
            elapsed_ptr=self.tau if elapsed_ptr is None else elapsed_ptr,
            step_id_ptr=self._step_id,
            epsilon=self.epsilon,
            tau_max=self.tau_max,
            ACCUMULATE_TIME=1 if elapsed_ptr is not None else 0,
        )

    def _launch_rates(
        self,
        state,
        age,
        infectivity,
        *,
        use_compaction: bool,
    ) -> None:
        """Fuse one configured graph traversal with current-rate evaluation.

        Binding the traversal strategy at construction keeps the hot kernels
        specialized, while this one host dispatcher is shared by eager and
        captured execution.
        """
        if self.csr_strategy == "merge":
            self._pressure_scratch.zero_()
            if self._num_edges:
                merge_grid = (
                    triton.cdiv(self._num_edges, self.edges_per_merge_block),
                )
                _pressure_merge_kernel[merge_grid](
                    row_ptr_ptr=self.graph.row_ptr,
                    col_ind_ptr=self.graph.col_ind,
                    weights_ptr=self.graph.weights_storage,
                    infectivity_ptr=infectivity,
                    state_ptr=state,
                    pressure_ptr=self._pressure_scratch,
                    beta=self._beta,
                    N=self.num_nodes,
                    E=self._num_edges,
                    STATE_S=self._state_s,
                    STATE_I=self._state_i,
                    HAS_WEIGHTS=self.graph.has_weights,
                    TRANSMISSION_AGE_DEPENDENT=(
                        1 if self._transmission_age_dependent else 0
                    ),
                    EDGES_PER_BLOCK=self.edges_per_merge_block,
                    BSEARCH_ITERS=self._bsearch_iters,
                )

            BLOCK_SIZE = 128
            grid = (triton.cdiv(self.num_nodes, BLOCK_SIZE),)
            _flash_renewal_rate_from_pressure_kernel[grid](
                pressure_ptr=self._pressure_scratch,
                age_ptr=age,
                state_ptr=state,
                mu_ei=self._mu_ei,
                sig_ei=self._sig_ei,
                mu_ir=self._mu_ir,
                sig_ir=self._sig_ir,
                rates_ptr=self.rates,
                max_rate_partials_ptr=self._max_rate_partials,
                N=self.num_nodes,
                STATE_S=self._state_s,
                STATE_E=self._state_e,
                STATE_I=self._state_i,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        elif self.csr_strategy == "warp":
            grid = (triton.cdiv(self.num_nodes, self.nodes_per_block),)
            _flash_renewal_rate_wc_kernel[grid](
                row_ptr_ptr=self.graph.row_ptr,
                col_ind_ptr=self.graph.col_ind,
                weights_ptr=self.graph.weights_storage,
                infectivity_ptr=infectivity,
                age_ptr=age,
                state_ptr=state,
                beta=self._beta,
                mu_ei=self._mu_ei,
                sig_ei=self._sig_ei,
                mu_ir=self._mu_ir,
                sig_ir=self._sig_ir,
                rates_ptr=self.rates,
                max_rate_partials_ptr=self._max_rate_partials,
                N=self.num_nodes,
                STATE_S=self._state_s,
                STATE_E=self._state_e,
                STATE_I=self._state_i,
                NODES_PER_BLOCK=self.nodes_per_block,
                LANES_PER_NODE=self.lanes_per_node,
                HAS_WEIGHTS=self.graph.has_weights,
                TRANSMISSION_AGE_DEPENDENT=(
                    1 if self._transmission_age_dependent else 0
                ),
            )
        else:
            BLOCK_SIZE = 128
            grid = (triton.cdiv(self.num_nodes, BLOCK_SIZE),)
            _flash_renewal_rate_kernel[grid](
                row_ptr_ptr=self.graph.row_ptr,
                col_ind_ptr=self.graph.col_ind,
                weights_ptr=self.graph.weights_storage,
                infectivity_ptr=infectivity,
                age_ptr=age,
                state_ptr=state,
                beta=self._beta,
                mu_ei=self._mu_ei,
                sig_ei=self._sig_ei,
                mu_ir=self._mu_ir,
                sig_ir=self._sig_ir,
                rates_ptr=self.rates,
                max_rate_partials_ptr=self._max_rate_partials,
                active_nodes_ptr=self._active_nodes_dummy,
                num_active_ptr=self._num_active_dummy,
                N=self.num_nodes,
                STATE_S=self._state_s,
                STATE_E=self._state_e,
                STATE_I=self._state_i,
                HAS_WEIGHTS=self.graph.has_weights,
                TRANSMISSION_AGE_DEPENDENT=(
                    1 if self._transmission_age_dependent else 0
                ),
                USE_COMPACTION=1 if use_compaction else 0,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    def _launch_transition(
        self,
        state,
        age,
        next_state,
        next_age,
        next_infectivity,
    ) -> None:
        """Sample current rates and write a complete next-node buffer."""
        BLOCK_SIZE = 128
        grid = (triton.cdiv(self.num_nodes, BLOCK_SIZE),)
        _flash_renewal_transition_kernel[grid](
            age_ptr=age,
            state_ptr=state,
            rates_ptr=self.rates,
            beta=self._beta,
            mu_ir=self._mu_ir,
            sig_ir=self._sig_ir,
            tau_ptr=self.tau,
            rng_seed_ptr=self._rng_seed,
            step_id_ptr=self._step_id,
            next_state_ptr=next_state,
            next_age_ptr=next_age,
            next_infectivity_ptr=next_infectivity,
            N=self.num_nodes,
            STATE_S=self._state_s,
            STATE_E=self._state_e,
            STATE_I=self._state_i,
            STATE_R=self._state_r,
            TRANSMISSION_AGE_DEPENDENT=(
                1 if self._transmission_age_dependent else 0
            ),
            MIXED_PRECISION=1 if self._use_mixed_precision else 0,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    def _step_impl(self) -> torch.Tensor:
        """Execute one eager, current-rate adaptive step."""
        self.graph._assert_unchanged(
            self._graph_signature, owner=type(self).__name__
        )
        self._launch_rates(
            self.state,
            self.age,
            self._infectivity,
            use_compaction=False,
        )
        self._compute_tau()
        self._launch_transition(
            self.state,
            self.age,
            self.next_state,
            self.next_age,
            self._next_infectivity,
        )

        # Swap buffers (pointer swap for eager — zero overhead)
        self.state, self.next_state = self.next_state, self.state
        self.age, self.next_age = self.next_age, self.age
        self._infectivity, self._next_infectivity = (
            self._next_infectivity,
            self._infectivity,
        )
        return self.tau

    def step(self) -> Tuple[float, torch.Tensor]:
        """
        Execute one fused simulation step.

        Returns:
            Tuple of (elapsed_time, current_state).
        """
        with torch.cuda.device(self.device):
            tau = float(self._step_impl().item())
        if not math.isfinite(tau) or tau <= 0.0:
            raise FloatingPointError(
                "Fused renewal tau is non-finite or non-positive; check model "
                "parameters, edge weights, ages, and aggregate weighted degree"
            )
        self.current_time += tau
        self.total_steps += 1
        return tau, self.state

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

    Captures rate evaluation, tau reduction, and transition as a CUDA Graph.

    Construction-time warmup state is reset after graph capture so the first
    user-supplied initial condition starts from a clean RNG and buffer state.
    """

    def __init__(
        self,
        *args,
        steps_per_launch: int = 50,
        use_active_compaction: bool = False,
        **kwargs,
    ):
        if not isinstance(use_active_compaction, bool):
            raise TypeError("use_active_compaction must be a bool")
        super().__init__(*args, **kwargs)

        # Force even steps_per_launch for double-buffer unrolling
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
        if self.steps_per_launch % 2 != 0:
            self.steps_per_launch += 1
        if self.steps_per_launch * self.tau_max > torch.finfo(torch.float64).max:
            raise ValueError(
                "rounded steps_per_launch * tau_max must fit in the fp64 "
                "CUDA Graph elapsed-time accumulator"
            )

        self.step_time_accumulator = torch.zeros(
            1, device=self.device, dtype=torch.float64
        )

        # --------------------------------------------------------------
        # Active-node compaction (Fixed-Grid, Early-Exit pattern)
        # --------------------------------------------------------------
        # Goal: mask node work and memory operations for recovered nodes,
        # recovering the back-end of the epidemic where R dominates.
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

        self._active_nodes_buffer = None
        self._num_active_device = None
        if self._use_active_compaction:
            # Size the active-node buffer to N + BLOCK_SIZE so the last fused
            # block can issue a masked load without touching unmapped memory.
            # The base class's two scalar dummies remain in place when
            # compaction is off; allocating this array unconditionally used to
            # waste 4*N persistent bytes on the default path.
            _SAFE_PAD = 128
            self._active_nodes_buffer = torch.zeros(
                self.num_nodes + _SAFE_PAD,
                device=self.device,
                dtype=torch.int32,
            )
            # Write directly into the static buffer instead of constructing a
            # second N-element arange allocation during engine startup.
            torch.arange(
                self.num_nodes,
                device=self.device,
                dtype=torch.int32,
                out=self._active_nodes_buffer[: self.num_nodes],
            )
            self._num_active_device = torch.tensor(
                [self.num_nodes], device=self.device, dtype=torch.int32
            )
            self._active_nodes_dummy = self._active_nodes_buffer
            self._num_active_dummy = self._num_active_device

        self.graph_exec = None
        self._capture_graph()

    def _refresh_active_set(self) -> None:
        """Rebuild the active-node list (state != R) into the static buffer.

        ``torch.nonzero`` determines a dynamic output shape and therefore
        triggers one D2H synchronization per CUDA Graph replay window. The
        index gather and in-place copy stay on-device, preserving the pointer
        captured by the graph.
        """
        if not self._use_active_compaction:
            return
        active_idx = torch.nonzero(
            self.state != self._state_r, as_tuple=False
        ).squeeze(-1).to(torch.int32)
        num = int(active_idx.numel())
        if num > 0:
            self._active_nodes_buffer[:num].copy_(active_idx)
        self._num_active_device.fill_(num)

    def _static_step_between(
        self,
        state,
        age,
        infectivity,
        next_state,
        next_age,
        next_infectivity,
    ) -> None:
        """Captured step between explicit ping-pong buffers."""
        self._launch_rates(
            state,
            age,
            infectivity,
            use_compaction=self._use_active_compaction,
        )
        self._compute_tau(elapsed_ptr=self.step_time_accumulator)
        self._launch_transition(
            state,
            age,
            next_state,
            next_age,
            next_infectivity,
        )

    def _capture_graph(self) -> None:
        """Capture the construction-time graph and restore the clean state.

        This private method is called only by ``__init__``, before callers can
        supply an initial condition. Avoiding an O(N) snapshot here removes a
        20--32*N byte construction-time peak; ``reset`` restores every mutable
        simulation tensor after warmup/capture instead.
        """
        # Warm and capture on the engine's actual device rather than relying on
        # the process-wide ambient CUDA device (important for cuda:1+).
        with torch.cuda.device(self.device):
            # Warm both directions of the ping-pong pair. Each pair returns the
            # current state to the public A buffers, matching replay boundaries.
            for _ in range(3):
                self._static_step_between(
                    self.state,
                    self.age,
                    self._infectivity,
                    self.next_state,
                    self.next_age,
                    self._next_infectivity,
                )
                self._static_step_between(
                    self.next_state,
                    self.next_age,
                    self._next_infectivity,
                    self.state,
                    self.age,
                    self._infectivity,
                )
            torch.cuda.synchronize(self.device)

            # Capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(self.steps_per_launch // 2):
                    self._static_step_between(
                        self.state,
                        self.age,
                        self._infectivity,
                        self.next_state,
                        self.next_age,
                        self._next_infectivity,
                    )
                    self._static_step_between(
                        self.next_state,
                        self.next_age,
                        self._next_infectivity,
                        self.state,
                        self.age,
                        self._infectivity,
                    )

            self.graph_exec = g

            # Construction began from the engine's canonical empty state.
            # Restore it without cloning dense arrays before capture.
            self.reset()
            self.step_time_accumulator.zero_()

    def step(self) -> Tuple[float, torch.Tensor]:
        """Execute steps_per_launch steps via CUDA Graph replay."""
        self.graph._assert_unchanged(
            self._graph_signature, owner=type(self).__name__
        )
        # Active-node compaction refresh: happens OUTSIDE the captured
        # graph. This is the only place we touch _active_nodes_buffer /
        # _num_active_device; the captured graph reads them at fixed
        # pointers via the USE_COMPACTION constexpr path.
        self._refresh_active_set()

        # Correctness contract for compaction: the rate kernel only
        # writes rate[i] for active i. Inactive (R) entries of the
        # rates buffer would otherwise retain STALE values from the
        # step at which that node last appeared in the active list
        # (when it was still E or I with nonzero rate). Compact max partials
        # exclude inactive entries, but public ``rates`` remains authoritative;
        # zeroing once per replay window pins every inactive position to its
        # current absorbing-state rate. The transition phase still writes every
        # node into the alternate buffer, so compacted ping-pong execution
        # cannot retain stale infectious state or infectivity.
        if self._use_active_compaction:
            self.rates.zero_()

        with torch.cuda.device(self.device):
            self.step_time_accumulator.zero_()
            self.graph_exec.replay()
            elapsed = float(self.step_time_accumulator.item())
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise FloatingPointError(
                "Fused renewal CUDA Graph elapsed time is non-finite or "
                "non-positive; check model parameters, edge weights, ages, "
                "and aggregate weighted degree"
            )
        self.current_time += elapsed
        self.total_steps += self.steps_per_launch

        return elapsed, self.state
