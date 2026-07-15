"""Independent trajectories sharing one graph and node-major state layout.

``ReferenceEnsembleEngine`` establishes the ensemble semantics used by the GPU
fast path: stateful tensors are node-major ``[N, R]``, every replica has its own
adaptive time step and random stream, and the incoming CSR is stored exactly
once.  The pure-PyTorch graph gather materializes an ``[E, R]`` temporary, so
this class is a reference/fallback rather than the throughput implementation.
``EnsembleEngine`` preserves that stochastic contract and broadcasts CSR
metadata across replica lanes. For the exact built-in constant-transmission
SEIR model it executes a specialized multi-kernel step: the tiled graph kernel
evaluates current renewal rates without a dense pressure round trip and reads
one packed infectious-state word per 32 replica lanes, a device reduction and
Triton finalizer select each replica's time step, and a tiled Triton transition
kernel advances state and age while maintaining that bitmap. Generic models
keep the separate tiled pressure and reference model/transition phases.
"""

from __future__ import annotations

import copy
import math
import operator

import torch

from ..core.ensemble_reference import (
    reference_ensemble_infectivity_csr,
    reference_ensemble_influence_csr,
)
from ..core.graph import as_csr
from ..core.host_rng import (
    UINT64_MASK,
    _fill_splitmix_counter_,
    _splitmix_uniform_,
    normalize_seed,
    offset_seed,
    project_seed,
    signed_int64,
    splitmix64_word,
)
from ..utils import (
    is_markovian,
    validate_compartment,
    validate_fp32_control,
    validate_model_contract,
    validate_population_count,
)


_INIT_DOMAIN = 0x243F6A8885A308D3
_EVENT_DOMAIN = 0x13198A2E03707344


def _replica_seed(base_seed: int, replica: int, domain: int) -> int:
    """Derive a stable full-width torch.Generator seed for one stream domain."""
    word = (base_seed & UINT64_MASK) ^ domain
    word ^= (int(replica) * 0x9E3779B97F4A7C15) & UINT64_MASK
    return splitmix64_word(word)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _has_builtin_seir_contract(model) -> bool:
    """Recognize the exact, unshadowed built-in fixed-state SEIR contract."""
    from ..config import supports_fused_renewal

    # Scalar and ensemble kernels hard-code the same built-in equations. Keep
    # one eligibility predicate so subclass, instance-shadow, and class-level
    # monkeypatch handling cannot drift between the two fast paths.
    return supports_fused_renewal(model)


def _supports_builtin_seir_rate_fusion(model) -> bool:
    """Require the built-in contract with prepared device scalar parameters."""
    if not _has_builtin_seir_contract(model):
        return False
    return all(
        isinstance(getattr(model, name, None), torch.Tensor)
        for name in ("_beta_t", "_mu_ei", "_sig_ei", "_mu_ir", "_sig_ir")
    )


class ReferenceEnsembleEngine:
    """Independent Markovian or renewal trajectories over one shared CSR.

    Args:
        graph: Canonical :class:`~flashspread.core.graph.GraphCSR` (or wrapper
            exposing ``.csr``). It is retained once, never copied per replica.
        model: A model accepted by the ordinary Markovian or renewal engine.
        replicas: Number of statistically independent trajectories.
        device: Tensor device. CPU is the intended correctness/fallback path.
        seed: Base seed. Replica and stream-domain ids are mixed into it.
        epsilon: Renewal adaptive-step bound.
        max_prob, theta, tau_min: Markovian adaptive-step controls.
        tau_max: Maximum step for either family.

    Notes:
        Authoritative node tensors use ``[N, R]`` layout. The ordinary
        ``full`` storage profile also keeps pressure, next-state, probability,
        mask, and random scratch in that layout. The specialized
        ``fused_seir`` profile replaces those dense compatibility buffers with
        one derived int32 infectious bitmap of shape
        ``[N, ceil(R / 32)]``: one bit per node-replica, rounded to whole
        replica words. ``tau`` and ``current_time`` are ``[R]`` because coupling
        all replicas to one global maximum rate would change their stochastic
        processes. Tau is fp32 like device rates; elapsed clocks accumulate in
        fp64 so small adaptive steps do not stop advancing at long horizons.
        Renewal ages remain fp32 for the hot path; simulations whose physical
        time approaches fp32 range should rescale their time unit.
    """

    def __init__(
        self,
        graph,
        model,
        replicas: int,
        *,
        device: str | torch.device | None = None,
        seed: int = 12345,
        epsilon: float = 0.03,
        max_prob: float = 0.1,
        theta: float = 0.01,
        tau_min: float = 1e-6,
        tau_max: float = 1.0,
        _storage_profile: str = "full",
    ):
        self.replicas = _positive_integer("replicas", replicas)
        graph_device = (graph.csr if hasattr(graph, "csr") else graph).device
        requested_device = torch.device(graph_device if device is None else device)
        self.graph = as_csr(graph, requested_device)
        # ``torch.device("cuda")`` is intentionally unindexed until an
        # allocation or transfer resolves it. The canonical CSR has completed
        # that transfer, so use its physical tensor device for every engine
        # allocation and public device attribute.
        self.device = self.graph.row_ptr.device
        self._graph_signature = self.graph._mutation_signature()
        self.num_nodes = int(self.graph.num_nodes)
        if self.num_nodes <= 0:
            raise ValueError("ensemble simulation requires a non-empty graph")

        self.model = copy.copy(model)
        self._markovian = is_markovian(self.model)
        required_methods = (
            ("prepare", "compute_rates", "apply_transitions")
            if self._markovian
            else ("compute_rates", "apply_transitions")
        )
        self.num_states, self.inducer_states = validate_model_contract(
            self.model,
            markovian=self._markovian,
            methods=required_methods,
        )
        if hasattr(self.model, "prepare"):
            self.model.prepare(self.device)

        if _storage_profile not in ("full", "fused_seir"):
            raise ValueError("_storage_profile must be either 'full' or 'fused_seir'")
        if _storage_profile == "fused_seir" and (
            not _has_builtin_seir_contract(self.model) or self.model.transmission_mode != "constant"
        ):
            raise ValueError(
                "the fused_seir storage profile requires the exact built-in "
                "constant-transmission SEIR model"
            )
        self._storage_profile = _storage_profile

        self.tau_max = validate_fp32_control("tau_max", tau_max, positive=True)
        if self._markovian:
            self.epsilon = epsilon
            self.max_prob = validate_fp32_control("max_prob", max_prob, positive=True)
            self.theta = validate_fp32_control("theta", theta, positive=True)
            self.tau_min = validate_fp32_control("tau_min", tau_min, positive=True)
        else:
            self.epsilon = validate_fp32_control("epsilon", epsilon, positive=True)
            self.max_prob = max_prob
            self.theta = theta
            self.tau_min = tau_min
        if self._markovian:
            if self.max_prob >= 1.0:
                raise ValueError("max_prob must be finite and in (0, 1)")
            if self.theta > 1.0:
                raise ValueError("theta is a target fraction and must be <= 1")
            if self.tau_min > self.tau_max:
                raise ValueError("require 0 < tau_min <= tau_max")

        self._transmission_age_dependent = (
            not self._markovian
            and getattr(self.model, "transmission_mode", "constant") == "age_dependent"
        )
        if self._transmission_age_dependent:
            missing = [
                name
                for name in ("compute_infectivity", "compute_rates_nonmarkov")
                if not callable(getattr(self.model, name, None))
            ]
            if missing:
                raise TypeError(
                    "age-dependent ensemble transmission requires model methods: "
                    + ", ".join(missing)
                )

        shape = (self.num_nodes, self.replicas)
        self.state = torch.zeros(shape, device=self.device, dtype=torch.int32)
        self.rates = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self._infectious_mask = None
        if self._storage_profile == "full":
            self.next_state = torch.zeros_like(self.state)
            self.pressure = torch.zeros_like(self.rates)
            self.event_prob = torch.zeros_like(self.rates)
            self.event_mask = torch.zeros(shape, device=self.device, dtype=torch.bool)
            self.rand_buffer = torch.zeros(shape, device=self.device, dtype=torch.float64)
        else:
            self._infectious_mask = torch.zeros(
                (self.num_nodes, (self.replicas + 31) // 32),
                device=self.device,
                dtype=torch.int32,
            )
            self.next_state = None
            self.pressure = None
            self.event_prob = None
            self.event_mask = None
            self.rand_buffer = None
        self._infectious_state_signature = (
            self._state_mutation_signature() if self._infectious_mask is not None else None
        )
        self.tau = torch.full(
            (self.replicas,), self.tau_max, device=self.device, dtype=torch.float32
        )
        self.current_time = torch.zeros(self.replicas, device=self.device, dtype=torch.float64)
        self.total_steps = 0
        self.total_events = torch.zeros(self.replicas, device=self.device, dtype=torch.int64)

        self.age = None
        self._infectivity = None
        self.seed_counter = None
        if not self._markovian:
            self.age = torch.zeros(shape, device=self.device, dtype=torch.float32)
            if self._transmission_age_dependent:
                self._infectivity = torch.zeros_like(self.age)
            if self._storage_profile == "full":
                self.seed_counter = torch.empty(shape, device=self.device, dtype=torch.int64)

        self._base_seed = normalize_seed(seed)
        self._init_generators = [torch.Generator(device=self.device) for _ in range(self.replicas)]
        self._event_generators = (
            [torch.Generator(device=self.device) for _ in range(self.replicas)]
            if self._markovian
            else None
        )
        self._reset_rng(self._base_seed)

    @property
    def is_markovian(self) -> bool:
        return self._markovian

    @property
    def storage_profile(self) -> str:
        """Resident scratch policy selected when the engine was constructed."""
        return self._storage_profile

    def _state_mutation_signature(self) -> tuple[int, int, int]:
        """Return the cheap host signature guarding derived state scratch."""
        return (
            id(self.state),
            self.state.untyped_storage().data_ptr(),
            self.state._version,
        )

    def _record_infectious_state_signature(self) -> None:
        """Mark the packed infectious bitmap as current for public state."""
        if self._infectious_mask is not None:
            self._infectious_state_signature = self._state_mutation_signature()

    def mark_state_dirty(self) -> None:
        """Invalidate derived state after writes PyTorch cannot observe.

        Ordinary PyTorch mutations and replacement of :attr:`state` are
        detected automatically.  Call this method after writing the state
        storage through ``.data``, DLPack, a custom kernel, or another alias
        that does not increment PyTorch's tensor version counter.  The next
        fused rate evaluation will rebuild its packed infectious bitmap.

        Engines without derived state scratch accept this notification as a
        no-op, which lets callers use the same protocol for every ensemble
        backend.
        """
        if self._infectious_mask is not None:
            self._infectious_state_signature = None

    def _reset_rng(self, effective_seed: int) -> None:
        for replica, generator in enumerate(self._init_generators):
            generator.manual_seed(_replica_seed(effective_seed, replica, _INIT_DOMAIN))
        if self._markovian:
            for replica, generator in enumerate(self._event_generators):
                generator.manual_seed(_replica_seed(effective_seed, replica, _EVENT_DOMAIN))
        elif self.seed_counter is not None:
            self._fill_seed_counter(effective_seed)

    def reseed(self, seed: int) -> None:
        """Reset every private stream without changing simulation tensors."""
        self._base_seed = normalize_seed(seed)
        self._reset_rng(self._base_seed)

    def reset(self, episode: int | None = None) -> None:
        """Clear all replicas and reset or episode-shift their private streams."""
        effective_seed = offset_seed(
            self._base_seed,
            episode if episode is not None else 0,
            name="episode",
        )
        self.state.zero_()
        self.rates.zero_()
        for scratch in (
            self._infectious_mask,
            self.next_state,
            self.pressure,
            self.event_prob,
            self.event_mask,
            self.rand_buffer,
        ):
            if scratch is not None:
                scratch.zero_()
        self._record_infectious_state_signature()
        self.tau.fill_(self.tau_max)
        self.current_time.zero_()
        self.total_events.zero_()
        self.total_steps = 0
        if self.age is not None:
            self.age.zero_()
        if self._infectivity is not None:
            self._infectivity.zero_()
        self._reset_rng(effective_seed)

    def _fill_seed_counter(self, effective_seed: int) -> None:
        if self.seed_counter is None:
            raise RuntimeError(
                "renewal counter scratch is unavailable in the compact fused_seir storage profile"
            )
        _fill_splitmix_counter_(self.seed_counter, effective_seed)

    def _renewal_uniform(self) -> torch.Tensor:
        """Open, exact 52-bit midpoint uniforms for all renewal lanes."""
        if self.seed_counter is None or self.rand_buffer is None:
            raise RuntimeError(
                "renewal uniform scratch is unavailable in the compact fused_seir storage profile"
            )
        return _splitmix_uniform_(self.seed_counter, self.rand_buffer)

    def _markovian_uniform(self) -> torch.Tensor:
        for replica, generator in enumerate(self._event_generators):
            self.rand_buffer[:, replica].random_(
                0,
                1 << 52,
                generator=generator,
            )
        self.rand_buffer.add_(0.5).mul_(2.0**-52)
        return self.rand_buffer

    def _normalize_counts(self, num_infected) -> tuple[int, ...]:
        if isinstance(num_infected, (str, bytes)):
            raw = None
        else:
            try:
                raw = tuple(num_infected)
            except TypeError:
                raw = None
        if raw is not None:
            if len(raw) != self.replicas:
                raise ValueError(f"num_infected sequence must have length {self.replicas}")
            return tuple(validate_population_count(value, self.num_nodes) for value in raw)
        value = validate_population_count(num_infected, self.num_nodes)
        return (value,) * self.replicas

    def seed_infection(self, num_infected, state: int | None = None) -> None:
        """Independently sample the requested infected count in every replica."""
        self._validate_graph_storage()
        if state is None:
            state = getattr(
                self.model,
                "infected" if self._markovian else "exposed",
                1,
            )
        state = validate_compartment(state, self.num_states)
        counts = self._normalize_counts(num_infected)
        for replica, count in enumerate(counts):
            indices = torch.randperm(
                self.num_nodes,
                device=self.device,
                generator=self._init_generators[replica],
            )[:count]
            self.state[indices, replica] = state
            if self.age is not None:
                self.age[indices, replica] = 0.0
        if self.next_state is not None:
            self.next_state.copy_(self.state)
        self._compute_rates()

    def _normalize_state(self, initial_state) -> torch.Tensor:
        state = torch.as_tensor(initial_state, device=self.device)
        if state.dim() == 1 and state.shape[0] == self.num_nodes:
            state = state[:, None].expand(-1, self.replicas)
        elif tuple(state.shape) != (self.num_nodes, self.replicas):
            raise ValueError(
                "initial_state must have node-major shape "
                f"[{self.num_nodes}, {self.replicas}] or [{self.num_nodes}], "
                f"got {tuple(state.shape)}"
            )
        if state.dtype == torch.bool or state.dtype.is_floating_point or state.dtype.is_complex:
            raise TypeError("initial_state must use an integer dtype")
        if state.numel() and (int(state.min()) < 0 or int(state.max()) >= self.num_states):
            raise ValueError(f"initial_state values must lie in [0, {self.num_states})")
        return state.to(torch.int32)

    def _normalize_age(self, initial_age) -> torch.Tensor:
        age = torch.as_tensor(initial_age, device=self.device, dtype=torch.float32)
        if age.dim() == 1 and age.shape[0] == self.num_nodes:
            age = age[:, None].expand(-1, self.replicas)
        elif tuple(age.shape) != (self.num_nodes, self.replicas):
            raise ValueError(
                "initial_age must have node-major shape "
                f"[{self.num_nodes}, {self.replicas}] or [{self.num_nodes}], "
                f"got {tuple(age.shape)}"
            )
        if not bool(torch.isfinite(age).all()) or bool((age < 0.0).any()):
            raise ValueError("initial_age must be finite and non-negative")
        return age

    def set_initial_state(self, initial_state, initial_age=None) -> None:
        """Set a shared ``[N]`` or explicit node-major ``[N, R]`` condition."""
        self._validate_graph_storage()
        # Validate both inputs before mutating either authoritative tensor.
        # In particular, a bad renewal age must not leave a newly copied state
        # paired with the old age field.
        normalized_state = self._normalize_state(initial_state)
        if self._markovian:
            if initial_age is not None:
                raise ValueError("initial_age is only valid for renewal ensembles")
            normalized_age = None
        elif initial_age is None:
            normalized_age = None
        else:
            normalized_age = self._normalize_age(initial_age)

        self.state.copy_(normalized_state)
        if self.next_state is not None:
            self.next_state.copy_(self.state)
        if not self._markovian:
            if normalized_age is None:
                self.age.zero_()
            else:
                self.age.copy_(normalized_age)
        self._compute_rates()

    def _gather_state_pressure(self) -> None:
        """Gather weighted inducer counts into the persistent pressure buffer."""
        self.pressure.copy_(
            reference_ensemble_influence_csr(self.graph, self.state, self.inducer_states)
        )

    def _gather_infectivity_pressure(self) -> None:
        """Gather floating source infectivity into the pressure buffer."""
        self.pressure.copy_(reference_ensemble_infectivity_csr(self.graph, self._infectivity))

    def _compute_rates(self) -> None:
        self._validate_graph_storage()
        if self._transmission_age_dependent:
            self.model.compute_infectivity(
                self.age.reshape(-1),
                self.state.reshape(-1),
                out=self._infectivity.reshape(-1),
            )
            self._gather_infectivity_pressure()
            evaluator = self.model.compute_rates_nonmarkov
            evaluator(
                self.age.reshape(-1),
                self.state.reshape(-1),
                self.pressure.reshape(-1),
                out=self.rates.reshape(-1),
            )
            return

        self._gather_state_pressure()
        if self._markovian:
            self.model.compute_rates(
                self.state.reshape(-1),
                self.pressure.reshape(-1),
                out=self.rates.reshape(-1),
            )
        else:
            self.model.compute_rates(
                self.age.reshape(-1),
                self.state.reshape(-1),
                self.pressure.reshape(-1),
                out=self.rates.reshape(-1),
            )

    def _choose_tau(self) -> None:
        min_rate, max_rate = torch.aminmax(self.rates, dim=0)
        if not bool(torch.isfinite(min_rate).all()) or not bool(torch.isfinite(max_rate).all()):
            raise FloatingPointError("ensemble transition rates must be finite")
        if bool((min_rate < 0.0).any()):
            raise ValueError("ensemble transition rates must be non-negative")
        tau_max = torch.full_like(max_rate, self.tau_max)
        if self._markovian:
            total_rate = self.rates.sum(dim=0)
            if not bool(torch.isfinite(total_rate).all()):
                raise FloatingPointError("ensemble total transition rates must be finite")
            event_target = self.theta * self.num_nodes / total_rate
            event_target = torch.clamp(event_target, min=self.tau_min)
            probability_bound = -math.log1p(-self.max_prob) / max_rate
            selected = torch.minimum(event_target, probability_bound)
            selected = torch.minimum(selected, tau_max)
            chosen = torch.where(total_rate <= 0.0, tau_max, selected)
        else:
            candidate = self.epsilon / max_rate
            selected = torch.minimum(candidate, tau_max)
            chosen = torch.where(max_rate == 0.0, tau_max, selected)
        if not bool(torch.isfinite(chosen).all()) or bool((chosen <= 0.0).any()):
            raise FloatingPointError("ensemble selected tau must be finite and positive")
        self.tau.copy_(chosen)

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance every replica once and return ``(tau[R], state[N, R])``."""
        if self._storage_profile != "full":
            raise RuntimeError(
                "the compact fused_seir storage profile requires the "
                "specialized EnsembleEngine step"
            )
        self._compute_rates()
        self._choose_tau()
        self.event_prob.copy_(self.rates)
        self.event_prob.mul_(-self.tau[None, :]).expm1_().neg_()
        uniforms = self._markovian_uniform() if self._markovian else self._renewal_uniform()
        torch.lt(uniforms, self.event_prob, out=self.event_mask)

        self.model.apply_transitions(
            self.state.reshape(-1),
            self.event_mask.reshape(-1),
            out=self.next_state.reshape(-1),
        )
        self.total_events.add_(self.event_mask.sum(dim=0))
        if not self._markovian:
            self.age.add_(self.tau[None, :])
            self.age.masked_fill_(self.next_state != self.state, 0.0)
        self.state.copy_(self.next_state)
        self.current_time.add_(self.tau)
        self.total_steps += 1
        return self.tau, self.state

    def _validate_graph_storage(self) -> None:
        """Reject graph pointer/content changes after engine construction."""
        self.graph._assert_unchanged(self._graph_signature, owner=type(self).__name__)

    def count_by_state(self) -> torch.Tensor:
        """Return replica-major counts with shape ``[R, num_states]``."""
        return torch.stack(
            [(self.state == state).sum(dim=0) for state in range(self.num_states)],
            dim=1,
        )

    def count_infected(self) -> torch.Tensor:
        """Return inducer counts for each replica as ``int64[R]``."""
        infected = torch.zeros(self.replicas, device=self.device, dtype=torch.int64)
        for state in self.inducer_states:
            infected.add_((self.state == state).sum(dim=0))
        return infected


class EnsembleEngine(ReferenceEnsembleEngine):
    """CUDA ensemble engine with CSR metadata shared across replica tiles.

    Generic models inherit the reference adaptive-step and transition path and
    replace only the two pressure gathers. The exact, unmodified built-in
    non-Markovian SEIR model with constant transmission uses a narrower fast
    path: one tiled kernel fuses susceptible CSR traversal with current-rate
    evaluation, broadcasts packed infectious-state words across replica lanes,
    and emits compact per-replica min/max partials while retaining public rates.
    Device reductions read those 128-node partials instead of rereading the
    full ``[N, R]`` rate matrix; a Triton finalizer selects time steps, and a
    tiled counter-based transition kernel updates state, age, and changed
    bitmap bits. Event partials are reduced on device before clocks and counters
    are committed. Both paths avoid the reference implementation's ``[E, R]``
    temporary and retain node-major ``[N, R]`` public tensors.

    Args:
        graph: Canonical incoming CSR, retained once for every replica.
        model: Markovian or renewal model accepted by the reference engine.
        replicas: Number of statistically independent trajectories.
        device: CUDA device. CPU callers should use
            :class:`ReferenceEnsembleEngine`.
        nodes_per_program: Power-of-two target-node tile, at most 32.
        replicas_per_tile: Power-of-two replica tile, at most 32. ``None``
            selects the smallest power of two covering ``replicas``, capped at
            32. The product of the two tile dimensions may not exceed 512.

    Notes:
        The specialized SEIR step is a sequence of device phases, not one
        monolithic kernel or a captured CUDA Graph. It performs one host status
        read after finalizing time steps; invalid rates abort transactionally
        before state, age, clocks, or event counts change. Each trajectory still
        chooses its own time step and advances an independent random stream.
        The fast stream is reproducible and tile-invariant but is not promised
        to be bitwise identical to the reference PyTorch stream. ``rates`` is
        authoritative. This path reports ``storage_profile == 'fused_seir'``
        and retains one int32 infectious bitmap with shape
        ``[N, ceil(R / 32)]`` (one bit per node-replica plus word padding). The
        transition kernel maintains this derived scratch, and every fused rate
        evaluation first checks a host mutation signature so ordinary PyTorch
        modifications of public ``state`` are repacked before use. Writes
        through ``.data``, DLPack, custom kernels, or independent storage
        aliases must be followed by :meth:`mark_state_dirty`, because PyTorch
        does not expose a cheap host-visible version change for those writes.
        Compatibility scratch such as ``pressure``, ``next_state``, event
        probabilities/masks, and dense random counters remains unallocated
        (``None``), rather than exposing stale intermediates. The dead minimum-
        bound storage is reinterpreted as int32 event partials after validation;
        this temporal alias saves one ``[ceil(N / 128), R]`` allocation.
    """

    def __init__(
        self,
        graph,
        model,
        replicas: int,
        *,
        device: str | torch.device | None = None,
        seed: int = 12345,
        epsilon: float = 0.03,
        max_prob: float = 0.1,
        theta: float = 0.01,
        tau_min: float = 1e-6,
        tau_max: float = 1.0,
        nodes_per_program: int = 8,
        replicas_per_tile: int | None = None,
    ):
        self._uses_fused_seir_rates = False
        replica_count = _positive_integer("replicas", replicas)
        graph_device = (graph.csr if hasattr(graph, "csr") else graph).device
        resolved_device = torch.device(graph_device if device is None else device)
        if resolved_device.type != "cuda":
            raise ValueError("EnsembleEngine requires a CUDA device")

        # Importing the optional GPU module is intentionally delayed until this
        # concrete engine is constructed. Importing ReferenceEnsembleEngine on
        # a CPU-only installation therefore never imports Triton.
        from ..core import flash_ensemble

        nodes_per_program = flash_ensemble._positive_power_of_two(
            "nodes_per_program", nodes_per_program, maximum=32
        )
        if replicas_per_tile is None:
            replicas_per_tile = flash_ensemble._default_replica_tile(replica_count)
        replicas_per_tile = flash_ensemble._positive_power_of_two(
            "replicas_per_tile", replicas_per_tile, maximum=32
        )
        if nodes_per_program * replicas_per_tile > 512:
            raise ValueError(
                "nodes_per_program * replicas_per_tile must be <= 512 to bound "
                "the fp32 accumulator tile"
            )
        if not flash_ensemble._HAS_TRITON:
            raise RuntimeError("Triton is required for EnsembleEngine") from (
                flash_ensemble._TRITON_IMPORT_ERROR
            )

        compact_seir_storage = (
            _has_builtin_seir_contract(model) and model.transmission_mode == "constant"
        )

        self.nodes_per_program = nodes_per_program
        self.replicas_per_tile = replicas_per_tile
        super().__init__(
            graph,
            model,
            replica_count,
            device=resolved_device,
            seed=seed,
            epsilon=epsilon,
            max_prob=max_prob,
            theta=theta,
            tau_min=tau_min,
            tau_max=tau_max,
            _storage_profile=("fused_seir" if compact_seir_storage else "full"),
        )
        # Age-dependent shedding evaluates a source-node hazard for every
        # infectious edge lane. The low-level kernel supports it for direct
        # benchmarking, but the engine retains its established infectivity
        # pre-pass until real-GPU profiling demonstrates a win.
        self._uses_fused_seir_rates = (
            not self._markovian
            and self.age is not None
            and not self._transmission_age_dependent
            and _supports_builtin_seir_rate_fusion(self.model)
        )
        self._uses_fused_seir_step = self._uses_fused_seir_rates
        if compact_seir_storage and not self._uses_fused_seir_step:
            raise RuntimeError("built-in SEIR fusion eligibility changed during model preparation")
        if self._uses_fused_seir_step:
            self._initialize_fused_seir_step()

    def _initialize_fused_seir_step(self) -> None:
        """Allocate fixed-address reduction, transaction, and event scratch."""
        if self._infectious_mask is None:
            raise RuntimeError("fused SEIR execution requires packed infectious-mask scratch")
        from ..core.flash_ensemble import (
            _RATE_BOUND_NODES_PER_PARTIAL,
            _rate_bound_partial_shape,
        )

        self._rate_bound_nodes_per_partial = _RATE_BOUND_NODES_PER_PARTIAL
        rate_bound_shape = _rate_bound_partial_shape(self.num_nodes, self.replicas)
        self._min_rate_partials = torch.empty(
            rate_bound_shape,
            device=self.device,
            dtype=torch.float32,
        )
        self._max_rate_partials = torch.empty_like(self._min_rate_partials)
        self._min_rate = torch.empty(self.replicas, device=self.device, dtype=torch.float32)
        self._max_rate = torch.empty_like(self._min_rate)
        self._tau_candidate = torch.empty_like(self._min_rate)
        self._invalid_step = torch.zeros(self.replicas, device=self.device, dtype=torch.int32)
        self._step_status = torch.zeros((), device=self.device, dtype=torch.int32)
        self._transition_nodes_per_program = 128
        self._transition_replicas_per_tile = min(
            4,
            1 << max(0, math.ceil(math.log2(self.replicas))),
        )
        node_blocks = (
            self.num_nodes + self._transition_nodes_per_program - 1
        ) // self._transition_nodes_per_program
        event_shape = (node_blocks, self.replicas)
        if event_shape != rate_bound_shape:
            raise RuntimeError(
                "rate-bound and transition partial shapes must match for "
                "temporal scratch reuse"
            )
        # The minimum bounds are dead after their device reduction and tau
        # validation. Reinterpret that same 4-byte contiguous storage for event
        # counts, which the transition overwrites before the next rate phase.
        # The two kernels never receive both aliases in one call.
        self._event_partials = self._min_rate_partials.view(torch.int32)
        self._step_events = torch.empty(self.replicas, device=self.device, dtype=torch.int64)
        self._event_seed = torch.empty((), device=self.device, dtype=torch.int64)
        # This scalar denotes the next accepted step. Starting at one matches
        # the single-trajectory fused engines and leaves failed steps unused.
        self._step_id = torch.ones((), device=self.device, dtype=torch.int64)
        self._reset_fused_event_stream(self._base_seed)

    def _reset_fused_event_stream(self, effective_seed: int) -> None:
        self._event_seed.fill_(signed_int64(project_seed(effective_seed, _EVENT_DOMAIN)))
        self._step_id.fill_(1)

    def reseed(self, seed: int) -> None:
        super().reseed(seed)
        if self._uses_fused_seir_step:
            self._reset_fused_event_stream(self._base_seed)

    def reset(self, episode: int | None = None) -> None:
        super().reset(episode=episode)
        if not self._uses_fused_seir_step:
            return
        effective_seed = offset_seed(
            self._base_seed,
            episode if episode is not None else 0,
            name="episode",
        )
        self._min_rate_partials.zero_()
        self._max_rate_partials.zero_()
        self._min_rate.zero_()
        self._max_rate.zero_()
        self._tau_candidate.zero_()
        self._invalid_step.zero_()
        self._step_status.zero_()
        self._step_events.zero_()
        self._reset_fused_event_stream(effective_seed)

    def _compute_rates(self) -> None:
        if not self._uses_fused_seir_rates:
            super()._compute_rates()
            return

        self._validate_graph_storage()
        from ..core.flash_ensemble import (
            ensemble_seir_renewal_rates_csr,
            pack_ensemble_infectious_mask,
        )

        # ``state`` is public and may have been mutated between steps. Normal
        # PyTorch writes increment ``_version``; replacing the tensor changes
        # its storage pointer. Accepted fused transitions maintain the mask and
        # record this signature themselves, so the steady-state path avoids an
        # otherwise redundant full state-read pass.
        state_signature = self._state_mutation_signature()
        if state_signature != self._infectious_state_signature:
            pack_ensemble_infectious_mask(
                self.state,
                out=self._infectious_mask,
            )
            self._infectious_state_signature = state_signature

        ensemble_seir_renewal_rates_csr(
            self.graph,
            self.state,
            self.age,
            beta=self.model._beta_t,
            mu_ei=self.model._mu_ei,
            sig_ei=self.model._sig_ei,
            mu_ir=self.model._mu_ir,
            sig_ir=self.model._sig_ir,
            transmission_age_dependent=False,
            infectious_mask=self._infectious_mask,
            rate_bounds=(self._min_rate_partials, self._max_rate_partials),
            out=self.rates,
            nodes_per_program=self.nodes_per_program,
            replicas_per_tile=self.replicas_per_tile,
        )

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._uses_fused_seir_step:
            return super().step()

        from ..core.flash_ensemble_step import (
            finalize_ensemble_renewal_tau,
            transition_ensemble_seir,
        )

        with torch.cuda.device(self.device):
            self._compute_rates()
            torch.amin(
                self._min_rate_partials,
                dim=0,
                out=self._min_rate,
            )
            torch.amax(
                self._max_rate_partials,
                dim=0,
                out=self._max_rate,
            )
            finalize_ensemble_renewal_tau(
                self._min_rate,
                self._max_rate,
                self._tau_candidate,
                self._invalid_step,
                epsilon=self.epsilon,
                tau_max=self.tau_max,
            )
            torch.amax(
                self._invalid_step,
                dim=0,
                out=self._step_status,
            )
            status = int(self._step_status.item())
            if status == 3:
                raise FloatingPointError("ensemble transition rates must be finite")
            if status == 2:
                raise ValueError("ensemble transition rates must be non-negative")
            if status != 0:
                raise FloatingPointError("ensemble selected tau must be finite and positive")

            transition_ensemble_seir(
                self.state,
                self.age,
                self.rates,
                self._tau_candidate,
                self._event_seed,
                self._step_id,
                self._event_partials,
                infectious_mask=self._infectious_mask,
                nodes_per_program=self._transition_nodes_per_program,
                replicas_per_tile=self._transition_replicas_per_tile,
            )
            self._record_infectious_state_signature()
            torch.sum(
                self._event_partials,
                dim=0,
                dtype=torch.int64,
                out=self._step_events,
            )
            self.tau.copy_(self._tau_candidate)
            self.current_time.add_(self._tau_candidate)
            self.total_events.add_(self._step_events)
            self._step_id.add_(1)

        self.total_steps += 1
        return self.tau, self.state

    def _gather_state_pressure(self) -> None:
        from ..core.flash_ensemble import ensemble_influence_csr

        ensemble_influence_csr(
            self.graph,
            self.state,
            self.inducer_states,
            out=self.pressure,
            nodes_per_program=self.nodes_per_program,
            replicas_per_tile=self.replicas_per_tile,
        )

    def _gather_infectivity_pressure(self) -> None:
        from ..core.flash_ensemble import ensemble_infectivity_csr

        ensemble_infectivity_csr(
            self.graph,
            self._infectivity,
            out=self.pressure,
            nodes_per_program=self.nodes_per_program,
            replicas_per_tile=self.replicas_per_tile,
        )


__all__ = ["EnsembleEngine", "ReferenceEnsembleEngine"]
