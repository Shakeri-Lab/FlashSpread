"""
Simulator: the one-obvious-way entry point to FlashSpread.

``Simulator`` is a thin facade over the engine factories in
:mod:`flashspread.engines`. It picks the right engine for the model (renewal for
age-dependent SEIR, Markovian for SIS/SIR), resolves the device, owns the seed,
and adds a high-level ``run(until, record_every) -> Trajectory`` loop so callers
do not hand-roll ``while current_time < T: step()``.

It also hides two inconsistencies in the underlying engines: their ``step()``
methods return different second elements (state vs. event count), and the model
``compute_rates`` signatures differ. The facade never calls ``compute_rates``
itself, so both model families work through one interface.

Power users keep full access to the engine zoo via ``flashspread.engines`` and
can inject a pre-built engine with ``engine=``.

Example:
    >>> import flashspread as fs
    >>> g = fs.regular_graph(2000, degree=10, seed=0, device="cpu")
    >>> model = fs.SEIRModel(beta=0.3)
    >>> sim = fs.Simulator(g, model, seed=0).seed_infection(20)
    >>> traj = sim.run(until=20.0, record_every=1.0)   # doctest: +SKIP
    >>> traj.peak_infected, traj.final_attack_rate     # doctest: +SKIP
"""

from __future__ import annotations

import numpy as np

from .config import EngineConfig
from .trajectory import Trajectory
from .utils import (
    is_markovian,
    resolve_device,
    seed_everything,
    state_names,
    validate_compartment,
)


def _same_device(requested, actual) -> bool:
    """Whether two devices name the same physical storage.

    An unindexed request such as ``torch.device("cuda")`` is not equal to the
    ``cuda:0`` an engine rebinds itself to, and neither is their string form, so
    resolve the default index before comparing. Torch is imported lazily: this
    helper only ever runs once a live engine exists.
    """
    import torch

    requested = torch.device(requested)
    actual = torch.device(actual)
    if requested.type != actual.type:
        return False
    if requested.index is None or actual.index is None:
        # One side left the index to PyTorch. Resolve it the same way PyTorch
        # would for that device type rather than declaring a mismatch.
        default = (
            torch.cuda.current_device()
            if requested.type == "cuda" and torch.cuda.is_available()
            else 0
        )
        return (requested.index if requested.index is not None else default) == (
            actual.index if actual.index is not None else default
        )
    return requested.index == actual.index


class Simulator:
    """High-level driver that selects and owns the appropriate engine.

    Args:
        graph: a :class:`GraphCSR` or a generated graph exposing one as ``csr``.
        model: a compartmental model (``SISModel``, ``SIRModel``, ``SEIRModel``).
        device: ``"cuda"``, ``"cpu"``, a ``torch.device``, or None to auto-detect.
        seed: base RNG seed, for reproducible runs.
        config: optional immutable :class:`EngineConfig`. It replaces legacy
            engine-selection keywords; passing both is an error.
        engine: optional pre-built engine to use verbatim (escape hatch); when
            given, engine auto-selection is skipped.
        **engine_kwargs: forwarded to the engine factory (e.g. ``epsilon``,
            ``tau_max``, ``steps_per_launch``, ``use_cuda_graph``).

    Note:
        CPU execution is a correctness/reference path for both engine families.
        The Triton kernels and CUDA Graph batching are selected on CUDA.
    """

    def __init__(
        self,
        graph,
        model,
        *,
        device=None,
        seed: int | None = None,
        config: EngineConfig | None = None,
        engine=None,
        **engine_kwargs,
    ):
        if config is not None and engine_kwargs:
            raise ValueError(
                "pass either config=EngineConfig(...) or legacy engine keyword "
                "arguments, not both"
            )
        if engine is not None and device is None and hasattr(engine, "device"):
            device = engine.device
        self.device = resolve_device(device)
        self.graph = graph
        self.model = model
        self._seed = seed
        self.config = config
        self._engine_kwargs = engine_kwargs
        self._engine_override = engine
        self._build()

    def _build(self) -> None:
        # Initial-condition bookkeeping. seed_infection installs an *initial*
        # condition, so it must know whether the simulation has advanced and
        # which compartments it has already populated.
        self._has_stepped = False
        self._seeded_compartments: set[int] = set()

        if self._seed is not None:
            seed_everything(self._seed)
        if self._engine_override is not None:
            self._engine = self._engine_override
            self._validate_engine_override()
            return

        # Import engine dispatch only when auto-building a simulation. An
        # injected engine remains a genuinely lazy escape hatch.
        from .engines import create_engine

        self._engine = create_engine(
            self.graph,
            self.model,
            device=str(self.device),
            config=self.config,
            seed=self._seed,
            **self._engine_kwargs,
        )

    def _validate_engine_override(self) -> None:
        """Ensure facade metadata describes the injected engine truthfully."""
        engine = self._engine
        if not hasattr(engine, "num_nodes"):
            raise TypeError("an injected engine must expose num_nodes")
        graph_csr = self.graph.csr if hasattr(self.graph, "csr") else self.graph
        if not hasattr(graph_csr, "num_nodes"):
            raise TypeError("graph must expose num_nodes directly or through .csr")
        if int(engine.num_nodes) != int(graph_csr.num_nodes):
            raise ValueError(
                "injected engine and graph disagree on num_nodes: "
                f"{engine.num_nodes} != {graph_csr.num_nodes}"
            )
        if hasattr(engine, "device"):
            # Compare canonical devices, not their strings. Every engine rebinds
            # itself to physical CSR storage (``cuda`` -> ``cuda:0``), so a raw
            # string compare rejected the natural spelling
            # ``Simulator(graph, model, device="cuda", engine=prebuilt)`` on a
            # correct configuration. torch.device equality treats an unindexed
            # request as distinct too, so resolve the index explicitly.
            if not _same_device(self.device, engine.device):
                raise ValueError(
                    "injected engine and Simulator disagree on device: "
                    f"{engine.device} != {self.device}"
                )
            # Adopt the engine's concrete device once validated, so the public
            # attribute names the storage the simulation actually runs on.
            self.device = engine.device
        engine_model = getattr(engine, "model", None)
        if engine_model is not None:
            if is_markovian(engine_model) != is_markovian(self.model):
                raise ValueError("injected engine uses the wrong model family")
            if getattr(engine_model, "num_states", None) != getattr(
                self.model, "num_states", None
            ):
                raise ValueError("injected engine model disagrees on num_states")
            engine_inducers = tuple(getattr(engine_model, "inducer_states", ()))
            facade_inducers = tuple(getattr(self.model, "inducer_states", ()))
            if engine_inducers != facade_inducers:
                raise ValueError("injected engine model disagrees on inducer_states")
            if tuple(state_names(engine_model)) != tuple(state_names(self.model)):
                raise ValueError("injected engine model disagrees on compartment mapping")

    # ---- underlying engine (power users) -----------------------------------
    @property
    def engine(self):
        """The engine instance this facade is driving."""
        return self._engine

    @property
    def num_nodes(self) -> int:
        """Population size from the canonical engine graph."""
        return int(self._engine.num_nodes)

    # ---- control surface ---------------------------------------------------
    def seed_infection(self, num_infected: int, state: int | None = None) -> "Simulator":
        """Seed ``num_infected`` initial infections. Returns self for chaining.

        The target compartment defaults to each model's natural entry state
        (Exposed for SEIR, Infected for SIS/SIR).

        This sets an *initial* condition, so it is rejected once the simulation
        has advanced. Seeding mid-run used to be accepted silently, injecting
        fresh-age nodes into a live epidemic while the clock and any already
        returned :class:`Trajectory` stayed untouched. Call
        :meth:`reset` first, or use :meth:`set_initial_state` to install a
        complete condition.
        """
        if self._has_stepped:
            raise RuntimeError(
                "seed_infection sets an initial condition and cannot be called "
                f"after the simulation has advanced (t={self.current_time!r}). "
                "Call reset() first, or use set_initial_state(...)."
            )
        compartment = (
            self._default_seed_compartment()
            if state is None
            else validate_compartment(state, self.model.num_states)
        )
        if compartment in self._seeded_compartments:
            raise RuntimeError(
                f"compartment {compartment} has already been seeded. Repeat "
                "calls are not additive: the second draw can select nodes the "
                "first already moved, so the requested totals are silently "
                "under-delivered. Seed the full count in one call, or build the "
                "exact condition with set_initial_state(...)."
            )
        self._engine.seed_infection(num_infected, state)
        self._seeded_compartments.add(compartment)
        return self

    def _default_seed_compartment(self) -> int:
        """The compartment ``seed_infection`` targets when none is given."""
        if is_markovian(self.model):
            return int(getattr(self.model, "infected", 1))
        return int(getattr(self.model, "exposed", 1))

    def set_initial_state(self, state, age=None) -> "Simulator":
        """Set a complete initial condition and return ``self``.

        Renewal engines accept optional node ages; Markovian engines reject an
        age array because holding time is not part of their state.
        """
        if age is None:
            self._engine.set_initial_state(state)
        elif is_markovian(self.model):
            raise ValueError("age is only valid for renewal/non-Markovian models")
        else:
            self._engine.set_initial_state(state, age)
        return self

    def step(self) -> float:
        """Advance the simulation and return the elapsed simulated time.

        Batched (CUDA-Graph) engines advance several internal steps per call, so
        one ``step()`` may return the summed duration of that window.
        """
        tau, _ = self._engine.step()      # engines differ in their 2nd element
        self._has_stepped = True
        return float(tau)

    def reset(self, episode: int | None = None) -> "Simulator":
        """Reset engine state. Returns self.

        With ``episode``, the RNG is reseeded to a mixed derivation of
        ``(base_seed, episode)`` so successive RL episodes are independent rather
        than replaying one stream. ``episode=0`` and ``episode=None`` both
        reproduce the base stream bitwise. The derivation is deliberately not
        ``base_seed + episode``: that made ``(100, episode=1)`` and
        ``(101, episode=0)`` the same stream, so a seed x episode sweep drew far
        fewer distinct streams than it appeared to.
        """
        if self._seed is not None and episode is None:
            seed_everything(self._seed)
        self._engine.reset(episode=episode)
        # A reset returns to t=0 with an empty population, so a fresh initial
        # condition is legitimate again.
        self._has_stepped = False
        self._seeded_compartments.clear()
        return self

    # ---- observables -------------------------------------------------------
    @property
    def current_time(self) -> float:
        return float(self._engine.current_time)

    def counts(self) -> np.ndarray:
        """Per-compartment population counts, as an int64 host array."""
        return self._engine.count_by_state().detach().cpu().numpy().astype(np.int64)

    def num_infected(self) -> int:
        """Number of nodes currently in an inducer (infectious) state."""
        return int(self._engine.count_infected())

    # ---- high-level run ----------------------------------------------------
    @property
    def steps_per_launch(self) -> int:
        """Internal simulation steps advanced per :meth:`step` call.

        1 for eager engines; the CUDA-Graph batch size (default 50) for batched
        ones. This is the granularity at which ``run(until=...)`` can stop, and
        it is the *effective* window: the fused engine rounds an odd request up
        to the next even number, so this may exceed what you asked for.
        """
        return int(getattr(self._engine, "steps_per_launch", 1))

    def run(self, until: float, record_every: float = 1.0,
            seed: int | None = None) -> Trajectory:
        """Run to ``until`` (simulated time), sampling every ``record_every``.

        Records the initial state, then approximately every ``record_every`` time
        units, then the final state.

        .. warning::
           **The stop time is granular, not exact.** A batched (CUDA-Graph)
           engine advances ``steps_per_launch`` internal steps per call and
           cannot be interrupted mid-window, so the run stops at the first
           sample *at or past* ``until`` and can overshoot it by up to one
           window (e.g. asking for ``until=50`` with the default
           ``steps_per_launch=50`` may finish near t=55).

           ``Trajectory.times[-1]`` is always the true end time -- never assume
           it equals ``until``. For a tight horizon, build the Simulator with a
           smaller window or an eager engine::

               fs.Simulator(g, m, steps_per_launch=2)     # tightest batched stop
               fs.Simulator(g, m, use_cuda_graph=False)   # one-step granularity

           The fused CUDA-Graph engine unrolls a double buffer, so it rounds an
           odd ``steps_per_launch`` **up to the next even number** -- asking for
           1 silently gives you 2. Read :attr:`steps_per_launch` back to see the
           window you actually got.

        Returns:
            A :class:`~flashspread.Trajectory`.
        """
        if seed is not None:
            seed_everything(seed)
            if hasattr(self._engine, "reseed"):
                self._engine.reseed(seed)

        if not np.isfinite(until):
            raise ValueError(f"until must be finite, got {until}")
        if until < self.current_time:
            raise ValueError(
                f"until ({until}) is before current_time ({self.current_time})"
            )
        if not np.isfinite(record_every) or record_every <= 0.0:
            raise ValueError(
                f"record_every must be finite and positive, got {record_every}"
            )

        names = tuple(state_names(self.model))
        inducers = tuple(int(s) for s in self.model.inducer_states)
        susceptible = validate_compartment(
            getattr(self.model, "susceptible", 0), len(names)
        )
        times: list[float] = []
        rows: list[np.ndarray] = []

        def record() -> None:
            times.append(self.current_time)
            rows.append(self.counts())

        record()                                   # initial state
        next_rec = self.current_time + record_every
        last_t = self.current_time

        while self.current_time < until:
            self.step()
            if self.current_time <= last_t:
                raise RuntimeError(
                    "Simulation time did not advance; check that epsilon > 0 "
                    "and tau_max > 0."
                )
            last_t = self.current_time
            if self.current_time >= next_rec:
                record()
                next_rec = self.current_time + record_every

        if times[-1] != self.current_time:
            record()                               # final state

        return Trajectory(
            times=np.asarray(times, dtype=float),
            counts=np.vstack(rows).astype(np.int64),
            state_names=names,
            inducer_states=inducers,
            num_nodes=self.num_nodes,
            susceptible_state=susceptible,
        )

    def __repr__(self) -> str:
        kind = "markovian" if is_markovian(self.model) else "renewal"
        return (f"Simulator(engine={type(self._engine).__name__}, kind={kind}, "
                f"device={self.device}, N={self.num_nodes}, "
                f"t={self.current_time:.3g})")
