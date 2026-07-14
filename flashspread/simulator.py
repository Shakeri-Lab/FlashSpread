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

from .engines import create_markovian_engine, create_renewal_engine
from .trajectory import Trajectory
from .utils import is_markovian, resolve_device, seed_everything, state_names


class Simulator:
    """High-level driver that selects and owns the appropriate engine.

    Args:
        graph: a FlashSpread graph (e.g. from :func:`flashspread.regular_graph`)
            or anything exposing ``csr``/``row_ptr`` and ``num_nodes``.
        model: a compartmental model (``SISModel``, ``SIRModel``, ``SEIRModel``).
        device: ``"cuda"``, ``"cpu"``, a ``torch.device``, or None to auto-detect.
        seed: base RNG seed, for reproducible runs.
        engine: optional pre-built engine to use verbatim (escape hatch); when
            given, engine auto-selection is skipped.
        **engine_kwargs: forwarded to the engine factory (e.g. ``epsilon``,
            ``tau_max``, ``steps_per_launch``, ``use_cuda_graph``).

    Note:
        The Markovian engine (SIS/SIR) requires a CUDA device; the renewal
        engine (SEIR) also runs on CPU via its reference-influence fallback.
    """

    def __init__(self, graph, model, *, device=None, seed: int | None = None,
                 engine=None, **engine_kwargs):
        self.device = resolve_device(device)
        self.graph = graph
        self.model = model
        self._seed = seed
        self._engine_kwargs = engine_kwargs
        self._engine_override = engine
        self._build()

    def _build(self) -> None:
        if self._seed is not None:
            seed_everything(self._seed)
        if self._engine_override is not None:
            self._engine = self._engine_override
            return
        kwargs = dict(self._engine_kwargs)
        if self._seed is not None:
            kwargs.setdefault("seed", self._seed)

        if is_markovian(self.model):
            if self.device.type != "cuda":
                raise RuntimeError(
                    "The Markovian engine (SIS/SIR) requires a CUDA device: its "
                    "influence kernel is GPU-only. Either use device='cuda', or "
                    "use an age-dependent model (SEIRModel), which has a CPU "
                    "fallback."
                )
            self._engine = create_markovian_engine(
                self.graph, self.model, device=str(self.device), **kwargs)
            return

        if self.device.type != "cuda":
            # The fused, CUDA-Graph and infectivity-kernel renewal paths are all
            # GPU-only. The plain RenewalEngine has a reference-influence CPU
            # fallback, so select it when there is no CUDA device.
            kwargs.setdefault("use_cuda_graph", False)
            kwargs.setdefault("use_fused", False)
            kwargs.setdefault("nonmarkov_edges", False)
        self._engine = create_renewal_engine(
            self.graph, self.model, device=str(self.device), **kwargs)

    # ---- underlying engine (power users) -----------------------------------
    @property
    def engine(self):
        """The engine instance this facade is driving."""
        return self._engine

    # ---- control surface ---------------------------------------------------
    def seed_infection(self, num_infected: int, state: int | None = None) -> "Simulator":
        """Seed ``num_infected`` initial infections. Returns self for chaining.

        The target compartment defaults to each model's natural entry state
        (Exposed for SEIR, Infected for SIS/SIR).
        """
        self._engine.seed_infection(num_infected, state)
        return self

    def step(self) -> float:
        """Advance the simulation and return the elapsed simulated time.

        Batched (CUDA-Graph) engines advance several internal steps per call, so
        one ``step()`` may return the summed duration of that window.
        """
        tau, _ = self._engine.step()      # engines differ in their 2nd element
        return float(tau)

    def reset(self, episode: int | None = None) -> "Simulator":
        """Reset engine state. Returns self.

        With ``episode``, the RNG is reseeded to ``base_seed + episode`` so
        successive RL episodes are independent rather than replaying one stream.
        """
        if self._seed is not None and episode is None:
            seed_everything(self._seed)
        self._engine.reset(episode=episode)
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
               fs.Simulator(g, m, use_cuda_graph=False)   # eager, exact

           The fused CUDA-Graph engine unrolls a double buffer, so it rounds an
           odd ``steps_per_launch`` **up to the next even number** -- asking for
           1 silently gives you 2. Read :attr:`steps_per_launch` back to see the
           window you actually got.

        Returns:
            A :class:`~flashspread.Trajectory`.
        """
        if seed is not None:
            seed_everything(seed)

        names = tuple(state_names(self.model))
        inducers = tuple(int(s) for s in self.model.inducer_states)
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
            num_nodes=int(self.graph.num_nodes),
        )

    def __repr__(self) -> str:
        kind = "markovian" if is_markovian(self.model) else "renewal"
        return (f"Simulator(engine={type(self._engine).__name__}, kind={kind}, "
                f"device={self.device}, N={self.graph.num_nodes}, "
                f"t={self.current_time:.3g})")
