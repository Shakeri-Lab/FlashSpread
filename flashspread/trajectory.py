"""
Trajectory: a lightweight, dependency-free result object for a simulation run.

Holds the recorded time points and per-state population counts, plus a few
convenience observables (peak infection, attack rate). Backed by NumPy arrays
only -- no pandas dependency -- but ``to_dict()`` drops straight into a
``pandas.DataFrame`` for users who want one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Recorded output of :meth:`flashspread.Simulator.run`.

    Attributes:
        times: ``[T]`` recorded time points.
        counts: ``[T, num_states]`` population counts per compartment.
        state_names: names of the compartments, index-aligned with columns
            of ``counts`` (e.g. ``("S", "E", "I", "R")``).
        inducer_states: state indices that count as "infectious".
        num_nodes: total population size.
        susceptible_state: column index of the susceptible compartment. The
            default of 0 preserves direct construction for conventional
            S-first models.
    """

    times: np.ndarray
    counts: np.ndarray
    state_names: tuple[str, ...]
    inducer_states: tuple[int, ...]
    num_nodes: int
    susceptible_state: int = 0

    @property
    def infected(self) -> np.ndarray:
        """Infectious count per time point (summed over inducer states)."""
        return self.counts[:, list(self.inducer_states)].sum(axis=1)

    @property
    def peak_infected(self) -> int:
        """Maximum simultaneous infectious count over the run."""
        return int(self.infected.max())

    @property
    def peak_time(self) -> float:
        """Time at which the infectious count peaks."""
        return float(self.times[int(self.infected.argmax())])

    @property
    def final_attack_rate(self) -> float:
        """Final non-susceptible fraction: ``(N - final S) / N``.

        This is cumulative attack rate for absorbing SIR/SEIR dynamics. For
        SIS, where nodes return to S, use :attr:`final_prevalence`; cumulative
        ever-infected history is not recoverable from compartment counts.
        """
        final_susceptible = self.counts[-1, self.susceptible_state]
        return float((self.num_nodes - final_susceptible) / self.num_nodes)

    @property
    def final_prevalence(self) -> float:
        """Fraction infectious at the final recorded time."""
        return float(self.infected[-1] / self.num_nodes)

    def to_dict(self) -> dict:
        """Column-oriented dict (``{"time": ..., "S": ..., ...}``).

        Handy for ``pandas.DataFrame(traj.to_dict())`` or JSON export.
        """
        out = {"time": self.times}
        for i, name in enumerate(self.state_names):
            out[name] = self.counts[:, i]
        return out

    def __getitem__(self, name: str) -> np.ndarray:
        """Access a compartment's time series by name, e.g. ``traj["I"]``."""
        return self.counts[:, self.state_names.index(name)]

    def __len__(self) -> int:
        return int(self.times.shape[0])
