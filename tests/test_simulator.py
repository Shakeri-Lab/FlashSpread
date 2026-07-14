#!/usr/bin/env python
"""
Tests for the public Simulator/Trajectory facade (CPU).

The renewal (SEIR) path runs on CPU via the reference-influence fallback, so the
facade is exercised end-to-end here without a GPU. The Markovian path is GPU-only
and is covered by a clear-error test plus the gpu-marked smoke tests.
"""

import numpy as np
import pytest

import flashspread as fs


def _graph(n=400, d=8, seed=1):
    return fs.regular_graph(n, degree=d, seed=seed, device="cpu")


def _seir():
    return fs.SEIRModel(beta=0.25, mean_ei=5.0, median_ei=4.0,
                        mean_ir=7.5, median_ir=5.0)


class TestPublicSurface:
    def test_blessed_names_exported(self):
        for name in ("Simulator", "Trajectory", "regular_graph", "barabasi_albert",
                     "watts_strogatz", "geometric", "SISModel", "SIRModel",
                     "SEIRModel", "check_env"):
            assert name in fs.__all__, f"{name} missing from __all__"
            assert hasattr(fs, name)

    def test_engine_zoo_still_importable(self):
        # Back-compat: demoted from __all__, but must still import.
        from flashspread import MarkovianEngine, RenewalEngine  # noqa: F401
        from flashspread.engines import create_renewal_engine  # noqa: F401


class TestSimulatorCPU:
    def test_seir_runs_on_cpu_and_conserves_population(self):
        g = _graph()
        sim = fs.Simulator(g, _seir(), device="cpu", seed=0).seed_infection(40)
        traj = sim.run(until=20.0, record_every=2.0)
        assert isinstance(traj, fs.Trajectory)
        # population conserved at every recorded sample
        assert (traj.counts.sum(axis=1) == g.num_nodes).all()
        assert len(traj) >= 2
        assert traj.times[0] == 0.0
        assert traj.times[-1] >= 20.0          # runs to at least `until`

    def test_trajectory_observables(self):
        g = _graph()
        sim = fs.Simulator(g, _seir(), device="cpu", seed=0).seed_infection(40)
        traj = sim.run(until=20.0, record_every=2.0)
        assert traj.state_names[:1] == ("S",)
        assert 0 <= traj.peak_infected <= g.num_nodes
        assert 0.0 <= traj.final_attack_rate <= 1.0
        # named access + dict export line up with the counts matrix
        assert np.array_equal(traj["S"], traj.counts[:, 0])
        assert set(traj.to_dict()) >= {"time", "S"}

    def test_reproducible_same_seed(self):
        g = _graph()
        a = fs.Simulator(g, _seir(), device="cpu", seed=7).seed_infection(40).run(until=10.0)
        b = fs.Simulator(g, _seir(), device="cpu", seed=7).seed_infection(40).run(until=10.0)
        assert np.array_equal(a.counts, b.counts)

    def test_chaining_returns_self(self):
        g = _graph()
        sim = fs.Simulator(g, _seir(), device="cpu", seed=0)
        assert sim.seed_infection(10) is sim
        assert sim.reset() is sim

    def test_step_returns_elapsed_time(self):
        g = _graph()
        sim = fs.Simulator(g, _seir(), device="cpu", seed=0).seed_infection(10)
        t0 = sim.current_time
        tau = sim.step()
        assert tau > 0
        assert sim.current_time > t0

    def test_steps_per_launch_exposed(self):
        # CPU renewal path is eager -> window of 1 (so run() stops tightly).
        g = _graph()
        sim = fs.Simulator(g, _seir(), device="cpu", seed=0)
        assert sim.steps_per_launch == 1

    def test_markovian_on_cpu_gives_clear_error(self):
        g = _graph()
        with pytest.raises(RuntimeError, match="requires a CUDA device"):
            fs.Simulator(g, fs.SISModel(beta=0.5, delta=0.2), device="cpu")

    def test_engine_escape_hatch(self):
        from flashspread.engines.renewal import RenewalEngine
        g, m = _graph(), _seir()
        eng = RenewalEngine(g, m, device="cpu", seed=0)
        sim = fs.Simulator(g, m, device="cpu", engine=eng)
        assert sim.engine is eng


class TestGraphConstructors:
    def test_regular_graph_is_symmetric_and_seeded(self):
        g = fs.regular_graph(300, degree=6, seed=3, device="cpu")
        assert g.num_edges == 300 * 6                 # both directions stored
        deg = g.csr.row_ptr[1:] - g.csr.row_ptr[:-1]
        assert int(deg.min()) == 6 and int(deg.max()) == 6
        g2 = fs.regular_graph(300, degree=6, seed=3, device="cpu")
        assert g.num_edges == g2.num_edges


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
