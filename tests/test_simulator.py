#!/usr/bin/env python
"""
Tests for the public Simulator/Trajectory facade (CPU).

The renewal (SEIR) path runs on CPU via the reference-influence fallback, so the
facade is exercised end-to-end here without a GPU. The Markovian path is GPU-only
and is covered by a clear-error test plus the gpu-marked smoke tests.
"""

import numpy as np
import pytest
import torch

import flashspread as fs


def _graph(n=400, d=8, seed=1):
    return fs.regular_graph(n, degree=d, seed=seed, device="cpu")


def _seir():
    return fs.SEIRModel(beta=0.25, mean_ei=5.0, median_ei=4.0,
                        mean_ir=7.5, median_ir=5.0)


class TestPublicSurface:
    def test_blessed_names_exported(self):
        for name in ("Simulator", "EngineConfig", "Trajectory", "regular_graph", "barabasi_albert",
                     "watts_strogatz", "geometric", "SISModel", "SIRModel",
                     "SEIRModel", "GraphCSR", "from_edges", "from_csr", "check_env"):
            assert name in fs.__all__, f"{name} missing from __all__"
            assert hasattr(fs, name)

    def test_engine_zoo_still_importable(self):
        # Back-compat: demoted from __all__, but must still import.
        from flashspread import (  # noqa: F401
            MarkovianEngine,
            MarkovianEngineCUDAGraph,
            RenewalEngine,
        )
        from flashspread.engines import create_renewal_engine  # noqa: F401

    def test_public_utility_boolean_flags_are_strict(self, tmp_path):
        with pytest.raises(TypeError, match="verbose"):
            fs.check_env(verbose="False")
        from flashspread.core.network import load_edges

        path = tmp_path / "edges.txt"
        path.write_text("0 1\n")
        with pytest.raises(TypeError, match="return_weights"):
            load_edges(path, return_weights="False")


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

    def test_trajectory_direct_construction_keeps_susceptible_zero_default(self):
        traj = fs.Trajectory(
            np.array([0.0, 1.0]),
            np.array([[9, 1], [7, 3]]),
            ("S", "I"),
            (1,),
            10,
        )

        assert traj.susceptible_state == 0
        assert traj.final_attack_rate == pytest.approx(0.3)

    def test_trajectory_uses_custom_model_susceptible_compartment(self):
        class RemappedSI:
            is_markovian = False
            infected = 0
            susceptible = 1
            num_states = 2
            inducer_states = [infected]
            transmission_mode = "constant"

            def prepare(self, device):
                pass

            def compute_rates(self, age, state, pressure, out=None):
                return torch.zeros_like(age) if out is None else out.zero_()

            def apply_transitions(self, state, event_mask, out=None):
                return state.clone() if out is None else out.copy_(state)

        graph = _graph(n=4, d=2)
        sim = fs.Simulator(graph, RemappedSI(), device="cpu", seed=0)
        sim.set_initial_state(
            torch.tensor([0, 1, 1, 1], dtype=torch.int32),
            age=torch.zeros(4),
        )

        traj = sim.run(until=0.0)

        assert traj.state_names == ("I", "S")
        assert traj.susceptible_state == 1
        assert traj.final_attack_rate == pytest.approx(0.25)

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

    def test_markovian_runs_on_cpu_and_conserves_population(self):
        g = _graph()
        sim = fs.Simulator(
            g, fs.SISModel(beta=0.5, delta=0.2), device="cpu", seed=4
        ).seed_infection(20)
        traj = sim.run(until=3.0, record_every=0.5)
        assert (traj.counts.sum(axis=1) == g.num_nodes).all()
        assert sim.engine._cpu_fallback

    def test_age_dependent_transmission_uses_cpu_reference_path(self):
        g = _graph(n=80, d=4)
        model = fs.SEIRModel(transmission_mode="age_dependent")
        sim = fs.Simulator(g, model, device="cpu", seed=4).seed_infection(8)
        assert type(sim.engine).__name__ == "RenewalEngineNonMarkov"
        assert sim.step() > 0.0

    def test_generic_renewal_model_protocol_runs_on_reference_engine(self):
        class AgeDependentSI:
            is_markovian = False
            susceptible = 0
            infected = 1
            num_states = 2
            inducer_states = [1]
            transmission_mode = "constant"

            def prepare(self, device):
                self.beta = torch.tensor(0.2, device=device)

            def compute_rates(self, age, state, pressure, out=None):
                out = torch.zeros_like(age) if out is None else out.zero_()
                out.copy_(torch.where(state == 0, self.beta * pressure, out))
                # A genuinely age-dependent I->S Weibull hazard (shape=2).
                out.copy_(torch.where(state == 1, 0.5 * age, out))
                return out

            def apply_transitions(self, state, event_mask, out=None):
                out = state.clone() if out is None else out.copy_(state)
                out[event_mask & (state == 0)] = 1
                out[event_mask & (state == 1)] = 0
                return out

        graph = _graph(n=20, d=4)
        sim = fs.Simulator(graph, AgeDependentSI(), device="cpu", seed=2)
        initial = torch.zeros(20, dtype=torch.int32)
        initial[0] = 1
        sim.set_initial_state(initial, age=torch.ones(20))
        assert type(sim.engine).__name__ == "RenewalEngine"
        assert sim.step() > 0.0

    def test_initial_state_bounds_are_checked_before_int32_narrowing(self):
        g = _graph(n=4, d=2)
        sim = fs.Simulator(g, _seir(), device="cpu", seed=0)
        invalid = torch.tensor([0, 0, 0, 2**32], dtype=torch.int64)
        with pytest.raises(ValueError, match="values"):
            sim.set_initial_state(invalid)

    def test_seed_reaches_private_engine_rng(self):
        g = _graph()
        a = fs.Simulator(g, _seir(), device="cpu", seed=7)
        b = fs.Simulator(g, _seir(), device="cpu", seed=999)
        assert a.engine._seed == 7
        assert b.engine._seed == 999
        a.seed_infection(40)
        b.seed_infection(40)
        assert not np.array_equal(
            a.engine.state.cpu().numpy(), b.engine.state.cpu().numpy()
        )

    @pytest.mark.parametrize("seed", [-1, 2**64 - 1])
    def test_facade_accepts_full_pytorch_seed_range(self, seed):
        sim = fs.Simulator(
            _graph(n=20, d=4), _seir(), device="cpu", seed=seed
        )
        assert sim.engine._seed == 2**64 - 1

    def test_engine_options_are_forwarded_and_unknowns_rejected(self):
        g = _graph()
        sim = fs.Simulator(
            g,
            _seir(),
            device="cpu",
            seed=17,
            epsilon=0.07,
            tau_max=0.4,
        )
        assert sim.engine.epsilon == pytest.approx(0.07)
        assert sim.engine.tau_max == pytest.approx(0.4)
        assert sim.engine._seed == 17
        with pytest.raises(TypeError, match="unexpected keyword"):
            fs.Simulator(g, _seir(), device="cpu", misspelled_option=True)

    def test_factory_does_not_mutate_shared_model(self):
        from flashspread.engines import create_renewal_engine

        g = _graph()
        model = _seir()
        engine = create_renewal_engine(
            g,
            model,
            device="cpu",
            use_cuda_graph=False,
            use_fused=False,
            nonmarkov_edges=True,
            transmission_mode="age_dependent",
            seed=2,
        )
        assert model.transmission_mode == "constant"
        assert engine.model.transmission_mode == "age_dependent"

    def test_explicit_config_selects_reference_backend(self):
        g = _graph()
        config = fs.EngineConfig(
            backend="reference",
            execution="eager",
            epsilon=0.02,
            tau_max=0.25,
        )
        sim = fs.Simulator(g, _seir(), device="cpu", seed=5, config=config)
        assert type(sim.engine).__name__ == "RenewalEngine"
        assert sim.engine.epsilon == pytest.approx(0.02)
        assert sim.engine.tau_max == pytest.approx(0.25)

    def test_config_rejects_conflicting_legacy_flags(self):
        g = _graph()
        with pytest.raises(ValueError, match="either config"):
            fs.Simulator(
                g,
                _seir(),
                device="cpu",
                config=fs.EngineConfig(),
                use_fused=False,
            )

    def test_config_rejects_irrelevant_markovian_choices(self):
        g = _graph()
        with pytest.raises(ValueError, match="traversal"):
            fs.Simulator(
                g,
                fs.SISModel(),
                device="cpu",
                config=fs.EngineConfig(traversal="merge"),
            )
        with pytest.raises(ValueError, match="backend"):
            fs.Simulator(
                g,
                fs.SISModel(),
                device="cpu",
                config=fs.EngineConfig(backend="reference"),
            )

    def test_compaction_auto_traversal_resolves_to_thread(self):
        config = fs.EngineConfig(compact=True)
        resolved = config.resolve(
            torch.device("cuda"),
            markovian=False,
            model=_seir(),
        )
        assert resolved["csr_strategy"] == "thread"

    def test_mixed_precision_warp_is_a_valid_compiled_combination(self):
        config = fs.EngineConfig(precision="mixed", traversal="warp")
        resolved = config.resolve(
            torch.device("cuda"), markovian=False, model=_seir()
        )
        assert resolved["use_mixed_precision"]
        assert resolved["csr_strategy"] == "warp"

    def test_explicit_markov_cuda_graph_config_forwards_batch(self):
        resolved = fs.EngineConfig(
            execution="cuda_graph", batch_steps=17
        ).resolve(
            torch.device("cuda"), markovian=True, model=fs.SISModel()
        )
        assert resolved["use_cuda_graph"]
        assert resolved["steps_per_launch"] == 17

    def test_auto_markov_config_preserves_eager_step_granularity(self):
        resolved = fs.EngineConfig().resolve(
            torch.device("cuda"), markovian=True, model=fs.SISModel()
        )
        assert not resolved["use_cuda_graph"]
        assert resolved["steps_per_launch"] == 50

    def test_auto_config_falls_back_for_generic_renewal_model(self):
        class GenericRenewal:
            is_markovian = False
            num_states = 2
            inducer_states = [1]
            transmission_mode = "constant"

            def prepare(self, device):
                pass

            def compute_rates(self, age, state, pressure, out=None):
                return out

            def apply_transitions(self, state, event_mask, out=None):
                return out

        resolved = fs.EngineConfig().resolve(
            torch.device("cuda"), markovian=False, model=GenericRenewal()
        )
        assert not resolved["use_fused"]
        assert not resolved["use_cuda_graph"]

    def test_engine_escape_hatch(self):
        from flashspread.engines.renewal import RenewalEngine
        g, m = _graph(), _seir()
        eng = RenewalEngine(g, m, device="cpu", seed=0)
        sim = fs.Simulator(g, m, device="cpu", engine=eng)
        assert sim.engine is eng

    def test_csr_only_graph_wrapper_runs_and_reports_population(self):
        class Wrapper:
            def __init__(self, csr):
                self.csr = csr

        graph = Wrapper(_graph(n=20, d=4).csr)
        sim = fs.Simulator(graph, _seir(), device="cpu", seed=3).seed_infection(2)
        assert "N=20" in repr(sim)
        trajectory = sim.run(until=0.1)
        assert trajectory.num_nodes == 20

    def test_engine_escape_hatch_rejects_population_mismatch(self):
        from flashspread.engines.renewal import RenewalEngine

        model = _seir()
        engine = RenewalEngine(_graph(n=20, d=4), model, device="cpu", seed=0)
        with pytest.raises(ValueError, match="num_nodes"):
            fs.Simulator(_graph(n=30, d=4), model, device="cpu", engine=engine)

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            ({"batch_steps": 1.5}, TypeError),
            ({"batch_steps": float("nan")}, TypeError),
            ({"batch_steps": True}, TypeError),
            ({"compact": "no"}, TypeError),
            ({"nodes_per_block": True}, TypeError),
            ({"nodes_per_block": 1.5}, TypeError),
            ({"epsilon": "0.1"}, TypeError),
            ({"tau_max": 1e300}, ValueError),
            ({"theta": 1.1}, ValueError),
        ],
    )
    def test_config_rejects_non_integral_and_non_boolean_fields(self, kwargs, error):
        with pytest.raises(error):
            fs.EngineConfig(**kwargs)

    def test_renewal_config_does_not_apply_markov_tau_floor_relationship(self):
        config = fs.EngineConfig(tau_max=1e-7, tau_min=1e-6)
        resolved = config.resolve(torch.device("cpu"), markovian=False, model=_seir())
        assert resolved["tau_max"] == pytest.approx(1e-7)
        with pytest.raises(ValueError, match="tau_min"):
            config.resolve(torch.device("cpu"), markovian=True, model=fs.SISModel())


class TestGraphConstructors:
    def test_regular_graph_is_symmetric_and_seeded(self):
        g = fs.regular_graph(300, degree=6, seed=3, device="cpu")
        assert g.num_edges == 300 * 6                 # both directions stored
        deg = g.csr.row_ptr[1:] - g.csr.row_ptr[:-1]
        assert int(deg.min()) == 6 and int(deg.max()) == 6
        g2 = fs.regular_graph(300, degree=6, seed=3, device="cpu")
        assert g.num_edges == g2.num_edges

    def test_tensor_constructors_return_canonical_csr(self):
        edges = fs.from_edges(
            np.array([[0, 1], [1, 0]]), num_nodes=2, device="cpu"
        )
        assert isinstance(edges, fs.GraphCSR)
        graph = fs.from_csr(
            edges.row_ptr, edges.col_ind, weights=edges.weights, device="cpu"
        )
        assert graph.csr is graph


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
