#!/usr/bin/env python
"""
Pure-CPU unit tests for FlashSpread's math/logic layer.

These exercise the parts that do not need a GPU (model rate computation,
hazard functions, the reference scatter-add influence, environment probing)
so that CI on a CPU runner runs real assertions rather than skipping
everything. GPU kernel parity is covered by the ``gpu``-marked smoke tests.
"""

import pytest
import torch

CPU = torch.device("cpu")


# ------------------------------------------------------------- models -------
class TestModelRates:
    def test_sis_rates(self):
        from flashspread.models import SISModel

        model = SISModel(beta=0.5, delta=1.0)
        model.prepare(CPU)
        state = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        influence = torch.tensor([1.0, 2.0, 0.0, 0.0])
        rates = model.compute_rates(state, influence)
        assert rates[0].item() == pytest.approx(0.5)   # beta * 1
        assert rates[1].item() == pytest.approx(1.0)   # beta * 2
        assert rates[2].item() == pytest.approx(1.0)   # delta
        assert rates[3].item() == pytest.approx(1.0)

    def test_sir_rates(self):
        from flashspread.models import SIRModel

        model = SIRModel(beta=0.4, gamma=0.2)
        model.prepare(CPU)
        state = torch.tensor([0, 1, 2], dtype=torch.int32)   # S, I, R
        influence = torch.tensor([3.0, 0.0, 0.0])
        rates = model.compute_rates(state, influence)
        assert rates[0].item() == pytest.approx(1.2)   # beta * 3
        assert rates[1].item() == pytest.approx(0.2)   # gamma
        assert rates[2].item() == pytest.approx(0.0)   # R is absorbing

    def test_seir_hazards_positive_finite(self):
        from flashspread.models import SEIRModel

        model = SEIRModel(beta=0.3, mean_ei=5.0, median_ei=4.0,
                          mean_ir=3.9, median_ir=1.5)
        model.prepare(CPU)
        age = torch.tensor([0.1, 1.0, 5.0, 10.0, 50.0])
        state = torch.full((5,), model.exposed, dtype=torch.int32)
        pressure = torch.zeros(5)
        rates = model.compute_rates(age, state, pressure)
        assert torch.all(rates > 0)
        assert torch.all(torch.isfinite(rates))

    def test_seir_shared_rate_evaluator_preserves_both_pressure_contracts(self):
        from flashspread.models import SEIRModel

        model = SEIRModel(beta=0.25)
        model.prepare(CPU)
        state = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        age = torch.tensor([0.0, 2.0, 1.5, 0.0])
        pressure = torch.tensor([4.0, 7.0, 9.0, 11.0])

        model.sparse_hazard = True
        ordinary_sparse = model.compute_rates(age, state, pressure)
        scaled_sparse = model.compute_rates_nonmarkov(age, state, pressure)
        model.sparse_hazard = False
        ordinary_dense = model.compute_rates(age, state, pressure)
        scaled_dense = model.compute_rates_nonmarkov(age, state, pressure)

        torch.testing.assert_close(ordinary_dense, ordinary_sparse)
        torch.testing.assert_close(scaled_dense, scaled_sparse)
        assert ordinary_sparse[0].item() == pytest.approx(1.0)
        assert scaled_sparse[0].item() == pytest.approx(4.0)

    def test_population_conserved_by_transition(self):
        from flashspread.models import SIRModel

        model = SIRModel()
        state = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
        mask = torch.tensor([True, True, False, False])
        new = model.apply_transitions(state, mask)
        # S->I and I->R only; total count unchanged
        assert new.numel() == state.numel()
        assert new.tolist() == [1, 2, 1, 2]

    @pytest.mark.parametrize(
        "constructor,kwargs,error",
        [
            ("sis", {"beta": True}, TypeError),
            ("sir", {"gamma": "0.1"}, TypeError),
            ("seir", {"beta": 1e300}, ValueError),
            ("seir", {"mean_ei": 1e-50}, ValueError),
        ],
    )
    def test_model_controls_are_strict_fp32_scalars(
        self, constructor, kwargs, error
    ):
        from flashspread.models import SEIRModel, SIRModel, SISModel

        constructors = {"sis": SISModel, "sir": SIRModel, "seir": SEIRModel}
        with pytest.raises(error):
            constructors[constructor](**kwargs)

    def test_seir_prepare_avoids_fp32_mean_over_median_overflow(self):
        from flashspread.models import SEIRModel

        model = SEIRModel(mean_ei=1e30, median_ei=1e-30)
        model.prepare(CPU)
        assert torch.isfinite(model._mu_ei)
        assert torch.isfinite(model._sig_ei)


def test_markovian_tau_respects_probability_bound_even_below_tau_min():
    from flashspread.core.graph import GraphCSR
    from flashspread.engines.markovian import MarkovianEngine
    from flashspread.models import SISModel

    graph = GraphCSR(
        torch.tensor([[0], [1]], dtype=torch.int64),
        2,
        weights=torch.tensor([100.0]),
    )
    engine = MarkovianEngine(
        graph,
        SISModel(beta=1.0, delta=0.1),
        device="cpu",
        max_prob=0.1,
        theta=1.0,
        tau_min=1.0,
        tau_max=2.0,
        seed=0,
    )
    engine.set_initial_state(torch.tensor([1, 0]))
    tau, _ = engine.step()
    realized_bound = 1.0 - torch.exp(torch.tensor(-100.0 * tau)).item()
    assert realized_bound <= 0.1 + 1e-6


def test_markovian_transition_rng_uses_float64_midpoints(monkeypatch):
    from flashspread import GraphCSR, SISModel
    from flashspread.engines.markovian import MarkovianEngine

    graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
    engine = MarkovianEngine(
        graph, SISModel(beta=0.0, delta=1e-13), device="cpu", seed=3
    )
    engine.set_initial_state(torch.tensor([1, 0]))
    observed = {}
    original = torch.Tensor.random_

    def recording_random_(tensor, *args, **kwargs):
        observed["dtype"] = tensor.dtype
        observed["bounds"] = args[:2]
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "random_", recording_random_)
    engine.step()
    assert observed["dtype"] == torch.float64
    assert observed["bounds"] == (0, 1 << 52)


def test_markovian_nonfinite_aggregate_rate_fails_loudly():
    from flashspread import GraphCSR, SISModel
    from flashspread.engines.markovian import MarkovianEngine

    graph = GraphCSR(
        torch.tensor([[0, 1], [2, 2]]),
        3,
        weights=torch.tensor([3.0e38, 3.0e38]),
    )
    engine = MarkovianEngine(
        graph, SISModel(beta=1.0, delta=0.1), device="cpu"
    )
    engine.set_initial_state(torch.tensor([1, 1, 0]))
    with pytest.raises(FloatingPointError, match="non-finite"):
        engine.step()


def test_custom_markovian_negative_rates_fail_loudly():
    from flashspread import GraphCSR
    from flashspread.engines.markovian import MarkovianEngine

    class NegativeRateModel:
        is_markovian = True
        num_states = 2
        susceptible = 0
        infected = 1
        inducer_states = (1,)

        def prepare(self, device):
            return None

        def compute_rates(self, state, influence, out=None):
            values = torch.tensor([-1.0, 1.0], device=state.device)
            if out is None:
                return values
            out.copy_(values)
            return out

        def apply_transitions(self, state, event_mask, out=None):
            target = state.clone() if out is None else out.copy_(state)
            target[event_mask] = 1 - target[event_mask]
            return target

    graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
    engine = MarkovianEngine(graph, NegativeRateModel(), device="cpu")
    engine.set_initial_state(torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="non-negative"):
        engine.step()


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"max_prob": True}, TypeError),
        ({"theta": True}, TypeError),
        ({"tau_min": True}, TypeError),
        ({"tau_max": True}, TypeError),
        ({"theta": 1e-50}, ValueError),
        ({"tau_max": 1e300}, ValueError),
    ],
)
def test_markovian_controls_preserve_strict_types_and_fp32_range(kwargs, error):
    from flashspread import GraphCSR, SISModel
    from flashspread.engines.markovian import MarkovianEngine

    graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
    with pytest.raises(error):
        MarkovianEngine(graph, SISModel(), device="cpu", **kwargs)


def test_markovian_reset_rejects_nonintegral_episode_without_mutating_state():
    from flashspread import GraphCSR, SISModel
    from flashspread.engines.markovian import MarkovianEngine

    graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
    engine = MarkovianEngine(graph, SISModel(), device="cpu", seed=3)
    engine.set_initial_state(torch.tensor([1, 0]))
    before = engine.state.clone()
    with pytest.raises(TypeError, match="episode"):
        engine.reset(episode=1.5)
    assert torch.equal(engine.state, before)


def test_generic_markovian_tau_underflow_is_rejected_before_sampling():
    from flashspread import GraphCSR
    from flashspread.engines.markovian import MarkovianEngine

    class MaxRateModel:
        is_markovian = True
        num_states = 2
        susceptible = 0
        infected = 1
        inducer_states = (1,)

        def prepare(self, device):
            return None

        def compute_rates(self, state, influence, out=None):
            out.zero_()
            out[0] = torch.finfo(torch.float32).max
            return out

        def apply_transitions(self, state, event_mask, out=None):
            return out.copy_(state)

    graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
    engine = MarkovianEngine(
        graph,
        MaxRateModel(),
        device="cpu",
        max_prob=2.0**-149,
        theta=1.0,
    )
    engine.set_initial_state(torch.tensor([0, 1]))
    before = engine.state.clone()
    with pytest.raises(ValueError, match="selected tau"):
        engine.step()
    assert torch.equal(engine.state, before)


def test_high_seed_bits_change_markov_cpu_stream():
    from flashspread import GraphCSR, SISModel
    from flashspread.engines.markovian import MarkovianEngine

    node = torch.arange(64)
    graph = GraphCSR(torch.stack((node, (node + 1) % 64)), 64)
    low = MarkovianEngine(graph, SISModel(), device="cpu", seed=0)
    high = MarkovianEngine(graph, SISModel(), device="cpu", seed=2**32)
    low.seed_infection(8)
    high.seed_infection(8)
    assert not torch.equal(low.state, high.state)


def test_model_family_marker_must_be_boolean():
    from flashspread.utils import is_markovian

    model = type("BadMarker", (), {"is_markovian": "False"})()
    with pytest.raises(TypeError, match="is_markovian"):
        is_markovian(model)


# ------------------------------------------------------------ hazards -------
class TestHazards:
    def test_models_star_export_preserves_stable_hazard(self):
        import flashspread.models as models

        namespace = {}
        exec("from flashspread.models import *", namespace)
        assert "lognormal_hazard_stable" in models.__all__
        assert namespace["lognormal_hazard_stable"] is models.lognormal_hazard_stable

    def test_lognormal_stable_matches_basic(self):
        from flashspread.models.hazards import (
            lognormal_hazard, lognormal_hazard_stable,
        )
        import math

        age = torch.linspace(0.1, 20.0, 50)
        mean, median = 5.0, 4.0
        mu = torch.tensor(math.log(median))
        sigma = torch.tensor(math.sqrt(2.0 * math.log(mean / median)))
        h_stable = lognormal_hazard_stable(age, mu, sigma)
        h_basic = lognormal_hazard(age, mean, median)
        assert torch.allclose(h_stable, h_basic, rtol=1e-3, atol=1e-4)
        assert torch.all(h_stable > 0)

    def test_erfcx_rational_approx(self):
        from flashspread.models.hazards import erfcx_rational_approx

        z = torch.linspace(-6.0, 30.0, 500)
        ref = torch.special.erfcx(z)
        approx = erfcx_rational_approx(z)
        valid = ref > 1e-20
        rel = ((approx[valid] - ref[valid]) / ref[valid]).abs()
        assert rel.max().item() < 5e-4

    def test_gamma_hazard_and_factory(self):
        from flashspread.models import build_hazard_from_params, gamma_hazard

        age = torch.linspace(0.1, 5.0, 20)
        direct = gamma_hazard(age, shape=2.0, rate=0.5)
        factory = build_hazard_from_params("gamma", shape=2.0, rate=0.5)
        assert torch.allclose(factory(age, torch.zeros_like(age)), direct)
        assert torch.isfinite(direct).all()
        with pytest.raises(ValueError):
            gamma_hazard(age, shape=0.0, rate=1.0)

    def test_gamma_tail_does_not_collapse_when_survival_is_tiny(self):
        from flashspread.models import gamma_hazard

        # For Gamma(shape=2, rate=1), h(t) = t / (1 + t).
        age = torch.tensor([50.0, 1000.0])
        expected = age / (1.0 + age)
        torch.testing.assert_close(
            gamma_hazard(age, shape=2.0, rate=1.0),
            expected,
            rtol=2e-6,
            atol=0.0,
        )
        large_shape = gamma_hazard(
            torch.tensor([15_000.0]), shape=10_000.0, rate=1.0
        )
        torch.testing.assert_close(
            large_shape,
            torch.tensor([0.33353314]),
            rtol=2e-6,
            atol=0.0,
        )

    def test_hazard_factory_is_stable_and_validates_rates(self):
        from flashspread.models import build_hazard_from_params

        params = {"mean": 5.0, "median": 4.0}
        age = torch.tensor([1000.0])
        pressure = torch.zeros_like(age)
        host = build_hazard_from_params("lognormal", **params)(age, pressure)
        explicit = build_hazard_from_params("lognormal", device="cpu", **params)(
            age, pressure
        )
        torch.testing.assert_close(host, explicit)
        assert host.item() > 1e-3
        for hazard_type, kwargs in (
            ("weibull", {"shape": 0.0, "scale": 1.0}),
            ("constant", {"rate": -1.0}),
            ("network", {"beta": -1.0}),
        ):
            with pytest.raises(ValueError):
                build_hazard_from_params(hazard_type, **kwargs)
        for hazard_type, kwargs in (
            ("lognormal", {"mean": "5.0", "median": 4.0}),
            ("weibull", {"shape": True, "scale": 1.0}),
            ("gamma", {"shape": 2.0, "rate": "0.5"}),
            ("network", {"beta": True}),
            ("constant", {"rate": "0.2"}),
        ):
            with pytest.raises(TypeError):
                build_hazard_from_params(hazard_type, **kwargs)


# -------------------------------------------------- reference influence -----
class TestReferenceInfluence:
    def test_reference_influence_known_case(self):
        from flashspread.core.flash_neighbor import reference_influence

        # Edges (src -> dst): 0->1, 0->2, 1->2.  Node 0 is infectious.
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
        states = torch.tensor([1, 0, 0], dtype=torch.int32)  # node 0 = I
        infl = reference_influence(edge_index, 3, states, [1])
        # influence[i] = # infectious in-neighbours
        assert infl.tolist() == [0.0, 1.0, 1.0]

    def test_graph_contract_rejects_boolean_node_count_and_orientation(self):
        from flashspread import GraphCSR

        edges = torch.empty((2, 0), dtype=torch.int64)
        with pytest.raises(TypeError, match="num_nodes"):
            GraphCSR(edges, True)
        with pytest.raises(TypeError, match="incoming"):
            GraphCSR(edges, 1, incoming="yes")
        with pytest.raises(TypeError, match="incoming"):
            GraphCSR.from_csr(
                torch.tensor([0, 0], dtype=torch.int32),
                torch.empty(0, dtype=torch.int32),
                incoming=1,
            )

    def test_reference_influence_infectivity_weighted(self):
        from flashspread.core.flash_neighbor import reference_influence_infectivity

        edge_index = torch.tensor([[0, 1], [1, 2]])   # 0->1, 1->2
        infectivity = torch.tensor([0.5, 0.25, 0.0])
        weights = torch.tensor([2.0, 4.0])
        out = reference_influence_infectivity(edge_index, 3, infectivity, weights)
        # influence[1] = w(0->1)*inf[0] = 2*0.5 = 1.0 ; influence[2] = 4*0.25 = 1.0
        assert out.tolist() == pytest.approx([0.0, 1.0, 1.0])


# ------------------------------------------------------------- env ----------
class TestCheckEnv:
    def test_check_env_keys(self):
        import flashspread

        info = flashspread.check_env(verbose=False)
        for key in ("flashspread", "torch", "cuda_available", "triton",
                    "networkx", "scipy", "gpu_ready"):
            assert key in info
        assert isinstance(info["gpu_ready"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
