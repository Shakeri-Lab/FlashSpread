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

    def test_population_conserved_by_transition(self):
        from flashspread.models import SIRModel

        model = SIRModel()
        state = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
        mask = torch.tensor([True, True, False, False])
        new = model.apply_transitions(state, mask)
        # S->I and I->R only; total count unchanged
        assert new.numel() == state.numel()
        assert new.tolist() == [1, 2, 1, 2]


# ------------------------------------------------------------ hazards -------
class TestHazards:
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
