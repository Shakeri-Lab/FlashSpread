#!/usr/bin/env python
"""
Phase 0 correctness & reproducibility regression tests (CPU-only).

These guard the fixes for the paper-affecting bugs found in the audit:
  C1     symmetric undirected graph generators (full in-degree)
  C2     reproducible seed_infection from the engine seed
  C4     reset(episode=) reseeds the RNG streams
  C6     SEIR mean>median>0 validation
  R1     epsilon>0 and num_infected bounds validation
  NEW-2  GraphCSR node-index bounds checking

Everything here runs on CPU (RenewalEngine has a reference-influence CPU
fallback), so it executes in ordinary CI without a GPU.
"""

import pytest
import torch

from flashspread.core.graph import GraphCSR
from flashspread.core.network import FixedDegreeGraph
from flashspread.models import SEIRModel
from flashspread.engines.renewal import RenewalEngine


# ---------------------------------------------------------------- C1 --------
class TestSymmetricGenerators:
    def test_regular_graph_full_degree(self):
        """Default (symmetric) FixedDegreeGraph gives exact in-degree d."""
        n, d = 500, 8
        g = FixedDegreeGraph(n, d, device="cpu", seed=0)
        assert g.num_edges == n * d
        deg = g.csr.row_ptr[1:] - g.csr.row_ptr[:-1]
        assert int(deg.min()) == d and int(deg.max()) == d
        assert int((deg == 0).sum()) == 0

    def test_symmetric_false_reproduces_legacy_half_degree(self):
        """The opt-out flag restores the pre-v1.1 half-degree behaviour."""
        n, d = 500, 8
        g = FixedDegreeGraph(n, d, device="cpu", symmetric=False, seed=0)
        assert g.num_edges == n * d // 2

    def test_seed_reproducible(self):
        a = FixedDegreeGraph(400, 6, device="cpu", seed=42).edge_index
        b = FixedDegreeGraph(400, 6, device="cpu", seed=42).edge_index
        assert torch.equal(a, b)


# ---------------------------------------------------------------- C6 --------
class TestSEIRValidation:
    def test_mean_less_than_median_raises(self):
        with pytest.raises(ValueError):
            SEIRModel(mean_ir=1.0, median_ir=2.0)

    def test_zero_median_raises(self):
        with pytest.raises(ValueError):
            SEIRModel(median_ei=0.0)

    def test_valid_model_constructs(self):
        SEIRModel(mean_ei=5.0, median_ei=4.0, mean_ir=3.9, median_ir=1.5)


# ------------------------------------------------------------- NEW-2 --------
class TestGraphCSRBounds:
    def test_out_of_range_index_raises(self):
        ei = torch.tensor([[0, 1], [1, 5]])  # node 5 >= num_nodes=2
        with pytest.raises(ValueError):
            GraphCSR(ei, num_nodes=2)

    def test_negative_index_raises(self):
        ei = torch.tensor([[0, -1], [1, 0]])
        with pytest.raises(ValueError):
            GraphCSR(ei, num_nodes=2)

    def test_valid_graph_ok(self):
        assert GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2).num_edges == 2


# --------------------------------------------------------- C2 / R1 / C4 -----
class TestEngineReproducibility:
    def _graph(self):
        return FixedDegreeGraph(300, 8, device="cpu", seed=1)

    def test_seed_infection_reproducible(self):
        g = self._graph()
        e1 = RenewalEngine(g, SEIRModel(), device="cpu", seed=7)
        e2 = RenewalEngine(g, SEIRModel(), device="cpu", seed=7)
        e1.seed_infection(30)
        e2.seed_infection(30)
        assert torch.equal(e1.state, e2.state)

    def test_different_seed_differs(self):
        g = self._graph()
        e1 = RenewalEngine(g, SEIRModel(), device="cpu", seed=7)
        e2 = RenewalEngine(g, SEIRModel(), device="cpu", seed=999)
        e1.seed_infection(30)
        e2.seed_infection(30)
        assert not torch.equal(e1.state, e2.state)

    def test_reset_reproduces_and_episode_differs(self):
        g = self._graph()
        e = RenewalEngine(g, SEIRModel(), device="cpu", seed=7)
        e.seed_infection(30)
        base = e.state.clone()
        e.reset()
        e.seed_infection(30)
        assert torch.equal(e.state, base)          # C4: plain reset reproduces
        e.reset(episode=1)
        e.seed_infection(30)
        assert not torch.equal(e.state, base)       # C4: episode shift decorrelates

    def test_epsilon_validation(self):
        g = self._graph()
        with pytest.raises(ValueError):
            RenewalEngine(g, SEIRModel(), device="cpu", epsilon=0.0)

    def test_num_infected_bounds(self):
        g = self._graph()
        e = RenewalEngine(g, SEIRModel(), device="cpu", seed=1)
        with pytest.raises(ValueError):
            e.seed_infection(g.num_nodes + 1)

    def test_cpu_run_conserves_population(self):
        g = self._graph()
        e = RenewalEngine(g, SEIRModel(), device="cpu", seed=3)
        e.seed_infection(20)
        for _ in range(15):
            tau, _ = e.step()
            assert tau > 0
        assert int(e.count_by_state().sum()) == g.num_nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
