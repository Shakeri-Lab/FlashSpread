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
from flashspread.core.flash_neighbor import (
    reference_influence_csr,
    reference_influence_infectivity_csr,
)
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

    @pytest.mark.parametrize("field", ["mean_ei", "median_ei", "mean_ir", "median_ir"])
    def test_nonfinite_dwell_parameters_raise(self, field):
        with pytest.raises(ValueError, match="finite"):
            SEIRModel(**{field: float("inf")})

    def test_valid_model_constructs(self):
        SEIRModel(mean_ei=5.0, median_ei=4.0, mean_ir=3.9, median_ir=1.5)

    def test_invalid_beta_and_transmission_mode_raise(self):
        with pytest.raises(ValueError, match="beta"):
            SEIRModel(beta=-0.1)
        with pytest.raises(ValueError, match="transmission_mode"):
            SEIRModel(transmission_mode="typo")


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

    def test_float_indices_rejected(self):
        with pytest.raises(TypeError, match="integer dtype"):
            GraphCSR(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), 2)

    def test_weight_shape_and_values_validated(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match=r"shape \[E\]"):
            GraphCSR(edges, 2, weights=torch.ones(1, 2))
        with pytest.raises(ValueError, match="non-negative"):
            GraphCSR(edges, 2, weights=torch.tensor([1.0, -1.0]))


class TestCanonicalWeightedCSR:
    @staticmethod
    def _graph():
        # Unsorted, directed, weighted, duplicate edge, self-loop, isolated 4.
        edges = torch.tensor(
            [[2, 0, 1, 2, 0, 3], [0, 2, 2, 0, 2, 3]], dtype=torch.int64
        )
        weights = torch.tensor([3.0, 5.0, 7.0, 11.0, 13.0, 17.0])
        return GraphCSR(edges, 5, weights=weights), edges, weights

    def test_csr_reference_matches_independent_oracle(self):
        graph, edges, weights = self._graph()
        state = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int32)
        expected = torch.zeros(5)
        for edge, weight in zip(edges.t(), weights):
            source, target = edge.tolist()
            if state[source] == 1:
                expected[target] += weight
        actual = reference_influence_csr(graph, state, 1)
        assert torch.equal(actual, expected)

    def test_infectivity_reference_and_transpose_preserve_weights(self):
        graph, edges, weights = self._graph()
        infectivity = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0])
        expected_in = torch.zeros(5)
        expected_out = torch.zeros(5)
        for edge, weight in zip(edges.t(), weights):
            source, target = edge.tolist()
            expected_in[target] += weight * infectivity[source]
            expected_out[source] += weight * infectivity[target]
        assert torch.equal(
            reference_influence_infectivity_csr(graph, infectivity), expected_in
        )
        assert torch.equal(
            reference_influence_infectivity_csr(graph.transpose(), infectivity),
            expected_out,
        )

    def test_direct_csr_runs_cpu_renewal(self):
        graph, _, _ = self._graph()
        engine = RenewalEngine(graph, SEIRModel(), device="cpu", seed=3)
        engine.seed_infection(1)
        tau, _ = engine.step()
        assert tau > 0.0
        assert int(engine.count_by_state().sum()) == graph.num_nodes

    def test_unit_weights_are_symbolic_not_edge_sized(self):
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        graph = GraphCSR(edges, 4)
        assert not graph.has_weights
        assert graph.weights_storage.numel() == 1
        assert graph.weights.shape == (4,)
        assert graph.weights.is_contiguous()
        assert graph.weights.sum().item() == 4.0

    def test_exposed_unit_weights_materialize_and_mutate_consistently(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        graph = GraphCSR(edges, 2)
        graph.weights[0] = 3.0
        assert graph.has_weights
        assert graph.weights_storage.tolist() == [3.0, 1.0]
        state = torch.tensor([1, 1], dtype=torch.int32)
        assert reference_influence_csr(graph, state, 1).tolist() == [3.0, 1.0]

    def test_weight_assignment_rejects_invalid_values_and_invalidates_transpose(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        graph = GraphCSR(edges, 2)
        first = graph.transpose()
        with pytest.raises(ValueError, match="finite"):
            graph.weights = torch.tensor([float("nan"), 1.0])
        graph.weights = torch.tensor([2.0, 4.0])
        assert graph.transpose() is not first

    def test_exposed_weight_alias_disables_stale_transpose_cache(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        graph = GraphCSR(edges, 2)
        exposed = graph.weights
        first = graph.transpose()
        exposed[0] = 3.0
        rebuilt = graph.transpose()
        assert rebuilt is not first
        state = torch.tensor([1, 1], dtype=torch.int32)
        assert reference_influence_csr(rebuilt, state, 1).tolist() == [1.0, 3.0]

    def test_kernel_weight_alias_mutation_invalidates_cached_transpose(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        graph = GraphCSR(edges, 2, weights=torch.tensor([2.0, 4.0]))
        storage = graph.weights_storage
        first = graph.transpose()
        storage[0] = 9.0
        assert graph.transpose() is not first

    def test_mutated_transpose_view_is_not_reused(self):
        graph = GraphCSR(
            torch.tensor([[0, 1], [1, 2]]),
            3,
            weights=torch.tensor([1.0, 2.0]),
        )
        cached = graph.transpose()
        cached.col_ind[0] = 2

        rebuilt = graph.transpose()

        assert rebuilt is not cached
        assert torch.equal(rebuilt.to_edge_index(), graph.to_edge_index())

    def test_repeated_bf16_conversion_owns_weight_storage(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        graph = GraphCSR(
            edges, 2, weights=torch.tensor([2.0, 4.0])
        ).to_bf16_weights()
        converted = graph.to_bf16_weights()
        first = converted.transpose()
        graph.weights_storage[0] = 9.0
        assert converted.weights_storage[0].item() == 4.0
        assert converted.transpose() is first

    def test_from_csr_owns_caller_weight_storage(self):
        row = torch.tensor([0, 1, 2], dtype=torch.int32)
        col = torch.tensor([1, 0], dtype=torch.int32)
        weights = torch.tensor([2.0, 4.0])
        graph = GraphCSR.from_csr(row, col, weights=weights)
        weights[0] = 99.0
        assert graph.weights.tolist() == [2.0, 4.0]

    def test_with_weights_returns_replacement_without_mutating_original(self):
        graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
        replacement = graph.with_weights(torch.tensor([2.0, 4.0]))

        assert graph.has_weights is False
        assert replacement.has_weights is True
        assert replacement.weights.tolist() == [2.0, 4.0]
        assert replacement.row_ptr.data_ptr() == graph.row_ptr.data_ptr()
        assert replacement.col_ind.data_ptr() == graph.col_ind.data_ptr()

    @pytest.mark.parametrize(
        "engine_kind", ["renewal", "renewal_nonmarkov", "markov"]
    )
    def test_engine_rejects_graph_mutation_after_binding(self, engine_kind):
        graph = GraphCSR(
            torch.tensor([[0, 1], [1, 0]]),
            2,
            weights=torch.tensor([2.0, 4.0]),
        )
        if engine_kind == "renewal":
            engine = RenewalEngine(graph, SEIRModel(), device="cpu")
        elif engine_kind == "renewal_nonmarkov":
            from flashspread.engines.renewal import RenewalEngineNonMarkov

            engine = RenewalEngineNonMarkov(
                graph, SEIRModel(), device="cpu"
            )
        else:
            from flashspread import SISModel
            from flashspread.engines.markovian import MarkovianEngine

            engine = MarkovianEngine(graph, SISModel(), device="cpu")
        graph.weights_storage[0] = 3.0
        with pytest.raises(RuntimeError, match="GraphCSR storage changed"):
            engine.step()

    def test_index_mutation_and_symbolic_weight_materialization_are_rejected(self):
        graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
        engine = RenewalEngine(graph, SEIRModel(), device="cpu")
        graph.col_ind[0] = 0
        with pytest.raises(RuntimeError, match="GraphCSR storage changed"):
            engine.step()

        graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
        engine = RenewalEngine(graph, SEIRModel(), device="cpu")
        _ = graph.weights
        with pytest.raises(RuntimeError, match="GraphCSR storage changed"):
            engine.step()

    def test_captured_replay_checks_graph_before_touching_cuda(self):
        graph = GraphCSR(
            torch.tensor([[0, 1], [1, 0]]),
            2,
            weights=torch.tensor([2.0, 4.0]),
        )
        engine = RenewalEngine(graph, SEIRModel(), device="cpu")
        graph.weights_storage[0] = 3.0
        with pytest.raises(RuntimeError, match="GraphCSR storage changed"):
            engine._replay_cuda_graph()

    def test_inference_mode_graph_storage_is_normalized_to_versioned_tensors(self):
        with torch.inference_mode():
            graph = GraphCSR.from_csr(
                torch.tensor([0, 1, 2], dtype=torch.int32),
                torch.tensor([1, 0], dtype=torch.int32),
            )
        assert not torch.is_inference(graph.row_ptr)
        assert not torch.is_inference(graph.col_ind)
        assert not torch.is_inference(graph.weights_storage)
        engine = RenewalEngine(graph, SEIRModel(), device="cpu")
        assert engine.step()[0] > 0.0

    def test_direct_csr_constructor_avoids_coo_roundtrip(self):
        graph, _, _ = self._graph()
        rebuilt = GraphCSR.from_csr(
            graph.row_ptr, graph.col_ind, weights=graph.weights
        )
        assert torch.equal(rebuilt.row_ptr, graph.row_ptr)
        assert torch.equal(rebuilt.col_ind, graph.col_ind)
        assert torch.equal(rebuilt.weights, graph.weights)

    def test_weighted_edge_file_loads_without_loss(self, tmp_path):
        from flashspread import load_graph

        path = tmp_path / "weighted.edgelist"
        path.write_text("0 1 2.5\n1 0 4.0\n")
        graph = load_graph(path, num_nodes=2)
        assert graph.has_weights
        assert sorted(graph.weights.tolist()) == [2.5, 4.0]


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

    def test_high_seed_bits_change_cpu_initialization_stream(self):
        graph = self._graph()
        low = RenewalEngine(graph, SEIRModel(), device="cpu", seed=0)
        high = RenewalEngine(graph, SEIRModel(), device="cpu", seed=2**32)
        low.seed_infection(30)
        high.seed_infection(30)
        assert not torch.equal(low.state, high.state)

    @pytest.mark.parametrize("seed", [2**63, 2**64 - 1, -1])
    def test_full_torch_seed_range_is_supported(self, seed):
        g = self._graph()
        engine = RenewalEngine(g, SEIRModel(), device="cpu", seed=seed)
        engine.seed_infection(3)
        assert engine.count_by_state().sum().item() == g.num_nodes

    def test_boolean_seed_is_rejected(self):
        with pytest.raises(TypeError, match="seed"):
            RenewalEngine(self._graph(), SEIRModel(), device="cpu", seed=True)

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

    @pytest.mark.parametrize("episode", [True, 1.5, "2"])
    def test_reset_rejects_invalid_episode_without_mutating_state(self, episode):
        engine = RenewalEngine(self._graph(), SEIRModel(), device="cpu", seed=7)
        engine.seed_infection(30)
        before = engine.state.clone()
        with pytest.raises(TypeError, match="episode"):
            engine.reset(episode=episode)
        assert torch.equal(engine.state, before)

    def test_epsilon_validation(self):
        g = self._graph()
        with pytest.raises(ValueError):
            RenewalEngine(g, SEIRModel(), device="cpu", epsilon=0.0)
        with pytest.raises(ValueError, match="epsilon"):
            RenewalEngine(g, SEIRModel(), device="cpu", epsilon=float("inf"))
        with pytest.raises(ValueError, match="tau_max"):
            RenewalEngine(g, SEIRModel(), device="cpu", tau_max=float("inf"))
        with pytest.raises(TypeError, match="epsilon"):
            RenewalEngine(g, SEIRModel(), device="cpu", epsilon=True)
        with pytest.raises(TypeError, match="tau_max"):
            RenewalEngine(g, SEIRModel(), device="cpu", tau_max=True)
        with pytest.raises(ValueError, match="representable"):
            RenewalEngine(g, SEIRModel(), device="cpu", epsilon=1e-50)
        with pytest.raises(ValueError, match="representable"):
            RenewalEngine(g, SEIRModel(), device="cpu", tau_max=1e300)

    def test_tiny_positive_rate_still_obeys_epsilon_bound(self):
        class TinyRateModel:
            is_markovian = False
            num_states = 2
            inducer_states = [1]

            def prepare(self, device):
                pass

            def compute_rates(self, age, state, pressure, out=None):
                out.fill_(5e-10)
                return out

            def apply_transitions(self, state, event_mask, out=None):
                out.copy_(state)
                return out

        graph = GraphCSR(
            torch.empty((2, 0), dtype=torch.int64),
            2,
        )
        engine = RenewalEngine(
            graph,
            TinyRateModel(),
            device="cpu",
            epsilon=0.03,
            tau_max=1e12,
        )

        tau, _ = engine.step()

        assert tau == pytest.approx(0.03 / 5e-10, rel=2e-6)
        assert engine.rates.max().item() * tau <= 0.03 * (1.0 + 2e-6)

    def test_mixed_negative_renewal_rates_fail_loudly(self):
        class NegativeRateModel:
            is_markovian = False
            num_states = 2
            inducer_states = [1]

            def compute_rates(self, age, state, pressure, out=None):
                out.copy_(torch.tensor([-1.0, 1.0], device=state.device))
                return out

            def apply_transitions(self, state, event_mask, out=None):
                return out.copy_(state)

        graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
        engine = RenewalEngine(graph, NegativeRateModel(), device="cpu")
        state_before = engine.state.clone()
        age_before = engine.age.clone()
        seed_before = engine.seed_counter.clone()
        with pytest.raises(FloatingPointError, match="tau"):
            engine.step()
        assert torch.equal(engine.state, state_before)
        assert torch.equal(engine.age, age_before)
        assert torch.equal(engine.seed_counter, seed_before)

    def test_renewal_tau_underflow_fails_without_advancing_rng_or_state(self):
        class MaxRateModel:
            is_markovian = False
            num_states = 2
            inducer_states = [1]

            def compute_rates(self, age, state, pressure, out=None):
                return out.fill_(torch.finfo(torch.float32).max)

            def apply_transitions(self, state, event_mask, out=None):
                return out.copy_(state)

        graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
        engine = RenewalEngine(
            graph, MaxRateModel(), device="cpu", epsilon=2.0**-149
        )
        state_before = engine.state.clone()
        age_before = engine.age.clone()
        seed_before = engine.seed_counter.clone()
        with pytest.raises(FloatingPointError, match="tau"):
            engine.step()
        assert torch.equal(engine.state, state_before)
        assert torch.equal(engine.age, age_before)
        assert torch.equal(engine.seed_counter, seed_before)

    def test_reference_rng_uses_open_high_resolution_uniforms(self):
        g = self._graph()
        engine = RenewalEngine(g, SEIRModel(), device="cpu", seed=7)
        engine.seed_counter.fill_(0)
        engine.seed_counter[:3] = torch.tensor([0, 1, -1])
        values = engine._rand_uniform(engine.rand_buffer)
        assert values.dtype == torch.float64
        assert bool((values > 0.0).all()) and bool((values < 1.0).all())

        golden_words = (
            16294208416658607535,
            10451216379200822465,
            16490336266968443936,
        )
        expected = torch.tensor(
            [
                ((word >> 12) + 0.5) * 2.0**-52
                for word in golden_words
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(values[:3], expected, rtol=0.0, atol=0.0)

    def test_num_infected_bounds(self):
        g = self._graph()
        e = RenewalEngine(g, SEIRModel(), device="cpu", seed=1)
        with pytest.raises(ValueError):
            e.seed_infection(g.num_nodes + 1)
        with pytest.raises(TypeError, match="integer"):
            e.seed_infection(1.5)
        with pytest.raises(TypeError, match="integer"):
            e.seed_infection(1, state=1.5)

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
