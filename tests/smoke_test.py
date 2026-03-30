#!/usr/bin/env python
"""
Smoke tests for FlashSpread package.

These tests verify basic functionality without requiring extensive computation.
Run with: python -m pytest tests/smoke_test.py -v
"""

import pytest
import torch


# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestGraphCSR:
    """Test GraphCSR construction and basic properties."""

    def test_csr_construction(self):
        """Verify CSR format is constructed correctly."""
        from flashspread.core.graph import GraphCSR

        # Simple triangle graph: 0 <-> 1 <-> 2 <-> 0
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2]
        ], dtype=torch.long, device="cuda")

        graph = GraphCSR(edge_index, num_nodes=3, incoming=True)

        assert graph.num_nodes == 3
        assert graph.num_edges == 6
        assert graph.row_ptr.shape == (4,)
        assert graph.col_ind.shape == (6,)

    def test_csr_weights(self):
        """Verify edge weights are handled correctly."""
        from flashspread.core.graph import GraphCSR

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device="cuda")
        weights = torch.tensor([2.0, 3.0], device="cuda")

        graph = GraphCSR(edge_index, num_nodes=2, weights=weights)

        assert graph.weights.sum().item() == pytest.approx(5.0)


class TestFlashNeighbor:
    """Test FlashNeighbor kernel against reference implementation."""

    def test_influence_computation(self):
        """Verify FlashNeighbor matches reference scatter_add."""
        from flashspread.core.graph import GraphCSR
        from flashspread.core.flash_neighbor import FlashNeighbor, reference_influence

        # Create a small test graph
        num_nodes = 100
        # Random edges
        torch.manual_seed(42)
        num_edges = 500
        src = torch.randint(0, num_nodes, (num_edges,), device="cuda")
        dst = torch.randint(0, num_nodes, (num_edges,), device="cuda")
        edge_index = torch.stack([src, dst])

        # Random states (0 = S, 1 = I)
        states = torch.randint(0, 2, (num_nodes,), device="cuda", dtype=torch.int32)
        inducer_state = 1  # Infected

        # Build CSR and FlashNeighbor
        graph = GraphCSR(edge_index, num_nodes, incoming=True)
        flash = FlashNeighbor(graph, [inducer_state])

        # Compute with both methods
        flash_result = flash.compute_influence(states)
        ref_result = reference_influence(edge_index, num_nodes, states, [inducer_state])

        # Compare
        diff = (flash_result - ref_result).abs().max().item()
        assert diff < 1e-5, f"FlashNeighbor differs from reference by {diff}"


class TestSISModel:
    """Test SIS model rate computation."""

    def test_sis_rates(self):
        """Verify SIS rate computation."""
        from flashspread.models import SISModel

        model = SISModel(beta=0.5, delta=1.0)
        model.prepare(torch.device("cuda"))

        # Test with 4 nodes: 2 susceptible, 2 infected
        state = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int32)
        influence = torch.tensor([1.0, 2.0, 0.0, 0.0], device="cuda")

        rates = model.compute_rates(state, influence)

        # S nodes: rate = beta * influence
        assert rates[0].item() == pytest.approx(0.5)  # 0.5 * 1.0
        assert rates[1].item() == pytest.approx(1.0)  # 0.5 * 2.0

        # I nodes: rate = delta
        assert rates[2].item() == pytest.approx(1.0)
        assert rates[3].item() == pytest.approx(1.0)


class TestSEIRModel:
    """Test SEIR model with age-dependent hazards."""

    def test_seir_hazard_positive(self):
        """Verify SEIR hazards are positive and finite."""
        from flashspread.models import SEIRModel

        model = SEIRModel(
            beta=0.3, mean_ei=5.0, median_ei=4.0, mean_ir=3.9, median_ir=1.5
        )
        model.prepare(torch.device("cuda"))

        # Various ages
        age = torch.tensor([0.1, 1.0, 5.0, 10.0, 50.0], device="cuda")
        state = torch.tensor([1, 1, 1, 1, 1], device="cuda", dtype=torch.int32)  # All E
        pressure = torch.zeros(5, device="cuda")

        rates = model.compute_rates(age, state, pressure)

        assert torch.all(rates > 0), "Hazards should be positive"
        assert torch.all(torch.isfinite(rates)), "Hazards should be finite"


class TestMarkovianEngine:
    """Test Markovian engine basic operation."""

    def test_engine_step(self):
        """Verify engine can run steps without error."""
        from flashspread import MarkovianEngine, SISModel, FixedDegreeGraph

        graph = FixedDegreeGraph(100, 10, device="cuda")
        model = SISModel(beta=0.5, delta=1.0)
        engine = MarkovianEngine(graph, model, device="cuda")

        engine.seed_infection(10)
        initial_infected = engine.count_infected()

        # Run a few steps
        for _ in range(10):
            tau, events = engine.step()
            assert tau > 0, "Time step should be positive"

        # State should still be valid
        counts = engine.count_by_state()
        assert counts.sum().item() == 100, "Population should be conserved"

    def test_engine_reset(self):
        """Verify engine reset works."""
        from flashspread import MarkovianEngine, SISModel, FixedDegreeGraph

        graph = FixedDegreeGraph(100, 10, device="cuda")
        model = SISModel()
        engine = MarkovianEngine(graph, model, device="cuda")

        engine.seed_infection(10)
        for _ in range(5):
            engine.step()

        engine.reset()
        assert engine.current_time == 0.0
        assert engine.count_infected() == 0


class TestRenewalEngine:
    """Test Renewal engine basic operation."""

    def test_renewal_step(self):
        """Verify renewal engine can run steps."""
        from flashspread import RenewalEngine, SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(100, 10, device="cuda")
        model = SEIRModel(beta=0.3, mean_ei=5.0, median_ei=4.0, mean_ir=3.9, median_ir=1.5)
        engine = RenewalEngine(graph, model, device="cuda")

        engine.seed_infection(10, state=model.exposed)

        # Run steps
        for _ in range(10):
            tau, state = engine.step()
            assert tau > 0

        # Check population conservation
        counts = engine.count_by_state()
        assert counts.sum().item() == 100

    def test_age_reset(self):
        """Verify age resets on transition (renewal property)."""
        from flashspread import RenewalEngine, SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(100, 10, device="cuda")
        model = SEIRModel(beta=0.3, mean_ei=5.0, median_ei=4.0, mean_ir=3.9, median_ir=1.5)
        engine = RenewalEngine(graph, model, device="cuda")

        # Set all to Exposed with various ages
        engine.state.fill_(model.exposed)
        engine.age = torch.rand(100, device="cuda") * 10

        initial_ages = engine.age.clone()

        # Run one step
        engine.step()

        # Nodes that transitioned should have age reset
        transitioned = engine.state != model.exposed
        if transitioned.any():
            # Check that transitioned nodes have age < initial
            # (they reset to 0 then advanced by tau)
            assert (engine.age[transitioned] < initial_ages[transitioned]).all()


class TestBF16Weights:
    """Test BF16 weight downcasting (Phase 1)."""

    def test_bf16_influence_accuracy(self):
        """FlashNeighbor with bf16 weights matches fp32 within tolerance."""
        from flashspread import FlashNeighbor, FixedDegreeGraph

        graph = FixedDegreeGraph(1000, 10, device="cuda")
        states = torch.zeros(1000, device="cuda", dtype=torch.int32)
        states[:100] = 1  # 100 infected

        # FP32 baseline
        fn_fp32 = FlashNeighbor(graph.csr, inducer_states=[1])
        result_fp32 = fn_fp32.compute_influence(states).clone()

        # BF16 weights
        graph_bf16 = graph.csr.to_bf16_weights()
        fn_bf16 = FlashNeighbor(graph_bf16, inducer_states=[1])
        result_bf16 = fn_bf16.compute_influence(states)

        assert torch.allclose(result_fp32, result_bf16, atol=1e-3)

    def test_renewal_engine_bf16(self):
        """RenewalEngine with bf16_weights runs without error."""
        from flashspread import RenewalEngine, SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(100, 10, device="cuda")
        model = SEIRModel(beta=0.3)
        engine = RenewalEngine(graph, model, device="cuda", bf16_weights=True)
        engine.seed_infection(10, state=model.exposed)

        for _ in range(5):
            tau, state = engine.step()
            assert tau > 0

        counts = engine.count_by_state()
        assert counts.sum().item() == 100


class TestFlashNeighborInfectivity:
    """Test infectivity-weighted kernel (Phase 2)."""

    def test_infectivity_kernel_vs_reference(self):
        """FlashNeighborInfectivity matches reference scatter_add."""
        from flashspread.core.flash_neighbor import (
            FlashNeighborInfectivity,
            reference_influence_infectivity,
        )
        from flashspread import FixedDegreeGraph

        graph = FixedDegreeGraph(500, 10, device="cuda")

        # Random infectivity values
        infectivity = torch.rand(500, device="cuda", dtype=torch.float32)
        infectivity[100:] = 0.0  # Only first 100 are "infectious"

        # Triton kernel
        fn_inf = FlashNeighborInfectivity(graph.csr)
        result_triton = fn_inf.compute_influence(infectivity).clone()

        # Reference
        result_ref = reference_influence_infectivity(
            graph.edge_index, 500, infectivity,
            weights=graph.csr.weights,
        )

        assert torch.allclose(result_triton, result_ref, atol=1e-4), \
            f"Max diff: {(result_triton - result_ref).abs().max().item()}"

    def test_seir_compute_infectivity(self):
        """SEIRModel.compute_infectivity produces non-zero for I-nodes only."""
        from flashspread import SEIRModel

        model = SEIRModel(beta=0.3)
        model.prepare(torch.device("cuda"))

        state = torch.zeros(100, device="cuda", dtype=torch.int32)
        state[10:30] = 2  # 20 nodes infected
        age = torch.ones(100, device="cuda") * 3.0  # age = 3 days

        infectivity = model.compute_infectivity(age, state)

        # Only I-nodes should have non-zero infectivity
        assert (infectivity[:10] == 0).all()
        assert (infectivity[10:30] > 0).all()
        assert (infectivity[30:] == 0).all()


class TestRenewalEngineNonMarkov:
    """Test non-Markovian edge engine (Phase 2)."""

    def test_nonmarkov_step(self):
        """RenewalEngineNonMarkov runs steps and conserves population."""
        from flashspread.engines.renewal import RenewalEngineNonMarkov
        from flashspread import SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(100, 10, device="cuda")
        model = SEIRModel(beta=0.3)
        engine = RenewalEngineNonMarkov(graph, model, device="cuda")
        engine.seed_infection(10, state=model.exposed)

        for _ in range(10):
            tau, state = engine.step()
            assert tau > 0

        counts = engine.count_by_state()
        assert counts.sum().item() == 100

    def test_nonmarkov_cudagraph(self):
        """RenewalEngineNonMarkovCUDAGraph runs and conserves population."""
        from flashspread.engines.renewal import RenewalEngineNonMarkovCUDAGraph
        from flashspread import SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(200, 10, device="cuda")
        model = SEIRModel(beta=0.3)
        engine = RenewalEngineNonMarkovCUDAGraph(
            graph, model, device="cuda", steps_per_launch=10
        )
        engine.seed_infection(20, state=model.exposed)

        elapsed, state = engine.step()
        assert elapsed > 0

        counts = engine.count_by_state()
        assert counts.sum().item() == 200


class TestErfcxApprox:
    """Test erfcx rational approximation (Phase 3)."""

    def test_erfcx_accuracy(self):
        """erfcx_rational_approx matches torch.special.erfcx."""
        from flashspread.models.hazards import erfcx_rational_approx

        z = torch.linspace(-8, 40, 1000, device="cuda")
        ref = torch.special.erfcx(z)
        approx = erfcx_rational_approx(z)

        # Relative error where ref > 1e-20
        valid = ref > 1e-20
        rel_err = ((approx[valid] - ref[valid]) / ref[valid]).abs()
        # 5e-4 tolerance: the approximation uses 1-erf(z) which loses
        # precision near the boundary; this is acceptable for tau-leaping
        assert rel_err.max().item() < 5e-4, \
            f"Max relative error: {rel_err.max().item()}"


class TestRenewalEngineFused:
    """Test fused Triton kernel engine (Phase 3)."""

    def test_fused_step(self):
        """RenewalEngineFused runs steps and conserves population."""
        from flashspread.engines.renewal_fused import RenewalEngineFused
        from flashspread import SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(100, 10, device="cuda")
        model = SEIRModel(beta=0.3)
        engine = RenewalEngineFused(graph, model, device="cuda")
        engine.seed_infection(10, state=model.exposed)

        for _ in range(10):
            tau, state = engine.step()
            assert tau > 0

        counts = engine.count_by_state()
        assert counts.sum().item() == 100

    def test_fused_cudagraph(self):
        """RenewalEngineFusedCUDAGraph runs and conserves population."""
        from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph
        from flashspread import SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(200, 10, device="cuda")
        model = SEIRModel(beta=0.3)
        engine = RenewalEngineFusedCUDAGraph(
            graph, model, device="cuda", steps_per_launch=10
        )
        engine.seed_infection(20, state=model.exposed)

        elapsed, state = engine.step()
        assert elapsed > 0

        counts = engine.count_by_state()
        assert counts.sum().item() == 200


class TestFactoryFunction:
    """Test updated create_renewal_engine factory."""

    def test_factory_default_is_fused_cg(self):
        """Default factory creates FusedCUDAGraph engine."""
        from flashspread.engines import create_renewal_engine
        from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph
        from flashspread import SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(100, 10, device="cuda")
        model = SEIRModel(beta=0.3)
        engine = create_renewal_engine(graph, model)
        assert isinstance(engine, RenewalEngineFusedCUDAGraph)

    def test_factory_unfused_nonmarkov(self):
        """Factory with use_fused=False creates NonMarkov engine."""
        from flashspread.engines import create_renewal_engine
        from flashspread.engines.renewal import RenewalEngineNonMarkovCUDAGraph
        from flashspread import SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(100, 10, device="cuda")
        model = SEIRModel(beta=0.3)
        engine = create_renewal_engine(
            graph, model, use_fused=False, use_cuda_graph=True
        )
        assert isinstance(engine, RenewalEngineNonMarkovCUDAGraph)

    def test_factory_bf16(self):
        """Factory with bf16_weights creates engine without error."""
        from flashspread.engines import create_renewal_engine
        from flashspread import SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(100, 10, device="cuda")
        model = SEIRModel(beta=0.3)
        engine = create_renewal_engine(
            graph, model, bf16_weights=True, use_cuda_graph=False
        )
        engine.seed_infection(5, state=model.exposed)
        tau, state = engine.step()
        assert tau > 0


class TestNetworkGeneration:
    """Test network generation utilities."""

    def test_fixed_degree(self):
        """Test FixedDegreeGraph creation."""
        from flashspread.core.network import FixedDegreeGraph

        graph = FixedDegreeGraph(1000, 10, device="cuda")

        assert graph.num_nodes == 1000
        # Each node has degree 10, undirected -> 2 * edges in directed form
        assert graph.num_edges > 0

    def test_random_geometric(self):
        """Test RandomGeometricGraph creation."""
        from flashspread.core.network import RandomGeometricGraph

        graph = RandomGeometricGraph(1000, 0.1, device="cuda")

        assert graph.num_nodes == 1000
        assert graph.num_edges > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
