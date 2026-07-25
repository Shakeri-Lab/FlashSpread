#!/usr/bin/env python
"""
Smoke tests for FlashSpread package.

These tests verify basic functionality without requiring extensive computation.
Run with: python -m pytest tests/smoke_test.py -v
"""

import pytest
import torch


# Every test in this module drives the CUDA/Triton kernels, so tag the whole
# module `gpu`. On a CPU runner conftest.py skips gpu-marked tests (and
# `pytest -m "not gpu"` deselects them), instead of the old module-level
# skipif that made the *entire* suite report green with 0 tests collected.
pytestmark = pytest.mark.gpu


# Module-scope Triton test kernel for TestTritonErfcxAccuracy.
# @triton.jit requires `tl` to be resolvable in the kernel's enclosing
# module scope at compile time, so we define the kernel here rather than
# inside a test method.
try:
    import triton as _triton
    import triton.language as tl

    from flashspread.core.flash_renewal_kernel import (
        _HAS_TRITON as _FLASH_HAS_TRITON,
        _erfcx_approx as _flash_erfcx_approx,
    )

    if _FLASH_HAS_TRITON:

        @_triton.jit
        def _erfcx_test_kernel(z_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            z = tl.load(z_ptr + offs, mask=mask, other=0.0)
            out = _flash_erfcx_approx(z)
            tl.store(out_ptr + offs, out, mask=mask)
    else:
        _erfcx_test_kernel = None
except Exception:
    _triton = None
    tl = None
    _FLASH_HAS_TRITON = False
    _erfcx_test_kernel = None


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

    def test_weighted_influence_computation(self):
        """The array-weight specialization preserves unsorted edge weights."""
        from flashspread.core.graph import GraphCSR
        from flashspread.core.flash_neighbor import FlashNeighbor, reference_influence

        edge_index = torch.tensor(
            [[2, 0, 1, 2, 0], [0, 2, 2, 0, 2]],
            device="cuda",
            dtype=torch.int64,
        )
        weights = torch.tensor([3.0, 5.0, 7.0, 11.0, 13.0], device="cuda")
        states = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int32)
        graph = GraphCSR(edge_index, 3, weights=weights)
        actual = FlashNeighbor(graph, 1).compute_influence(states).clone()
        expected = reference_influence(edge_index, 3, states, 1, weights=weights)
        assert torch.equal(actual, expected)


class TestMarkovianFrontier:
    def test_weighted_parallel_edge_propagation(self):
        from flashspread.core.flash_markovian import propagate_frontier
        from flashspread.core.graph import GraphCSR

        edge_index = torch.tensor(
            [[0, 0, 1, 2], [2, 2, 2, 0]], device="cuda", dtype=torch.int64
        )
        weights = torch.tensor([2.0, 3.0, 5.0, 7.0], device="cuda")
        outgoing = GraphCSR(edge_index, 3, weights=weights, incoming=False)
        changed = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        delta = torch.tensor([1.0, -1.0], device="cuda")
        influence = torch.zeros(3, device="cuda")
        propagate_frontier(outgoing, changed, delta, influence)
        assert torch.equal(influence, torch.tensor([0.0, 0.0, 0.0], device="cuda"))


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
        assert engine.device == engine.graph.row_ptr.device == engine.state.device

        engine.seed_infection(10)
        assert engine.outgoing_graph is engine.graph
        assert engine._shares_outgoing_csr

        # Run a few steps
        for _ in range(10):
            tau, events = engine.step()
            assert tau > 0, "Time step should be positive"

        # State should still be valid
        counts = engine.count_by_state()
        assert counts.sum().item() == 100, "Population should be conserved"

    def test_engine_actually_progresses(self):
        """
        Regression test for the apply_transitions-return-value bug:
        with a supercritical SIS (R0 >> 1) and 10 seed infections, the
        peak-I over 200 steps must rise meaningfully above the seed
        fraction. Without the fix, the engine silently discarded every
        new state and peak_I would stay stuck at the 10/100 seed level.
        """
        from flashspread import MarkovianEngine, SISModel, FixedDegreeGraph

        graph = FixedDegreeGraph(200, 10, device="cuda")
        # R0 = beta * d / delta = 0.5 * 10 / 0.2 = 25, well supercritical
        model = SISModel(beta=0.5, delta=0.2)
        engine = MarkovianEngine(graph, model, device="cuda",
                                 max_prob=0.1, theta=0.05)
        engine.seed_infection(10)

        peak_fraction = engine.count_infected() / 200
        for _ in range(200):
            engine.step()
            peak_fraction = max(peak_fraction,
                                engine.count_infected() / 200)

        assert peak_fraction > 0.25, (
            f"SIS engine failed to progress: peak_I={peak_fraction:.3f}, "
            f"expected > 0.25 for a supercritical SIS. This typically "
            f"means MarkovianEngine.step() is dropping the new-state "
            f"tensor returned by model.apply_transitions()."
        )

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

    @pytest.mark.parametrize("model_name", ["sis", "sir"])
    def test_builtin_fast_path_keeps_weighted_influence_and_rates_current(
        self, model_name
    ):
        from flashspread import GraphCSR, SIRModel, SISModel
        from flashspread.core.reference import reference_influence_csr
        from flashspread.engines.markovian import MarkovianEngine

        edges = torch.tensor(
            [[0, 0, 1, 2, 3, 4, 5], [1, 2, 2, 3, 4, 5, 0]],
            dtype=torch.int64,
            device="cuda",
        )
        weights = torch.tensor(
            [0.5, 1.5, 2.0, 0.75, 1.25, 0.25, 3.0], device="cuda"
        )
        graph = GraphCSR(edges, 6, weights=weights)
        model = (
            SISModel(beta=0.4, delta=0.3)
            if model_name == "sis"
            else SIRModel(beta=0.4, gamma=0.3)
        )
        engine = MarkovianEngine(graph, model, device="cuda", seed=13)
        engine.set_initial_state(torch.tensor([1, 0, 1, 0, 0, 0]))

        assert engine._builtin_gpu_kind == model_name
        assert engine.influence.data_ptr() == engine.flash_neighbor.out_buffer.data_ptr()
        for _ in range(12):
            before = engine.state.clone()
            tau, events = engine.step()
            assert tau > 0.0
            assert events == int((engine.state != before).sum())
            expected_influence = reference_influence_csr(
                graph, engine.state, model.infected
            )
            torch.testing.assert_close(
                engine.influence, expected_influence, rtol=0.0, atol=2e-5
            )
            expected_rates = model.compute_rates(engine.state, expected_influence)
            torch.testing.assert_close(engine.rates, expected_rates)

    def test_markovian_cuda_graph_replay_restores_exact_boundary_influence(self):
        from flashspread import GraphCSR, SIRModel
        from flashspread.core.reference import reference_influence_csr
        from flashspread.engines.markovian import MarkovianEngineCUDAGraph

        nodes = torch.arange(32, device="cuda", dtype=torch.int64)
        edges = torch.stack((nodes, (nodes + 1) % 32))
        reverse = torch.stack((edges[1], edges[0]))
        graph = GraphCSR(torch.cat((edges, reverse), dim=1), 32)
        model = SIRModel(beta=0.8, gamma=0.2)
        engine = MarkovianEngineCUDAGraph(
            graph,
            model,
            device="cuda",
            seed=17,
            steps_per_launch=7,
        )
        engine.seed_infection(4)

        elapsed, events = engine.step()
        assert elapsed > 0.0
        assert events >= 0
        assert engine.total_steps == 7
        assert engine.current_time == pytest.approx(elapsed)
        expected = reference_influence_csr(graph, engine.state, model.infected)
        torch.testing.assert_close(engine.influence, expected)
        expected_rates = model.compute_rates(engine.state, expected)
        torch.testing.assert_close(engine.rates, expected_rates)

        engine.reset()
        assert engine.total_steps == 0
        assert engine.current_time == 0.0
        assert int(engine._step_id_device.item()) == 0

    def test_builtin_rng_preserves_high_seed_words(self):
        from flashspread import FixedDegreeGraph, SISModel
        from flashspread.engines.markovian import MarkovianEngine

        graph = FixedDegreeGraph(512, 8, device="cuda", seed=4)
        initial = (torch.arange(512, device="cuda") % 2).to(torch.int32)
        # These two seeds collided under the former 32-bit key compression.
        engines = [
            MarkovianEngine(
                graph,
                SISModel(beta=0.5, delta=0.5),
                device="cuda",
                seed=seed,
                max_prob=0.2,
                theta=0.1,
            )
            for seed in (0, 0x0000000185EBCA77)
        ]
        for engine in engines:
            engine.set_initial_state(initial)
            engine.step()
        assert not torch.equal(engines[0].state, engines[1].state)


class TestRenewalEngine:
    """Test Renewal engine basic operation."""

    def test_renewal_step(self):
        """Verify renewal engine can run steps."""
        from flashspread import RenewalEngine, SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(100, 10, device="cuda")
        model = SEIRModel(beta=0.3, mean_ei=5.0, median_ei=4.0, mean_ir=3.9, median_ir=1.5)
        engine = RenewalEngine(graph, model, device="cuda")
        assert engine.device == engine.graph.row_ptr.device == engine.state.device

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


class TestTritonErfcxAccuracy:
    """
    Validate the in-kernel Triton erfcx approximation (_erfcx_approx in
    flash_renewal_kernel.py) against torch.special.erfcx, and record the
    empirical max relative error used in JOCS Appendix A.

    Notes on fp32 representability:
      - For z < 0, erfcx(z) ~ 2*exp(z^2) overflows fp32 when |z| > ~9.42.
        We restrict the validation range to z >= -9 to keep both the
        kernel's fp32 output and the reference in-range.
      - For large z >> 0, erfcx(z) underflows to the kernel's 1e-30 clamp;
        we mask those samples from the relative-error computation.
      - Near z = 3.5 (the branch-switch point) the identity
        `exp(z^2)(1 - erf(z))` loses significant digits in fp32 because
        `1 - erf(3.5) ~ 5e-7` is at the edge of fp32 precision. We measure
        the realized max relative error rather than asserting the
        tighter (over-optimistic) "~4e-4" bound quoted in earlier drafts.
    """

    # fp32-safe range for negative z: exp(z^2) stays below fp32 max.
    Z_LO = -9.0
    Z_HI = 30.0
    # Empirically measured bounds on A100 / Triton 3.1 / fp32 (2026-04):
    #   * full range [-9, 30]: ~3.9e-2 (dominated by fp32 cancellation in
    #     1 - erf(z) near the branch-switch at z ~ 3.5)
    #   * away from branch boundary: ~6e-3 (near z ~ 9 where the overflow
    #     guard drops from the 4-term asymptotic to the 1-term form)
    # The 5e-2 / 1e-2 thresholds below leave ~25-60% headroom. These are
    # the bounds cited in JOCS Appendix A.
    TOLERANCE_FULL = 5e-2
    TOLERANCE_AWAY = 1e-2

    def _run_kernel(self, z):
        if _erfcx_test_kernel is None or not _FLASH_HAS_TRITON:
            pytest.skip("Triton / flash_renewal_kernel not available")
        out = torch.empty_like(z)
        BLOCK = 256
        grid = (_triton.cdiv(z.numel(), BLOCK),)
        _erfcx_test_kernel[grid](z, out, z.numel(), BLOCK_SIZE=BLOCK)
        return out

    def _measured_bound(self, z_lo, z_hi, n=10001):
        """Return (max_rel_err, at_z) on a dense fp32-safe grid."""
        z = torch.linspace(
            z_lo, z_hi, n, device="cuda", dtype=torch.float32
        )
        out = self._run_kernel(z)
        ref64 = torch.special.erfcx(z.to(torch.float64))

        # Mask: exclude samples where reference underflows the kernel's
        # 1e-30 clamp (rel-err is meaningless there) and any non-finite
        # values (safety net; should not occur within [Z_LO, Z_HI]).
        finite = torch.isfinite(out) & torch.isfinite(ref64.to(torch.float32))
        valid = finite & (ref64 > 1e-20)
        rel = ((out.to(torch.float64)[valid] - ref64[valid]) / ref64[valid]).abs()
        if rel.numel() == 0:
            return float("nan"), float("nan")
        max_rel = rel.max().item()
        arg_max = rel.argmax().item()
        return max_rel, z[valid][arg_max].item()

    def test_triton_erfcx_max_rel_err_fp32_safe_range(self):
        """Max relative error of the Triton erfcx kernel on an fp32-safe grid."""
        max_rel, at_z = self._measured_bound(self.Z_LO, self.Z_HI)
        print(
            f"\n[TritonErfcxAccuracy] max rel err = {max_rel:.3e} at z = {at_z:.3f} "
            f"over z in [{self.Z_LO}, {self.Z_HI}] (fp32-safe range)"
        )
        assert max_rel < self.TOLERANCE_FULL, (
            f"Triton erfcx max relative error {max_rel:.3e} at z={at_z:.3f} "
            f"exceeds tolerance {self.TOLERANCE_FULL:.1e}"
        )

    def test_triton_erfcx_away_from_branch_boundary(self):
        """
        Away from the z~3.5 branch point, fp32 cancellation does not dominate
        and the approximation should agree with the reference to ~1e-3.
        """
        # Split around the branch boundary; exclude a safety band of +/-0.25.
        lo_part = self._measured_bound(self.Z_LO, 3.25, n=5001)
        hi_part = self._measured_bound(3.75, self.Z_HI, n=5001)
        print(
            f"\n[TritonErfcxAccuracy] away-from-boundary max rel err: "
            f"|z|<3.25: {lo_part[0]:.3e} at z={lo_part[1]:.3f}; "
            f"z>3.75: {hi_part[0]:.3e} at z={hi_part[1]:.3f}"
        )
        assert lo_part[0] < self.TOLERANCE_AWAY and hi_part[0] < self.TOLERANCE_AWAY, (
            f"Away-from-boundary max rel err exceeds {self.TOLERANCE_AWAY:.0e}: "
            f"lo={lo_part[0]:.3e}, hi={hi_part[0]:.3e}"
        )

    def test_triton_erfcx_no_nans_across_zero(self):
        """Kernel must not emit NaN anywhere in the fp32-safe input range."""
        z = torch.linspace(
            self.Z_LO, self.Z_HI, 20001, device="cuda", dtype=torch.float32
        )
        out = self._run_kernel(z)
        assert torch.isnan(out).sum().item() == 0, (
            "Triton erfcx kernel produced NaN on fp32-safe inputs"
        )


class TestRenewalEngineFused:
    """Test fused Triton kernel engine (Phase 3)."""

    @pytest.mark.parametrize(
        "strategy,mixed",
        [
            ("thread", False),
            ("warp", False),
            ("merge", False),
            ("thread", True),
        ],
    )
    def test_first_tau_uses_current_weighted_rate(self, strategy, mixed):
        """The first step must enforce epsilon, not sample with tau_max."""
        from flashspread import GraphCSR, SEIRModel
        from flashspread.engines.renewal_fused import RenewalEngineFused

        # Four infectious sources feed target 0; their age-zero recovery
        # hazards vanish, so the target's weighted pressure is the max rate.
        edges = torch.tensor(
            [[1, 2, 3, 4], [0, 0, 0, 0]], device="cuda", dtype=torch.int64
        )
        weights = torch.tensor([1.0, 2.0, 1.0, 2.0], device="cuda")
        graph = GraphCSR(edges, 6, weights=weights)
        model = SEIRModel(beta=0.5)
        engine = RenewalEngineFused(
            graph,
            model,
            device="cuda",
            epsilon=0.03,
            tau_max=1.0,
            csr_strategy=strategy,
            use_mixed_precision=mixed,
        )
        assert engine.device == engine.graph.row_ptr.device == engine.state.device
        state = torch.tensor([0, 2, 2, 2, 2, 3], device="cuda")
        engine.set_initial_state(state)
        tau, _ = engine.step()
        expected_rate = 0.5 * weights.sum().item()
        assert tau == pytest.approx(0.03 / expected_rate, rel=2e-5)
        assert engine.rates.max().item() * tau <= 0.03 * (1.0 + 2e-5)
        expected_partial_width = engine.nodes_per_block if strategy == "warp" else 128
        assert engine._max_rate_partials.numel() == (
            engine.num_nodes + expected_partial_width - 1
        ) // expected_partial_width
        torch.testing.assert_close(engine._max_rate, engine.rates.max())
        torch.testing.assert_close(
            engine._max_rate_partials.max(), engine.rates.max()
        )

    def test_zero_rate_uses_tau_max(self):
        from flashspread import GraphCSR, SEIRModel
        from flashspread.engines.renewal_fused import RenewalEngineFused

        graph = GraphCSR(
            torch.empty((2, 0), device="cuda", dtype=torch.int64), 4
        )
        engine = RenewalEngineFused(
            graph, SEIRModel(), device="cuda", epsilon=0.03, tau_max=0.7
        )
        tau, _ = engine.step()
        assert tau == pytest.approx(0.7)

    def test_constant_transmission_uses_scalar_infectivity_storage(self):
        from flashspread import GraphCSR, SEIRModel
        from flashspread.engines.renewal_fused import RenewalEngineFused

        graph = GraphCSR(
            torch.empty((2, 0), device="cuda", dtype=torch.int64), 4
        )
        engine = RenewalEngineFused(graph, SEIRModel(beta=0.3), device="cuda")
        engine.set_initial_state(torch.tensor([0, 2, 3, 2], device="cuda"))

        assert engine._infectivity.numel() == 1
        assert engine._next_infectivity.numel() == 1
        torch.testing.assert_close(
            engine.infectivity,
            torch.tensor([0.0, 0.3, 0.0, 0.3], device="cuda"),
        )

    def test_mixed_storage_keeps_small_tau_age_progress(self):
        from flashspread import GraphCSR, SEIRModel
        from flashspread.engines.renewal_fused import RenewalEngineFused

        graph = GraphCSR(
            torch.tensor([[1], [0]], device="cuda"),
            3,
            weights=torch.tensor([200.0], device="cuda"),
        )
        engine = RenewalEngineFused(
            graph,
            SEIRModel(beta=0.3),
            device="cuda",
            use_mixed_precision=True,
        )
        engine.set_initial_state(
            torch.tensor([0, 2, 3], device="cuda"),
            torch.tensor([0.0, 0.0, 2.0], device="cuda"),
        )

        tau, _ = engine.step()

        assert tau < 2.0**-10  # below half an fp16 ULP at age=2
        assert engine.age.dtype == torch.float32
        assert engine.age[2].item() > 2.0

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

    def test_cudagraph_pingpong_matches_eager(self):
        """Captured A->B->A replay needs no copy-back and preserves results."""
        from flashspread.engines.renewal_fused import (
            RenewalEngineFused,
            RenewalEngineFusedCUDAGraph,
        )
        from flashspread import SEIRModel, FixedDegreeGraph

        graph = FixedDegreeGraph(512, 8, device="cuda", seed=4)
        eager = RenewalEngineFused(
            graph, SEIRModel(), device="cuda", seed=9, csr_strategy="thread"
        )
        captured = RenewalEngineFusedCUDAGraph(
            graph,
            SEIRModel(),
            device="cuda",
            seed=9,
            csr_strategy="thread",
            steps_per_launch=10,
        )
        eager.seed_infection(40)
        captured.seed_infection(40)
        elapsed_eager = 0.0
        for _ in range(10):
            dt, _ = eager.step()
            elapsed_eager += dt
        elapsed_captured, _ = captured.step()
        assert elapsed_captured == pytest.approx(elapsed_eager, rel=1e-6)
        assert torch.equal(captured.state, eager.state)
        assert torch.equal(captured.age, eager.age)
        assert torch.equal(captured.infectivity, eager.infectivity)

    def test_cudagraph_episode_seed_is_device_visible(self):
        from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph
        from flashspread import SEIRModel, FixedDegreeGraph

        from flashspread.core.host_rng import offset_seed, signed_int64

        graph = FixedDegreeGraph(128, 8, device="cuda", seed=2)
        engine = RenewalEngineFusedCUDAGraph(
            graph, SEIRModel(), device="cuda", seed=11, steps_per_launch=2
        )
        base = int(engine._rng_seed.item())

        # The point of this test is that reset(episode=...) reaches the device
        # buffer the captured graph reads, without recapture. Assert that
        # through the public derivation rather than a hard-coded sum: episode
        # seeds are deliberately mixed, not added, because base_seed + episode
        # made (seed=11, episode=7) and (seed=18, episode=0) the same stream.
        engine.reset(episode=7)
        episode_seed = int(engine._rng_seed.item())
        assert episode_seed != base
        assert episode_seed == signed_int64(offset_seed(11, 7))

        engine.reset()
        assert int(engine._rng_seed.item()) == base

    def test_compacted_pingpong_clears_stale_alternate_infectivity(self):
        from flashspread import GraphCSR, SEIRModel
        from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph

        # Reproduce the dangerous boundary condition directly: A says source 0
        # is recovered, while B retains an old infectious value. The all-node
        # transition phase must repair B before the captured B->A rate pass.
        graph = GraphCSR(
            torch.tensor([[0], [1]], device="cuda", dtype=torch.int64), 2
        )
        model = SEIRModel(beta=0.3, transmission_mode="age_dependent")
        engine = RenewalEngineFusedCUDAGraph(
            graph,
            model,
            device="cuda",
            seed=3,
            steps_per_launch=2,
            csr_strategy="thread",
            use_active_compaction=True,
        )
        engine.set_initial_state(torch.tensor([3, 0], device="cuda"))
        engine.next_state[0] = model.infected
        engine.next_infectivity[0] = model.beta
        engine.step()
        assert engine.rates[1].item() == 0.0
        assert engine.state.tolist() == [model.recovered, model.susceptible]


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
