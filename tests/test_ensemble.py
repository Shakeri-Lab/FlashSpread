from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ._triton_support import triton_interpreter_skip_reason

from flashspread.core.ensemble_reference import (
    reference_ensemble_infectivity_csr,
    reference_ensemble_influence_csr,
)
from flashspread.core.graph import GraphCSR
from flashspread.engines.ensemble import (
    EnsembleEngine,
    ReferenceEnsembleEngine,
    _has_builtin_seir_contract,
    _supports_builtin_seir_rate_fusion,
)
from flashspread.models import SEIRModel, SISModel

_REQUIRES_TRITON_INTERPRETER = pytest.mark.skipif(
    triton_interpreter_skip_reason() is not None,
    reason=triton_interpreter_skip_reason() or "",
)


def _weighted_graph() -> GraphCSR:
    # Directed, weighted, irregular, with a duplicate 0->2 edge.
    edges = torch.tensor([[0, 2, 1, 0, 3, 0], [2, 0, 2, 2, 1, 3]], dtype=torch.int64)
    weights = torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    return GraphCSR(edges, 4, weights=weights)


def _ring_graph(num_nodes: int = 40) -> GraphCSR:
    node = torch.arange(num_nodes)
    edges = torch.stack((node, (node + 1) % num_nodes))
    return GraphCSR(edges, num_nodes)


@pytest.mark.parametrize("recorded_device", ["cuda:0", "cuda"])
def test_graph_to_uses_resolved_tensor_device_for_unindexed_cuda(
    monkeypatch,
    recorded_device,
):
    import flashspread.core.graph as graph_core

    class ResolvedCudaTensor:
        device = torch.device("cuda:0")

        def __init__(self):
            self.requests = []

        def to(self, device):
            self.requests.append(torch.device(device))
            # Model PyTorch resolving an abstract ``cuda`` request on transfer.
            return self

    row_ptr = ResolvedCudaTensor()
    col_ind = ResolvedCudaTensor()
    weights = ResolvedCudaTensor()
    graph = object.__new__(GraphCSR)
    graph.device = torch.device(recorded_device)
    graph.num_nodes = 2
    graph.incoming = True
    graph._transpose_cache = None
    graph._transpose_cache_signature = None
    graph.row_ptr = row_ptr
    graph.col_ind = col_ind
    graph.has_weights = False
    graph._weights = weights
    monkeypatch.setattr(graph_core, "_ensure_versioned", lambda tensor: tensor)

    moved = graph.to("cuda")

    assert moved is not graph
    assert moved.device == torch.device("cuda:0")
    assert moved.device == moved.row_ptr.device == moved.col_ind.device
    assert moved.device == moved.weights_storage.device
    assert row_ptr.requests == [torch.device("cuda")]
    assert col_ind.requests == [torch.device("cuda")]
    assert weights.requests == [torch.device("cuda")]


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
    reason="requires CUDA and the optional Triton dependency",
)
def test_unindexed_cuda_ensemble_uses_canonical_csr_device():
    from flashspread.graphs import regular_graph

    graph = regular_graph(8, 2, device="cuda", algorithm="circulant")
    engine = EnsembleEngine(
        graph,
        SEIRModel(),
        replicas=3,
        device="cuda",
    )
    concrete = torch.device("cuda", torch.cuda.current_device())

    assert engine.device == concrete
    assert engine.graph.device == concrete
    assert engine.graph.row_ptr.device == concrete
    assert engine.state.device == engine.age.device == concrete
    initial = torch.full((8,), engine.model.recovered, dtype=torch.int32)
    engine.set_initial_state(initial, torch.zeros(8, dtype=torch.float32))
    assert torch.equal(engine.rates, torch.zeros_like(engine.rates))


def test_reference_ensemble_gathers_match_independent_oracles():
    graph = _weighted_graph()
    state = torch.tensor(
        [
            [1, 0, 2, 1, 0],
            [0, 1, 2, 0, 1],
            [2, 1, 0, 1, 2],
            [1, 2, 1, 0, 0],
        ],
        dtype=torch.int32,
    )
    infectivity = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 10.0

    actual_state = reference_ensemble_influence_csr(graph, state, [1, 2])
    actual_payload = reference_ensemble_infectivity_csr(graph, infectivity)
    expected_state = torch.zeros_like(actual_state)
    expected_payload = torch.zeros_like(actual_payload)
    edge_index = graph.to_edge_index()
    for edge, weight in zip(edge_index.t(), graph.weights):
        source, target = edge.tolist()
        expected_state[target].add_(
            ((state[source] == 1) | (state[source] == 2)).to(torch.float32) * weight
        )
        expected_payload[target].add_(infectivity[source] * weight)

    torch.testing.assert_close(actual_state, expected_state)
    torch.testing.assert_close(actual_payload, expected_payload)


def test_reference_engine_owns_one_graph_and_uses_node_major_layout():
    graph = _ring_graph()
    engine = ReferenceEnsembleEngine(graph, SISModel(), replicas=7, seed=4)

    assert engine.storage_profile == "full"
    assert engine.graph is graph
    assert engine.state.shape == (graph.num_nodes, 7)
    assert engine.state.stride() == (7, 1)
    assert engine.rates.stride() == (7, 1)
    assert engine.tau.shape == (7,)
    assert engine.current_time.shape == (7,)
    assert engine.current_time.dtype == torch.float64
    assert engine.age is None
    assert engine._infectious_mask is None


def test_reference_engine_and_gpu_gathers_have_lazy_public_mappings():
    import flashspread
    import flashspread.core as core
    import flashspread.engines as engines

    assert flashspread.ReferenceEnsembleEngine is ReferenceEnsembleEngine
    assert flashspread.EnsembleEngine is EnsembleEngine
    assert engines.ReferenceEnsembleEngine is ReferenceEnsembleEngine
    assert engines.EnsembleEngine is EnsembleEngine
    assert callable(core.ensemble_influence_csr)
    assert callable(core.ensemble_infectivity_csr)
    assert callable(core.pack_ensemble_infectious_mask)


def test_importing_ensemble_engine_types_does_not_import_triton():
    repository = Path(__file__).resolve().parents[1]
    script = """
import sys
import flashspread
assert 'triton' not in sys.modules
from flashspread import EnsembleEngine, ReferenceEnsembleEngine
assert EnsembleEngine.__name__ == 'EnsembleEngine'
assert ReferenceEnsembleEngine.__name__ == 'ReferenceEnsembleEngine'
assert 'triton' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_ensemble_engines_does_not_import_triton():
    repository = Path(__file__).resolve().parents[1]
    script = r"""
import sys
from flashspread.engines.ensemble import EnsembleEngine, ReferenceEnsembleEngine
assert EnsembleEngine is not None and ReferenceEnsembleEngine is not None
assert "triton" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tiled_engine_hooks_reuse_persistent_output_and_forward_tiles(monkeypatch):
    import flashspread.core.flash_ensemble as flash_ensemble

    engine = object.__new__(EnsembleEngine)
    engine.graph = object()
    engine.state = torch.tensor([[0, 1], [2, 1]], dtype=torch.int32)
    engine.inducer_states = (1, 2)
    engine._infectivity = torch.ones((2, 2), dtype=torch.float32)
    engine.pressure = torch.empty((2, 2), dtype=torch.float32)
    engine.nodes_per_program = 4
    engine.replicas_per_tile = 2
    output_pointer = engine.pressure.data_ptr()
    calls = []

    def fake_state(graph, state, inducers, **kwargs):
        calls.append(("state", graph, state, inducers, kwargs))
        return kwargs["out"].fill_(3.0)

    def fake_payload(graph, infectivity, **kwargs):
        calls.append(("payload", graph, infectivity, kwargs))
        return kwargs["out"].fill_(5.0)

    monkeypatch.setattr(flash_ensemble, "ensemble_influence_csr", fake_state)
    monkeypatch.setattr(flash_ensemble, "ensemble_infectivity_csr", fake_payload)
    engine._gather_state_pressure()
    assert torch.equal(engine.pressure, torch.full((2, 2), 3.0))
    engine._gather_infectivity_pressure()
    assert torch.equal(engine.pressure, torch.full((2, 2), 5.0))
    assert engine.pressure.data_ptr() == output_pointer
    assert calls[0][3] == (1, 2)
    for call in calls:
        kwargs = call[-1]
        assert kwargs["out"] is engine.pressure
        assert kwargs["nodes_per_program"] == 4
        assert kwargs["replicas_per_tile"] == 2


def test_builtin_seir_rate_fusion_detection_is_semantics_strict():
    model = SEIRModel()
    assert _has_builtin_seir_contract(model)
    assert not _supports_builtin_seir_rate_fusion(model)
    model.prepare(torch.device("cpu"))
    assert _supports_builtin_seir_rate_fusion(model)

    age_dependent = SEIRModel(transmission_mode="age_dependent")
    assert _has_builtin_seir_contract(age_dependent)
    age_dependent.prepare(torch.device("cpu"))
    assert _supports_builtin_seir_rate_fusion(age_dependent)

    class DerivedSEIR(SEIRModel):
        pass

    derived = DerivedSEIR()
    assert not _has_builtin_seir_contract(derived)
    derived.prepare(torch.device("cpu"))
    assert not _supports_builtin_seir_rate_fusion(derived)

    shadowed = SEIRModel()
    shadowed.compute_rates = lambda *args, **kwargs: None
    assert not _has_builtin_seir_contract(shadowed)
    shadowed.prepare(torch.device("cpu"))
    assert not _supports_builtin_seir_rate_fusion(shadowed)

    shadowed_prepare = SEIRModel()
    shadowed_prepare.prepare = lambda *args, **kwargs: None
    assert not _has_builtin_seir_contract(shadowed_prepare)
    assert not _supports_builtin_seir_rate_fusion(shadowed_prepare)

    shadowed_transition = SEIRModel()
    shadowed_transition.apply_transitions = lambda *args, **kwargs: None
    assert not _has_builtin_seir_contract(shadowed_transition)
    shadowed_transition.prepare(torch.device("cpu"))
    assert not _supports_builtin_seir_rate_fusion(shadowed_transition)

    wrong_family = SEIRModel()
    wrong_family.is_markovian = True
    assert not _has_builtin_seir_contract(wrong_family)
    wrong_family.prepare(torch.device("cpu"))
    assert not _supports_builtin_seir_rate_fusion(wrong_family)

    remapped = SEIRModel()
    remapped.susceptible, remapped.recovered = 3, 0
    assert not _has_builtin_seir_contract(remapped)
    remapped.prepare(torch.device("cpu"))
    assert not _supports_builtin_seir_rate_fusion(remapped)

    missing_parameter = SEIRModel()
    missing_parameter.prepare(torch.device("cpu"))
    missing_parameter._mu_ir = None
    assert _has_builtin_seir_contract(missing_parameter)
    assert not _supports_builtin_seir_rate_fusion(missing_parameter)


def test_builtin_seir_detection_rejects_class_level_hook_monkeypatch(monkeypatch):
    model = SEIRModel()
    model.prepare(torch.device("cpu"))
    assert _supports_builtin_seir_rate_fusion(model)

    monkeypatch.setattr(SEIRModel, "apply_transitions", lambda *args, **kwargs: None)

    assert not _has_builtin_seir_contract(model)
    assert not _supports_builtin_seir_rate_fusion(model)


def test_compact_fused_seir_storage_omits_full_step_scratch_and_is_reset_safe():
    graph = _ring_graph(8)
    engine = ReferenceEnsembleEngine(
        graph,
        SEIRModel(),
        replicas=3,
        seed=19,
        _storage_profile="fused_seir",
    )

    assert engine.storage_profile == "fused_seir"
    with pytest.raises(AttributeError):
        engine.storage_profile = "full"
    for name in (
        "next_state",
        "pressure",
        "event_prob",
        "event_mask",
        "rand_buffer",
        "seed_counter",
        "_infectivity",
    ):
        assert getattr(engine, name) is None
    assert engine._infectious_mask.dtype == torch.int32
    assert engine._infectious_mask.shape == (8, 1)
    assert engine._infectious_state_signature == engine._state_mutation_signature()
    assert engine.state.shape == (8, 3)
    assert engine.rates.shape == (8, 3)
    assert engine.age.shape == (8, 3)
    with pytest.raises(RuntimeError, match="compact fused_seir"):
        engine._renewal_uniform()

    engine.state.fill_(2)
    engine.rates.fill_(3.0)
    engine.age.fill_(4.0)
    engine.tau.fill_(0.25)
    engine.current_time.fill_(5.0)
    engine.total_events.fill_(6)
    engine.total_steps = 7
    engine._infectious_mask.fill_(-1)
    mask_before_reseed = engine._infectious_mask.clone()
    signature_before_reseed = engine._infectious_state_signature
    engine.reseed(23)
    assert torch.equal(engine._infectious_mask, mask_before_reseed)
    assert engine._infectious_state_signature == signature_before_reseed
    engine.reset(episode=2)
    assert not bool(engine.state.any())
    assert not bool(engine.rates.any())
    assert not bool(engine.age.any())
    assert torch.equal(engine.tau, torch.ones(3))
    assert not bool(engine.current_time.any())
    assert not bool(engine.total_events.any())
    assert not bool(engine._infectious_mask.any())
    assert engine._infectious_state_signature == engine._state_mutation_signature()
    assert engine.total_steps == 0

    # The reference engine does not execute compact fused rates or transitions;
    # these calls isolate lifecycle handling for the optional compatibility
    # buffers used by EnsembleEngine in production.
    engine._compute_rates = lambda: None
    engine.seed_infection([1, 2, 3])
    assert (engine.state == engine.model.exposed).sum(dim=0).tolist() == [1, 2, 3]

    initial = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32)
    initial_age = torch.arange(8, dtype=torch.float32)
    engine.set_initial_state(initial, initial_age)
    assert torch.equal(engine.state, initial[:, None].expand(-1, 3))
    assert torch.equal(engine.age, initial_age[:, None].expand(-1, 3))

    # Validation is transactional across the authoritative state/age pair.
    state_before = engine.state.clone()
    age_before = engine.age.clone()
    signature_before = engine._state_mutation_signature()
    with pytest.raises(ValueError, match="finite and non-negative"):
        engine.set_initial_state(
            torch.full((8,), engine.model.recovered, dtype=torch.int32),
            torch.full((8,), -1.0),
        )
    assert torch.equal(engine.state, state_before)
    assert torch.equal(engine.age, age_before)
    assert engine._state_mutation_signature() == signature_before


@pytest.mark.parametrize(
    "replicas, packed_words",
    [(1, 1), (31, 1), (32, 1), (33, 2), (65, 3)],
)
def test_compact_fused_seir_storage_sizes_packed_mask(replicas, packed_words):
    engine = ReferenceEnsembleEngine(
        _ring_graph(5),
        SEIRModel(),
        replicas=replicas,
        _storage_profile="fused_seir",
    )

    assert engine.storage_profile == "fused_seir"
    assert engine._infectious_mask.shape == (5, packed_words)
    assert engine._infectious_mask.dtype == torch.int32
    assert engine._infectious_mask.untyped_storage().nbytes() == 5 * packed_words * 4


def test_fused_step_rate_bound_scratch_has_fixed_128_node_granularity():
    engine = object.__new__(EnsembleEngine)
    engine.num_nodes = 257
    engine.replicas = 3
    engine.device = torch.device("cpu")
    engine._base_seed = 19
    engine._infectious_mask = torch.zeros((257, 1), dtype=torch.int32)

    engine._initialize_fused_seir_step()

    assert engine._rate_bound_nodes_per_partial == 128
    assert engine._min_rate_partials.shape == (3, 3)
    assert engine._max_rate_partials.shape == (3, 3)
    assert engine._min_rate_partials.dtype == torch.float32
    assert engine._max_rate_partials.dtype == torch.float32
    assert engine._event_partials.shape == (3, 3)
    # Minimum bounds are dead after tau validation, so the transition can reuse
    # that exact 4-byte storage as int32 event counts. Maximum bounds remain
    # independent because both bound arrays are live during their reductions.
    assert (
        engine._min_rate_partials.untyped_storage().data_ptr()
        == engine._event_partials.untyped_storage().data_ptr()
    )
    assert engine._event_partials.dtype == torch.int32
    assert (
        engine._max_rate_partials.untyped_storage().data_ptr()
        != engine._event_partials.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize(
    "model",
    [SISModel(), SEIRModel(), SEIRModel(transmission_mode="age_dependent")],
)
def test_full_ensemble_profiles_allocate_no_packed_mask(model):
    engine = ReferenceEnsembleEngine(_ring_graph(5), model, replicas=33)

    assert engine.storage_profile == "full"
    assert engine._infectious_mask is None
    assert engine._infectious_state_signature is None


@pytest.mark.parametrize(
    "model",
    [
        SISModel(),
        SEIRModel(transmission_mode="age_dependent"),
    ],
)
def test_compact_fused_seir_storage_rejects_incompatible_models(model):
    with pytest.raises(ValueError, match="fused_seir"):
        ReferenceEnsembleEngine(
            _ring_graph(4),
            model,
            replicas=2,
            _storage_profile="fused_seir",
        )


def test_reference_ensemble_rejects_unknown_storage_profile():
    with pytest.raises(ValueError, match="storage_profile"):
        ReferenceEnsembleEngine(
            _ring_graph(4),
            SEIRModel(),
            replicas=2,
            _storage_profile="typo",
        )


def test_tiled_engine_builtin_rate_dispatch_reuses_rates_and_guards_graph(monkeypatch):
    import flashspread.core.flash_ensemble as flash_ensemble

    graph = _ring_graph(4)
    model = SEIRModel()
    model.prepare(torch.device("cpu"))
    engine = object.__new__(EnsembleEngine)
    engine._uses_fused_seir_rates = True
    engine.graph = graph
    engine._graph_signature = graph._mutation_signature()
    engine.state = torch.zeros((4, 3), dtype=torch.int32)
    engine.age = torch.zeros((4, 3), dtype=torch.float32)
    engine.rates = torch.empty((4, 3), dtype=torch.float32)
    engine._min_rate_partials = torch.empty((1, 3), dtype=torch.float32)
    engine._max_rate_partials = torch.empty((1, 3), dtype=torch.float32)
    engine._infectious_mask = torch.empty((4, 1), dtype=torch.int32)
    engine._infectious_state_signature = None
    engine.model = model
    engine.nodes_per_program = 4
    engine.replicas_per_tile = 2
    rates_pointer = engine.rates.data_ptr()
    calls = []

    def fake_pack(state, **kwargs):
        calls.append(("pack", state.clone(), kwargs))
        assert kwargs["out"] is engine._infectious_mask
        kwargs["out"].fill_(int((state == model.infected).sum()))
        return kwargs["out"]

    def fake_rates(graph_arg, state, age, **kwargs):
        calls.append(("rates", graph_arg, state, age, kwargs))
        assert kwargs["infectious_mask"] is engine._infectious_mask
        return kwargs["out"].fill_(7.0)

    monkeypatch.setattr(flash_ensemble, "pack_ensemble_infectious_mask", fake_pack)
    monkeypatch.setattr(flash_ensemble, "ensemble_seir_renewal_rates_csr", fake_rates)
    engine._compute_rates()
    assert torch.equal(engine.rates, torch.full((4, 3), 7.0))
    assert engine.rates.data_ptr() == rates_pointer
    assert [call[0] for call in calls] == ["pack", "rates"]
    assert not bool(calls[0][1].any())
    _, graph_arg, state, age, kwargs = calls[1]
    assert graph_arg is graph and state is engine.state and age is engine.age
    assert kwargs["out"] is engine.rates
    assert kwargs["beta"] is model._beta_t
    assert kwargs["mu_ei"] is model._mu_ei
    assert kwargs["sig_ei"] is model._sig_ei
    assert kwargs["mu_ir"] is model._mu_ir
    assert kwargs["sig_ir"] is model._sig_ir
    assert kwargs["transmission_age_dependent"] is False
    assert kwargs["infectious_mask"] is engine._infectious_mask
    assert kwargs["rate_bounds"][0] is engine._min_rate_partials
    assert kwargs["rate_bounds"][1] is engine._max_rate_partials
    assert kwargs["nodes_per_program"] == 4
    assert kwargs["replicas_per_tile"] == 2

    calls.clear()
    engine._compute_rates()
    assert [call[0] for call in calls] == ["rates"]

    # Tensor/storage identity is part of the signature as well as the version counter.
    # Replacing state with an otherwise identical, unmodified tensor must pack.
    engine.state = torch.zeros_like(engine.state)
    calls.clear()
    engine._compute_rates()
    assert [call[0] for call in calls] == ["pack", "rates"]

    # Public state remains authoritative. A mutation after the preceding rate
    # evaluation must be visible in the very next packed snapshot, while both
    # persistent output buffers retain their addresses.
    mask_pointer = engine._infectious_mask.data_ptr()
    engine.state[0, 0] = model.infected
    calls.clear()
    engine._compute_rates()
    assert [call[0] for call in calls] == ["pack", "rates"]
    assert calls[0][1][0, 0] == model.infected
    assert engine._infectious_mask.data_ptr() == mask_pointer
    assert engine.rates.data_ptr() == rates_pointer

    # Raw-storage writes do not necessarily increment the authoritative
    # tensor's PyTorch version counter. The explicit notification protocol
    # makes those advanced writes safe without imposing a dense scan on every
    # steady-state step.
    clean_signature = engine._infectious_state_signature
    engine.state.data[0, 1] = model.infected
    assert engine._state_mutation_signature() == clean_signature
    engine.mark_state_dirty()
    assert engine._infectious_state_signature is None
    calls.clear()
    engine._compute_rates()
    assert [call[0] for call in calls] == ["pack", "rates"]
    assert calls[0][1][0, 1] == model.infected

    graph.row_ptr[0] = graph.row_ptr[0]
    calls.clear()
    with pytest.raises(RuntimeError, match="GraphCSR storage changed"):
        engine._compute_rates()
    assert not calls


def _mock_fused_step_engine() -> EnsembleEngine:
    engine = object.__new__(EnsembleEngine)
    engine._uses_fused_seir_step = True
    engine.device = torch.device("cpu")
    engine.epsilon = 0.03
    engine.tau_max = 1.0
    engine.state = torch.tensor(
        [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]],
        dtype=torch.int32,
    )
    engine.age = torch.ones((4, 3), dtype=torch.float32)
    # Dense public rates deliberately differ from the compact bounds. The
    # production step must not reread this [N, R] tensor to select tau.
    engine.rates = torch.full((4, 3), 9.0, dtype=torch.float32)
    engine._min_rate_partials = torch.ones((1, 3), dtype=torch.float32)
    engine._max_rate_partials = torch.ones((1, 3), dtype=torch.float32)
    engine._min_rate = torch.empty(3, dtype=torch.float32)
    engine._max_rate = torch.empty(3, dtype=torch.float32)
    engine._tau_candidate = torch.empty(3, dtype=torch.float32)
    engine._invalid_step = torch.zeros(3, dtype=torch.int32)
    engine._step_status = torch.zeros((), dtype=torch.int32)
    engine._event_partials = torch.empty((1, 3), dtype=torch.int32)
    engine._step_events = torch.empty(3, dtype=torch.int64)
    engine._event_seed = torch.tensor(17, dtype=torch.int64)
    engine._step_id = torch.tensor(1, dtype=torch.int64)
    engine._infectious_mask = torch.zeros((4, 1), dtype=torch.int32)
    engine._infectious_state_signature = engine._state_mutation_signature()
    engine._transition_nodes_per_program = 128
    engine._transition_replicas_per_tile = 4
    engine.tau = torch.full((3,), 0.75, dtype=torch.float32)
    engine.current_time = torch.zeros(3, dtype=torch.float64)
    engine.total_events = torch.zeros(3, dtype=torch.int64)
    engine.total_steps = 0
    engine._compute_rates = lambda: None
    return engine


def test_fused_step_commits_clocks_events_and_rng_only_after_validation(monkeypatch):
    import flashspread.core.flash_ensemble_step as step_core

    engine = _mock_fused_step_engine()
    calls = []

    def fake_finalize(min_rate, max_rate, candidate, invalid, **kwargs):
        calls.append("finalize")
        torch.testing.assert_close(min_rate, torch.ones(3))
        torch.testing.assert_close(max_rate, torch.ones(3))
        candidate.copy_(torch.tensor([0.1, 0.2, 0.3]))
        invalid.zero_()
        return candidate, invalid

    def fake_transition(state, age, rates, tau, seed, step_id, partials, **kwargs):
        calls.append("transition")
        assert int(step_id) == 1
        assert int(seed) == 17
        assert kwargs["infectious_mask"] is engine._infectious_mask
        age.add_(tau[None, :])
        state[0, 0] = 1
        age[0, 0] = 0.0
        partials.copy_(torch.tensor([[1, 2, 0]], dtype=torch.int32))
        return state, age, partials

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(
        torch,
        "aminmax",
        lambda *args, **kwargs: pytest.fail("fused step reread dense public rates"),
    )
    monkeypatch.setattr(step_core, "finalize_ensemble_renewal_tau", fake_finalize)
    monkeypatch.setattr(step_core, "transition_ensemble_seir", fake_transition)
    tau, state = engine.step()

    assert calls == ["finalize", "transition"]
    torch.testing.assert_close(tau, torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(
        engine.current_time,
        torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
    )
    assert torch.equal(engine.total_events, torch.tensor([1, 2, 0]))
    assert engine.total_steps == 1
    assert int(engine._step_id) == 2
    assert engine._infectious_state_signature == engine._state_mutation_signature()
    assert state is engine.state and engine.state[0, 0] == 1
    assert engine.age[0, 0] == 0.0


@pytest.mark.parametrize(
    "status, error, message",
    [
        (3, FloatingPointError, "finite"),
        (2, ValueError, "non-negative"),
        (1, FloatingPointError, "selected tau"),
    ],
)
def test_fused_step_failure_is_all_replica_transactional(monkeypatch, status, error, message):
    import flashspread.core.flash_ensemble_step as step_core

    engine = _mock_fused_step_engine()
    snapshots = {
        "state": engine.state.clone(),
        "age": engine.age.clone(),
        "tau": engine.tau.clone(),
        "time": engine.current_time.clone(),
        "events": engine.total_events.clone(),
        "step_id": engine._step_id.clone(),
        "infectious_mask": engine._infectious_mask.clone(),
        "infectious_signature": engine._infectious_state_signature,
    }
    transition_called = False

    def fake_finalize(min_rate, max_rate, candidate, invalid, **kwargs):
        candidate.fill_(float("nan"))
        invalid.zero_()
        invalid[1] = status
        return candidate, invalid

    def forbidden_transition(*args, **kwargs):
        nonlocal transition_called
        transition_called = True

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(step_core, "finalize_ensemble_renewal_tau", fake_finalize)
    monkeypatch.setattr(step_core, "transition_ensemble_seir", forbidden_transition)
    with pytest.raises(error, match=message):
        engine.step()

    assert not transition_called
    assert torch.equal(engine.state, snapshots["state"])
    assert torch.equal(engine.age, snapshots["age"])
    assert torch.equal(engine.tau, snapshots["tau"])
    assert torch.equal(engine.current_time, snapshots["time"])
    assert torch.equal(engine.total_events, snapshots["events"])
    assert torch.equal(engine._step_id, snapshots["step_id"])
    assert torch.equal(engine._infectious_mask, snapshots["infectious_mask"])
    assert engine._infectious_state_signature == snapshots["infectious_signature"]
    assert engine.total_steps == 0


def test_markovian_replicas_choose_independent_tau():
    graph = _ring_graph(4)
    engine = ReferenceEnsembleEngine(
        graph,
        SISModel(beta=0.8, delta=0.2),
        replicas=2,
        theta=0.01,
        tau_max=1.0,
        seed=8,
    )
    state = torch.tensor([[1, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.int32)
    engine.set_initial_state(state)
    tau, _ = engine.step()

    assert tau[0] != tau[1]
    torch.testing.assert_close(engine.current_time, tau.to(torch.float64))
    assert engine.count_by_state().shape == (2, 2)
    assert torch.equal(engine.count_by_state().sum(dim=1), torch.tensor([4, 4]))


class _NoTransitionRenewal:
    is_markovian = False
    num_states = 2
    inducer_states = [1]

    def compute_rates(self, age, state, pressure, out=None):
        out.copy_(torch.where(state == 0, 0.1, 0.5))
        return out

    def apply_transitions(self, state, event_mask, out=None):
        out.copy_(state)
        return out


def test_renewal_replicas_have_independent_clocks_and_age_updates():
    graph = _ring_graph(5)
    engine = ReferenceEnsembleEngine(
        graph,
        _NoTransitionRenewal(),
        replicas=3,
        epsilon=0.03,
        tau_max=1.0,
        seed=9,
    )
    state = torch.tensor(
        [[0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 1, 0]],
        dtype=torch.int32,
    )
    engine.set_initial_state(state)
    tau, _ = engine.step()

    torch.testing.assert_close(tau, torch.tensor([0.3, 0.06, 0.06]))
    torch.testing.assert_close(engine.age, tau[None, :].expand(5, -1))
    torch.testing.assert_close(engine.current_time, tau.to(torch.float64))
    assert torch.equal(engine.total_events, engine.event_mask.sum(dim=0))


@pytest.mark.parametrize(
    "model",
    [
        SISModel(beta=0.8, delta=0.3),
        SEIRModel(beta=0.4),
        SEIRModel(beta=0.4, transmission_mode="age_dependent"),
    ],
)
def test_ensemble_reset_reproduces_and_episode_decorrelates(model):
    graph = _ring_graph(60)
    engine = ReferenceEnsembleEngine(graph, model, replicas=4, seed=17)

    def run_once():
        engine.seed_infection([8, 9, 10, 11])
        initial = engine.state.clone()
        for _ in range(8):
            engine.step()
        return initial, engine.state.clone(), engine.current_time.clone()

    initial_a, state_a, time_a = run_once()
    engine.reset()
    initial_b, state_b, time_b = run_once()
    assert torch.equal(initial_a, initial_b)
    assert torch.equal(state_a, state_b)
    torch.testing.assert_close(time_a, time_b, rtol=0.0, atol=0.0)

    engine.reset(episode=1)
    initial_c, _, _ = run_once()
    assert not torch.equal(initial_a, initial_c)


def test_replica_random_streams_are_distinct_and_resettable():
    graph = _ring_graph(12)
    markov = ReferenceEnsembleEngine(graph, SISModel(), replicas=3, seed=7)
    first = markov._markovian_uniform().clone()
    assert not torch.equal(first[:, 0], first[:, 1])
    markov.reset()
    torch.testing.assert_close(markov._markovian_uniform(), first, rtol=0.0, atol=0.0)

    renewal = ReferenceEnsembleEngine(graph, SEIRModel(), replicas=3, seed=7)
    first = renewal._renewal_uniform().clone()
    assert bool((first > 0.0).all()) and bool((first < 1.0).all())
    assert not torch.equal(first[:, 0], first[:, 1])
    renewal.reset()
    torch.testing.assert_close(renewal._renewal_uniform(), first, rtol=0.0, atol=0.0)


def test_shared_one_dimensional_initial_state_broadcasts_over_replicas():
    graph = _ring_graph(6)
    engine = ReferenceEnsembleEngine(graph, SEIRModel(), replicas=3)
    initial = torch.tensor([0, 1, 2, 3, 0, 1])
    age = torch.arange(6, dtype=torch.float32)
    engine.set_initial_state(initial, age)
    assert torch.equal(engine.state, initial[:, None].expand(-1, 3))
    assert torch.equal(engine.age, age[:, None].expand(-1, 3))
    with pytest.raises(ValueError, match="node-major"):
        engine.set_initial_state(torch.zeros((3, 6), dtype=torch.int32))


def test_ensemble_count_and_episode_contracts_are_strict():
    graph = _ring_graph(6)
    engine = ReferenceEnsembleEngine(graph, SISModel(), replicas=3)
    engine.seed_infection(torch.tensor([1, 2, 3]))
    assert engine.count_infected().tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match="length 3"):
        engine.seed_infection([1, 2])
    with pytest.raises(TypeError, match="episode"):
        engine.reset(episode=1.5)


def test_ensemble_rejects_graph_mutation_before_initialization_or_step():
    graph = _ring_graph(6)
    engine = ReferenceEnsembleEngine(graph, SISModel(), replicas=2, device="cpu", seed=3)
    state_before = engine.state.clone()
    graph.col_ind[0] = graph.col_ind[0]
    with pytest.raises(RuntimeError, match="GraphCSR storage changed"):
        engine.seed_infection(1)
    assert torch.equal(engine.state, state_before)
    with pytest.raises(RuntimeError, match="GraphCSR storage changed"):
        engine.set_initial_state(torch.zeros(graph.num_nodes, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="GraphCSR storage changed"):
        engine.step()


def test_ensemble_tau_underflow_fails_before_rng_or_state_mutation():
    class MaxRateRenewal:
        is_markovian = False
        num_states = 2
        susceptible = 0
        infected = 1
        inducer_states = (1,)

        def compute_rates(self, age, state, pressure, out=None):
            return out.fill_(torch.finfo(torch.float32).max)

        def apply_transitions(self, state, event_mask, out=None):
            return out.copy_(state)

    engine = ReferenceEnsembleEngine(
        _ring_graph(4),
        MaxRateRenewal(),
        replicas=2,
        device="cpu",
        epsilon=2.0**-149,
    )
    state_before = engine.state.clone()
    seed_before = engine.seed_counter.clone()
    with pytest.raises(FloatingPointError, match="selected tau"):
        engine.step()
    assert torch.equal(engine.state, state_before)
    assert torch.equal(engine.seed_counter, seed_before)


def test_gpu_wrapper_alias_guard_rejects_shared_storage():
    from flashspread.core.flash_ensemble import _reject_output_alias

    values = torch.ones((4, 5), dtype=torch.float32)
    with pytest.raises(ValueError, match="share storage"):
        _reject_output_alias(values, (values,))
    with pytest.raises(ValueError, match="share storage"):
        _reject_output_alias(values[:, :2], (values[:, 2:],))
    _reject_output_alias(torch.empty_like(values), (values,))


def test_ensemble_launch_grid_flattens_replica_tiles_onto_grid_x():
    from flashspread.core.flash_ensemble import _flat_ensemble_grid

    assert _flat_ensemble_grid(3, 5, 2, 4) == (4,)

    # 65,536 replica tiles exceeded CUDA's grid.y ceiling in the previous 2-D
    # launch. The same logical work now remains a modest one-dimensional grid.
    grid = _flat_ensemble_grid(33, 32 * 65_536, 8, 32)
    assert grid == (5 * 65_536,)

    with pytest.raises(ValueError, match="grid-x capacity"):
        _flat_ensemble_grid(1, 1 << 31, 1, 1)


@pytest.mark.parametrize(
    "inducers, error",
    [
        (True, TypeError),
        ([], ValueError),
        ([1, 1], ValueError),
        ([-1], ValueError),
        ([1.5], TypeError),
        ("1", ValueError),
    ],
)
def test_gpu_ensemble_inducer_contract_is_strict_before_launch(inducers, error):
    from flashspread.core.flash_ensemble import ensemble_influence_csr

    graph = _ring_graph(4)
    state = torch.zeros((4, 2), dtype=torch.int32)
    with pytest.raises(error, match="inducer"):
        ensemble_influence_csr(graph, state, inducers)


def test_gpu_wrapper_normalizes_multi_inducer_contract_before_launch():
    from flashspread.core.flash_ensemble import _normalize_inducer_states

    assert _normalize_inducer_states(2) == (2,)
    assert _normalize_inducer_states([2, 4]) == (2, 4)
    with pytest.raises(ValueError, match="at least one"):
        _normalize_inducer_states([])
    with pytest.raises(ValueError, match="duplicates"):
        _normalize_inducer_states([1, 1])
    with pytest.raises(ValueError, match="integer"):
        _normalize_inducer_states("1")
    with pytest.raises(ValueError, match="non-negative"):
        _normalize_inducer_states([-1])


@pytest.mark.parametrize("beta", [True, "0.3", 1e300])
def test_gpu_ensemble_beta_is_a_strict_fp32_scalar(beta):
    from flashspread.core.flash_ensemble import ensemble_influence_csr

    graph = _ring_graph(4)
    state = torch.zeros((4, 2), dtype=torch.int32)
    with pytest.raises((TypeError, ValueError), match="beta"):
        ensemble_influence_csr(graph, state, 1, beta=beta)


@_REQUIRES_TRITON_INTERPRETER
def test_triton_ensemble_gather_interpreter_all_payload_weight_modes():
    """Exercise partial tiles and both constexpr branches without a GPU."""
    repository = Path(__file__).resolve().parents[1]
    script = r"""
import torch
from flashspread.core.flash_ensemble import (
    _ensemble_csr_gather_kernel,
    _flat_ensemble_grid,
)
from flashspread.core.graph import GraphCSR
from flashspread.core.ensemble_reference import (
    reference_ensemble_influence_csr,
    reference_ensemble_infectivity_csr,
)

edge_index = torch.tensor([[1, 2, 0, 0, 1], [0, 0, 1, 2, 2]])
state = torch.tensor(
    [[1, 0, 1, 0, 1], [0, 1, 1, 0, 0], [1, 1, 0, 1, 0]],
    dtype=torch.int32,
)
payload = torch.tensor(
    [[.1, .2, .3, .4, .5], [.6, .7, .8, .9, 1.], [1.1, 1.2, 1.3, 1.4, 1.5]],
)
for weighted in (False, True):
    weights = torch.tensor([2., 3., 4., 5., 6.]) if weighted else None
    graph = GraphCSR(edge_index, 3, weights=weights)
    for payload_mode in (False, True):
        values = payload if payload_mode else state
        expected = (
            reference_ensemble_infectivity_csr(graph, payload)
            if payload_mode
            else 2.0 * reference_ensemble_influence_csr(graph, state, 1)
        )
        outputs = []
        for node_tile, replica_tile in ((2, 4), (1, 2)):
            out = torch.empty((3, 5), dtype=torch.float32)
            grid = _flat_ensemble_grid(3, 5, node_tile, replica_tile)
            _ensemble_csr_gather_kernel[grid](
                graph.row_ptr,
                graph.col_ind,
                graph.weights_storage,
                values,
                values,
                out,
                2.0,
                3,
                5,
                INDUCER_STATE=1,
                PAYLOAD_MODE=int(payload_mode),
                HAS_WEIGHTS=int(graph.has_weights),
                ACCUMULATE=0,
                NODES_PER_PROGRAM=node_tile,
                REPLICAS_PER_TILE=replica_tile,
            )
            outputs.append(out)
            torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)

    # Ordered accumulation supports arbitrary multi-inducer model contracts
    # without materializing an [E, R] reference contribution tensor.
    expected = reference_ensemble_influence_csr(graph, state, [1, 0])
    outputs = []
    for node_tile, replica_tile in ((2, 4), (1, 2)):
        out = torch.empty((3, 5), dtype=torch.float32)
        for launch, inducer_state in enumerate((1, 0)):
            grid = _flat_ensemble_grid(3, 5, node_tile, replica_tile)
            _ensemble_csr_gather_kernel[grid](
                graph.row_ptr,
                graph.col_ind,
                graph.weights_storage,
                state,
                state,
                out,
                1.0,
                3,
                5,
                INDUCER_STATE=inducer_state,
                PAYLOAD_MODE=0,
                HAS_WEIGHTS=int(graph.has_weights),
                ACCUMULATE=int(launch != 0),
                NODES_PER_PROGRAM=node_tile,
                REPLICAS_PER_TILE=replica_tile,
            )
        outputs.append(out)
        torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
"""
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@_REQUIRES_TRITON_INTERPRETER
def test_triton_ensemble_seir_rate_interpreter_all_parameter_weight_modes():
    """Protect the fused rate path across partial tiles and all branches."""
    repository = Path(__file__).resolve().parents[1]
    script = r"""
import torch

from flashspread.core.ensemble_reference import (
    reference_ensemble_infectivity_csr,
    reference_ensemble_influence_csr,
)
from flashspread.core.flash_ensemble import (
    _pack_ensemble_infectious_mask_kernel,
    _ensemble_seir_renewal_rate_kernel,
    _flat_ensemble_grid,
)
from flashspread.core.graph import GraphCSR
from flashspread.models import SEIRModel

edge_index = torch.tensor(
    [
        [0, 2, 1, 0, 3, 0, 4, 2, 1, 4],
        [2, 0, 2, 2, 1, 3, 3, 3, 4, 4],
    ]
)
state = torch.tensor(
    [
        [2, 0, 3, 1, 2],
        [0, 2, 2, 3, 1],
        [0, 3, 0, 2, 2],
        [3, 2, 1, 0, 2],
        [2, 1, 3, 2, 0],
        [0, 1, 2, 3, 0],
    ],
    dtype=torch.int32,
).repeat(1, 7)
replicas = state.shape[1]
age = (
    torch.arange(6 * replicas, dtype=torch.float32).reshape(6, replicas) + 1.0
) / 29.0
age[0, 0] = 0.0
age[5, 1] = 0.0
age[5, 2] = 0.0

# R=35 exercises both the signed high bit of the first word and a partial
# second word. Packing itself is invariant to the node-program tile.
mask_words = (replicas + 31) // 32
expected_mask = torch.zeros((6, mask_words), dtype=torch.int64)
for node in range(6):
    for replica in range(replicas):
        if state[node, replica] == 2:
            expected_mask[node, replica // 32] |= 1 << (replica % 32)
expected_mask = expected_mask.to(torch.int32)
packed_outputs = []
for node_tile in (4, 2):
    packed = torch.empty((6, mask_words), dtype=torch.int32)
    pack_grid = _flat_ensemble_grid(6, mask_words, node_tile, 1)
    _pack_ensemble_infectious_mask_kernel[pack_grid](
        state,
        packed,
        6,
        replicas,
        NODES_PER_PROGRAM=node_tile,
    )
    assert torch.equal(packed, expected_mask)
    packed_outputs.append(packed)
assert torch.equal(packed_outputs[0], packed_outputs[1])
infectious_mask = packed_outputs[0]

for weighted in (False, True):
    weights = None
    if weighted:
        weights = torch.tensor([.5, 1., 2., .25, 1.5, 2., .5, 1., .25, 2.])
    graph = GraphCSR(edge_index, 6, weights=weights)
    for age_dependent in (False, True):
        mode = "age_dependent" if age_dependent else "constant"
        model = SEIRModel(beta=0.25, transmission_mode=mode)
        model.prepare(torch.device("cpu"))
        if age_dependent:
            infectivity = torch.empty_like(age)
            model.compute_infectivity(
                age.reshape(-1),
                state.reshape(-1),
                out=infectivity.reshape(-1),
            )
            pressure = reference_ensemble_infectivity_csr(graph, infectivity)
            expected = torch.empty_like(age)
            model.compute_rates_nonmarkov(
                age.reshape(-1),
                state.reshape(-1),
                pressure.reshape(-1),
                out=expected.reshape(-1),
            )
        else:
            pressure = reference_ensemble_influence_csr(graph, state, 2)
            expected = torch.empty_like(age)
            model.compute_rates(
                age.reshape(-1),
                state.reshape(-1),
                pressure.reshape(-1),
                out=expected.reshape(-1),
            )

        host_params = tuple(
            float(value)
            for value in (
                model._beta_t,
                model._mu_ei,
                model._sig_ei,
                model._mu_ir,
                model._sig_ir,
            )
        )
        tile_outputs = []
        for node_tile, replica_tile in ((4, 32), (2, 16)):
            outputs = []
            for params_on_device in (False, True):
                mask_outputs = []
                for use_mask in (False, True):
                    out = torch.empty_like(age)
                    pointers = (
                        (
                            model._beta_t,
                            model._mu_ei,
                            model._sig_ei,
                            model._mu_ir,
                            model._sig_ir,
                        )
                        if params_on_device
                        else (age,) * 5
                    )
                    scalars = (0.0,) * 5 if params_on_device else host_params
                    grid = _flat_ensemble_grid(6, replicas, node_tile, replica_tile)
                    _ensemble_seir_renewal_rate_kernel[grid](
                        graph.row_ptr,
                        graph.col_ind,
                        graph.weights_storage,
                        state,
                        infectious_mask if use_mask else state,
                        age,
                        out,
                        out,
                        out,
                        *pointers,
                        *scalars,
                        6,
                        replicas,
                        mask_words,
                        PARAMS_ON_DEVICE=int(params_on_device),
                        HAS_WEIGHTS=int(graph.has_weights),
                        USE_INFECTIOUS_MASK=int(use_mask),
                        TRANSMISSION_AGE_DEPENDENT=int(age_dependent),
                        EMIT_RATE_BOUNDS=0,
                        RATE_TILES_PER_PARTIAL=1,
                        NODES_PER_PROGRAM=node_tile,
                        REPLICAS_PER_TILE=replica_tile,
                    )
                    mask_outputs.append(out)
                    torch.testing.assert_close(out, expected, rtol=0.06, atol=1e-7)
                    assert torch.equal(
                        out[state == 3], torch.zeros_like(out[state == 3])
                    )
                    zero_age_hazard = ((state == 1) | (state == 2)) & (age == 0.0)
                    assert torch.equal(
                        out[zero_age_hazard], torch.zeros_like(out[zero_age_hazard])
                    )
                    zero_pressure = (state == 0) & (expected == 0.0)
                    assert torch.equal(
                        out[zero_pressure], torch.zeros_like(out[zero_pressure])
                    )
                torch.testing.assert_close(
                    mask_outputs[0], mask_outputs[1], rtol=0.0, atol=0.0
                )
                outputs.append(mask_outputs[0])
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
            tile_outputs.append(outputs[0])
        torch.testing.assert_close(
            tile_outputs[0],
            tile_outputs[1],
            rtol=0.0,
            atol=0.0,
        )

# The production launch groups the existing node tiles into 128-node bound
# partials. Exercise that loop, the compact outputs, and the explicit
# nonfinite sentinel without requiring a physical GPU.
node_tile = 2
replica_tile = 16
rate_tiles_per_partial = 128 // node_tile
bound_rows = (state.shape[0] + 127) // 128
bound_grid = _flat_ensemble_grid(
    state.shape[0], replicas, 128, replica_tile
)
out = torch.empty_like(age)
minimum_partials = torch.empty((bound_rows, replicas), dtype=torch.float32)
maximum_partials = torch.empty_like(minimum_partials)
_ensemble_seir_renewal_rate_kernel[bound_grid](
    graph.row_ptr,
    graph.col_ind,
    graph.weights_storage,
    state,
    infectious_mask,
    age,
    out,
    minimum_partials,
    maximum_partials,
    model._beta_t,
    model._mu_ei,
    model._sig_ei,
    model._mu_ir,
    model._sig_ir,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    state.shape[0],
    replicas,
    mask_words,
    PARAMS_ON_DEVICE=1,
    HAS_WEIGHTS=int(graph.has_weights),
    USE_INFECTIOUS_MASK=1,
    TRANSMISSION_AGE_DEPENDENT=1,
    EMIT_RATE_BOUNDS=1,
    RATE_TILES_PER_PARTIAL=rate_tiles_per_partial,
    NODES_PER_PROGRAM=node_tile,
    REPLICAS_PER_TILE=replica_tile,
)
torch.testing.assert_close(
    minimum_partials[0], torch.amin(out, dim=0), rtol=0.0, atol=0.0
)
torch.testing.assert_close(
    maximum_partials[0], torch.amax(out, dim=0), rtol=0.0, atol=0.0
)

poisoned_age = age.clone()
poisoned_age[0, 0] = float("nan")
_ensemble_seir_renewal_rate_kernel[bound_grid](
    graph.row_ptr,
    graph.col_ind,
    graph.weights_storage,
    state,
    infectious_mask,
    poisoned_age,
    out,
    minimum_partials,
    maximum_partials,
    model._beta_t,
    model._mu_ei,
    model._sig_ei,
    model._mu_ir,
    model._sig_ir,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    state.shape[0],
    replicas,
    mask_words,
    PARAMS_ON_DEVICE=1,
    HAS_WEIGHTS=int(graph.has_weights),
    USE_INFECTIOUS_MASK=1,
    TRANSMISSION_AGE_DEPENDENT=1,
    EMIT_RATE_BOUNDS=1,
    RATE_TILES_PER_PARTIAL=rate_tiles_per_partial,
    NODES_PER_PROGRAM=node_tile,
    REPLICAS_PER_TILE=replica_tile,
)
assert torch.isnan(out[0, 0])
assert torch.isneginf(minimum_partials[0, 0])
assert torch.isposinf(maximum_partials[0, 0])

# Two bound rows protect the 128-node grouping and the ragged final group, not
# merely the single-row case above. An edgeless graph keeps interpreter runtime
# focused on rate evaluation and compact-output addressing.
tail_nodes = 130
tail_replicas = 5
tail_graph = GraphCSR(torch.empty((2, 0), dtype=torch.int64), tail_nodes)
tail_state = (
    torch.arange(tail_nodes)[:, None] + torch.arange(tail_replicas)[None, :]
).remainder(4).to(torch.int32)
tail_age = (
    torch.arange(tail_nodes * tail_replicas, dtype=torch.float32)
    .reshape(tail_nodes, tail_replicas)
    .add_(1.0)
    .div_(101.0)
)
tail_model = SEIRModel(beta=0.25)
tail_model.prepare(torch.device("cpu"))
tail_out = torch.empty_like(tail_age)
tail_minimum = torch.empty((2, tail_replicas), dtype=torch.float32)
tail_maximum = torch.empty_like(tail_minimum)
tail_node_tile = 4
tail_replica_tile = 4
tail_grid = _flat_ensemble_grid(
    tail_nodes,
    tail_replicas,
    128,
    tail_replica_tile,
)
_ensemble_seir_renewal_rate_kernel[tail_grid](
    tail_graph.row_ptr,
    tail_graph.col_ind,
    tail_graph.weights_storage,
    tail_state,
    tail_state,
    tail_age,
    tail_out,
    tail_minimum,
    tail_maximum,
    tail_model._beta_t,
    tail_model._mu_ei,
    tail_model._sig_ei,
    tail_model._mu_ir,
    tail_model._sig_ir,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    tail_nodes,
    tail_replicas,
    (tail_replicas + 31) // 32,
    PARAMS_ON_DEVICE=1,
    HAS_WEIGHTS=0,
    USE_INFECTIOUS_MASK=0,
    TRANSMISSION_AGE_DEPENDENT=0,
    EMIT_RATE_BOUNDS=1,
    RATE_TILES_PER_PARTIAL=128 // tail_node_tile,
    NODES_PER_PROGRAM=tail_node_tile,
    REPLICAS_PER_TILE=tail_replica_tile,
)
for partial, (start, stop) in enumerate(((0, 128), (128, 130))):
    torch.testing.assert_close(
        tail_minimum[partial],
        torch.amin(tail_out[start:stop], dim=0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        tail_maximum[partial],
        torch.amax(tail_out[start:stop], dim=0),
        rtol=0.0,
        atol=0.0,
    )
"""
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@_REQUIRES_TRITON_INTERPRETER
def test_triton_ensemble_step_tail_interpreter_transaction_and_rng_contract():
    """Exercise status precedence, in-place ages, counts, and tile-stable RNG."""
    repository = Path(__file__).resolve().parents[1]
    script = r"""
import math
import torch

from flashspread.core.flash_ensemble_step import (
    _ensemble_finalize_renewal_tau_kernel,
    _ensemble_seir_transition_kernel,
)

fp32_max = torch.finfo(torch.float32).max
min_rate = torch.tensor([0.0, 0.5, -0.1, float("nan"), 0.0, 1.0, fp32_max])
max_rate = torch.tensor([0.0, 1.0, 1.0, 1.0, float("inf"), 0.0, fp32_max])
candidate = torch.empty(7)
status = torch.empty(7, dtype=torch.int32)
_ensemble_finalize_renewal_tau_kernel[(1,)](
    min_rate,
    max_rate,
    candidate,
    status,
    2.0**-149,
    0.75,
    7,
    BLOCK_SIZE=8,
)
assert status.tolist() == [0, 0, 2, 3, 3, 1, 1]
assert candidate[0] == 0.75
assert 0.0 < candidate[1] <= 0.75
assert bool(torch.isnan(candidate[2:]).all())

N, R = 17, 5
initial_state = (torch.arange(N * R).reshape(N, R) % 4).to(torch.int32)
initial_age = (torch.arange(N * R, dtype=torch.float32).reshape(N, R) + 1) / 11
rates = torch.full((N, R), 1.7, dtype=torch.float32)
tau = torch.tensor([0.2, 0.1, float("nan"), 0.3, 0.4])
event_seed = torch.tensor(-0x123456789ABCDE, dtype=torch.int64)
step_id = torch.tensor((1 << 62) + 19, dtype=torch.int64)

def pack_infectious(state):
    packed = torch.zeros(
        (state.shape[0], math.ceil(state.shape[1] / 32)),
        dtype=torch.int64,
    )
    for replica in range(state.shape[1]):
        packed[:, replica // 32] |= (
            (state[:, replica] == 2).to(torch.int64) << (replica % 32)
        )
    return packed.to(torch.int32)

results = []
for node_tile, replica_tile in ((8, 4), (4, 2)):
    state = initial_state.clone()
    age = initial_age.clone()
    infectious_mask = pack_infectious(state)
    partials = torch.empty(
        (math.ceil(N / node_tile), R), dtype=torch.int32
    )
    replica_tiles = math.ceil(R / replica_tile)
    grid = (math.ceil(N / node_tile) * replica_tiles,)
    _ensemble_seir_transition_kernel[grid](
        state,
        age,
        rates,
        tau,
        event_seed,
        step_id,
        partials,
        infectious_mask,
        N,
        R,
        replica_tiles,
        UPDATE_INFECTIOUS_MASK=1,
        NODES_PER_PROGRAM=node_tile,
        REPLICAS_PER_TILE=replica_tile,
    )
    changed = state != initial_state
    assert torch.equal(partials.sum(dim=0), changed.sum(dim=0))
    assert torch.equal(partials[:, 2], torch.zeros_like(partials[:, 2]))
    assert torch.equal(state[:, 2], initial_state[:, 2])
    torch.testing.assert_close(age[:, 2], initial_age[:, 2], rtol=0.0, atol=0.0)
    for replica in (0, 1, 3, 4):
        torch.testing.assert_close(
            age[changed[:, replica], replica],
            torch.zeros_like(age[changed[:, replica], replica]),
            rtol=0.0,
            atol=0.0,
        )
        unchanged = ~changed[:, replica]
        torch.testing.assert_close(
            age[unchanged, replica],
            initial_age[unchanged, replica] + tau[replica],
            rtol=0.0,
            atol=0.0,
        )
    assert torch.equal(infectious_mask, pack_infectious(state))
    results.append((state, age, changed.sum(dim=0), infectious_mask))

for left, right in zip(results[0], results[1]):
    torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

# The optional bitmap branch must not alter the established transition stream.
state = initial_state.clone()
age = initial_age.clone()
partials = torch.empty((math.ceil(N / 8), R), dtype=torch.int32)
_ensemble_seir_transition_kernel[(6,)](
    state,
    age,
    rates,
    tau,
    event_seed,
    step_id,
    partials,
    state,  # Compile-time dead pointer when bitmap maintenance is disabled.
    N,
    R,
    2,
    UPDATE_INFECTIOUS_MASK=0,
    NODES_PER_PROGRAM=8,
    REPLICAS_PER_TILE=4,
)
torch.testing.assert_close(state, results[0][0], rtol=0.0, atol=0.0)
torch.testing.assert_close(age, results[0][1], rtol=0.0, atol=0.0)
torch.testing.assert_close(
    partials.sum(dim=0), results[0][2], rtol=0.0, atol=0.0
)

# Identical replica inputs still receive distinct packed Philox counters. R=37
# also covers the sign bit and a partial second infectious-mask word.
state = torch.ones((64, 37), dtype=torch.int32)
age = torch.zeros((64, 37), dtype=torch.float32)
rates = torch.full((64, 37), 3.5, dtype=torch.float32)
tau = torch.full((37,), 0.2, dtype=torch.float32)
partials = torch.empty((8, 37), dtype=torch.int32)
infectious_mask = pack_infectious(state)
_ensemble_seir_transition_kernel[(80,)](
    state,
    age,
    rates,
    tau,
    event_seed,
    step_id,
    partials,
    infectious_mask,
    64,
    37,
    10,
    UPDATE_INFECTIOUS_MASK=1,
    NODES_PER_PROGRAM=8,
    REPLICAS_PER_TILE=4,
)
assert not torch.equal(state[:, 0], state[:, 1])
assert torch.equal(infectious_mask, pack_infectious(state))

# A zero-rate step has no state writes to perform, but every valid lane still
# advances its age. Exercise R=35 so the no-event path also spans bit 31, a
# partial final bitmap word, and several narrow replica tiles per word.
N, R = 9, 35
state = (torch.arange(N * R).reshape(N, R) % 4).to(torch.int32)
state_before = state.clone()
age = (torch.arange(N * R, dtype=torch.float32).reshape(N, R) + 1) / 17
age_before = age.clone()
rates = torch.zeros((N, R), dtype=torch.float32)
tau = torch.linspace(0.01, 0.35, R, dtype=torch.float32)
partials = torch.empty((math.ceil(N / 4), R), dtype=torch.int32)
infectious_mask = pack_infectious(state)
mask_before = infectious_mask.clone()
replica_tiles = math.ceil(R / 2)
_ensemble_seir_transition_kernel[(math.ceil(N / 4) * replica_tiles,)](
    state,
    age,
    rates,
    tau,
    event_seed,
    step_id,
    partials,
    infectious_mask,
    N,
    R,
    replica_tiles,
    UPDATE_INFECTIOUS_MASK=1,
    NODES_PER_PROGRAM=4,
    REPLICAS_PER_TILE=2,
)
assert torch.equal(state, state_before)
torch.testing.assert_close(age, age_before + tau[None, :], rtol=0.0, atol=0.0)
assert not bool(partials.any())
assert torch.equal(infectious_mask, mask_before)
"""
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
