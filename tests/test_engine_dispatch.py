"""Canonical scalar-engine plan and dispatch regressions."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from flashspread.config import EngineConfig, supports_fused_renewal
from flashspread.engines import (
    create_engine,
    create_markovian_engine,
    create_renewal_engine,
)
from flashspread.models import SEIRModel, SISModel


class _Spy:
    def __init__(self, graph, model, **kwargs):
        self.graph = graph
        self.model = model
        self.kwargs = kwargs


class _GenericRenewal:
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


class _SEIRSubclass(SEIRModel):
    """Semantically custom even when it inherits the built-in hooks verbatim."""


def test_bare_engines_import_remains_metadata_only():
    repository = Path(__file__).resolve().parents[1]
    script = """
import sys
import flashspread.engines as engines

assert callable(engines.create_engine)
assert "torch" not in sys.modules
assert "flashspread.config" not in sys.modules
assert "flashspread.engines.markovian" not in sys.modules
assert "flashspread.engines.renewal" not in sys.modules
assert "flashspread.engines.renewal_fused" not in sys.modules
assert "triton" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=dict(os.environ),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_protocol_complete_custom_cuda_resolution_stays_torch_free():
    repository = Path(__file__).resolve().parents[1]
    script = """
import sys
from flashspread.config import EngineConfig

class FakeCUDA:
    type = "cuda"

class CustomSEIR:
    is_markovian = False
    susceptible, exposed, infected, recovered = 0, 1, 2, 3
    num_states = 4
    inducer_states = (2,)
    transmission_mode = "constant"
    beta = 0.3
    _mu_ei = _sig_ei = _mu_ir = _sig_ir = None

    def prepare(self, device): pass
    def compute_rates(self, age, state, pressure, out=None): return out
    def apply_transitions(self, state, event_mask, out=None): return out

plan = EngineConfig()._resolve_plan(
    FakeCUDA(), markovian=False, model=CustomSEIR()
)
assert plan.options["use_fused"] is False
assert plan.options["use_cuda_graph"] is False
assert "torch" not in sys.modules
assert "flashspread.models.compartmental" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=repository,
        env=dict(os.environ),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_engine_config_dict_is_a_view_of_the_typed_plan():
    model = SEIRModel(transmission_mode="age_dependent")
    config = EngineConfig(
        backend="fused",
        execution="eager",
        traversal="warp",
        precision="bf16_weights",
        batch_steps=17,
        epsilon=0.02,
        tau_max=0.4,
    )
    device = torch.device("cuda:1")

    plan = config._resolve_plan(
        device,
        markovian=False,
        model=model,
        seed=29,
    )

    assert plan.device == device
    assert plan.seed == 29
    assert plan.family == "renewal"
    assert plan.options["use_fused"] is True
    assert plan.options["use_cuda_graph"] is False
    assert plan.factory_kwargs() == config.resolve(
        device,
        markovian=False,
        model=model,
    )


def test_canonical_dispatch_preserves_family_and_auto_backend_policy(monkeypatch):
    import flashspread.engines.markovian as markovian
    import flashspread.engines.renewal as renewal
    import flashspread.engines.renewal_fused as fused

    monkeypatch.setattr(markovian, "MarkovianEngine", _Spy)
    monkeypatch.setattr(renewal, "RenewalEngine", _Spy)
    monkeypatch.setattr(renewal, "RenewalEngineNonMarkov", _Spy)
    monkeypatch.setattr(fused, "RenewalEngineFusedCUDAGraph", _Spy)

    markov = create_engine(object(), SISModel(), device="cpu", seed=11)
    assert markov.kwargs["device"] == "cpu"
    assert markov.kwargs["seed"] == 11
    assert "steps_per_launch" not in markov.kwargs

    generic = create_engine(object(), _GenericRenewal(), device="cuda:1", seed=13)
    assert generic.kwargs["device"] == "cuda:1"
    assert generic.kwargs["seed"] == 13

    age_dependent = create_engine(
        object(),
        SEIRModel(transmission_mode="age_dependent"),
        device="cpu",
        seed=17,
    )
    assert age_dependent.kwargs["device"] == "cpu"
    assert age_dependent.model.transmission_mode == "age_dependent"

    fused_engine = create_engine(
        object(),
        SEIRModel(),
        device="cuda:1",
        seed=19,
    )
    assert fused_engine.kwargs["device"] == "cuda:1"
    assert fused_engine.kwargs["seed"] == 19
    assert fused_engine.kwargs["steps_per_launch"] == 50


@pytest.mark.parametrize(
    "variant",
    ["subclass", "shadowed-compute-rates", "shadowed-transition"],
)
def test_cuda_auto_fusion_falls_back_for_custom_seir_semantics(monkeypatch, variant):
    import flashspread.engines.renewal as renewal
    import flashspread.engines.renewal_fused as fused

    if variant == "subclass":
        model = _SEIRSubclass()
    else:
        model = SEIRModel()
        shadowed_method = (
            "compute_rates"
            if variant == "shadowed-compute-rates"
            else "apply_transitions"
        )
        setattr(
            model,
            shadowed_method,
            lambda *args, **kwargs: kwargs.get("out"),
        )

    class ForbiddenFused(_Spy):
        def __init__(self, *args, **kwargs):
            raise AssertionError("custom model reached the fused constructor")

    monkeypatch.setattr(renewal, "RenewalEngine", _Spy)
    monkeypatch.setattr(fused, "RenewalEngineFusedCUDAGraph", ForbiddenFused)

    engine = create_engine(object(), model, device="cuda:1", seed=31)

    assert isinstance(engine, _Spy)
    assert engine.kwargs["device"] == "cuda:1"
    assert engine.kwargs["seed"] == 31
    assert not supports_fused_renewal(model)


def test_explicit_fused_backend_rejects_custom_seir_before_allocation():
    with pytest.raises(TypeError, match="exact, unmodified"):
        create_renewal_engine(
            object(),
            _SEIRSubclass(),
            device="cuda",
            use_fused=True,
        )
    with pytest.raises(TypeError, match="exact, unmodified"):
        create_engine(
            object(),
            _SEIRSubclass(),
            device="cuda",
            config=EngineConfig(backend="fused", execution="eager"),
        )


def test_fused_capability_rejects_class_level_hook_monkeypatch(monkeypatch):
    model = SEIRModel()
    assert supports_fused_renewal(model)

    monkeypatch.setattr(SEIRModel, "compute_rates", lambda *args, **kwargs: None)

    assert not supports_fused_renewal(model)


def test_direct_fused_constructor_rejects_custom_seir_before_graph_allocation(
    monkeypatch,
):
    import flashspread.engines.renewal_fused as fused

    # The semantic guard precedes graph/CUDA allocation. Keep this regression
    # runnable on CPU-only installations where optional Triton is absent.
    monkeypatch.setattr(fused, "triton", object())
    with pytest.raises(TypeError, match="exact, unmodified"):
        fused.RenewalEngineFused(
            object(),
            _SEIRSubclass(),
            device="cuda",
        )


def test_config_dispatch_uses_the_same_constructor_and_validation(monkeypatch):
    import flashspread.engines.renewal as renewal

    monkeypatch.setattr(renewal, "RenewalEngine", _Spy)
    config = EngineConfig(
        backend="reference",
        execution="eager",
        epsilon=0.02,
        tau_max=0.4,
    )

    engine = create_engine(
        object(),
        SEIRModel(),
        device="cpu",
        config=config,
        seed=23,
    )

    assert engine.kwargs == {
        "device": "cpu",
        "epsilon": 0.02,
        "tau_max": 0.4,
        "seed": 23,
        "bf16_weights": False,
    }

    oversized = EngineConfig(execution="cuda_graph", batch_steps=4097)
    assert oversized.resolve(
        torch.device("cuda"), markovian=True, model=SISModel()
    )["steps_per_launch"] == 4097
    with pytest.raises(ValueError, match="<= 4096"):
        create_engine(
            object(),
            SISModel(),
            device="cuda",
            config=oversized,
        )


def test_canonical_dispatch_preserves_conflict_and_unknown_keyword_errors():
    with pytest.raises(ValueError, match="either config"):
        create_engine(
            object(),
            SEIRModel(),
            device="cpu",
            config=EngineConfig(),
            use_fused=False,
        )
    with pytest.raises(TypeError, match="create_renewal_engine.*unexpected keyword"):
        create_engine(
            object(),
            SEIRModel(),
            device="cpu",
            misspelled_option=True,
        )


def test_compatibility_adapter_signatures_remain_stable():
    assert tuple(inspect.signature(create_markovian_engine).parameters) == (
        "graph",
        "model",
        "device",
        "max_prob",
        "theta",
        "tau_min",
        "tau_max",
        "seed",
        "use_cuda_graph",
        "steps_per_launch",
    )
    assert tuple(inspect.signature(create_renewal_engine).parameters) == (
        "graph",
        "model",
        "device",
        "use_cuda_graph",
        "nonmarkov_edges",
        "use_fused",
        "bf16_weights",
        "transmission_mode",
        "epsilon",
        "tau_max",
        "steps_per_launch",
        "seed",
        "use_mixed_precision",
        "csr_strategy",
        "nodes_per_block",
        "lanes_per_node",
        "edges_per_merge_block",
        "warp_collaborative",
        "use_active_compaction",
    )
