"""Construction-time dispatch and option-forwarding regressions."""

import pytest
import torch

from flashspread.engines import (
    create_ensemble_engine,
    create_markovian_engine,
    create_renewal_engine,
)
from flashspread.models import SEIRModel, SISModel


class _Spy:
    def __init__(self, graph, model, **kwargs):
        self.graph = graph
        self.model = model
        self.kwargs = kwargs


class _EnsembleSpy(_Spy):
    def __init__(self, graph, model, replicas, **kwargs):
        super().__init__(graph, model, **kwargs)
        self.replicas = replicas


def test_fused_factory_forwards_every_performance_option(monkeypatch):
    import flashspread.engines.renewal_fused as fused

    monkeypatch.setattr(fused, "RenewalEngineFusedCUDAGraph", _Spy)
    model = SEIRModel()
    engine = create_renewal_engine(
        object(),
        model,
        device="cuda:1",
        seed=19,
        epsilon=0.02,
        tau_max=0.4,
        steps_per_launch=12,
        use_mixed_precision=True,
        csr_strategy="merge",
        nodes_per_block=4,
        lanes_per_node=16,
        edges_per_merge_block=1024,
        use_active_compaction=False,
        transmission_mode="age_dependent",
    )
    assert engine.kwargs == {
        "steps_per_launch": 12,
        "use_active_compaction": False,
        "use_mixed_precision": True,
        "csr_strategy": "merge",
        "nodes_per_block": 4,
        "lanes_per_node": 16,
        "edges_per_merge_block": 1024,
        "warp_collaborative": False,
        "device": "cuda:1",
        "epsilon": 0.02,
        "tau_max": 0.4,
        "seed": 19,
        "bf16_weights": False,
    }
    assert engine.model is not model
    assert engine.model.transmission_mode == "age_dependent"
    assert model.transmission_mode == "constant"


def test_markovian_factory_forwards_seed_and_tau_min(monkeypatch):
    import flashspread.engines.markovian as markovian

    monkeypatch.setattr(markovian, "MarkovianEngine", _Spy)
    engine = create_markovian_engine(
        object(),
        SISModel(),
        device="cuda",
        seed=23,
        max_prob=0.07,
        theta=0.03,
        tau_min=0.005,
        tau_max=0.8,
    )
    assert engine.kwargs == {
        "device": "cuda",
        "max_prob": 0.07,
        "theta": 0.03,
        "tau_min": 0.005,
        "tau_max": 0.8,
        "seed": 23,
    }


def test_ensemble_factory_auto_selects_cpu_reference():
    import torch

    from flashspread import GraphCSR
    from flashspread.engines.ensemble import ReferenceEnsembleEngine

    graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
    engine = create_ensemble_engine(
        graph,
        SISModel(),
        3,
        device="cpu",
        seed=31,
        tau_max=0.7,
    )
    assert isinstance(engine, ReferenceEnsembleEngine)
    assert engine.state.shape == (2, 3)
    assert engine._base_seed == 31
    assert engine.tau_max == pytest.approx(0.7)


def test_ensemble_factory_forwards_tiled_cuda_controls(monkeypatch):
    import flashspread.engines.ensemble as ensemble

    monkeypatch.setattr(ensemble, "EnsembleEngine", _EnsembleSpy)
    engine = create_ensemble_engine(
        object(),
        SEIRModel(),
        17,
        device="cuda:1",
        seed=37,
        epsilon=0.02,
        tau_max=0.4,
        nodes_per_program=4,
        replicas_per_tile=16,
    )
    assert engine.replicas == 17
    assert engine.kwargs == {
        "nodes_per_program": 4,
        "replicas_per_tile": 16,
        "device": torch.device("cuda:1"),
        "seed": 37,
        "epsilon": 0.02,
        "max_prob": 0.1,
        "theta": 0.01,
        "tau_min": 1e-6,
        "tau_max": 0.4,
    }


@pytest.mark.parametrize("backend", [True, "unknown"])
def test_ensemble_factory_rejects_invalid_backend(backend):
    error = TypeError if backend is True else ValueError
    with pytest.raises(error, match="backend"):
        create_ensemble_engine(
            object(), SISModel(), 2, device="cpu", backend=backend
        )


def test_ensemble_factory_rejects_tiled_cpu_and_inert_tuning():
    with pytest.raises(ValueError, match="CUDA"):
        create_ensemble_engine(
            object(), SISModel(), 2, device="cpu", backend="tiled"
        )
    with pytest.raises(ValueError, match="tile controls"):
        create_ensemble_engine(
            object(), SISModel(), 2, device="cpu", replicas_per_tile=2
        )


def test_markovian_factory_selects_cuda_graph_and_forwards_batch(monkeypatch):
    import flashspread.engines.markovian as markovian

    monkeypatch.setattr(markovian, "MarkovianEngineCUDAGraph", _Spy)
    model = SISModel()
    engine = create_markovian_engine(
        object(),
        model,
        device="cuda:1",
        seed=29,
        use_cuda_graph=True,
        steps_per_launch=17,
    )
    assert engine.kwargs["device"] == "cuda:1"
    assert engine.kwargs["seed"] == 29
    assert engine.kwargs["steps_per_launch"] == 17
    assert engine.model is not model


@pytest.mark.parametrize("steps", [0, -1])
def test_markovian_factory_rejects_invalid_batch_before_allocation(steps):
    with pytest.raises(ValueError, match="positive"):
        create_markovian_engine(
            object(), SISModel(), steps_per_launch=steps
        )


def test_markovian_factory_caps_literal_cuda_graph_unrolling():
    with pytest.raises(ValueError, match="<= 4096"):
        create_markovian_engine(
            object(),
            SISModel(),
            use_cuda_graph=True,
            steps_per_launch=4097,
        )


def test_markovian_cuda_graph_rejects_cpu_before_graph_allocation():
    from flashspread.engines.markovian import MarkovianEngineCUDAGraph

    with pytest.raises(RuntimeError, match="CUDA device"):
        MarkovianEngineCUDAGraph(object(), SISModel(), device="cpu")


@pytest.mark.parametrize(
    "name",
    [
        "use_cuda_graph",
        "nonmarkov_edges",
        "use_fused",
        "bf16_weights",
        "use_mixed_precision",
        "warp_collaborative",
        "use_active_compaction",
    ],
)
def test_renewal_factory_rejects_truthy_non_boolean_flags(name):
    with pytest.raises(TypeError, match=rf"{name} must be a bool"):
        create_renewal_engine(object(), SEIRModel(), **{name: "False"})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"use_fused": True, "nonmarkov_edges": False}, "requires"),
        ({"use_mixed_precision": True, "use_fused": False}, "requires"),
        ({"use_active_compaction": True, "use_cuda_graph": False}, "requires"),
        ({"transmission_mode": "typo"}, "transmission_mode"),
        ({"steps_per_launch": 0}, "positive"),
        ({"use_fused": False, "csr_strategy": "merge"}, "require use_fused"),
    ],
)
def test_invalid_factory_combinations_fail_before_allocation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        create_renewal_engine(object(), SEIRModel(), **kwargs)


def test_direct_engines_reject_wrong_model_family_and_duplicate_inducers():
    import torch

    from flashspread import GraphCSR
    from flashspread.engines.markovian import MarkovianEngine
    from flashspread.engines.renewal import RenewalEngine

    graph = GraphCSR(torch.tensor([[0, 1], [1, 0]]), 2)
    with pytest.raises(TypeError, match="wrong model family"):
        MarkovianEngine(graph, SEIRModel(), device="cpu")
    with pytest.raises(TypeError, match="wrong model family"):
        RenewalEngine(graph, SISModel(), device="cpu")
    model = SISModel()
    model.inducer_states = [model.infected, model.infected]
    with pytest.raises(ValueError, match="duplicates"):
        MarkovianEngine(graph, model, device="cpu")
