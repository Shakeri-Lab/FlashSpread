"""Compatibility and semantic-parity tests for the quarantined tunable engine."""

from __future__ import annotations

from contextlib import nullcontext
import math

import pytest
import torch

from flashspread.core.graph import GraphCSR
from flashspread.engines.renewal import RenewalEngine
from flashspread.engines.renewal_tunable import (
    RenewalEngineTunable,
    RenewalEngineTunableCUDAGraph,
    estimate_flops_per_step,
    estimate_memory_bytes_per_step,
)
from flashspread.models import SEIRModel


def _ring_graph() -> GraphCSR:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ],
        dtype=torch.int64,
    )
    return GraphCSR(edge_index, 4)


def _seir() -> SEIRModel:
    return SEIRModel(
        beta=0.3,
        mean_ei=5.0,
        median_ei=4.0,
        mean_ir=3.9,
        median_ir=1.5,
    )


@pytest.mark.parametrize("dense_pressure", [False, True])
def test_multiplier_one_matches_production_step_and_rng(dense_pressure):
    graph = _ring_graph()
    production = RenewalEngine(graph, _seir(), device="cpu", seed=91)
    tunable = RenewalEngineTunable(
        graph,
        _seir(),
        device="cpu",
        seed=91,
        compute_multiplier=1,
        dense_pressure=dense_pressure,
    )
    state = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    age = torch.tensor([0.0, 2.0, 3.0, 0.0])
    production.set_initial_state(state, age)
    tunable.set_initial_state(state, age)

    for _ in range(8):
        production_tau, _ = production.step()
        tunable_tau, _ = tunable.step()
        assert tunable_tau == production_tau
        assert torch.equal(tunable.state, production.state)
        assert torch.equal(tunable.age, production.age)
        assert torch.equal(tunable.rates, production.rates)
        assert torch.equal(tunable.event_mask, production.event_mask)
        assert torch.equal(tunable.seed_counter, production.seed_counter)

    assert tunable.current_time == production.current_time
    assert tunable.total_steps == production.total_steps
    if dense_pressure:
        assert torch.equal(tunable.pressure_dense, tunable.pressure)


class _ConstantRateModel:
    is_markovian = False
    susceptible = 0
    infected = 1
    num_states = 2
    inducer_states = [1]

    def __init__(self, rate: float):
        self.rate = rate
        self.calls = 0

    def prepare(self, device):
        pass

    def compute_rates(self, age, state, pressure, out=None):
        self.calls += 1
        out.fill_(self.rate)
        return out

    def apply_transitions(self, state, event_mask, out=None):
        return out.copy_(state)


def test_multiplier_repeats_only_rate_hook_and_dense_copy_resets():
    model = _ConstantRateModel(0.25)
    engine = RenewalEngineTunable(
        _ring_graph(),
        model,
        device="cpu",
        seed=8,
        compute_multiplier=3,
        dense_pressure=True,
    )
    engine.set_initial_state(torch.tensor([1, 0, 0, 0], dtype=torch.int32))

    tau, _ = engine.step()

    assert tau == pytest.approx(engine.epsilon / model.rate)
    assert model.calls == 3
    assert torch.equal(engine.pressure_dense, engine.pressure)
    assert engine.get_config() == {
        "engine_type": "RenewalEngineTunable",
        "num_nodes": 4,
        "epsilon": 0.03,
        "tau_max": 1.0,
        "compute_multiplier": 3,
        "dense_pressure": True,
        "timing_enabled": False,
    }

    engine.reset()
    assert not torch.count_nonzero(engine.pressure_dense)


@pytest.mark.parametrize("invalid_rate", [-1.0, float("nan")])
def test_invalid_rates_share_production_transactional_rng_semantics(invalid_rate):
    for engine_type in (RenewalEngine, RenewalEngineTunable):
        engine = engine_type(
            _ring_graph(),
            _ConstantRateModel(invalid_rate),
            device="cpu",
            seed=123,
        )
        engine.set_initial_state(torch.tensor([1, 0, 0, 0], dtype=torch.int32))
        state_before = engine.state.clone()
        age_before = engine.age.clone()
        counter_before = engine.seed_counter.clone()

        with pytest.raises(FloatingPointError, match="non-finite or non-positive"):
            engine.step()

        assert torch.equal(engine.state, state_before)
        assert torch.equal(engine.age, age_before)
        assert torch.equal(engine.seed_counter, counter_before)
        assert engine.current_time == 0.0
        assert engine.total_steps == 0


def test_cuda_graph_compatibility_reuses_production_static_step(monkeypatch):
    def skip_capture(self, steps_per_launch):
        self.steps_per_launch = steps_per_launch
        self.step_time_accumulator = torch.zeros(1, dtype=torch.float64)
        self.graph_exec = None

    monkeypatch.setattr(
        RenewalEngineTunableCUDAGraph,
        "_initialize_cuda_graph",
        skip_capture,
    )
    model = _ConstantRateModel(0.5)
    engine = RenewalEngineTunableCUDAGraph(
        _ring_graph(),
        model,
        device="cpu",
        compute_multiplier=2,
        dense_pressure=True,
        timing_enabled=True,
        steps_per_launch=7,
    )

    assert RenewalEngineTunableCUDAGraph._static_step is RenewalEngine._static_step
    assert engine.timing_enabled is False
    engine._static_step()
    assert math.isfinite(float(engine.step_time_accumulator))
    assert float(engine.step_time_accumulator) > 0.0
    assert model.calls == 2
    assert engine.get_config()["steps_per_launch"] == 7
    assert engine.get_config()["engine_type"] == "RenewalEngineTunableCUDAGraph"


def test_mocked_cuda_events_record_every_production_phase(monkeypatch):
    class FakeEvent:
        clock = 0

        def __init__(self, *, enable_timing):
            assert enable_timing is True
            self.timestamp = None

        def record(self):
            type(self).clock += 1
            self.timestamp = type(self).clock

        def synchronize(self):
            assert self.timestamp is not None

        def elapsed_time(self, end_event):
            return float(end_event.timestamp - self.timestamp)

    engine = RenewalEngineTunable(
        _ring_graph(),
        _ConstantRateModel(0.25),
        device="cpu",
        seed=17,
        timing_enabled=False,
    )
    engine.set_initial_state(torch.tensor([1, 0, 0, 0], dtype=torch.int32))

    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    engine.timing_enabled = True
    engine.device = torch.device("cuda")
    engine._initialize_timing()

    engine.step()

    expected = {"pressure", "hazard", "tau_select", "transition", "total"}
    assert set(engine._timing_records) == expected
    assert all(len(engine._timing_records[phase]) == 1 for phase in expected)
    stats = engine.get_timing_stats()
    assert set(stats) == expected
    assert all(stats[phase]["count"] == 1 for phase in expected)
    assert all(stats[phase]["mean_ms"] > 0.0 for phase in expected)

    engine.reset_timing()
    assert all(not records for records in engine._timing_records.values())


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("compute_multiplier", 0, ValueError),
        ("compute_multiplier", True, TypeError),
        ("dense_pressure", 1, TypeError),
        ("timing_enabled", 1, TypeError),
    ],
)
def test_synthetic_options_are_strict(keyword, value, error):
    with pytest.raises(error):
        RenewalEngineTunable(
            _ring_graph(),
            _ConstantRateModel(0.5),
            device="cpu",
            **{keyword: value},
        )


def test_historical_estimators_and_names_remain_available():
    flops = estimate_flops_per_step(10, 20, compute_multiplier=3, dense_hazard=True)
    traffic = estimate_memory_bytes_per_step(10, 20, dense_pressure=True)
    assert flops["hazard"] == 10 * 55 * 2 * 3
    assert flops["total"] == sum(value for key, value in flops.items() if key != "total")
    assert traffic["dense_pressure"] == 40
    assert traffic["total"] == sum(
        value for key, value in traffic.items() if key != "total"
    )


def test_engine_package_keeps_deprecated_lazy_import_names():
    import flashspread.engines as engines

    expected = {
        "RenewalEngineTunable": RenewalEngineTunable,
        "RenewalEngineTunableCUDAGraph": RenewalEngineTunableCUDAGraph,
        "estimate_flops_per_step": estimate_flops_per_step,
        "estimate_memory_bytes_per_step": estimate_memory_bytes_per_step,
    }
    for name, value in expected.items():
        with pytest.warns(DeprecationWarning, match="historical synthetic"):
            assert getattr(engines, name) is value
