"""Focused dispatch regression for the non-Markovian renewal pressure hook."""

from __future__ import annotations

import pytest
import torch

from flashspread.core.graph import GraphCSR
from flashspread.core.reference import reference_influence_infectivity_csr
from flashspread.engines.renewal import (
    RenewalEngine,
    RenewalEngineNonMarkov,
    RenewalEngineNonMarkovCUDAGraph,
)


class _TrackedNonMarkovModel:
    is_markovian = False
    susceptible = 0
    infected = 1
    num_states = 2
    inducer_states = [1]

    def __init__(self):
        self.calls = []
        self.rate_pressure = None

    def prepare(self, device):
        pass

    def compute_rates(self, age, state, pressure, out=None):
        raise AssertionError("binary-pressure rate evaluator must not be called")

    def compute_infectivity(self, age, state, out=None):
        self.calls.append("infectivity")
        out.copy_(torch.where(state == self.infected, age + 1.0, 0.0))
        return out

    def compute_rates_nonmarkov(self, age, state, pressure, out=None):
        self.calls.append("rates_nonmarkov")
        self.rate_pressure = pressure.clone()
        return out.fill_(0.25)

    def apply_transitions(self, state, event_mask, out=None):
        self.calls.append("transition")
        return out.copy_(state)


def _weighted_graph() -> GraphCSR:
    return GraphCSR(
        torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]],
            dtype=torch.int64,
        ),
        3,
        weights=torch.tensor([2.0, 3.0, 5.0, 7.0]),
    )


def test_inherited_step_dispatches_nonmarkov_pressure_rates_and_reset():
    model = _TrackedNonMarkovModel()
    engine = RenewalEngineNonMarkov(
        _weighted_graph(), model, device="cpu", seed=4
    )
    initial_state = torch.tensor([1, 0, 1], dtype=torch.int32)
    initial_age = torch.tensor([2.0, 0.0, 4.0])
    engine.set_initial_state(initial_state, initial_age)
    expected_infectivity = torch.tensor([3.0, 0.0, 5.0])
    expected_pressure = reference_influence_infectivity_csr(
        engine.graph, expected_infectivity
    )

    tau, _ = engine.step()

    assert RenewalEngineNonMarkov._step_impl is RenewalEngine._step_impl
    assert (
        RenewalEngineNonMarkovCUDAGraph._compute_pressure_phase
        is RenewalEngineNonMarkov._compute_pressure_phase
    )
    assert RenewalEngineNonMarkovCUDAGraph._static_step is RenewalEngine._static_step
    assert model.calls == ["infectivity", "rates_nonmarkov", "transition"]
    assert torch.equal(engine.infectivity, expected_infectivity)
    assert torch.equal(engine.pressure, expected_pressure)
    assert torch.equal(model.rate_pressure, expected_pressure)
    assert tau == pytest.approx(engine.epsilon / 0.25)
    assert torch.allclose(engine.age, initial_age + tau)

    engine.infectivity.fill_(9.0)
    engine.reset()
    assert not torch.count_nonzero(engine.infectivity)
