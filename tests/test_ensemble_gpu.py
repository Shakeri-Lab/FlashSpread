"""GPU coverage for the tiled shared-graph ensemble.

The ensemble subsystem previously had no ``gpu``-marked test at all, so a
sampler that never fired was invisible: the acceptance harness restores fixed
checkpoints and times one step, which costs exactly the same whether or not any
transition is accepted.

The tests here assert *observable stochastic progress* per replica, which is the
property a throughput benchmark cannot see. See ``test_rng_contract.py`` for the
device-free compile-time guard on the same failure mode.
"""

import pytest
import torch

from flashspread.core.graph import GraphCSR
from flashspread.engines import create_ensemble_engine
from flashspread.models import SEIRModel


pytestmark = pytest.mark.gpu


REPLICAS = 32


def _ring_graph(num_nodes: int, degree: int, device: str) -> GraphCSR:
    """Undirected ring lattice built straight into incoming CSR."""
    offsets = []
    for step in range(1, degree // 2 + 1):
        offsets += [step, -step]
    node = torch.arange(num_nodes, dtype=torch.int64).repeat_interleave(len(offsets))
    shift = torch.tensor(offsets, dtype=torch.int64).repeat(num_nodes)
    source = (node + shift) % num_nodes
    edge_index = torch.stack((source, node)).to(device)
    return GraphCSR(edge_index, num_nodes, incoming=True)


def _engine(num_nodes=4096, degree=8, seed=12345, beta=0.6):
    graph = _ring_graph(num_nodes, degree, "cuda")
    model = SEIRModel(beta=beta)
    engine = create_ensemble_engine(graph, model, replicas=REPLICAS, seed=seed)
    engine.seed_infection(num_nodes // 20, state=model.infected)
    return engine


def test_every_replica_accepts_transitions():
    """Each replica must make stochastic progress, not merely advance its clock.

    With a suppressed sampler every replica holds its seeded state forever while
    ``tau`` and ``current_time`` still advance, so asserting on the clock alone
    would pass. Assert on the state field instead.
    """
    engine = _engine()
    initial = engine.state.clone()
    for _ in range(40):
        engine.step()

    changed_per_replica = (engine.state != initial).sum(dim=0)
    assert changed_per_replica.shape == (REPLICAS,)
    assert int(changed_per_replica.min()) > 0, (
        "at least one replica accepted zero transitions over 40 steps; the "
        f"per-replica change counts were {changed_per_replica.tolist()}"
    )
    assert float(engine.current_time.min()) > 0.0


def test_replicas_are_independent_not_identical():
    """Shared graph, independent streams: no two replicas may coincide."""
    engine = _engine()
    for _ in range(40):
        engine.step()

    state = engine.state
    duplicates = [
        (i, j)
        for i in range(REPLICAS)
        for j in range(i + 1, REPLICAS)
        if torch.equal(state[:, i], state[:, j])
    ]
    assert not duplicates, f"replica pairs share an identical state field: {duplicates}"

    infected = (state == 2).sum(dim=0).float()
    assert float(infected.std()) > 0.0


def test_event_frequency_matches_the_sampled_probability():
    """Directly bound the accepted-event rate against its Bernoulli target.

    This is the assertion that fails when the Philox counter is widened: the
    realized rate collapses to ~p * 2**-32 rather than landing inside a
    binomial band around p.
    """
    engine = _engine()
    lanes = engine.state.numel()

    accepted = 0
    expected = 0.0
    for _ in range(12):
        before = engine.state.clone()
        tau, _ = engine.step()
        accepted += int((engine.state != before).sum())
        # ``rates`` is written by the rate phase and read by the transition
        # phase, so after step() it still holds the field that was sampled --
        # together with the tau step() returned, that is the exact Bernoulli
        # target for the transitions just applied.
        rates = engine.rates
        probability = -torch.expm1(-rates * tau.to(rates.dtype).unsqueeze(0))
        expected += float(probability.sum())

    assert expected > 0.0, "the rate field carried no probability mass"
    sigma = max(expected**0.5, 1.0)
    assert abs(accepted - expected) < 6.0 * sigma, (
        f"accepted {accepted} transitions against an expected {expected:.1f} "
        f"(+/- {sigma:.1f}) over {lanes} lanes"
    )


def test_reset_reproduces_and_episode_decorrelates():
    """``reset()`` must replay bitwise; ``reset(episode=k)`` must diverge."""
    engine = _engine()
    for _ in range(10):
        engine.step()
    baseline = engine.state.clone()

    engine.reset()
    engine.seed_infection(engine.num_nodes // 20, state=2)
    for _ in range(10):
        engine.step()
    assert torch.equal(engine.state, baseline)

    engine.reset(episode=1)
    engine.seed_infection(engine.num_nodes // 20, state=2)
    for _ in range(10):
        engine.step()
    assert not torch.equal(engine.state, baseline)


def test_infectious_bitmap_tracks_the_state_field():
    """The packed bitmap is a second source of truth; pin it to the state."""
    engine = _engine()
    mask = getattr(engine, "_infectious_mask", None)
    if mask is None:
        pytest.skip("this model/graph did not select the packed-bitmap path")

    for _ in range(15):
        engine.step()
        engine.count_by_state()  # refresh whatever the freshness guard requires

        expected = torch.zeros_like(mask)
        infectious = (engine.state == 2).to(torch.int32)
        for replica in range(REPLICAS):
            word, bit = divmod(replica, 32)
            expected[:, word] |= infectious[:, replica] << bit
        torch.testing.assert_close(mask, expected, rtol=0.0, atol=0.0)
