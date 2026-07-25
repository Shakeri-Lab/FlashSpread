"""Regression tests for the silent-wrong-answer guards added in this pass.

Each test here corresponds to a defect that produced a plausible-looking wrong
number rather than an error, which is why none of them was caught by the
existing suite:

* the Markovian engine aliased an incoming CSR as its outgoing CSR whenever the
  graph was structurally symmetric, applying ``w(v -> u)`` where ``w(u -> v)``
  was required;
* the Markovian Triton pipeline was selected by a bare ``type()`` check, so a
  model with a shadowed hook had its semantics replaced by the built-in kernel
  equations;
* episode seeds were derived by integer addition, so ``(base, episode)`` pairs
  with equal sums produced bitwise identical streams;
* ``seed_infection`` accepted mid-run calls and non-additive repeat calls.
"""

import pytest
import torch

from flashspread.config import supports_builtin_markovian, supports_fused_renewal
from flashspread.core.graph import GraphCSR
from flashspread.core.host_rng import offset_seed
from flashspread.engines.markovian import MarkovianEngine
from flashspread.models import SEIRModel, SIRModel, SISModel
from flashspread.simulator import Simulator


# --------------------------------------------------------------------------
# Symmetric-CSR aliasing must not survive non-unit weights
# --------------------------------------------------------------------------


def _ring(num_nodes: int, degree: int) -> torch.Tensor:
    offsets = [s for step in range(1, degree // 2 + 1) for s in (step, -step)]
    node = torch.arange(num_nodes, dtype=torch.int64).repeat_interleave(len(offsets))
    shift = torch.tensor(offsets, dtype=torch.int64).repeat(num_nodes)
    return torch.stack(((node + shift) % num_nodes, node))


def test_generated_graph_can_acquire_weights_through_the_public_setter():
    """Pin the reachability that makes the weighted-alias guard necessary.

    If this ever stops holding, the guard is dead code and the comment
    explaining it should be revisited -- but while it holds, a structurally
    symmetric graph really can reach the engine carrying asymmetric weights.
    """
    graph = pytest.importorskip("flashspread").regular_graph(
        64, degree=4, seed=0, device="cpu", algorithm="circulant"
    )
    from flashspread.core.network import _GeneratedGraph

    assert isinstance(graph, _GeneratedGraph) and graph.symmetric
    assert not graph.csr.has_weights
    graph.csr.weights = torch.arange(1, graph.csr.num_edges + 1, dtype=torch.float32)
    assert graph.csr.has_weights, (
        "the public weights setter must still be able to attach weights to a "
        "generated graph; the Markovian alias guard depends on that being possible"
    )


@pytest.mark.gpu
def test_weighted_symmetric_graph_does_not_alias_its_transpose():
    """Weighted graphs must get a real transpose, not the incoming rows."""
    import flashspread as fs

    graph = fs.regular_graph(
        256, degree=4, seed=0, device="cuda", algorithm="circulant"
    )
    weights = torch.arange(
        1, graph.csr.num_edges + 1, dtype=torch.float32, device="cuda"
    )
    graph.csr.weights = weights

    engine = MarkovianEngine(graph, SISModel(beta=0.3, delta=0.1), device="cuda")
    assert not engine._shares_outgoing_csr, (
        "a weighted structurally symmetric graph must not reuse its incoming "
        "CSR as the outgoing CSR: entry k of incoming row u is the edge "
        "col_ind[k] -> u, so the weights would be reversed"
    )
    reference = engine.graph.transpose()
    torch.testing.assert_close(
        engine.outgoing_graph.weights_storage,
        reference.weights_storage,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.gpu
def test_unweighted_symmetric_graph_still_shares_storage():
    """The memory optimization must survive for the case where it is valid."""
    import flashspread as fs

    graph = fs.regular_graph(
        256, degree=4, seed=0, device="cuda", algorithm="circulant"
    )
    engine = MarkovianEngine(graph, SISModel(beta=0.3, delta=0.1), device="cuda")
    assert engine._shares_outgoing_csr


# --------------------------------------------------------------------------
# The Markovian built-in gate must be as strict as the renewal one
# --------------------------------------------------------------------------


def test_exact_builtin_markovian_models_are_accepted():
    assert supports_builtin_markovian(SISModel(beta=0.5, delta=0.2)) == "sis"
    assert supports_builtin_markovian(SIRModel(beta=0.5, gamma=0.2)) == "sir"
    # Wrong family: the renewal model must not resolve to a Markovian kernel.
    assert supports_builtin_markovian(SEIRModel(beta=0.3)) is None


def test_subclass_is_rejected():
    class TweakedSIS(SISModel):
        def apply_transitions(self, state, event_mask, out=None):  # pragma: no cover
            raise AssertionError("the built-in kernel must not replace this")

    assert supports_builtin_markovian(TweakedSIS(beta=0.5, delta=0.2)) is None


def test_instance_shadowed_hook_is_rejected():
    """The kernel bypasses these hooks, so a shadow must force the generic path."""
    for hook in ("prepare", "compute_rates", "apply_transitions"):
        model = SISModel(beta=0.5, delta=0.2)
        setattr(model, hook, lambda *args, **kwargs: None)
        assert supports_builtin_markovian(model) is None, (
            f"an instance-shadowed {hook} must not be silently replaced by the "
            "built-in kernel equations"
        )


def test_mutated_state_ids_and_inducers_are_rejected():
    model = SISModel(beta=0.5, delta=0.2)
    model.infected = 7
    assert supports_builtin_markovian(model) is None

    model = SISModel(beta=0.5, delta=0.2)
    # The frontier kernel treats infected as the sole inducer; the generic
    # rebuild honours a wider set, so the two would silently disagree.
    model.inducer_states = [0, 1]
    assert supports_builtin_markovian(model) is None

    model = SISModel(beta=0.5, delta=0.2)
    model.num_states = 3
    assert supports_builtin_markovian(model) is None


def test_gate_mirrors_the_renewal_gate_shape():
    """Both gates must reject the same categories, so neither drifts stricter."""
    for factory in (lambda: SISModel(beta=0.5, delta=0.2), lambda: SEIRModel(beta=0.3)):
        shadowed = factory()
        shadowed.apply_transitions = lambda *args, **kwargs: None
        assert supports_builtin_markovian(shadowed) is None
        assert supports_fused_renewal(shadowed) is False


def test_cuda_graph_markovian_rejects_a_shadowed_model_without_a_device():
    """The strict gate must run before any CUDA requirement is evaluated."""
    from flashspread.engines.markovian import MarkovianEngineCUDAGraph

    model = SISModel(beta=0.5, delta=0.2)
    model.compute_rates = lambda *args, **kwargs: None
    graph = GraphCSR(_ring(32, 4), 32, incoming=True)
    with pytest.raises((TypeError, RuntimeError)):
        MarkovianEngineCUDAGraph(graph, model, device="cpu")


# --------------------------------------------------------------------------
# Episode seeds must be domain separated
# --------------------------------------------------------------------------


def test_episode_zero_reproduces_the_base_seed_bitwise():
    for base in (0, 1, 100, 2**63, 2**64 - 1):
        assert offset_seed(base, 0) == base


def test_equal_sum_seed_episode_pairs_no_longer_collide():
    """The exact aliasing that collapsed an S x E sweep to S + E - 1 streams."""
    assert offset_seed(100, 1) != offset_seed(101, 0)
    assert offset_seed(100, 2) != offset_seed(102, 0)
    assert offset_seed(5, 10) != offset_seed(10, 5)


def test_seed_episode_grid_is_pairwise_distinct():
    seeds = range(16)
    episodes = range(64)
    streams = {offset_seed(s, e) for s in seeds for e in episodes}
    assert len(streams) == len(seeds) * len(episodes)


def test_reset_episode_produces_a_different_trajectory():
    import flashspread as fs

    graph = fs.regular_graph(512, degree=6, seed=0, device="cpu", algorithm="circulant")
    model = SEIRModel(beta=0.4)

    def run(episode):
        sim = fs.Simulator(graph, model, device="cpu", seed=7)
        sim.reset(episode=episode)
        sim.seed_infection(40)
        sim.run(until=6.0, record_every=1.0)
        return sim.counts()

    baseline = run(None)
    assert (run(None) == baseline).all(), "reset() must replay the base stream"
    assert not (run(1) == baseline).all(), "reset(episode=1) must decorrelate"


# --------------------------------------------------------------------------
# seed_infection is an initial condition, not a mid-run mutation
# --------------------------------------------------------------------------


def _simulator():
    import flashspread as fs

    graph = fs.regular_graph(400, degree=6, seed=0, device="cpu", algorithm="circulant")
    return fs.Simulator(graph, SEIRModel(beta=0.4), device="cpu", seed=3)


def test_seed_infection_after_stepping_is_rejected():
    sim = _simulator().seed_infection(50)
    sim.run(until=2.0, record_every=1.0)
    with pytest.raises(RuntimeError, match="initial condition"):
        sim.seed_infection(50)


def test_repeat_seeding_of_one_compartment_is_rejected():
    """Repeat calls are not additive: the second draw can hit seeded nodes."""
    sim = _simulator().seed_infection(100)
    with pytest.raises(RuntimeError, match="already been seeded"):
        sim.seed_infection(100)


def test_seeding_a_second_distinct_compartment_is_allowed():
    sim = _simulator()
    sim.seed_infection(20, state=sim.model.exposed)
    sim.seed_infection(20, state=sim.model.infected)
    counts = sim.counts()
    assert counts[sim.model.exposed] > 0
    assert counts[sim.model.infected] > 0


def test_reset_permits_reseeding():
    sim = _simulator().seed_infection(50)
    sim.run(until=2.0, record_every=1.0)
    sim.reset()
    sim.seed_infection(50)  # must not raise
    assert sim.current_time == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Seed sampling must not cost O(N) storage, nor bias towards low node ids
# --------------------------------------------------------------------------


def test_small_populations_keep_the_historical_randperm_draw():
    """Below the threshold, every existing seeded initial condition is unchanged."""
    from flashspread.utils import _RANDPERM_MAX_NODES, sample_distinct_nodes

    for num_nodes in (1000, 100_000, _RANDPERM_MAX_NODES):
        drawn = sample_distinct_nodes(
            num_nodes, 10, device="cpu", generator=torch.Generator().manual_seed(5)
        )
        historical = torch.randperm(
            num_nodes, generator=torch.Generator().manual_seed(5)
        )[:10]
        assert torch.equal(drawn, historical), (
            f"the N={num_nodes} draw must stay bitwise identical so recorded "
            "initial conditions remain reproducible"
        )


def test_large_population_draw_is_distinct_in_range_and_reproducible():
    from flashspread.utils import _RANDPERM_MAX_NODES, sample_distinct_nodes

    num_nodes = 8 * _RANDPERM_MAX_NODES
    count = 512
    drawn = sample_distinct_nodes(
        num_nodes, count, device="cpu", generator=torch.Generator().manual_seed(9)
    )
    assert drawn.numel() == count
    assert drawn.unique().numel() == count
    assert int(drawn.min()) >= 0 and int(drawn.max()) < num_nodes

    again = sample_distinct_nodes(
        num_nodes, count, device="cpu", generator=torch.Generator().manual_seed(9)
    )
    assert torch.equal(drawn, again)
    assert sample_distinct_nodes(
        num_nodes, 0, device="cpu", generator=torch.Generator().manual_seed(9)
    ).numel() == 0


def test_large_population_draw_is_not_biased_towards_low_ids():
    """``torch.unique`` sorts, so a truncated prefix would skew low. Catch that."""
    from flashspread.utils import _RANDPERM_MAX_NODES, sample_distinct_nodes

    num_nodes = 8 * _RANDPERM_MAX_NODES
    generator = torch.Generator().manual_seed(3)
    lower = total = 0
    for _ in range(40):
        drawn = sample_distinct_nodes(
            num_nodes, 200, device="cpu", generator=generator
        )
        lower += int((drawn < num_nodes // 2).sum())
        total += drawn.numel()
    fraction = lower / total
    # 8000 Bernoulli(0.5) draws: 5 sigma is about 0.028.
    assert 0.45 < fraction < 0.55, (
        f"{fraction:.3f} of ids fell in the lower half; a sorted-prefix "
        "truncation bug looks exactly like this"
    )


def test_injected_engine_accepts_an_unindexed_device_request():
    """``device="cuda"`` plus an engine on ``cuda:0`` is a correct configuration."""
    from flashspread.engines.renewal import RenewalEngine

    graph = GraphCSR(_ring(64, 4), 64, incoming=True)
    model = SEIRModel(beta=0.3)
    engine = RenewalEngine(graph, model, device="cpu")
    sim = Simulator(graph, model, device="cpu", engine=engine)
    assert torch.device(sim.device) == torch.device(engine.device)
