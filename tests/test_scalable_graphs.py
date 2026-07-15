"""CPU tests for the bounded-memory direct-CSR regular workload."""

import math

import pytest
import torch

from flashspread.core import network
from flashspread.core.graph import GraphCSR
from flashspread.core.network import FixedDegreeGraph
from flashspread.graphs import regular_graph


def _adjacency(graph: FixedDegreeGraph) -> list[set[int]]:
    csr = graph.csr
    return [
        set(csr.col_ind[int(csr.row_ptr[i]) : int(csr.row_ptr[i + 1])].tolist())
        for i in range(csr.num_nodes)
    ]


@pytest.mark.parametrize(("n", "degree"), [(31, 8), (24, 5), (18, 1), (9, 0)])
def test_circulant_is_exact_simple_symmetric_regular(n, degree):
    graph = FixedDegreeGraph(
        n, degree, device="cpu", seed=42, algorithm="circulant"
    )
    adjacency = _adjacency(graph)

    assert graph.num_edges == n * degree
    assert torch.equal(
        graph.csr.row_ptr,
        torch.arange(0, n * degree + 1, degree, dtype=torch.int32)
        if degree
        else torch.zeros(n + 1, dtype=torch.int32),
    )
    for node, neighbors in enumerate(adjacency):
        assert len(neighbors) == degree
        assert node not in neighbors
        assert all(node in adjacency[neighbor] for neighbor in neighbors)
    assert graph.circulant_component_count == math.gcd(
        n, *graph.circulant_offsets
    )


def test_all_small_mathematically_valid_circulants_are_simple_and_symmetric():
    for n in range(1, 17):
        for degree in range(n):
            if (n * degree) % 2:
                continue
            graph = FixedDegreeGraph(
                n, degree, device="cpu", seed=11, algorithm="circulant"
            )
            adjacency = _adjacency(graph)
            assert all(len(neighbors) == degree for neighbors in adjacency)
            assert all(node not in adjacency[node] for node in range(n))
            assert all(
                node in adjacency[neighbor]
                for node in range(n)
                for neighbor in adjacency[node]
            )


def test_odd_degree_uses_antipodal_perfect_matching():
    n, degree = 24, 5
    graph = FixedDegreeGraph(
        n, degree, device="cpu", seed=3, algorithm="circulant"
    )
    adjacency = _adjacency(graph)
    assert graph.circulant_offsets[-1] == n // 2
    assert all((node + n // 2) % n in adjacency[node] for node in range(n))


def test_circulant_seed_is_reproducible_and_changes_offsets():
    first = FixedDegreeGraph(
        101, 8, device="cpu", seed=0, algorithm="circulant"
    )
    repeat = FixedDegreeGraph(
        101, 8, device="cpu", seed=0, algorithm="circulant"
    )
    other = FixedDegreeGraph(
        101, 8, device="cpu", seed=1, algorithm="circulant"
    )

    assert first.circulant_offsets == repeat.circulant_offsets
    assert torch.equal(first.csr.col_ind, repeat.csr.col_ind)
    assert first.circulant_offsets != other.circulant_offsets
    assert not torch.equal(first.csr.col_ind, other.csr.col_ind)


def test_circulant_uses_direct_csr_and_does_not_require_networkx(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("COO/NetworkX construction must not run")

    monkeypatch.setattr(network, "_require_networkx", fail)
    monkeypatch.setattr(GraphCSR, "__init__", fail)
    graph = FixedDegreeGraph(
        40, 6, device="cpu", seed=7, algorithm="circulant"
    )
    assert graph.num_edges == 240
    assert graph.construction_algorithm == "seeded_simple_circulant_direct_csr"

    public_graph = regular_graph(
        40, 6, device="cpu", seed=7, algorithm="circulant"
    )
    assert torch.equal(graph.csr.col_ind, public_graph.csr.col_ind)


@pytest.mark.parametrize(
    ("n", "degree", "match"),
    [
        (9, 3, "must be even"),
        (8, 8, "degree must satisfy"),
        (0, 0, "positive"),
    ],
)
def test_circulant_rejects_impossible_simple_regular_shapes(n, degree, match):
    with pytest.raises(ValueError, match=match):
        FixedDegreeGraph(
            n, degree, device="cpu", seed=0, algorithm="circulant"
        )


def test_circulant_rejects_legacy_one_way_mode_and_int32_overflow():
    with pytest.raises(ValueError, match="requires symmetric"):
        FixedDegreeGraph(
            20,
            4,
            device="cpu",
            symmetric=False,
            algorithm="circulant",
        )
    with pytest.raises(OverflowError, match="int32 CSR edge limit"):
        FixedDegreeGraph(
            300_000_000, 8, device="cpu", seed=0, algorithm="circulant"
        )


def test_large_memory_plan_without_large_allocation_and_small_resident_check():
    plan = network._circulant_memory_plan(100_000_000, 8)
    assert plan["directed_edges"] == 800_000_000
    assert plan["resident_csr_bytes"] == 3_600_000_008
    assert plan["fill_temporary_bytes_bound"] == 37_748_800
    assert plan["validation_temporary_bytes_bound"] == 100_000_000
    assert plan["peak_live_tensor_bytes_bound"] == 3_700_000_008

    graph = FixedDegreeGraph(
        1_000, 8, device="cpu", seed=2, algorithm="circulant"
    )
    actual_resident = sum(
        tensor.untyped_storage().nbytes()
        for tensor in (
            graph.csr.row_ptr,
            graph.csr.col_ind,
            graph.csr.weights_storage,
        )
    )
    assert actual_resident == graph.construction_memory_plan["resident_csr_bytes"]


def test_default_algorithm_keeps_networkx_random_regular_semantics():
    graph = FixedDegreeGraph(30, 4, device="cpu", seed=5)
    assert graph.algorithm == "networkx"
    assert graph.construction_algorithm == "networkx_random_regular_via_coo"
