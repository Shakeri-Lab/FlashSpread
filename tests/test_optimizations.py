"""Exact regressions for device-native CSR reordering utilities."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from flashspread.core.graph import GraphCSR
from flashspread.core.optimizations import (
    apply_permutation_to_graph,
    compute_graph_bandwidth,
    reorder_graph_rcm,
    reverse_cuthill_mckee,
)


def _example(device: str = "cpu"):
    row_ptr = torch.tensor([0, 2, 3, 6, 6], dtype=torch.int32, device=device)
    col_ind = torch.tensor([1, 3, 0, 1, 2, 3], dtype=torch.int32, device=device)
    weights = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float64, device=device
    )
    # SciPy convention: new row 0 comes from old row 2, and so on.
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.int64, device=device)
    return row_ptr, col_ind, weights, perm


@pytest.mark.parametrize("weighted", [False, True])
def test_apply_permutation_uses_scipy_new_to_old_contract(weighted):
    row_ptr, col_ind, weights, perm = _example()
    supplied_weights = weights if weighted else None

    new_row_ptr, new_col_ind, new_weights = apply_permutation_to_graph(
        row_ptr, col_ind, supplied_weights, perm
    )

    assert torch.equal(new_row_ptr, torch.tensor([0, 3, 5, 5, 6], dtype=torch.int32))
    assert torch.equal(new_col_ind, torch.tensor([3, 0, 2, 3, 2, 1], dtype=torch.int32))
    assert new_row_ptr.dtype == row_ptr.dtype
    assert new_col_ind.dtype == col_ind.dtype
    assert new_row_ptr.device == row_ptr.device
    assert new_col_ind.device == col_ind.device
    if weighted:
        assert new_weights is not None
        assert new_weights.dtype == weights.dtype
        assert torch.equal(new_weights, weights[[3, 4, 5, 0, 1, 2]])
    else:
        assert new_weights is None


def test_empty_csr_permutation_and_bandwidth():
    row_ptr = torch.tensor([0], dtype=torch.int32)
    col_ind = torch.empty(0, dtype=torch.int32)
    perm = torch.empty(0, dtype=torch.int64)

    new_row_ptr, new_col_ind, new_weights = apply_permutation_to_graph(
        row_ptr, col_ind, None, perm
    )

    assert torch.equal(new_row_ptr, row_ptr)
    assert torch.equal(new_col_ind, col_ind)
    assert new_weights is None
    assert compute_graph_bandwidth(row_ptr, col_ind) == 0


def test_bandwidth_handles_empty_rows_without_materializing_on_the_host():
    row_ptr = torch.tensor([0, 0, 2, 2, 3], dtype=torch.int64)
    col_ind = torch.tensor([0, 3, 1], dtype=torch.int64)

    assert compute_graph_bandwidth(row_ptr, col_ind) == 2


def test_rcm_uses_boolean_structure_data(monkeypatch):
    """Exercise the optional boundary without relying on the host SciPy ABI."""
    observed = {}
    sparse = ModuleType("scipy.sparse")
    csgraph = ModuleType("scipy.sparse.csgraph")

    def fake_csr_matrix(parts, *, shape):
        data, columns, pointers = parts
        observed.update(data=data, columns=columns, pointers=pointers, shape=shape)
        return object()

    def fake_rcm(_adjacency):
        return np.array([1, 0], dtype=np.int32)

    sparse.csr_matrix = fake_csr_matrix
    csgraph.reverse_cuthill_mckee = fake_rcm
    scipy = ModuleType("scipy")
    scipy.sparse = sparse
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.sparse", sparse)
    monkeypatch.setitem(sys.modules, "scipy.sparse.csgraph", csgraph)

    permutation = reverse_cuthill_mckee(
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
    )

    assert observed["data"].dtype == np.bool_
    assert observed["shape"] == (2, 2)
    assert torch.equal(permutation, torch.tensor([1, 0], dtype=torch.int32))


def test_int64_inputs_are_exact_across_small_processing_chunks(monkeypatch):
    from flashspread.core import optimizations

    monkeypatch.setattr(optimizations, "_EDGE_CHUNK_SIZE", 2)
    row_ptr, col_ind, weights, perm = _example()
    row_ptr = row_ptr.to(torch.int64)
    col_ind = col_ind.to(torch.int64)

    new_row_ptr, new_col_ind, new_weights = apply_permutation_to_graph(
        row_ptr, col_ind, weights, perm
    )

    assert torch.equal(new_row_ptr, torch.tensor([0, 3, 5, 5, 6]))
    assert torch.equal(new_col_ind, torch.tensor([3, 0, 2, 3, 2, 1]))
    assert torch.equal(new_weights, weights[[3, 4, 5, 0, 1, 2]])
    assert compute_graph_bandwidth(row_ptr, col_ind) == 3


@pytest.mark.parametrize(
    ("row_ptr", "col_ind"),
    [
        ([1], []),
        ([0, 1], []),
        ([0, 2, 1], [0]),
        ([0, 1], [1]),
    ],
)
def test_invalid_csr_contents_are_rejected(row_ptr, col_ind):
    row_ptr = torch.tensor(row_ptr, dtype=torch.int32)
    col_ind = torch.tensor(col_ind, dtype=torch.int32)
    perm = torch.arange(row_ptr.numel() - 1, dtype=torch.int64)

    with pytest.raises(ValueError, match="invalid CSR"):
        apply_permutation_to_graph(row_ptr, col_ind, None, perm)
    with pytest.raises(ValueError, match="invalid CSR"):
        compute_graph_bandwidth(row_ptr, col_ind)


@pytest.mark.parametrize(
    "perm",
    [
        torch.tensor([0, 1, 1, 3]),
        torch.tensor([0, 1, 2, 4]),
        torch.tensor([-1, 1, 2, 3]),
    ],
)
def test_permutation_must_be_a_bijection(perm):
    row_ptr, col_ind, _, _ = _example()

    with pytest.raises(ValueError, match="exactly once"):
        apply_permutation_to_graph(row_ptr, col_ind, None, perm)


def test_permutation_and_weight_metadata_are_validated():
    row_ptr, col_ind, _, _ = _example()

    with pytest.raises(ValueError, match="shape"):
        apply_permutation_to_graph(row_ptr, col_ind, None, torch.arange(3))
    with pytest.raises(TypeError, match="int32 or int64"):
        apply_permutation_to_graph(row_ptr, col_ind, None, torch.arange(4.0))
    with pytest.raises(ValueError, match="weights must have shape"):
        apply_permutation_to_graph(row_ptr, col_ind, torch.ones(5), torch.arange(4))


@pytest.mark.parametrize("weighted", [False, True])
def test_reorder_graph_rcm_preserves_symbolic_or_explicit_weights(monkeypatch, weighted):
    row_ptr, col_ind, weights, fixed_perm = _example()
    graph = GraphCSR.from_csr(
        row_ptr,
        col_ind,
        weights=weights.to(torch.float32) * 10 if weighted else None,
    )

    from flashspread.core import optimizations

    original_apply = optimizations.apply_permutation_to_graph
    observed_weights = []

    def recording_apply(row_ptr, col_ind, supplied_weights, perm):
        observed_weights.append(supplied_weights)
        return original_apply(row_ptr, col_ind, supplied_weights, perm)

    monkeypatch.setattr(
        optimizations,
        "reverse_cuthill_mckee",
        lambda row_ptr, col_ind: fixed_perm,
    )
    monkeypatch.setattr(optimizations, "apply_permutation_to_graph", recording_apply)
    reordered, perm = reorder_graph_rcm(graph)

    expected = original_apply(
        graph.row_ptr,
        graph.col_ind,
        graph.weights_storage if weighted else None,
        perm,
    )
    assert torch.equal(reordered.row_ptr, expected[0])
    assert torch.equal(reordered.col_ind, expected[1])
    assert len(observed_weights) == 1
    if weighted:
        assert observed_weights[0] is graph.weights_storage
        assert reordered.has_weights
        assert torch.equal(reordered.weights_storage, expected[2])
    else:
        assert observed_weights == [None]
        assert not reordered.has_weights
        assert reordered.weights_storage.numel() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_permutation_and_bandwidth_stay_on_cuda():
    row_ptr, col_ind, weights, perm = _example("cuda")

    reordered = apply_permutation_to_graph(row_ptr, col_ind, weights, perm)

    assert all(tensor is not None and tensor.is_cuda for tensor in reordered)
    assert torch.equal(
        reordered[0].cpu(), torch.tensor([0, 3, 5, 5, 6], dtype=torch.int32)
    )
    assert compute_graph_bandwidth(row_ptr, col_ind) == 3
