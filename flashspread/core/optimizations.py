"""
Optimization utilities for FlashSpread engines.

This module provides graph reordering and other optimizations
to improve cache locality and reduce memory traffic.
"""

from typing import Any, Tuple

import numpy as np
import torch


_CSR_DTYPES = (torch.int32, torch.int64)
_INT32_MAX = 2**31 - 1
_EDGE_CHUNK_SIZE = 1 << 20


def _csr_shape(row_ptr: torch.Tensor, col_ind: torch.Tensor) -> tuple[int, int]:
    """Validate structural CSR metadata without moving tensor data to the host."""
    if not isinstance(row_ptr, torch.Tensor) or not isinstance(col_ind, torch.Tensor):
        raise TypeError("row_ptr and col_ind must be tensors")
    if row_ptr.dim() != 1 or col_ind.dim() != 1:
        raise ValueError("row_ptr and col_ind must be one-dimensional")
    if row_ptr.numel() == 0:
        raise ValueError("row_ptr must contain at least the initial zero")
    if row_ptr.dtype not in _CSR_DTYPES:
        raise TypeError("row_ptr must use int32 or int64")
    if col_ind.dtype not in _CSR_DTYPES:
        raise TypeError("col_ind must use int32 or int64")
    if row_ptr.device != col_ind.device:
        raise ValueError("row_ptr and col_ind must be on the same device")
    num_nodes, num_edges = row_ptr.numel() - 1, col_ind.numel()
    if num_nodes > _INT32_MAX or num_edges > _INT32_MAX:
        raise OverflowError("CSR shape exceeds the package's int32 index limit")
    return num_nodes, num_edges


def _validate_csr(row_ptr: torch.Tensor, col_ind: torch.Tensor) -> tuple[int, int]:
    """Validate CSR contents with one device-to-host status read."""
    num_nodes, num_edges = _csr_shape(row_ptr, col_ind)
    valid = (
        (row_ptr[0] == 0)
        & (row_ptr[-1] == num_edges)
        & torch.all(row_ptr[1:] >= row_ptr[:-1])
    )
    for start in range(0, num_edges, _EDGE_CHUNK_SIZE):
        columns = col_ind[start : start + _EDGE_CHUNK_SIZE]
        valid = valid & torch.all((columns >= 0) & (columns < num_nodes))
    if not bool(valid):
        raise ValueError(
            "invalid CSR: row_ptr must start at 0, be non-decreasing, end at "
            "len(col_ind), and col_ind values must be valid node indices"
        )
    return num_nodes, num_edges


def reverse_cuthill_mckee(row_ptr: torch.Tensor, col_ind: torch.Tensor) -> torch.Tensor:
    """
    Compute Reverse Cuthill-McKee ordering for better cache locality.

    RCM reordering reduces the bandwidth of the adjacency matrix,
    which improves cache utilization during graph traversal.

    Args:
        row_ptr: CSR row pointer tensor [N+1]
        col_ind: CSR column indices tensor [E]

    Returns:
        Permutation tensor ``perm`` with SciPy's convention
        ``perm[new_index] = old_index``.
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee as scipy_rcm
    except ImportError:
        raise ImportError("scipy is required for RCM reordering")

    # Convert to scipy sparse matrix
    N, _ = _validate_csr(row_ptr, col_ind)
    row_ptr_np = row_ptr.cpu().numpy()
    col_ind_np = col_ind.cpu().numpy()

    # Create CSR matrix with unit weights
    # RCM reads only the sparsity structure. Boolean data uses one byte per
    # edge and remains structurally idempotent if SciPy symmetrizes a graph
    # containing duplicate edges (unlike a narrow integer that can overflow).
    data = np.ones(len(col_ind_np), dtype=np.bool_)
    adj = csr_matrix((data, col_ind_np, row_ptr_np), shape=(N, N))

    # Compute RCM ordering
    perm = scipy_rcm(adj)

    return torch.from_numpy(perm.copy()).to(row_ptr.device)


def apply_permutation_to_graph(
    row_ptr: torch.Tensor,
    col_ind: torch.Tensor,
    weights: torch.Tensor | None,
    perm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Apply a permutation to reorder graph nodes.

    Args:
        row_ptr: Original CSR row pointer [N+1]
        col_ind: Original CSR column indices [E]
        weights: Original edge weights [E], or None for symbolic unit weights.
        perm: SciPy-style permutation with ``perm[new_index] = old_index``.

    Returns:
        Tuple of (new_row_ptr, new_col_ind, new_weights). ``new_weights`` is
        None when ``weights`` is None.
    """
    N, E = _validate_csr(row_ptr, col_ind)
    device = row_ptr.device

    if not isinstance(perm, torch.Tensor):
        raise TypeError("perm must be a tensor")
    if perm.dim() != 1 or perm.numel() != N:
        raise ValueError(f"perm must have shape [{N}]")
    if perm.dtype not in _CSR_DTYPES:
        raise TypeError("perm must use int32 or int64")
    if perm.device != device:
        raise ValueError("perm must be on the same device as the CSR tensors")
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a tensor or None")
        if weights.dim() != 1 or weights.numel() != E:
            raise ValueError(f"weights must have shape [{E}]")
        if weights.device != device:
            raise ValueError("weights must be on the same device as the CSR tensors")

    perm_values = perm
    if N:
        counts = torch.bincount(perm_values.clamp(0, N - 1), minlength=N)
        valid_perm = (
            torch.all((perm_values >= 0) & (perm_values < N))
            & torch.all(counts == 1)
        )
        if not bool(valid_perm):
            raise ValueError("perm must contain each node index exactly once")

    perm_index = perm.to(torch.int32)
    inverse = torch.empty(N, device=device, dtype=torch.int32)
    inverse[perm_index] = torch.arange(N, device=device, dtype=torch.int32)

    degrees = row_ptr[1:] - row_ptr[:-1]
    new_degrees = degrees[perm_index]
    new_row_ptr = torch.empty(N + 1, device=device, dtype=row_ptr.dtype)
    new_row_ptr[0] = 0
    new_row_ptr[1:] = torch.cumsum(new_degrees, dim=0, dtype=row_ptr.dtype)

    # In each new row, old and new edge positions differ by one row-constant
    # offset. Repeating those offsets produces the old edge gather order
    # directly, without sorting or retaining several edge-sized index arrays.
    row_offsets = (row_ptr[perm_index] - new_row_ptr[:-1]).to(torch.int32)
    old_edge_positions = torch.repeat_interleave(
        row_offsets,
        new_degrees,
        output_size=E,
    )
    for start in range(0, E, _EDGE_CHUNK_SIZE):
        end = min(start + _EDGE_CHUNK_SIZE, E)
        old_edge_positions[start:end].add_(
            torch.arange(start, end, device=device, dtype=torch.int32)
        )

    old_columns = col_ind[old_edge_positions]
    new_weights = weights[old_edge_positions] if weights is not None else None
    del old_edge_positions
    remapped_columns = inverse[old_columns.to(torch.int32)]
    del old_columns
    new_col_ind = remapped_columns.to(col_ind.dtype)

    return new_row_ptr, new_col_ind, new_weights


def reorder_graph_rcm(graph_csr) -> Tuple[Any, torch.Tensor]:
    """
    Reorder a GraphCSR using Reverse Cuthill-McKee ordering.

    Args:
        graph_csr: GraphCSR object

    Returns:
        Tuple of (reordered_graph_csr, permutation), where the permutation
        follows SciPy's ``perm[new_index] = old_index`` convention.
    """
    from .graph import GraphCSR

    perm = reverse_cuthill_mckee(graph_csr.row_ptr, graph_csr.col_ind)

    had_weights = graph_csr.has_weights
    source_weights = graph_csr.weights_storage if had_weights else None
    new_row_ptr, new_col_ind, new_weights = apply_permutation_to_graph(
        graph_csr.row_ptr, graph_csr.col_ind, source_weights, perm
    )

    # Re-enter through the invariant-checking CSR constructor rather than
    # manually recreating a partially initialized GraphCSR object.
    reordered = GraphCSR.from_csr(
        new_row_ptr,
        new_col_ind,
        weights=new_weights if had_weights else None,
        incoming=getattr(graph_csr, "incoming", True),
    )

    return reordered, perm


def compute_graph_bandwidth(row_ptr: torch.Tensor, col_ind: torch.Tensor) -> int:
    """
    Compute the bandwidth of a graph's adjacency matrix.

    Bandwidth = max |i - j| for all edges (i, j)
    Lower bandwidth means better cache locality.

    Args:
        row_ptr: CSR row pointer [N+1]
        col_ind: CSR column indices [E]

    Returns:
        Integer bandwidth
    """
    N, E = _csr_shape(row_ptr, col_ind)
    valid = (
        (row_ptr[0] == 0)
        & (row_ptr[-1] == E)
        & torch.all(row_ptr[1:] >= row_ptr[:-1])
    )

    bandwidth = torch.zeros((), device=row_ptr.device, dtype=torch.int64)
    for start in range(0, E, _EDGE_CHUNK_SIZE):
        end = min(start + _EDGE_CHUNK_SIZE, E)
        columns = col_ind[start:end]
        valid = valid & torch.all((columns >= 0) & (columns < N))
        edge_positions = torch.arange(
            start, end, device=row_ptr.device, dtype=row_ptr.dtype
        )
        rows = torch.searchsorted(row_ptr[1:], edge_positions, right=True)
        rows.sub_(columns)
        rows.abs_()
        bandwidth = torch.maximum(bandwidth, rows.amax())

    # Encode validation and result in one scalar so this preprocessing helper
    # performs exactly one device-to-host value read.
    result = int(torch.where(valid, bandwidth, -torch.ones_like(bandwidth)).item())
    if result < 0:
        raise ValueError(
            "invalid CSR: row_ptr must start at 0, be non-decreasing, end at "
            "len(col_ind), and col_ind values must be valid node indices"
        )
    return result


class OptimizationConfig:
    """Configuration for engine optimizations."""

    def __init__(
        self,
        use_rcm_reordering: bool = False,
        use_fused_ops: bool = False,
        flash_neighbor_block_size: int = 128,
        use_persistent_rng: bool = True,
    ):
        """
        Initialize optimization configuration.

        Args:
            use_rcm_reordering: Apply RCM graph reordering for cache locality
            use_fused_ops: Use fused PyTorch operations where possible
            flash_neighbor_block_size: Block size for FlashNeighbor kernel
            use_persistent_rng: Use persistent RNG state (faster)
        """
        self.use_rcm_reordering = use_rcm_reordering
        self.use_fused_ops = use_fused_ops
        self.flash_neighbor_block_size = flash_neighbor_block_size
        self.use_persistent_rng = use_persistent_rng

    def to_dict(self) -> dict:
        return {
            "use_rcm_reordering": self.use_rcm_reordering,
            "use_fused_ops": self.use_fused_ops,
            "flash_neighbor_block_size": self.flash_neighbor_block_size,
            "use_persistent_rng": self.use_persistent_rng,
        }

    @classmethod
    def default(cls) -> "OptimizationConfig":
        """Return default (no optimizations) config."""
        return cls()

    @classmethod
    def optimized(cls) -> "OptimizationConfig":
        """Return fully optimized config."""
        return cls(
            use_rcm_reordering=True,
            use_fused_ops=True,
            flash_neighbor_block_size=256,
            use_persistent_rng=True,
        )
