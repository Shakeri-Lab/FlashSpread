"""
Optimization utilities for FlashSpread engines.

This module provides graph reordering and other optimizations
to improve cache locality and reduce memory traffic.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def reverse_cuthill_mckee(row_ptr: torch.Tensor, col_ind: torch.Tensor) -> torch.Tensor:
    """
    Compute Reverse Cuthill-McKee ordering for better cache locality.

    RCM reordering reduces the bandwidth of the adjacency matrix,
    which improves cache utilization during graph traversal.

    Args:
        row_ptr: CSR row pointer tensor [N+1]
        col_ind: CSR column indices tensor [E]

    Returns:
        Permutation tensor [N] mapping old indices to new indices
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee as scipy_rcm
    except ImportError:
        raise ImportError("scipy is required for RCM reordering")

    # Convert to scipy sparse matrix
    N = row_ptr.shape[0] - 1
    row_ptr_np = row_ptr.cpu().numpy()
    col_ind_np = col_ind.cpu().numpy()

    # Create CSR matrix with unit weights
    data = np.ones(len(col_ind_np), dtype=np.float32)
    adj = csr_matrix((data, col_ind_np, row_ptr_np), shape=(N, N))

    # Compute RCM ordering
    perm = scipy_rcm(adj)

    return torch.from_numpy(perm.copy()).to(row_ptr.device)


def apply_permutation_to_graph(
    row_ptr: torch.Tensor,
    col_ind: torch.Tensor,
    weights: torch.Tensor,
    perm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a permutation to reorder graph nodes.

    Args:
        row_ptr: Original CSR row pointer [N+1]
        col_ind: Original CSR column indices [E]
        weights: Original edge weights [E]
        perm: Permutation mapping old -> new indices [N]

    Returns:
        Tuple of (new_row_ptr, new_col_ind, new_weights)
    """
    N = row_ptr.shape[0] - 1
    device = row_ptr.device

    # Compute inverse permutation
    inv_perm = torch.zeros_like(perm)
    inv_perm[perm] = torch.arange(N, device=device, dtype=perm.dtype)

    # Build new CSR by iterating in new order
    new_row_ptr = torch.zeros(N + 1, device=device, dtype=row_ptr.dtype)

    # Count edges per new node
    for new_idx in range(N):
        old_idx = perm[new_idx].item()
        new_row_ptr[new_idx + 1] = row_ptr[old_idx + 1] - row_ptr[old_idx]

    # Cumulative sum for row pointers
    new_row_ptr = torch.cumsum(new_row_ptr, dim=0)

    # Build new col_ind and weights
    E = col_ind.shape[0]
    new_col_ind = torch.zeros(E, device=device, dtype=col_ind.dtype)
    new_weights = torch.zeros(E, device=device, dtype=weights.dtype)

    write_ptr = 0
    for new_idx in range(N):
        old_idx = perm[new_idx].item()
        start = row_ptr[old_idx].item()
        end = row_ptr[old_idx + 1].item()

        for j in range(start, end):
            old_neighbor = col_ind[j].item()
            new_neighbor = inv_perm[old_neighbor].item()
            new_col_ind[write_ptr] = new_neighbor
            new_weights[write_ptr] = weights[j]
            write_ptr += 1

    return new_row_ptr, new_col_ind, new_weights


def reorder_graph_rcm(graph_csr) -> Tuple[any, torch.Tensor]:
    """
    Reorder a GraphCSR using Reverse Cuthill-McKee ordering.

    Args:
        graph_csr: GraphCSR object

    Returns:
        Tuple of (reordered_graph_csr, permutation)
    """
    from .graph import GraphCSR

    perm = reverse_cuthill_mckee(graph_csr.row_ptr, graph_csr.col_ind)

    new_row_ptr, new_col_ind, new_weights = apply_permutation_to_graph(
        graph_csr.row_ptr, graph_csr.col_ind, graph_csr.weights, perm
    )

    # Create new GraphCSR (simplified - may need adjustment based on actual class)
    reordered = GraphCSR.__new__(GraphCSR)
    reordered.row_ptr = new_row_ptr
    reordered.col_ind = new_col_ind
    reordered.weights = new_weights
    reordered.num_nodes = graph_csr.num_nodes
    # num_edges is a read-only property on GraphCSR (derived from
    # col_ind.numel()); copying the backing arrays is sufficient.
    reordered.device = graph_csr.device

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
    N = row_ptr.shape[0] - 1
    max_diff = 0

    row_ptr_cpu = row_ptr.cpu()
    col_ind_cpu = col_ind.cpu()

    for i in range(N):
        start = row_ptr_cpu[i].item()
        end = row_ptr_cpu[i + 1].item()
        for j_idx in range(start, end):
            j = col_ind_cpu[j_idx].item()
            diff = abs(i - j)
            if diff > max_diff:
                max_diff = diff

    return max_diff


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
