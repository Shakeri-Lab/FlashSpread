"""
Convenience graph constructors for the public API.

Thin, keyword-friendly wrappers over the generator classes in
:mod:`flashspread.core.network`, adding automatic device resolution and a
``seed`` argument. They build symmetric (undirected) contact networks by
default; pass ``symmetric=False`` only to reproduce the legacy half-degree
behaviour.

The default generators require NetworkX (``pip install flashspread[graph]``).
The explicit ``regular_graph(..., algorithm="circulant")`` path is CSR-native
and has no NetworkX dependency.
"""

from __future__ import annotations

import torch

from .core.graph import GraphCSR
from .core.network import (
    BarabasiAlbertGraph,
    FixedDegreeGraph,
    RandomGeometricGraph,
    WattsStrogatzGraph,
)
from .utils import resolve_device


def from_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    weights: torch.Tensor | None = None,
    device=None,
) -> GraphCSR:
    """Build the canonical graph directly from ``[source, target]`` edges."""
    edge_index = torch.as_tensor(edge_index)
    target = resolve_device(device) if device is not None else edge_index.device
    edge_index = edge_index.to(target)
    if weights is not None:
        weights = torch.as_tensor(weights, device=target)
    return GraphCSR(edge_index, num_nodes, weights=weights, incoming=True)


def from_csr(
    row_ptr: torch.Tensor,
    col_ind: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    device=None,
) -> GraphCSR:
    """Build the canonical graph from incoming CSR without a COO conversion."""
    row_ptr = torch.as_tensor(row_ptr)
    col_ind = torch.as_tensor(col_ind)
    target = resolve_device(device) if device is not None else row_ptr.device
    row_ptr = row_ptr.to(target)
    col_ind = col_ind.to(target)
    if weights is not None:
        weights = torch.as_tensor(weights, device=target)
    return GraphCSR.from_csr(row_ptr, col_ind, weights=weights, incoming=True)


def regular_graph(
    n: int,
    degree: int,
    *,
    symmetric: bool = True,
    seed: int | None = None,
    device=None,
    algorithm: str = "networkx",
) -> FixedDegreeGraph:
    """Build a ``degree``-regular graph on ``n`` nodes.

    ``algorithm="networkx"`` preserves the existing random-regular semantics.
    ``algorithm="circulant"`` builds a seeded exact-simple undirected
    circulant directly in int32 CSR with bounded temporary storage. A
    circulant is useful for very large performance workloads but is not a
    uniform random-regular graph.
    """
    return FixedDegreeGraph(
        n,
        degree,
        device=resolve_device(device),
        symmetric=symmetric,
        seed=seed,
        algorithm=algorithm,
    )


def barabasi_albert(n: int, m: int, *, symmetric: bool = True,
                    seed: int | None = None, device=None) -> BarabasiAlbertGraph:
    """Barabasi-Albert scale-free graph (``m`` attachments per new node)."""
    return BarabasiAlbertGraph(
        n, m, device=resolve_device(device), symmetric=symmetric, seed=seed
    )


def watts_strogatz(n: int, k: int, p: float, *, symmetric: bool = True,
                   seed: int | None = None, device=None) -> WattsStrogatzGraph:
    """Watts-Strogatz small-world graph (ring degree ``k``, rewiring prob ``p``)."""
    return WattsStrogatzGraph(
        n, k, p, device=resolve_device(device), symmetric=symmetric, seed=seed
    )


def geometric(n: int, radius: float, *, symmetric: bool = True,
              seed: int | None = None, device=None) -> RandomGeometricGraph:
    """Random geometric graph (nodes in a unit square, edges within ``radius``)."""
    return RandomGeometricGraph(
        n, radius, device=resolve_device(device), symmetric=symmetric, seed=seed
    )
