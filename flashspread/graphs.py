"""
Convenience graph constructors for the public API.

Thin, keyword-friendly wrappers over the generator classes in
:mod:`flashspread.core.network`, adding automatic device resolution and a
``seed`` argument. They build symmetric (undirected) contact networks by
default; pass ``symmetric=False`` to reproduce the pre-v1.1 half-degree
behaviour.

These require NetworkX (``pip install flashspread[graph]``); a clear error is
raised at call time if it is missing.
"""

from __future__ import annotations

from .core.network import (
    BarabasiAlbertGraph,
    FixedDegreeGraph,
    RandomGeometricGraph,
    WattsStrogatzGraph,
)
from .utils import resolve_device


def regular_graph(n: int, degree: int, *, symmetric: bool = True,
                  seed: int | None = None, device=None) -> FixedDegreeGraph:
    """Random ``degree``-regular graph on ``n`` nodes (uniform contact structure)."""
    return FixedDegreeGraph(
        n, degree, device=resolve_device(device), symmetric=symmetric, seed=seed
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
