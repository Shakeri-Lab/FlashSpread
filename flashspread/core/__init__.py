"""
Core components for FlashSpread: graph structures, kernels, and network utilities.
"""

from .graph import GraphCSR
from .flash_neighbor import FlashNeighbor
from .network import (
    FixedDegreeGraph,
    RandomGeometricGraph,
    BarabasiAlbertGraph,
    WattsStrogatzGraph,
    load_edges,
    save_edges_txt,
)

__all__ = [
    "GraphCSR",
    "FlashNeighbor",
    "FixedDegreeGraph",
    "RandomGeometricGraph",
    "BarabasiAlbertGraph",
    "WattsStrogatzGraph",
    "load_edges",
    "save_edges_txt",
]
