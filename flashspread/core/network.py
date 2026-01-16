"""
Network generation and I/O utilities for FlashSpread.

This module provides convenience functions for creating common network
topologies and loading/saving edge lists.
"""

import torch
import networkx as nx
import numpy as np
from typing import Tuple

from .graph import GraphCSR


def _edge_index_from_networkx(graph: nx.Graph, directed: bool = True) -> torch.Tensor:
    """Convert NetworkX graph to edge_index tensor."""
    edges = list(graph.edges())
    if not directed and not graph.is_directed():
        # Add reverse edges for undirected graphs
        edges = edges + [(v, u) for (u, v) in edges]

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class FixedDegreeGraph:
    """
    Random regular graph with fixed degree for each node.

    This topology is commonly used in epidemic simulation benchmarks
    as it provides uniform contact structure.
    """

    def __init__(self, num_nodes: int, degree: int, device: str | torch.device = "cuda"):
        """
        Create a random regular graph.

        Args:
            num_nodes: Number of nodes.
            degree: Degree of each node (must be even and < num_nodes).
            device: Device for tensors.
        """
        self.num_nodes = num_nodes
        self.degree = degree
        self.device = torch.device(device)

        # Create random regular graph
        self._nx_graph = nx.random_regular_graph(degree, num_nodes)
        self._edge_index = _edge_index_from_networkx(self._nx_graph).to(self.device)

        # Build CSR for efficient kernel access
        self._csr = GraphCSR(self._edge_index, num_nodes, incoming=True)

    @property
    def edge_index(self) -> torch.Tensor:
        """Return edge_index tensor [2, E]."""
        return self._edge_index

    @property
    def csr(self) -> GraphCSR:
        """Return incoming CSR representation."""
        return self._csr

    @property
    def num_edges(self) -> int:
        """Return number of directed edges."""
        return self._edge_index.size(1)


class RandomGeometricGraph:
    """
    Random geometric graph with spatial locality.

    Nodes are placed uniformly in a unit square, and edges connect
    nodes within a given radius. This creates high clustering.
    """

    def __init__(
        self, num_nodes: int, radius: float, device: str | torch.device = "cuda"
    ):
        """
        Create a random geometric graph.

        Args:
            num_nodes: Number of nodes.
            radius: Connection radius.
            device: Device for tensors.
        """
        self.num_nodes = num_nodes
        self.radius = radius
        self.device = torch.device(device)

        self._nx_graph = nx.random_geometric_graph(num_nodes, radius)
        self._edge_index = _edge_index_from_networkx(self._nx_graph).to(self.device)
        self._csr = GraphCSR(self._edge_index, num_nodes, incoming=True)

    @property
    def edge_index(self) -> torch.Tensor:
        return self._edge_index

    @property
    def csr(self) -> GraphCSR:
        return self._csr

    @property
    def num_edges(self) -> int:
        return self._edge_index.size(1)


class BarabasiAlbertGraph:
    """
    Scale-free network using the Barabasi-Albert preferential attachment model.

    This creates networks with power-law degree distributions, which are
    common in real-world social and biological networks.
    """

    def __init__(
        self, num_nodes: int, num_attachments: int, device: str | torch.device = "cuda"
    ):
        """
        Create a Barabasi-Albert graph.

        Args:
            num_nodes: Number of nodes.
            num_attachments: Number of edges to attach from each new node.
            device: Device for tensors.
        """
        self.num_nodes = num_nodes
        self.num_attachments = num_attachments
        self.device = torch.device(device)

        self._nx_graph = nx.barabasi_albert_graph(num_nodes, num_attachments)
        self._edge_index = _edge_index_from_networkx(self._nx_graph).to(self.device)
        self._csr = GraphCSR(self._edge_index, num_nodes, incoming=True)

    @property
    def edge_index(self) -> torch.Tensor:
        return self._edge_index

    @property
    def csr(self) -> GraphCSR:
        return self._csr

    @property
    def num_edges(self) -> int:
        return self._edge_index.size(1)


class WattsStrogatzGraph:
    """
    Small-world network using the Watts-Strogatz model.

    Combines high clustering with short average path length.
    """

    def __init__(
        self,
        num_nodes: int,
        k: int,
        p: float,
        device: str | torch.device = "cuda",
    ):
        """
        Create a Watts-Strogatz small-world graph.

        Args:
            num_nodes: Number of nodes.
            k: Each node connected to k nearest neighbors in ring topology.
            p: Probability of rewiring each edge.
            device: Device for tensors.
        """
        self.num_nodes = num_nodes
        self.k = k
        self.p = p
        self.device = torch.device(device)

        self._nx_graph = nx.watts_strogatz_graph(num_nodes, k, p)
        self._edge_index = _edge_index_from_networkx(self._nx_graph).to(self.device)
        self._csr = GraphCSR(self._edge_index, num_nodes, incoming=True)

    @property
    def edge_index(self) -> torch.Tensor:
        return self._edge_index

    @property
    def csr(self) -> GraphCSR:
        return self._csr

    @property
    def num_edges(self) -> int:
        return self._edge_index.size(1)


def load_edges(
    filepath: str,
    num_nodes: int | None = None,
    base: int = 0,
    device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, int]:
    """
    Load edge list from text file.

    File format: one edge per line as "source target" or "source target weight".

    Args:
        filepath: Path to edge file.
        num_nodes: Number of nodes (inferred from edges if None).
        base: Index base (0 or 1) of the file.
        device: Device for returned tensor.

    Returns:
        Tuple of (edge_index [2, E], num_nodes).
    """
    edges = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src = int(parts[0]) - base
                dst = int(parts[1]) - base
                edges.append((src, dst))

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        return edge_index, num_nodes or 0

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    return edge_index, num_nodes


def save_edges_txt(
    filepath: str,
    edge_index: torch.Tensor,
    base: int = 0,
) -> None:
    """
    Save edge list to text file.

    Args:
        filepath: Output path.
        edge_index: [2, E] tensor of edges.
        base: Index base for output (0 or 1).
    """
    edge_index = edge_index.cpu()
    with open(filepath, "w") as f:
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item() + base
            dst = edge_index[1, i].item() + base
            f.write(f"{src} {dst}\n")


def create_graph_from_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    weights: torch.Tensor | None = None,
    device: str | torch.device = "cuda",
) -> GraphCSR:
    """
    Create GraphCSR from edge_index tensor.

    Args:
        edge_index: [2, E] tensor of edges.
        num_nodes: Number of nodes.
        weights: Optional edge weights.
        device: Target device.

    Returns:
        GraphCSR object.
    """
    edge_index = edge_index.to(device)
    if weights is not None:
        weights = weights.to(device)
    return GraphCSR(edge_index, num_nodes, weights=weights, incoming=True)
