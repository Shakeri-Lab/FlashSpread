"""
Network generation and I/O utilities for FlashSpread.

This module provides convenience functions for creating common network
topologies and loading/saving edge lists.
"""

import math
import operator
import random
import warnings
from typing import Tuple

import torch

from .graph import GraphCSR


# Keep the largest broadcast temporary below roughly 32 MiB (int64).  The
# row vector used alongside it raises the analytical live-temporary bound to
# at most 64 MiB, attained by a degree-one graph.  This is intentionally an
# edge budget rather than a node budget so high-degree construction stays
# bounded too.
_CIRCULANT_CHUNK_EDGE_LIMIT = 4 * 1024 * 1024


def _strict_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _circulant_memory_plan(
    num_nodes: int,
    degree: int,
    *,
    chunk_edge_limit: int = _CIRCULANT_CHUNK_EDGE_LIMIT,
) -> dict[str, int]:
    """Return an allocation model for the direct circulant CSR builder.

    ``peak_live_tensor_bytes_bound`` models live tensor storage, not caching
    allocator reservations.  It includes the final int32 CSR and symbolic
    unit weight, then the larger of the chunk-fill temporary and
    :meth:`GraphCSR.from_csr`'s one-byte-per-row monotonicity check.
    """
    n = operator.index(num_nodes)
    d = operator.index(degree)
    if n < 0 or d < 0 or chunk_edge_limit <= 0:
        raise ValueError("memory-plan dimensions and chunk limit must be non-negative")
    edge_count = n * d
    chunk_nodes = min(n, max(1, chunk_edge_limit // d)) if d else 0
    chunk_edges = chunk_nodes * d
    # int64 rows [C, 1], int64 broadcast block [C, d], and offsets [d].
    fill_temporary = 8 * (chunk_nodes + chunk_edges + d) if d else 0
    validation_temporary = n  # bool result of row_ptr[1:] < row_ptr[:-1]
    peak_temporary = max(fill_temporary, validation_temporary)
    resident = 4 * (n + 1) + 4 * edge_count + 4
    return {
        "num_nodes": n,
        "degree": d,
        "directed_edges": edge_count,
        "chunk_edge_limit": chunk_edge_limit,
        "max_chunk_nodes": chunk_nodes,
        "max_chunk_edges": chunk_edges,
        "resident_csr_bytes": resident,
        "fill_temporary_bytes_bound": fill_temporary,
        "validation_temporary_bytes_bound": validation_temporary,
        "peak_temporary_bytes_bound": peak_temporary,
        "peak_live_tensor_bytes_bound": resident + peak_temporary,
    }


def _circulant_offsets(
    num_nodes: int,
    degree: int,
    seed: int | None,
) -> tuple[int, ...]:
    """Choose signed offsets for a seeded simple undirected circulant graph."""
    pair_count = degree // 2
    # Exclude zero and, for even N, N/2: each selected offset contributes the
    # distinct pair (+offset, -offset).  An odd degree gets N/2 separately.
    candidates = range(1, (num_nodes + 1) // 2)
    if seed is not None:
        try:
            seed = operator.index(seed)
        except TypeError as exc:
            raise TypeError("seed must be an integer or None") from exc
        if isinstance(seed, bool):
            raise TypeError("seed must be an integer or None")
    positive = random.Random(seed).sample(candidates, pair_count)
    signed = tuple(value for offset in positive for value in (offset, -offset))
    if degree % 2:
        signed += (num_nodes // 2,)
    return signed


def _circulant_regular_csr(
    num_nodes: int,
    degree: int,
    *,
    device: torch.device,
    seed: int | None,
) -> tuple[GraphCSR, tuple[int, ...], dict[str, int]]:
    """Construct exact-simple symmetric regular CSR without COO intermediates."""
    try:
        n = operator.index(num_nodes)
        d = operator.index(degree)
    except TypeError as exc:
        raise TypeError("num_nodes and degree must be integers") from exc
    if isinstance(num_nodes, bool) or isinstance(degree, bool):
        raise TypeError("num_nodes and degree must be integers")
    if n <= 0:
        raise ValueError("num_nodes must be positive")
    if d < 0 or d >= n:
        raise ValueError("degree must satisfy 0 <= degree < num_nodes")
    if (n * d) % 2:
        raise ValueError(
            "num_nodes * degree must be even; odd degree requires even num_nodes"
        )
    edge_count = n * d
    if edge_count > torch.iinfo(torch.int32).max:
        raise OverflowError(
            "num_nodes * degree exceeds FlashSpread's int32 CSR edge limit"
        )

    offsets = _circulant_offsets(n, d, seed)
    plan = _circulant_memory_plan(n, d)
    row_ptr = torch.empty(n + 1, dtype=torch.int32, device=device)
    if d:
        torch.arange(0, edge_count + 1, d, out=row_ptr)
    else:
        row_ptr.zero_()
    col_ind = torch.empty(edge_count, dtype=torch.int32, device=device)

    if d:
        offset_tensor = torch.tensor(offsets, dtype=torch.int64, device=device)
        rows_per_chunk = plan["max_chunk_nodes"]
        columns = col_ind.view(n, d)
        for start in range(0, n, rows_per_chunk):
            stop = min(start + rows_per_chunk, n)
            rows = torch.arange(
                start, stop, dtype=torch.int64, device=device
            ).unsqueeze(1)
            neighbors = rows + offset_tensor.unsqueeze(0)
            neighbors.remainder_(n)
            # copy_ performs the checked-range int64 -> int32 conversion into
            # final storage; there is no second edge-sized output tensor.
            columns[start:stop].copy_(neighbors)
            # Release the old broadcast block before evaluating the next
            # iteration's RHS; otherwise two chunks can briefly overlap.
            del rows, neighbors
        del columns, offset_tensor

    return (
        GraphCSR.from_csr(row_ptr, col_ind, incoming=True),
        offsets,
        plan,
    )


def _require_networkx():
    """Import NetworkX lazily with an actionable error.

    NetworkX is only needed by the graph *generators*, not to import the
    package or run the engines, so it is an optional dependency (the
    ``graph`` extra). Importing it here keeps ``import flashspread`` working
    on a bare (CPU / no-GPU) install.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - trivial guard
        raise ImportError(
            "The graph generators require NetworkX. Install it with "
            "`pip install flashspread[graph]` (or `pip install networkx`)."
        ) from exc
    return nx


def _edge_index_from_networkx(graph, directed: bool = True) -> torch.Tensor:
    """Convert NetworkX graph to edge_index tensor."""
    edges = list(graph.edges())
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if not directed and not graph.is_directed():
        # Duplicate the compact tensor, not millions of Python tuples.
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    return edge_index


class _GeneratedGraph:
    """Shared compact storage for the legacy named generator classes."""

    def _initialize_graph(self, nx_graph) -> None:
        edge_index = _edge_index_from_networkx(
            nx_graph, directed=not self.symmetric
        ).to(self.device)
        self._csr = GraphCSR(edge_index, self.num_nodes, incoming=True)
        # Do not retain a second 64-bit COO graph. Reconstruct/cache it only if
        # a compatibility caller explicitly requests ``edge_index``.
        self._edge_index_cache = None

    @property
    def edge_index(self) -> torch.Tensor:
        if self._edge_index_cache is None:
            self._edge_index_cache = self._csr.to_edge_index()
        return self._edge_index_cache

    @property
    def csr(self) -> GraphCSR:
        return self._csr

    @property
    def num_edges(self) -> int:
        return self._csr.num_edges


class FixedDegreeGraph(_GeneratedGraph):
    """
    Degree-regular graph with either NetworkX or direct-CSR construction.

    This topology is commonly used in epidemic simulation benchmarks
    as it provides uniform contact structure.
    """

    def __init__(
        self,
        num_nodes: int,
        degree: int,
        device: str | torch.device = "cuda",
        symmetric: bool = True,
        seed: int | None = None,
        algorithm: str = "networkx",
    ):
        """
        Create a random regular graph.

        Args:
            num_nodes: Number of nodes.
            degree: Degree of each node. It must be smaller than ``num_nodes``
                and ``num_nodes * degree`` must be even.
            device: Device for tensors.
            symmetric: If True (default), store both directions of every
                undirected edge so each node's in-degree equals ``degree``
                (a proper undirected contact network). If False, keep the
                legacy behaviour where NetworkX enumerates each edge once,
                giving asymmetric ~degree/2 in-degree — retained only for
                historical result reproduction.
            seed: Generator seed (reproducibility).
            algorithm: ``"networkx"`` (default) retains the uniform-ish
                random-regular NetworkX topology. ``"circulant"`` builds a
                seeded deterministic circulant directly in CSR. The latter
                is exact-simple and degree-regular but is not a uniform sample
                from all regular graphs; it is intended for memory-scalable
                performance workloads.
        """
        self.num_nodes = num_nodes
        self.degree = degree
        self.device = torch.device(device)
        self.symmetric = _strict_bool("symmetric", symmetric)
        if algorithm not in {"networkx", "circulant"}:
            raise ValueError("algorithm must be 'networkx' or 'circulant'")
        self.algorithm = algorithm

        if algorithm == "circulant":
            if not self.symmetric:
                raise ValueError(
                    "algorithm='circulant' requires symmetric=True; use "
                    "algorithm='networkx' for the legacy one-way edge listing"
                )
            self._csr, self.circulant_offsets, self.construction_memory_plan = (
                _circulant_regular_csr(
                    num_nodes,
                    degree,
                    device=self.device,
                    seed=seed,
                )
            )
            self._edge_index_cache = None
            self.construction_algorithm = "seeded_simple_circulant_direct_csr"
            self.circulant_component_count = math.gcd(
                self.num_nodes, *self.circulant_offsets
            )
        else:
            nx = _require_networkx()
            self._initialize_graph(
                nx.random_regular_graph(degree, num_nodes, seed=seed)
            )
            self.construction_algorithm = "networkx_random_regular_via_coo"


class RandomGeometricGraph(_GeneratedGraph):
    """
    Random geometric graph with spatial locality.

    Nodes are placed uniformly in a unit square, and edges connect
    nodes within a given radius. This creates high clustering.
    """

    def __init__(
        self,
        num_nodes: int,
        radius: float,
        device: str | torch.device = "cuda",
        symmetric: bool = True,
        seed: int | None = None,
    ):
        """
        Create a random geometric graph.

        Args:
            num_nodes: Number of nodes.
            radius: Connection radius.
            device: Device for tensors.
            symmetric: If True (default), store both directions of every
                undirected edge so adjacency is symmetric. See
                :class:`FixedDegreeGraph` for the rationale.
            seed: Random seed for the NetworkX generator (reproducibility).
        """
        self.num_nodes = num_nodes
        self.radius = radius
        self.device = torch.device(device)
        self.symmetric = _strict_bool("symmetric", symmetric)

        nx = _require_networkx()
        self._initialize_graph(nx.random_geometric_graph(num_nodes, radius, seed=seed))


class BarabasiAlbertGraph(_GeneratedGraph):
    """
    Scale-free network using the Barabasi-Albert preferential attachment model.

    This creates networks with power-law degree distributions, which are
    common in real-world social and biological networks.
    """

    def __init__(
        self,
        num_nodes: int,
        num_attachments: int,
        device: str | torch.device = "cuda",
        symmetric: bool = True,
        seed: int | None = None,
    ):
        """
        Create a Barabasi-Albert graph.

        Args:
            num_nodes: Number of nodes.
            num_attachments: Number of edges to attach from each new node.
            device: Device for tensors.
            symmetric: If True (default), store both directions of every
                undirected edge so adjacency is symmetric. See
                :class:`FixedDegreeGraph` for the rationale.
            seed: Random seed for the NetworkX generator (reproducibility).
        """
        self.num_nodes = num_nodes
        self.num_attachments = num_attachments
        self.device = torch.device(device)
        self.symmetric = _strict_bool("symmetric", symmetric)

        nx = _require_networkx()
        self._initialize_graph(
            nx.barabasi_albert_graph(num_nodes, num_attachments, seed=seed)
        )


class WattsStrogatzGraph(_GeneratedGraph):
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
        symmetric: bool = True,
        seed: int | None = None,
    ):
        """
        Create a Watts-Strogatz small-world graph.

        Args:
            num_nodes: Number of nodes.
            k: Each node connected to k nearest neighbors in ring topology.
            p: Probability of rewiring each edge.
            device: Device for tensors.
            symmetric: If True (default), store both directions of every
                undirected edge so adjacency is symmetric. See
                :class:`FixedDegreeGraph` for the rationale.
            seed: Random seed for the NetworkX generator (reproducibility).
        """
        self.num_nodes = num_nodes
        self.k = k
        self.p = p
        self.device = torch.device(device)
        self.symmetric = _strict_bool("symmetric", symmetric)

        nx = _require_networkx()
        self._initialize_graph(nx.watts_strogatz_graph(num_nodes, k, p, seed=seed))


def load_edges(
    filepath: str,
    num_nodes: int | None = None,
    base: int = 0,
    device: str | torch.device = "cpu",
    return_weights: bool = False,
) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Load edge list from text file.

    File format: one edge per line as "source target" or "source target weight".

    Args:
        filepath: Path to edge file.
        num_nodes: Number of nodes (inferred from edges if None).
        base: Index base (0 or 1) of the file.
        device: Device for returned tensor.
        return_weights: If true, also return a float32 ``[E]`` weight tensor.

    Returns:
        ``(edge_index, num_nodes)`` or ``(edge_index, num_nodes, weights)``.
    """
    if not isinstance(return_weights, bool):
        raise TypeError("return_weights must be a bool")
    edges = []
    weights = []
    saw_weight = False
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src = int(parts[0]) - base
                dst = int(parts[1]) - base
                edges.append((src, dst))
                if len(parts) >= 3:
                    weights.append(float(parts[2]))
                    saw_weight = True
                else:
                    weights.append(1.0)

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        if return_weights:
            return edge_index, num_nodes or 0, torch.empty(0, device=device)
        return edge_index, num_nodes or 0

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    if return_weights:
        return edge_index, num_nodes, weight_tensor
    if saw_weight:
        warnings.warn(
            "weighted edge columns were present but return_weights=False; "
            "pass return_weights=True or use load_graph()",
            UserWarning,
            stacklevel=2,
        )
    return edge_index, num_nodes


def load_graph(
    filepath: str,
    num_nodes: int | None = None,
    base: int = 0,
    device: str | torch.device = "cpu",
) -> GraphCSR:
    """Load an edge-list file directly into canonical incoming CSR."""
    edge_index, resolved_nodes, weights = load_edges(
        filepath,
        num_nodes=num_nodes,
        base=base,
        device=device,
        return_weights=True,
    )
    return GraphCSR(edge_index, resolved_nodes, weights=weights, incoming=True)


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
