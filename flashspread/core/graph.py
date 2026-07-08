"""
Graph data structures optimized for GPU-accelerated epidemic simulation.

The GraphCSR class provides compressed sparse row (CSR) format storage
for efficient gather-based parallelism in the FlashNeighbor kernel.
"""

import torch


class GraphCSR:
    """
    Immutable graph structure in CSR format optimized for GPU gather kernels.

    The CSR format enables efficient traversal of neighbors for each node,
    which is the primary operation in epidemic simulation (computing influence
    from infectious neighbors).

    Attributes:
        num_nodes: Total number of nodes in the graph.
        device: PyTorch device where tensors are stored.
        row_ptr: (N+1,) tensor of row pointers into col_ind.
        col_ind: (E,) tensor of neighbor indices.
        weights: (E,) tensor of edge weights.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        weights: torch.Tensor | None = None,
        incoming: bool = True,
    ):
        """
        Construct CSR representation from edge list.

        Args:
            edge_index: [2, E] tensor where edge_index[0] are sources and
                       edge_index[1] are targets.
            num_nodes: Total number of nodes in the graph.
            weights: Optional [E] tensor of edge weights. Defaults to 1.0.
            incoming: If True, build CSR indexed by target nodes (for gather).
                     If False, build CSR indexed by source nodes (for scatter).
        """
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E]")

        self.device = edge_index.device
        self.num_nodes = int(num_nodes)

        # Bounds-check node ids. Without this an out-of-range *source*
        # silently corrupts col_ind (later an OOB gather in the kernel ->
        # illegal memory access) and an out-of-range *target* makes the
        # bincount/cumsum below raise a cryptic shape error. One-time
        # construction cost; not on any hot path.
        if edge_index.numel() > 0:
            lo = int(edge_index.min())
            hi = int(edge_index.max())
            if lo < 0 or hi >= self.num_nodes:
                raise ValueError(
                    f"edge_index node ids must be in [0, {self.num_nodes}); "
                    f"got range [{lo}, {hi}]. Check num_nodes or the index base."
                )

        # For incoming=True: we want to iterate over incoming neighbors
        # So we sort by target (edge_index[1]) and store sources
        if incoming:
            src = edge_index[1].to(torch.int64)  # Sort key: targets
            dst = edge_index[0].to(torch.int32)  # Store: sources
        else:
            src = edge_index[0].to(torch.int64)  # Sort key: sources
            dst = edge_index[1].to(torch.int32)  # Store: targets

        # Sort edges by the key (src)
        sorted_indices = torch.argsort(src)
        src_sorted = src[sorted_indices]
        self.col_ind = dst[sorted_indices].contiguous()

        # Handle weights
        if weights is None:
            self.weights = torch.ones(
                self.col_ind.size(0), device=self.device, dtype=torch.float32
            )
        else:
            if weights.numel() != edge_index.size(1):
                raise ValueError("weights must have length E to match edge_index")
            self.weights = weights.to(self.device)[sorted_indices].contiguous().float()

        # Build row pointers from degree counts
        degrees = torch.bincount(src_sorted, minlength=self.num_nodes)
        row_ptr = torch.zeros(self.num_nodes + 1, device=self.device, dtype=torch.int32)
        torch.cumsum(degrees, dim=0, out=row_ptr[1:])
        self.row_ptr = row_ptr.contiguous()

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        return self.col_ind.numel()

    def to_bf16_weights(self) -> "GraphCSR":
        """Return a copy with weights downcast to bfloat16.

        Halves memory traffic for the weight array during FlashNeighbor
        traversal. Safe for epidemic simulation where weights are typically
        small integers or unit values.
        """
        new_graph = object.__new__(GraphCSR)
        new_graph.device = self.device
        new_graph.num_nodes = self.num_nodes
        new_graph.row_ptr = self.row_ptr
        new_graph.col_ind = self.col_ind
        new_graph.weights = self.weights.to(torch.bfloat16)
        return new_graph

    def to(self, device: torch.device | str) -> "GraphCSR":
        """Move graph to specified device."""
        device = torch.device(device)
        if device == self.device:
            return self

        new_graph = object.__new__(GraphCSR)
        new_graph.device = device
        new_graph.num_nodes = self.num_nodes
        new_graph.row_ptr = self.row_ptr.to(device)
        new_graph.col_ind = self.col_ind.to(device)
        new_graph.weights = self.weights.to(device)
        return new_graph


class DualGraphCSR:
    """
    Dual CSR representation storing both incoming and outgoing edge structures.

    The Markovian engine requires both:
    - Incoming CSR: For FlashNeighbor gather operations
    - Outgoing CSR: For sparse incremental updates (Inertial mode)

    This class maintains both representations to avoid runtime transposition.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        weights: torch.Tensor | None = None,
    ):
        """
        Construct dual CSR from edge list.

        Args:
            edge_index: [2, E] tensor of edges.
            num_nodes: Total number of nodes.
            weights: Optional edge weights.
        """
        self.incoming = GraphCSR(edge_index, num_nodes, weights, incoming=True)
        self.outgoing = GraphCSR(edge_index, num_nodes, weights, incoming=False)
        self.num_nodes = num_nodes
        self.device = edge_index.device

    def to(self, device: torch.device | str) -> "DualGraphCSR":
        """Move both graphs to specified device."""
        device = torch.device(device)
        if device == self.device:
            return self

        new_dual = object.__new__(DualGraphCSR)
        new_dual.incoming = self.incoming.to(device)
        new_dual.outgoing = self.outgoing.to(device)
        new_dual.num_nodes = self.num_nodes
        new_dual.device = device
        return new_dual
