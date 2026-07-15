"""Compact graph storage used by every simulation engine.

``GraphCSR`` is the runtime graph contract.  Rows are targets by default, so a
row contains the source nodes whose state/infectivity contributes to that
target.  Engines should not retain a second COO edge list merely for fallback
code: :meth:`GraphCSR.to_edge_index` reconstructs one when an interoperability
boundary genuinely needs it, and :meth:`GraphCSR.transpose` builds the
outgoing view required by sparse propagation.
"""

import operator
import warnings

import torch


_INT32_MAX = torch.iinfo(torch.int32).max


def _ensure_versioned(tensor: torch.Tensor) -> torch.Tensor:
    """Return storage whose in-place mutation counter can be inspected."""
    if not torch.is_inference(tensor):
        return tensor
    # Inference tensors deliberately omit version counters. Engine pointer and
    # specialization guards need those counters, so pay a copy only for graphs
    # constructed under an ambient inference_mode context.
    with torch.inference_mode(False):
        return tensor.clone()


class GraphCSR:
    """
    Compact graph structure in CSR format optimized for GPU gather kernels.

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
        if edge_index.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(
                "edge_index must use an integer dtype; "
                f"got {edge_index.dtype}"
            )

        self.device = edge_index.device
        if isinstance(num_nodes, bool):
            raise TypeError("num_nodes must be an integer, not bool")
        try:
            self.num_nodes = operator.index(num_nodes)
        except TypeError as exc:
            raise TypeError("num_nodes must be an integer") from exc
        if not isinstance(incoming, bool):
            raise TypeError("incoming must be a bool")
        self.incoming = incoming
        self._transpose_cache = None
        self._transpose_cache_signature = None
        if self.num_nodes < 0:
            raise ValueError(f"num_nodes must be non-negative, got {self.num_nodes}")
        if self.num_nodes > _INT32_MAX:
            raise OverflowError("num_nodes exceeds the int32 CSR index limit")
        if edge_index.size(1) > _INT32_MAX:
            raise OverflowError("edge count exceeds the int32 CSR offset limit")

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
            self.has_weights = False
            self._weights = torch.ones(1, device=self.device, dtype=torch.float32)
        else:
            if weights.dim() != 1 or weights.numel() != edge_index.size(1):
                raise ValueError("weights must have shape [E] to match edge_index")
            weights = weights.to(self.device, dtype=torch.float32)
            if not bool(torch.isfinite(weights).all()):
                raise ValueError("weights must be finite")
            if bool((weights < 0).any()):
                raise ValueError("weights must be non-negative")
            sorted_weights = weights[sorted_indices].contiguous()
            self.has_weights = not bool((sorted_weights == 1.0).all())
            self._weights = (
                sorted_weights
                if self.has_weights
                else torch.ones(1, device=self.device, dtype=torch.float32)
            )

        # Build row pointers from degree counts
        degrees = torch.bincount(src_sorted, minlength=self.num_nodes)
        row_ptr = torch.zeros(self.num_nodes + 1, device=self.device, dtype=torch.int32)
        torch.cumsum(degrees, dim=0, out=row_ptr[1:])
        self.row_ptr = row_ptr.contiguous()
        self.row_ptr = _ensure_versioned(self.row_ptr)
        self.col_ind = _ensure_versioned(self.col_ind)
        self._weights = _ensure_versioned(self._weights)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        return self.col_ind.numel()

    @property
    def weights_storage(self) -> torch.Tensor:
        """Physical edge-weight storage used by kernels (one scalar if unit)."""
        return self._weights

    @property
    def weights(self) -> torch.Tensor:
        """Contiguous public ``[E]`` edge weights.

        Unit weights stay symbolic until this compatibility property is
        requested. At that boundary they are materialized deliberately: a
        writable zero-stride expansion would let one indexed assignment change
        every logical edge while GPU kernels still compile unit weights in.
        Runtime kernels use :attr:`weights_storage` and never trigger this cost.
        """
        if not self.has_weights:
            with torch.inference_mode(False):
                self._weights = torch.ones(
                    self.num_edges,
                    device=self.device,
                    dtype=self._weights.dtype,
                )
            self.has_weights = True
        # The returned tensor is writable for legacy compatibility. Transpose
        # cache lookups compare PyTorch mutation-version counters, so a later
        # indexed write through this alias is detected before reuse.
        return self._weights

    @weights.setter
    def weights(self, values: torch.Tensor) -> None:
        """Compatibility setter used by legacy graph-reordering experiments."""
        values = torch.as_tensor(values)
        if values.dim() != 1 or values.numel() != self.num_edges:
            raise ValueError("weights must have shape [E]")
        values = (
            values.to(device=self.col_ind.device, dtype=torch.float32)
            .contiguous()
            .clone()
        )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("weights must be finite")
        if bool((values < 0).any()):
            raise ValueError("weights must be non-negative")
        self.has_weights = not bool((values == 1.0).all())
        self._weights = _ensure_versioned(
            values
            if self.has_weights
            else torch.ones(1, device=self.col_ind.device, dtype=torch.float32)
        )
        cached = getattr(self, "_transpose_cache", None)
        self._transpose_cache = None
        self._transpose_cache_signature = None
        if cached is not None:
            cached._transpose_cache = None
            cached._transpose_cache_signature = None

    @property
    def csr(self) -> "GraphCSR":
        """Return self, allowing ``GraphCSR`` as the canonical public graph."""
        return self

    @property
    def edge_index(self) -> torch.Tensor:
        """Compatibility COO view, reconstructed on demand.

        Simulation engines never use this property. Prefer CSR-native APIs or
        :meth:`to_edge_index` at explicit interoperability boundaries.
        """
        return self.to_edge_index()

    @classmethod
    def from_csr(
        cls,
        row_ptr: torch.Tensor,
        col_ind: torch.Tensor,
        weights: torch.Tensor | None = None,
        *,
        incoming: bool = True,
    ) -> "GraphCSR":
        """Construct directly from validated CSR arrays without sorting COO.

        This is the scalable entry point for graph generators and datasets
        that already produce CSR. Contiguous int32 ``row_ptr``/``col_ind`` are
        retained zero-copy to avoid an edge-sized construction duplicate,
        except that inference-mode tensors are copied to restore mutation
        counters. The caller must therefore treat retained source tensors as
        immutable. Weights are always owned by the new graph.
        """
        if row_ptr.dim() != 1 or col_ind.dim() != 1:
            raise ValueError("row_ptr and col_ind must be one-dimensional")
        if row_ptr.numel() == 0:
            raise ValueError("row_ptr must contain at least the initial zero")
        if row_ptr.dtype not in (torch.int32, torch.int64):
            raise TypeError("row_ptr must use int32 or int64")
        if col_ind.dtype not in (torch.int32, torch.int64):
            raise TypeError("col_ind must use int32 or int64")
        if row_ptr.device != col_ind.device:
            raise ValueError("row_ptr and col_ind must be on the same device")

        num_nodes = int(row_ptr.numel() - 1)
        # Comparisons do not require widening an int32 row pointer. Avoiding an
        # N-sized int64 validation copy matters for the direct large-CSR path.
        if int(row_ptr[0]) != 0 or int(row_ptr[-1]) != col_ind.numel():
            raise ValueError("row_ptr must start at 0 and end at len(col_ind)")
        if bool((row_ptr[1:] < row_ptr[:-1]).any()):
            raise ValueError("row_ptr must be non-decreasing")
        if num_nodes > _INT32_MAX or col_ind.numel() > _INT32_MAX:
            raise OverflowError("CSR shape exceeds the package's int32 index limit")
        if col_ind.numel():
            lo, hi = int(col_ind.min()), int(col_ind.max())
            if lo < 0 or hi >= num_nodes:
                raise ValueError(
                    f"col_ind values must lie in [0, {num_nodes}), got [{lo}, {hi}]"
                )

        obj = cls.__new__(cls)
        obj.device = row_ptr.device
        obj.num_nodes = num_nodes
        if not isinstance(incoming, bool):
            raise TypeError("incoming must be a bool")
        obj.incoming = incoming
        obj._transpose_cache = None
        obj._transpose_cache_signature = None
        obj.row_ptr = _ensure_versioned(row_ptr.to(torch.int32).contiguous())
        obj.col_ind = _ensure_versioned(col_ind.to(torch.int32).contiguous())
        if weights is None:
            obj.has_weights = False
            obj._weights = _ensure_versioned(
                torch.ones(1, device=obj.device, dtype=torch.float32)
            )
        else:
            if weights.dim() != 1 or weights.numel() != col_ind.numel():
                raise ValueError("weights must have shape [E]")
            values = weights.to(device=obj.device, dtype=torch.float32)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("weights must be finite")
            if bool((values < 0).any()):
                raise ValueError("weights must be non-negative")
            # Own the storage. Otherwise a caller can mutate the source tensor
            # after construction without invalidating derived orientations.
            values = values.contiguous().clone()
            obj.has_weights = not bool((values == 1.0).all())
            obj._weights = _ensure_versioned(
                values
                if obj.has_weights
                else torch.ones(1, device=obj.device, dtype=torch.float32)
            )
        return obj

    def to_edge_index(self) -> torch.Tensor:
        """Reconstruct a COO ``[source, target]`` edge list.

        This is intentionally a method rather than retained storage: keeping a
        64-bit COO tensor alongside CSR costs another 16 bytes per edge and is
        unnecessary during simulation.
        """
        degrees = (self.row_ptr[1:] - self.row_ptr[:-1]).to(torch.int64)
        rows = torch.repeat_interleave(
            torch.arange(self.num_nodes, device=self.device, dtype=torch.int64),
            degrees,
        )
        cols = self.col_ind.to(torch.int64)
        if self.incoming:
            return torch.stack((cols, rows))
        return torch.stack((rows, cols))

    def transpose(self) -> "GraphCSR":
        """Return the opposite CSR orientation for the same directed edges."""
        cached = getattr(self, "_transpose_cache", None)
        signature = self._mutation_signature()
        if (
            cached is not None
            and getattr(self, "_transpose_cache_signature", None) == signature
            and getattr(cached, "_transpose_cache", None) is self
            and getattr(cached, "_transpose_cache_signature", None)
            == cached._mutation_signature()
        ):
            return cached

        # Let PyTorch's native sparse transpose perform the counting/scatter in
        # C++/CUDA. The old COO reconstruction needed several int64 E-sized
        # temporaries plus an argsort, which is prohibitive at benchmark scale.
        values = (
            self._weights
            if self.has_weights
            else torch.ones(self.num_edges, device=self.device, dtype=torch.uint8)
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Sparse CSR tensor support is in beta state"
                )
                warnings.filterwarnings(
                    "ignore", message="Sparse invariant checks are implicitly disabled"
                )
                matrix = torch.sparse_csr_tensor(
                    self.row_ptr,
                    self.col_ind,
                    values,
                    size=(self.num_nodes, self.num_nodes),
                    device=self.device,
                    check_invariants=False,
                )
                transposed = matrix.transpose(0, 1).to_sparse_csr()
            result = GraphCSR.from_csr(
                transposed.crow_indices(),
                transposed.col_indices(),
                weights=transposed.values() if self.has_weights else None,
                incoming=not self.incoming,
            )
        except (NotImplementedError, RuntimeError):
            # Conservative backend fallback (notably for older CUDA sparse
            # builds). Correct but uses larger temporary COO storage.
            result = GraphCSR(
                self.to_edge_index(),
                self.num_nodes,
                weights=self._weights if self.has_weights else None,
                incoming=not self.incoming,
            )
        # A caller may have mutated the previously returned opposite view.
        # Detach that stale reciprocal link before installing a fresh pair.
        if cached is not None and getattr(cached, "_transpose_cache", None) is self:
            cached._transpose_cache = None
            cached._transpose_cache_signature = None
        self._transpose_cache = result
        self._transpose_cache_signature = signature
        result._transpose_cache = self
        result._transpose_cache_signature = result._mutation_signature()
        return result

    def _mutation_signature(self) -> tuple:
        """Identity/version fingerprint for tensors underlying cached views."""
        def tensor_signature(tensor: torch.Tensor) -> tuple:
            return (
                id(tensor),
                int(tensor._version),
                tuple(tensor.shape),
                tensor.dtype,
                tensor.device,
            )

        return (
            tensor_signature(self.row_ptr),
            tensor_signature(self.col_ind),
            tensor_signature(self._weights),
            bool(self.has_weights),
            bool(self.incoming),
            int(self.num_nodes),
        )

    def _assert_unchanged(self, expected: tuple, *, owner: str) -> None:
        """Reject mutation after an engine/kernel has bound graph storage."""
        if self._mutation_signature() != expected:
            raise RuntimeError(
                f"GraphCSR storage changed after {owner} construction. Graph "
                "indices and weights are immutable while an engine/kernel is "
                "bound; create graph.with_weights(...) and construct a new "
                "engine instead."
            )

    def with_weights(self, weights: torch.Tensor | None) -> "GraphCSR":
        """Return a new graph sharing immutable indices with replacement weights."""
        return GraphCSR.from_csr(
            self.row_ptr,
            self.col_ind,
            weights=weights,
            incoming=self.incoming,
        )

    def to_bf16_weights(self) -> "GraphCSR":
        """Return a copy with weights downcast to bfloat16.

        Halves memory traffic for the weight array during FlashNeighbor
        traversal. Safe for epidemic simulation where weights are typically
        small integers or unit values.
        """
        new_graph = object.__new__(GraphCSR)
        new_graph.num_nodes = self.num_nodes
        new_graph.incoming = self.incoming
        new_graph._transpose_cache = None
        new_graph._transpose_cache_signature = None
        new_graph.row_ptr = self.row_ptr
        new_graph.col_ind = self.col_ind
        # Device ownership is defined by physical CSR storage, never by an
        # unresolved request such as ``torch.device("cuda")``.
        new_graph.device = new_graph.row_ptr.device
        new_graph.has_weights = self.has_weights
        new_graph._weights = _ensure_versioned(
            self._weights.to(torch.bfloat16).clone()
        )
        return new_graph

    def to(self, device: torch.device | str) -> "GraphCSR":
        """Move graph to specified device."""
        device = torch.device(device)
        if device == self.device and self.device == self.row_ptr.device:
            return self

        new_graph = object.__new__(GraphCSR)
        new_graph.num_nodes = self.num_nodes
        new_graph.incoming = self.incoming
        new_graph._transpose_cache = None
        new_graph._transpose_cache_signature = None
        new_graph.row_ptr = _ensure_versioned(self.row_ptr.to(device))
        new_graph.col_ind = _ensure_versioned(self.col_ind.to(device))
        # An unindexed CUDA request resolves only when PyTorch moves a tensor
        # (for example, ``cuda`` -> ``cuda:0``). Derive metadata from that
        # result so graph.device always matches every CSR tensor device.
        new_graph.device = new_graph.row_ptr.device
        new_graph.has_weights = self.has_weights
        new_graph._weights = _ensure_versioned(self._weights.to(device))
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
        self.outgoing = self.incoming.transpose()
        self.num_nodes = num_nodes
        self.device = edge_index.device

    def to(self, device: torch.device | str) -> "DualGraphCSR":
        """Move both graphs to specified device."""
        device = torch.device(device)
        if device == self.device and self.device == self.incoming.device:
            return self

        new_dual = object.__new__(DualGraphCSR)
        new_dual.incoming = self.incoming.to(device)
        new_dual.outgoing = new_dual.incoming.transpose()
        new_dual.num_nodes = self.num_nodes
        new_dual.device = new_dual.incoming.device
        return new_dual


def as_csr(graph, device: torch.device | str | None = None) -> GraphCSR:
    """Normalize a public graph object to the package's incoming CSR contract.

    Accepted inputs are a :class:`GraphCSR` or an object exposing ``.csr``.
    Keeping this conversion in one place removes subtly different graph
    duck-typing branches from each engine.
    """
    csr = graph.csr if hasattr(graph, "csr") else graph
    if not isinstance(csr, GraphCSR):
        raise TypeError("graph must be a GraphCSR or expose a GraphCSR as .csr")
    # Legacy compatibility objects built via ``GraphCSR.__new__`` did not
    # record orientation; those objects were always incoming.
    if not getattr(csr, "incoming", True):
        csr = csr.transpose()
    return csr if device is None else csr.to(device)
