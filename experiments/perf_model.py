"""CPU-testable storage and traffic accounting for performance experiments.

The quantities in this module are deliberately *models*, not profiler
counters.  Storage accounting uses physical tensor storages, so aliases and
views are counted once.  Renewal traffic counts logical bytes requested by the
current non-compacted thread/warp fast path.  Hardware caches can make actual
HBM traffic smaller (or cache-line effects can make it larger), so publication
rooflines should use Nsight Compute counters and retain these values as an
auditable lower-level cross-check.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import operator
from typing import Literal

import torch


Transmission = Literal["constant", "age_dependent"]


@dataclass(frozen=True, slots=True)
class StorageUsage:
    """Physical storage occupied by a collection of dense tensors."""

    num_storages: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class GraphStorage:
    """Physical storage of one int32-style CSR orientation."""

    row_pointer_bytes: int
    column_index_bytes: int
    weight_storage_bytes: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class MarkovReductionScratch:
    """Exact persistent reduction hierarchy used by built-in Markov kernels."""

    num_nodes: int
    node_block: int
    reduction_block: int
    level_sizes: tuple[int, ...]
    rate_sum_bytes: int
    rate_max_bytes: int
    event_count_bytes: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class RenewalTraffic:
    """Logical byte requests for one current-rate renewal step.

    Pressure remains in registers. Rates are the sole dense intermediate and
    are written once and read once by the transition. The rate programs also
    write compact maximum partials, which the ordering reduction reads instead
    of rereading the dense public rate array.
    """

    rate_node_read_bytes: int
    row_pointer_read_bytes: int
    column_index_read_bytes: int
    source_payload_read_bytes: int
    weight_read_bytes: int
    rate_write_bytes: int
    rate_max_partial_write_bytes: int
    rate_reduction_read_bytes: int
    transition_read_bytes: int
    transition_write_bytes: int
    total_bytes: int

    @property
    def node_bytes(self) -> int:
        """Bytes not proportional to susceptible rows or their edges."""
        return (
            self.rate_node_read_bytes
            + self.rate_write_bytes
            + self.rate_max_partial_write_bytes
            + self.rate_reduction_read_bytes
            + self.transition_read_bytes
            + self.transition_write_bytes
        )

    @property
    def susceptible_graph_bytes(self) -> int:
        """Row and edge bytes proportional to the susceptible target set."""
        return (
            self.row_pointer_read_bytes
            + self.column_index_read_bytes
            + self.source_payload_read_bytes
            + self.weight_read_bytes
        )


def _count(name: str, value: int) -> int:
    """Return a validated non-negative integer count."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer count, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer count") from exc
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _item_size(name: str, value: int) -> int:
    """Return a validated positive byte width."""
    result = _count(name, value)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def unique_storage_usage(tensors: Iterable[torch.Tensor]) -> StorageUsage:
    """Count unique physical storages behind dense tensors.

    A slice, reshape, detached tensor, or repeated reference shares the same
    storage and is counted once.  Callers should pass CSR component tensors
    explicitly; sparse container tensors do not expose one representative
    storage for all of their index/value arrays.
    """
    seen: set[tuple[str, int | None, int]] = set()
    total_bytes = 0

    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("unique_storage_usage expects torch.Tensor values")
        if tensor.layout != torch.strided:
            raise TypeError(
                "unique_storage_usage expects dense strided tensors; "
                "pass sparse tensor components explicitly"
            )
        storage = tensor.untyped_storage()
        storage_bytes = int(storage.nbytes())
        if storage_bytes == 0:
            continue
        key = (tensor.device.type, tensor.device.index, int(storage.data_ptr()))
        if key in seen:
            continue
        seen.add(key)
        total_bytes += storage_bytes

    return StorageUsage(num_storages=len(seen), total_bytes=total_bytes)


def graph_csr_storage(
    *,
    num_nodes: int,
    num_edges: int,
    has_weights: bool,
    index_bytes: int = 4,
    weight_bytes: int = 4,
) -> GraphStorage:
    """Model physical bytes for one canonical CSR orientation.

    Unit weights are symbolic in :class:`flashspread.core.graph.GraphCSR` and
    occupy one scalar, not ``E`` scalars.  This function mirrors that storage
    contract without touching the materializing public ``graph.weights``
    compatibility property.
    """
    n = _count("num_nodes", num_nodes)
    edges = _count("num_edges", num_edges)
    index_size = _item_size("index_bytes", index_bytes)
    weight_size = _item_size("weight_bytes", weight_bytes)
    if not isinstance(has_weights, bool):
        raise TypeError("has_weights must be bool")

    row_pointer_bytes = (n + 1) * index_size
    column_index_bytes = edges * index_size
    weight_storage_bytes = (edges if has_weights else 1) * weight_size
    total = row_pointer_bytes + column_index_bytes + weight_storage_bytes
    return GraphStorage(
        row_pointer_bytes=row_pointer_bytes,
        column_index_bytes=column_index_bytes,
        weight_storage_bytes=weight_storage_bytes,
        total_bytes=total,
    )


def markov_reduction_scratch(
    num_nodes: int,
    *,
    node_block: int = 128,
    reduction_block: int = 1024,
) -> MarkovReductionScratch:
    """Account for the fixed Markov sum/max/event reduction arrays.

    The rate kernel emits one partial per ``node_block`` nodes. Each following
    level reduces ``reduction_block`` inputs to one output until one scalar
    remains. The engine retains fp32 sum and max arrays plus one int64 event
    array at every level, hence ``(4 + 4 + 8) * sum(level_sizes)`` bytes.
    This excludes graph/state storage, CUDA Graph pools, compiler workspaces,
    and allocator reservation.
    """
    n = _count("num_nodes", num_nodes)
    if n == 0:
        raise ValueError("num_nodes must be positive")
    node_width = _item_size("node_block", node_block)
    reduction_width = _item_size("reduction_block", reduction_block)
    if reduction_width < 2:
        raise ValueError("reduction_block must be at least 2")
    sizes = [(n + node_width - 1) // node_width]
    while sizes[-1] > 1:
        sizes.append(
            (sizes[-1] + reduction_width - 1) // reduction_width
        )
    partials = sum(sizes)
    rate_sum_bytes = 4 * partials
    rate_max_bytes = 4 * partials
    event_count_bytes = 8 * partials
    return MarkovReductionScratch(
        num_nodes=n,
        node_block=node_width,
        reduction_block=reduction_width,
        level_sizes=tuple(sizes),
        rate_sum_bytes=rate_sum_bytes,
        rate_max_bytes=rate_max_bytes,
        event_count_bytes=event_count_bytes,
        total_bytes=rate_sum_bytes + rate_max_bytes + event_count_bytes,
    )


def renewal_logical_traffic(
    *,
    num_nodes: int,
    num_edges: int,
    susceptible_nodes: int,
    susceptible_edges: int,
    hazard_nodes: int | None = None,
    transmission: Transmission = "constant",
    has_weights: bool = False,
    state_bytes: int = 4,
    age_bytes: int = 4,
    infectivity_bytes: int = 4,
    weight_bytes: int = 4,
    index_bytes: int = 4,
    rate_bytes: int = 4,
    rate_nodes_per_partial: int = 128,
) -> RenewalTraffic:
    """Model one non-compacted thread/warp renewal step.

    ``susceptible_edges`` is the exact number of incoming CSR entries owned by
    susceptible targets, not the graph's total edge count.  The current rate
    kernels load row pointers and traverse adjacency only for those targets.
    ``hazard_nodes`` is the number of E/I nodes whose age is loaded; S/R ages
    are masked out. It defaults to N for a conservative/back-compatible bound.

    Constant transmission gathers source state; age-dependent transmission
    gathers source infectivity.  Symbolic unit weights set ``has_weights`` to
    false and therefore issue no per-edge weight reads.

    ``rate_nodes_per_partial`` is the number of node rates reduced by one rate
    program: 128 for the production thread/merge tails and
    ``nodes_per_block`` for the warp traversal. This model intentionally
    excludes scalar tau/RNG traffic, reduction scratch beyond its logical
    partial read/write, allocator effects, cache-line amplification, active
    compaction, and the merge strategy's pressure scratch/atomics/searches.
    """
    n = _count("num_nodes", num_nodes)
    edges = _count("num_edges", num_edges)
    n_s = _count("susceptible_nodes", susceptible_nodes)
    e_s = _count("susceptible_edges", susceptible_edges)
    n_h = n if hazard_nodes is None else _count("hazard_nodes", hazard_nodes)
    if n_s > n:
        raise ValueError("susceptible_nodes cannot exceed num_nodes")
    if n_h > n:
        raise ValueError("hazard_nodes cannot exceed num_nodes")
    if e_s > edges:
        raise ValueError("susceptible_edges cannot exceed num_edges")
    if transmission not in ("constant", "age_dependent"):
        raise ValueError(
            "transmission must be 'constant' or 'age_dependent', "
            f"got {transmission!r}"
        )
    if not isinstance(has_weights, bool):
        raise TypeError("has_weights must be bool")

    state_size = _item_size("state_bytes", state_bytes)
    age_size = _item_size("age_bytes", age_bytes)
    infectivity_size = _item_size("infectivity_bytes", infectivity_bytes)
    weight_size = _item_size("weight_bytes", weight_bytes)
    index_size = _item_size("index_bytes", index_bytes)
    rate_size = _item_size("rate_bytes", rate_bytes)
    partial_width = _item_size(
        "rate_nodes_per_partial", rate_nodes_per_partial
    )

    source_size = state_size if transmission == "constant" else infectivity_size

    # Phase 1: state is required for every node; age only for E/I hazards.
    rate_node_read_bytes = state_size * n + age_size * n_h
    row_pointer_read_bytes = 2 * index_size * n_s
    column_index_read_bytes = index_size * e_s
    source_payload_read_bytes = source_size * e_s
    weight_read_bytes = weight_size * e_s if has_weights else 0
    rate_write_bytes = rate_size * n

    # Global ordering point: each rate program writes one maximum, then the
    # reduction reads that compact array to choose the same step's tau.
    rate_partial_count = (n + partial_width - 1) // partial_width
    rate_max_partial_write_bytes = rate_size * rate_partial_count
    rate_reduction_read_bytes = rate_size * rate_partial_count

    # Phase 2: read current state/age/rate and write complete ping-pong output.
    # Constant transmission derives source shedding from state and beta, so it
    # does not carry or write a dense infectivity buffer.
    transition_read_bytes = (state_size + age_size + rate_size) * n
    transition_write_size = state_size + age_size
    if transmission == "age_dependent":
        transition_write_size += infectivity_size
    transition_write_bytes = transition_write_size * n

    components = (
        rate_node_read_bytes,
        row_pointer_read_bytes,
        column_index_read_bytes,
        source_payload_read_bytes,
        weight_read_bytes,
        rate_write_bytes,
        rate_max_partial_write_bytes,
        rate_reduction_read_bytes,
        transition_read_bytes,
        transition_write_bytes,
    )
    return RenewalTraffic(
        rate_node_read_bytes=rate_node_read_bytes,
        row_pointer_read_bytes=row_pointer_read_bytes,
        column_index_read_bytes=column_index_read_bytes,
        source_payload_read_bytes=source_payload_read_bytes,
        weight_read_bytes=weight_read_bytes,
        rate_write_bytes=rate_write_bytes,
        rate_max_partial_write_bytes=rate_max_partial_write_bytes,
        rate_reduction_read_bytes=rate_reduction_read_bytes,
        transition_read_bytes=transition_read_bytes,
        transition_write_bytes=transition_write_bytes,
        total_bytes=sum(components),
    )
