"""Logical traffic accounting for node-major trajectory ensembles.

The pressure accounting mirrors the implemented standalone Triton gather: a program processes one
replica tile, so row pointers, column indices and optional weights are requested
once for that tile, while source payloads are requested once per active
replica.  A partially filled final tile therefore still incurs one structural
graph read but only issues masked payload loads for its real replicas.

The renewal-step composition models the dominant graph, rate, reduction, and
transition traffic. For the exact built-in non-Markovian constant-transmission
SEIR model, ``EnsembleEngine`` implements this structure as a multi-phase step:
the tiled kernel emits rates and one min/max pair per 128 nodes without storing
pressure, reductions over those compact partials plus a Triton finalizer select
per-replica time steps, and a tiled Triton kernel samples and applies
transitions. Generic and age-dependent-transmission models retain the separate
fp32 pressure and reference transition phases.

This accounting is not a claim that the whole step is one kernel. It omits
lower-order per-replica controls and event-partial read/write traffic, while
reporting the event-partial resident storage needed to quantify its temporal
alias with rate bounds. The compact int8/bf16 defaults describe the intended
end-to-end target. To approximate the current exact-SEIR path, pass its actual
storage widths (int32 state, fp32 age and rates, and the graph's stored
weight/index widths). ``ReferenceEnsembleEngine`` remains the pure-PyTorch
oracle.

For the implemented constant-transmission fast path, ``constant_source_encoding``
can model one uint32 infectious-state bitmap word per node and 32 replicas.
The graph then requests one packed word per edge in each susceptible tile union,
instead of one source-state element per active replica. Steady-state transition
atomics maintain this bitmap incrementally. The renewal model can separately
account for a full state-to-bitmap refresh after initialization or detected
external state mutation. Its resident bitmap footprint is reported separately
and is not added to per-step traffic.

These are logical byte requests, not HBM counters or measured performance.
Cache lines, compiler load coalescing and reduction scratch must still be
measured with Nsight Compute.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Literal

import torch


Transmission = Literal["constant", "age_dependent"]
ConstantSourceEncoding = Literal["state", "packed_bitmap"]


def _count(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer count, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer count") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _size(name: str, value: int) -> int:
    value = _count(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _counts(name: str, values, expected: int) -> tuple[int, ...]:
    try:
        result = tuple(_count(name, value) for value in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of counts") from exc
    if len(result) != expected:
        raise ValueError(f"{name} must contain {expected} values")
    return result


@dataclass(frozen=True, slots=True)
class EnsembleActivity:
    """State-dependent counts needed for exact masked graph accounting."""

    num_nodes: int
    num_edges: int
    replicas: int
    replicas_per_tile: int
    tile_count: int
    replica_susceptible_nodes: tuple[int, ...]
    replica_susceptible_edges: tuple[int, ...]
    replica_hazard_nodes: tuple[int, ...]
    tile_susceptible_union_nodes: tuple[int, ...]
    tile_susceptible_union_edges: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EnsemblePressureTraffic:
    """Logical bytes and useful accumulation FLOPs for one graph phase."""

    constant_source_encoding: ConstantSourceEncoding
    replicas: int
    replicas_per_tile: int
    tile_count: int
    row_pointer_read_bytes: int
    column_index_read_bytes: int
    weight_read_bytes: int
    source_payload_read_bytes: int
    packed_source_word_read_bytes: int
    packed_bitmap_resident_bytes: int
    output_write_bytes: int
    pressure_flops: int
    total_bytes: int

    @property
    def graph_metadata_bytes(self) -> int:
        return self.row_pointer_read_bytes + self.column_index_read_bytes + self.weight_read_bytes

    @property
    def arithmetic_intensity(self) -> float:
        return self.pressure_flops / self.total_bytes if self.total_bytes else 0.0

    @property
    def average_bytes_per_replica(self) -> float:
        return self.total_bytes / self.replicas


@dataclass(frozen=True, slots=True)
class EnsembleRenewalTraffic:
    """Dominant logical bytes for a multi-phase renewal ensemble step."""

    pressure: EnsemblePressureTraffic
    bitmap_pack_read_bytes: int
    bitmap_pack_write_bytes: int
    bitmap_atomic_updates: int | None
    bitmap_atomic_read_modify_write_bytes: int
    rate_node_read_bytes: int
    rate_write_bytes: int
    rate_bound_nodes_per_partial: int | None
    rate_bound_partial_count: int
    rate_bound_partial_resident_bytes: int
    event_partial_resident_bytes: int
    rate_event_temporally_shared_bytes: int
    step_partial_resident_bytes: int
    rate_bound_partial_write_bytes: int
    rate_reduction_read_bytes: int
    transition_read_bytes: int
    transition_state_updates: int | None
    transition_state_write_bytes: int
    transition_age_write_bytes: int
    transition_infectivity_write_bytes: int
    transition_write_bytes: int
    total_bytes: int

    @property
    def average_bytes_per_replica(self) -> float:
        return self.total_bytes / self.pressure.replicas


def renewal_activity_from_state(
    row_ptr: torch.Tensor,
    state: torch.Tensor,
    *,
    susceptible: int,
    exposed: int,
    infected: int,
    replicas_per_tile: int,
    node_chunk_size: int = 262_144,
) -> EnsembleActivity:
    """Derive exact per-replica and per-tile activity from ``state[N, R]``.

    Per-tile union counts matter: a row's index/weight metadata is loaded once
    whenever *any* replica in that tile is susceptible. Source payload loads
    remain masked per replica and therefore use the per-replica edge counts.
    This helper is for checkpoints/accounting, not the timed simulation path.
    It streams node chunks so scratch is ``O(node_chunk_size * R)`` rather than
    materializing multiple ``[N, R]`` masks at roofline scale.
    """
    if row_ptr.dim() != 1 or row_ptr.dtype not in (torch.int32, torch.int64):
        raise TypeError("row_ptr must be a one-dimensional int32/int64 tensor")
    if state.dim() != 2 or state.shape[0] + 1 != row_ptr.numel():
        raise ValueError("state must have node-major shape [len(row_ptr)-1, R]")
    if state.dtype == torch.bool or state.dtype.is_floating_point or state.dtype.is_complex:
        raise TypeError("state must use an integer dtype")
    if state.device != row_ptr.device:
        raise ValueError("row_ptr and state must be on the same device")
    replicas = int(state.shape[1])
    if replicas <= 0:
        raise ValueError("state must contain at least one replica")
    tile = _size("replicas_per_tile", replicas_per_tile)
    if tile & (tile - 1):
        raise ValueError("replicas_per_tile must be a power of two")
    if tile > 32:
        raise ValueError("replicas_per_tile must be <= 32")
    chunk_size = _size("node_chunk_size", node_chunk_size)
    tile_count = math.ceil(replicas / tile)
    replica_nodes_acc = torch.zeros(replicas, device=state.device, dtype=torch.int64)
    replica_edges_acc = torch.zeros_like(replica_nodes_acc)
    hazard_nodes_acc = torch.zeros_like(replica_nodes_acc)
    tile_nodes_acc = torch.zeros(tile_count, device=state.device, dtype=torch.int64)
    tile_edges_acc = torch.zeros_like(tile_nodes_acc)

    for start in range(0, state.shape[0], chunk_size):
        stop = min(start + chunk_size, state.shape[0])
        state_chunk = state[start:stop]
        degrees = row_ptr[start + 1 : stop + 1] - row_ptr[start:stop]
        susceptible_mask = state_chunk == susceptible
        replica_nodes_acc.add_(susceptible_mask.sum(dim=0, dtype=torch.int64))
        # Index one replica at a time: scratch stays at one bool [chunk,R]
        # mask plus one selected degree vector, not an int64 [chunk,R] product.
        for replica in range(replicas):
            replica_edges_acc[replica].add_(
                degrees[susceptible_mask[:, replica]].sum(dtype=torch.int64)
            )
        for tile_index, first in enumerate(range(0, replicas, tile)):
            union = susceptible_mask[:, first : first + tile].any(dim=1)
            tile_nodes_acc[tile_index].add_(union.sum(dtype=torch.int64))
            tile_edges_acc[tile_index].add_(degrees[union].sum(dtype=torch.int64))
        del susceptible_mask

        hazard_mask = state_chunk == exposed
        hazard_mask.logical_or_(state_chunk == infected)
        hazard_nodes_acc.add_(hazard_mask.sum(dim=0, dtype=torch.int64))

    replica_nodes = tuple(replica_nodes_acc.cpu().tolist())
    replica_edges = tuple(replica_edges_acc.cpu().tolist())
    hazard_nodes = tuple(hazard_nodes_acc.cpu().tolist())
    tile_nodes = tuple(tile_nodes_acc.cpu().tolist())
    tile_edges = tuple(tile_edges_acc.cpu().tolist())
    return EnsembleActivity(
        num_nodes=int(state.shape[0]),
        num_edges=int(row_ptr[-1].item()),
        replicas=replicas,
        replicas_per_tile=tile,
        tile_count=tile_count,
        replica_susceptible_nodes=replica_nodes,
        replica_susceptible_edges=replica_edges,
        replica_hazard_nodes=hazard_nodes,
        tile_susceptible_union_nodes=tile_nodes,
        tile_susceptible_union_edges=tile_edges,
    )


def ensemble_pressure_logical_traffic(
    *,
    activity: EnsembleActivity,
    transmission: Transmission,
    has_weights: bool,
    state_bytes: int = 1,
    infectivity_bytes: int = 2,
    weight_bytes: int = 2,
    index_bytes: int = 4,
    output_bytes: int = 0,
    constant_source_encoding: ConstantSourceEncoding = "state",
    packed_word_bytes: int = 4,
) -> EnsemblePressureTraffic:
    """Model one susceptible-only node-major ensemble pressure gather.

    ``output_bytes=0`` is the production fused rate path (pressure remains in
    registers); use ``4`` to model the standalone validation gather's fp32
    output. The activity object makes the accounting exact even when replica
    states differ or the final replica tile is partially full.

    ``constant_source_encoding="packed_bitmap"`` stores inducer-state bits in
    fixed-width words. The graph reads one word per edge in each execution
    tile's susceptible-row union; several narrow execution tiles may therefore
    reread the same resident word. Resident bytes include padding in the final
    word, are a storage footprint rather than traffic, and are excluded from
    ``total_bytes``. Age-dependent accounting still reads per-replica
    infectivity and therefore rejects the packed constant-source encoding.
    """
    if transmission not in ("constant", "age_dependent"):
        raise ValueError("transmission must be 'constant' or 'age_dependent'")
    if not isinstance(has_weights, bool):
        raise TypeError("has_weights must be bool")
    if constant_source_encoding not in ("state", "packed_bitmap"):
        raise ValueError("constant_source_encoding must be 'state' or 'packed_bitmap'")
    if transmission == "age_dependent" and constant_source_encoding == "packed_bitmap":
        raise ValueError("packed_bitmap source encoding only supports constant transmission")
    state_size = _size("state_bytes", state_bytes)
    infectivity_size = _size("infectivity_bytes", infectivity_bytes)
    weight_size = _size("weight_bytes", weight_bytes)
    index_size = _size("index_bytes", index_bytes)
    output_size = _count("output_bytes", output_bytes)
    packed_word_size = _size("packed_word_bytes", packed_word_bytes)
    source_size = state_size if transmission == "constant" else infectivity_size

    n = _count("activity.num_nodes", activity.num_nodes)
    edges = _count("activity.num_edges", activity.num_edges)
    replicas = _size("activity.replicas", activity.replicas)
    tile = _size("activity.replicas_per_tile", activity.replicas_per_tile)
    if tile & (tile - 1) or tile > 32:
        raise ValueError("activity.replicas_per_tile must be a power of two <= 32")
    if constant_source_encoding == "packed_bitmap" and tile > 8 * packed_word_size:
        raise ValueError("packed_word_bytes cannot represent one bit per replica tile lane")
    expected_tiles = math.ceil(replicas / tile)
    if activity.tile_count != expected_tiles:
        raise ValueError("activity.tile_count is inconsistent with replicas/tile")
    replica_edges = _counts(
        "replica_susceptible_edges",
        activity.replica_susceptible_edges,
        replicas,
    )
    tile_nodes = _counts(
        "tile_susceptible_union_nodes",
        activity.tile_susceptible_union_nodes,
        expected_tiles,
    )
    tile_edges = _counts(
        "tile_susceptible_union_edges",
        activity.tile_susceptible_union_edges,
        expected_tiles,
    )
    replica_nodes = _counts(
        "replica_susceptible_nodes",
        activity.replica_susceptible_nodes,
        replicas,
    )
    if any(value > n for value in replica_nodes) or any(value > n for value in tile_nodes):
        raise ValueError("susceptible node counts cannot exceed activity.num_nodes")
    if any(value > edges for value in replica_edges) or any(value > edges for value in tile_edges):
        raise ValueError("susceptible edge counts cannot exceed activity.num_edges")

    row_bytes = 2 * index_size * sum(tile_nodes)
    column_bytes = index_size * sum(tile_edges)
    weight_read_bytes = weight_size * sum(tile_edges) if has_weights else 0
    if constant_source_encoding == "packed_bitmap":
        packed_source_word_read_bytes = packed_word_size * sum(tile_edges)
        source_bytes = packed_source_word_read_bytes
        replicas_per_word = 8 * packed_word_size
        resident_words = math.ceil(replicas / replicas_per_word)
        packed_bitmap_resident_bytes = packed_word_size * n * resident_words
    else:
        source_bytes = source_size * sum(replica_edges)
        packed_source_word_read_bytes = 0
        packed_bitmap_resident_bytes = 0
    output_write_bytes = output_size * n * replicas
    pressure_flops = (2 if has_weights else 1) * sum(replica_edges)
    components = row_bytes, column_bytes, weight_read_bytes, source_bytes
    return EnsemblePressureTraffic(
        constant_source_encoding=constant_source_encoding,
        replicas=replicas,
        replicas_per_tile=tile,
        tile_count=expected_tiles,
        row_pointer_read_bytes=row_bytes,
        column_index_read_bytes=column_bytes,
        weight_read_bytes=weight_read_bytes,
        source_payload_read_bytes=source_bytes,
        packed_source_word_read_bytes=packed_source_word_read_bytes,
        packed_bitmap_resident_bytes=packed_bitmap_resident_bytes,
        output_write_bytes=output_write_bytes,
        pressure_flops=pressure_flops,
        total_bytes=sum(components) + output_write_bytes,
    )


def ensemble_full_pressure_logical_traffic(
    *,
    num_nodes: int,
    num_edges: int,
    replicas: int,
    replicas_per_tile: int,
    transmission: Transmission,
    has_weights: bool,
    state_bytes: int = 1,
    infectivity_bytes: int = 2,
    weight_bytes: int = 2,
    index_bytes: int = 4,
    output_bytes: int = 4,
    constant_source_encoding: ConstantSourceEncoding = "state",
    packed_word_bytes: int = 4,
) -> EnsemblePressureTraffic:
    """Exact traffic for the standalone kernel, which gathers every CSR row."""
    n = _count("num_nodes", num_nodes)
    edges = _count("num_edges", num_edges)
    replicas = _size("replicas", replicas)
    tile = _size("replicas_per_tile", replicas_per_tile)
    if tile & (tile - 1) or tile > 32:
        raise ValueError("replicas_per_tile must be a power of two <= 32")
    tiles = math.ceil(replicas / tile)
    activity = EnsembleActivity(
        num_nodes=n,
        num_edges=edges,
        replicas=replicas,
        replicas_per_tile=tile,
        tile_count=tiles,
        replica_susceptible_nodes=(n,) * replicas,
        replica_susceptible_edges=(edges,) * replicas,
        replica_hazard_nodes=(0,) * replicas,
        tile_susceptible_union_nodes=(n,) * tiles,
        tile_susceptible_union_edges=(edges,) * tiles,
    )
    return ensemble_pressure_logical_traffic(
        activity=activity,
        transmission=transmission,
        has_weights=has_weights,
        state_bytes=state_bytes,
        infectivity_bytes=infectivity_bytes,
        weight_bytes=weight_bytes,
        index_bytes=index_bytes,
        output_bytes=output_bytes,
        constant_source_encoding=constant_source_encoding,
        packed_word_bytes=packed_word_bytes,
    )


def ensemble_renewal_logical_traffic(
    *,
    num_nodes: int,
    activity: EnsembleActivity,
    transmission: Transmission,
    has_weights: bool,
    state_bytes: int = 1,
    age_bytes: int = 4,
    infectivity_bytes: int = 2,
    weight_bytes: int = 2,
    index_bytes: int = 4,
    rate_bytes: int = 4,
    constant_source_encoding: ConstantSourceEncoding = "state",
    packed_word_bytes: int = 4,
    rate_bound_nodes_per_partial: int | None = None,
    bitmap_refresh: bool = False,
    bitmap_atomic_updates: int | None = None,
    transition_changed_events: int | None = None,
) -> EnsembleRenewalTraffic:
    """Model dominant rate/reduction/transition ensemble-step traffic.

    The exact built-in constant-transmission SEIR fast path executes these
    phases, but also has lower-order control and event-partial traffic omitted
    here. Defaults represent the intended compact GPU layout; pass the actual
    tensor byte widths when comparing with a current engine.

    Set ``rate_bound_nodes_per_partial`` for the fused production reduction.
    The rate kernel writes two fp32 bounds per node group and replica, and the
    following reductions read those two compact arrays. Their resident bytes
    are reported separately. At the implemented 128-node granularity, minimum-
    bound storage is dead before transition and is reinterpreted as the equal-
    shape int32 event-partial array; ``rate_event_temporally_shared_bytes``
    reports that alias and ``step_partial_resident_bytes`` counts it once.
    ``None`` retains the generic/reference dense-rate reread, so accounting does
    not silently apply the model-specific fast path to other engines.

    Packed constant-source accounting follows the incremental design: E->I and
    I->R transitions set or clear bits atomically, so a normal steady-state
    step does not scan and repack the dense state. Set ``bitmap_refresh=True``
    only for initialization or a step that repairs the bitmap after detected
    external state mutation. A refresh reads the full node-major state and
    writes the full node-major bitmap; its write bytes equal the separately
    reported resident footprint.

    Atomic traffic is state-dependent and is therefore omitted unless
    ``bitmap_atomic_updates`` supplies the number of E->I plus I->R updates.
    Each supplied update is modeled as one logical packed-word read-modify-write
    (one word read and one word write). Actual cache/HBM transactions require
    hardware-counter measurement.

    The implemented fused transition writes state only for lanes whose event
    changes their compartment, while age is written for every valid lane. Pass
    ``transition_changed_events`` to account for state writes exactly. ``None``
    preserves the historical all-lanes value as a conservative upper bound;
    the returned ``transition_state_updates`` remains ``None`` so an estimate
    cannot be mistaken for an observed event count. Age-dependent accounting
    retains its separate next-infectivity write component.
    """
    if not isinstance(bitmap_refresh, bool):
        raise TypeError("bitmap_refresh must be bool")
    if constant_source_encoding != "packed_bitmap":
        if bitmap_refresh:
            raise ValueError("bitmap_refresh requires packed_bitmap source encoding")
        if bitmap_atomic_updates is not None:
            raise ValueError("bitmap_atomic_updates requires packed_bitmap source encoding")
    n = _count("num_nodes", num_nodes)
    if n != activity.num_nodes:
        raise ValueError("num_nodes must match activity.num_nodes")
    replicas = _size("activity.replicas", activity.replicas)
    hazard_nodes = _counts("replica_hazard_nodes", activity.replica_hazard_nodes, replicas)
    if any(value > n for value in hazard_nodes):
        raise ValueError("replica hazard counts cannot exceed num_nodes")
    state_size = _size("state_bytes", state_bytes)
    age_size = _size("age_bytes", age_bytes)
    infectivity_size = _size("infectivity_bytes", infectivity_bytes)
    rate_size = _size("rate_bytes", rate_bytes)

    pressure = ensemble_pressure_logical_traffic(
        activity=activity,
        transmission=transmission,
        has_weights=has_weights,
        state_bytes=state_size,
        infectivity_bytes=infectivity_size,
        weight_bytes=weight_bytes,
        index_bytes=index_bytes,
        constant_source_encoding=constant_source_encoding,
        packed_word_bytes=packed_word_bytes,
    )
    lanes = n * replicas
    if constant_source_encoding == "packed_bitmap" and bitmap_refresh:
        bitmap_pack_read = state_size * lanes
        bitmap_pack_write = pressure.packed_bitmap_resident_bytes
    else:
        bitmap_pack_read = 0
        bitmap_pack_write = 0
    if bitmap_atomic_updates is None:
        atomic_updates = None
        bitmap_atomic_bytes = 0
    else:
        atomic_updates = _count("bitmap_atomic_updates", bitmap_atomic_updates)
        packed_word_size = _size("packed_word_bytes", packed_word_bytes)
        bitmap_atomic_bytes = 2 * packed_word_size * atomic_updates
    if transition_changed_events is None:
        state_updates = None
        modeled_state_updates = lanes
    else:
        state_updates = _count("transition_changed_events", transition_changed_events)
        if state_updates > lanes:
            raise ValueError(
                "transition_changed_events cannot exceed the number of node-replica lanes"
            )
        modeled_state_updates = state_updates
    if atomic_updates is not None and state_updates is not None and atomic_updates > state_updates:
        raise ValueError("bitmap_atomic_updates cannot exceed transition_changed_events")
    rate_node_read = state_size * lanes + age_size * sum(hazard_nodes)
    rate_write = rate_size * lanes
    if rate_bound_nodes_per_partial is None:
        bound_nodes = None
        bound_count = 0
        bound_resident = 0
        event_partial_resident = 0
        shared_partial_resident = 0
        step_partial_resident = 0
        bound_write = 0
        reduction_read = rate_size * lanes
    else:
        bound_nodes = _size(
            "rate_bound_nodes_per_partial",
            rate_bound_nodes_per_partial,
        )
        bound_count = math.ceil(n / bound_nodes) * replicas
        # Bound outputs are fixed fp32 even when ``rate_bytes`` is used to
        # explore a narrower hypothetical public-rate representation.
        bound_value_size = 4
        bound_resident = 2 * bound_value_size * bound_count
        event_partial_count = math.ceil(n / 128) * replicas
        event_partial_resident = 4 * event_partial_count
        shared_partial_resident = (
            event_partial_resident
            if bound_nodes == 128
            else 0
        )
        step_partial_resident = (
            bound_resident
            + event_partial_resident
            - shared_partial_resident
        )
        bound_write = bound_resident
        reduction_read = bound_resident
    transition_read = (state_size + age_size + rate_size) * lanes
    transition_state_write = state_size * modeled_state_updates
    transition_age_write = age_size * lanes
    transition_infectivity_write = (
        infectivity_size * lanes if transmission == "age_dependent" else 0
    )
    transition_write = transition_state_write + transition_age_write + transition_infectivity_write
    components = (
        pressure.total_bytes,
        bitmap_pack_read,
        bitmap_pack_write,
        bitmap_atomic_bytes,
        rate_node_read,
        rate_write,
        bound_write,
        reduction_read,
        transition_read,
        transition_write,
    )
    return EnsembleRenewalTraffic(
        pressure=pressure,
        bitmap_pack_read_bytes=bitmap_pack_read,
        bitmap_pack_write_bytes=bitmap_pack_write,
        bitmap_atomic_updates=atomic_updates,
        bitmap_atomic_read_modify_write_bytes=bitmap_atomic_bytes,
        rate_node_read_bytes=rate_node_read,
        rate_write_bytes=rate_write,
        rate_bound_nodes_per_partial=bound_nodes,
        rate_bound_partial_count=bound_count,
        rate_bound_partial_resident_bytes=bound_resident,
        event_partial_resident_bytes=event_partial_resident,
        rate_event_temporally_shared_bytes=shared_partial_resident,
        step_partial_resident_bytes=step_partial_resident,
        rate_bound_partial_write_bytes=bound_write,
        rate_reduction_read_bytes=reduction_read,
        transition_read_bytes=transition_read,
        transition_state_updates=state_updates,
        transition_state_write_bytes=transition_state_write,
        transition_age_write_bytes=transition_age_write,
        transition_infectivity_write_bytes=transition_infectivity_write,
        transition_write_bytes=transition_write,
        total_bytes=sum(components),
    )


__all__ = [
    "ConstantSourceEncoding",
    "EnsembleActivity",
    "EnsemblePressureTraffic",
    "EnsembleRenewalTraffic",
    "renewal_activity_from_state",
    "ensemble_pressure_logical_traffic",
    "ensemble_full_pressure_logical_traffic",
    "ensemble_renewal_logical_traffic",
]
