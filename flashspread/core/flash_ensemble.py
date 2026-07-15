"""Triton CSR gather foundation for graph-reusing trajectory ensembles.

The hot tensor layout is ``[node, replica]`` with replicas contiguous.  One
program owns a small target-node tile and a replica tile.  The two logical tile
coordinates are flattened onto ``grid.x`` so large ensembles are not capped by
CUDA's much smaller ``grid.y`` limit.  During each ragged CSR iteration the
program loads one neighbor index (and one optional edge weight) per target row,
then broadcasts that graph metadata across the replica lanes.  Consequently
graph-structure traffic is amortized by ``REPLICAS_PER_TILE``; source payload
traffic and useful accumulation work still scale with replicas.

The generic gathers remain available for arbitrary model contracts.  The
built-in renewal SEIR path additionally fuses current-rate evaluation into the
same traversal, eliminating the dense pressure intermediate while retaining
independent per-replica reductions, clocks, and random streams.
"""

from __future__ import annotations

import math
import operator

import torch

from .ensemble_reference import (
    reference_ensemble_infectivity_csr,
    reference_ensemble_influence_csr,
)
from .graph import as_csr
from ..utils import validate_fp32_control


# The production rate kernel emits one min/max pair per 128 nodes and replica.
# Every supported power-of-two node tile divides this exactly. A fixed reduction
# granularity keeps scratch/traffic independent of traversal tuning and matches
# the transition kernel's event-partial row granularity.
_RATE_BOUND_NODES_PER_PARTIAL = 128


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
    _TRITON_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional GPU dependency
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = exc


if _HAS_TRITON:
    # This helper is shared with the single-trajectory renewal kernels so the
    # two fast paths use the same fp32 tail approximation and age-zero limit.
    # ``flash_renewal_kernel`` depends only on the RNG primitives and does not
    # import this module, so importing the JIT helper here is cycle-free.
    from .flash_renewal_kernel import _lognormal_hazard_triton

    @triton.jit
    def _pack_ensemble_infectious_mask_kernel(
        state_ptr,
        infectious_mask_ptr,
        N,
        R,
        NODES_PER_PROGRAM: tl.constexpr,
    ):
        """Pack built-in ``I=2`` lanes into one uint32 bit pattern per word."""
        words = tl.cdiv(R, 32)
        program = tl.program_id(0)
        node_program = program // words
        word = program - node_program * words
        node = node_program * NODES_PER_PROGRAM + tl.arange(0, NODES_PER_PROGRAM)
        bit = tl.arange(0, 32)
        replica = word * 32 + bit
        lane_mask = (node < N)[:, None] & (replica < R)[None, :]
        offset = node.to(tl.int64)[:, None] * R + replica.to(tl.int64)[None, :]
        state = tl.load(state_ptr + offset, mask=lane_mask, other=0).to(tl.int32)

        # The output pointer is signed int32 for PyTorch compatibility, but
        # each element is deliberately a raw uint32 bit pattern. Summation is
        # equivalent to OR because every infectious lane contributes a unique
        # power of two, including bit 31.
        bit_value = tl.full([32], 1, tl.uint32) << bit
        packed = tl.sum(tl.where(state == 2, bit_value[None, :], 0), axis=1)
        output_offset = node.to(tl.int64) * words + word
        tl.store(
            infectious_mask_ptr + output_offset,
            packed,
            mask=node < N,
        )

    @triton.jit
    def _ensemble_csr_gather_kernel(
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        state_ptr,
        payload_ptr,
        output_ptr,
        beta,
        N,
        R,
        INDUCER_STATE: tl.constexpr,
        PAYLOAD_MODE: tl.constexpr,
        HAS_WEIGHTS: tl.constexpr,
        ACCUMULATE: tl.constexpr,
        NODES_PER_PROGRAM: tl.constexpr,
        REPLICAS_PER_TILE: tl.constexpr,
    ):
        """Gather a state-filtered or floating source payload over one CSR."""
        # Replica tiles are the fast coordinate of the flattened 1-D launch.
        # Derive their count from runtime R: exact ensemble cardinality does
        # not become a constexpr merely to decode the program id.
        replica_tiles = tl.cdiv(R, REPLICAS_PER_TILE)
        program = tl.program_id(0)
        node_program = program // replica_tiles
        replica_program = program - node_program * replica_tiles
        node = node_program * NODES_PER_PROGRAM + tl.arange(0, NODES_PER_PROGRAM)
        replica = replica_program * REPLICAS_PER_TILE + tl.arange(0, REPLICAS_PER_TILE)
        node_mask = node < N
        replica_mask = replica < R

        row_start = tl.load(row_ptr_ptr + node, mask=node_mask, other=0)
        row_end = tl.load(row_ptr_ptr + node + 1, mask=node_mask, other=0)
        edge = row_start
        pressure = tl.zeros([NODES_PER_PROGRAM, REPLICAS_PER_TILE], dtype=tl.float32)

        any_edge = tl.max((edge < row_end).to(tl.int32), axis=0)
        while any_edge != 0:
            edge_mask = node_mask & (edge < row_end)
            neighbor = tl.load(col_ind_ptr + edge, mask=edge_mask, other=0)

            # int64 pointer arithmetic keeps N*R addressable even though the
            # package's CSR indices themselves are intentionally int32.
            source_offset = neighbor.to(tl.int64)[:, None] * R + replica.to(tl.int64)[None, :]
            lane_mask = edge_mask[:, None] & replica_mask[None, :]
            if PAYLOAD_MODE:
                source_value = tl.load(
                    payload_ptr + source_offset,
                    mask=lane_mask,
                    other=0.0,
                ).to(tl.float32)
            else:
                source_state = tl.load(
                    state_ptr + source_offset,
                    mask=lane_mask,
                    other=INDUCER_STATE + 1,
                ).to(tl.int32)
                source_value = tl.where(source_state == INDUCER_STATE, beta, 0.0)

            if HAS_WEIGHTS:
                # Shape [nodes, 1]: one edge-weight load is broadcast across
                # all replica lanes rather than reissued R_TILE times.
                weight = tl.load(
                    weights_ptr + edge,
                    mask=edge_mask,
                    other=0.0,
                ).to(tl.float32)[:, None]
                source_value *= weight
            pressure += tl.where(lane_mask, source_value, 0.0)

            edge += 1
            any_edge = tl.max((edge < row_end).to(tl.int32), axis=0)

        output_offset = node.to(tl.int64)[:, None] * R + replica.to(tl.int64)[None, :]
        output_mask = node_mask[:, None] & replica_mask[None, :]
        if ACCUMULATE:
            pressure += tl.load(
                output_ptr + output_offset,
                mask=output_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(output_ptr + output_offset, pressure, mask=output_mask)

    @triton.jit
    def _ensemble_seir_renewal_rate_tile(
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        state_ptr,
        infectious_mask_ptr,
        age_ptr,
        node,
        replica,
        node_mask,
        replica_mask,
        beta_value,
        mu_ei_value,
        sig_ei_value,
        mu_ir_value,
        sig_ir_value,
        N,
        R,
        MASK_WORDS,
        replica_program,
        HAS_WEIGHTS: tl.constexpr,
        USE_INFECTIOUS_MASK: tl.constexpr,
        TRANSMISSION_AGE_DEPENDENT: tl.constexpr,
        REPLICAS_PER_TILE: tl.constexpr,
    ):
        """Evaluate one node/replica tile and return built-in SEIR rates."""
        lane_mask = node_mask[:, None] & replica_mask[None, :]
        offset = node.to(tl.int64)[:, None] * R + replica.to(tl.int64)[None, :]

        # Built-in SEIR compartment ids are deliberately fixed here. Keeping
        # them compile-time constants removes four runtime controls from the
        # package's model-specific ensemble fast path.
        state = tl.load(
            state_ptr + offset,
            mask=lane_mask,
            other=3,
        ).to(tl.int32)
        susceptible = lane_mask & (state == 0)
        exposed = lane_mask & (state == 1)
        infected = lane_mask & (state == 2)
        needs_age = exposed | infected
        age = tl.load(age_ptr + offset, mask=needs_age, other=0.0).to(tl.float32)

        # Row pointers, neighbor indices and optional weights are loaded once
        # per target row and broadcast over the contiguous replica tile. More
        # importantly, E/I/R-only row tiles do not touch the CSR at all.
        row_has_susceptible = tl.max(susceptible.to(tl.int32), axis=1) != 0
        active_row = node_mask & row_has_susceptible
        row_start = tl.load(row_ptr_ptr + node, mask=active_row, other=0)
        row_end = tl.load(row_ptr_ptr + node + 1, mask=active_row, other=0)
        edge = row_start
        pressure = tl.zeros_like(susceptible).to(tl.float32)

        any_edge = tl.max((active_row & (edge < row_end)).to(tl.int32), axis=0)
        while any_edge != 0:
            edge_mask = active_row & (edge < row_end)
            neighbor = tl.load(col_ind_ptr + edge, mask=edge_mask, other=0)
            source_offset = neighbor.to(tl.int64)[:, None] * R + replica.to(tl.int64)[None, :]
            source_lane = edge_mask[:, None] & susceptible
            if USE_INFECTIOUS_MASK:
                # REPLICAS_PER_TILE is a power-of-two divisor of 32, so each
                # tile lies wholly within one bitmap word. Load that word once
                # per edge/target row and broadcast its bits over the replica
                # lanes instead of loading one int32 source state per lane.
                mask_word = (replica_program * REPLICAS_PER_TILE) // 32
                packed = tl.load(
                    infectious_mask_ptr
                    + neighbor.to(tl.int64) * MASK_WORDS
                    + mask_word,
                    mask=edge_mask,
                    other=0,
                ).to(tl.uint32)
                bit = (replica - mask_word * 32).to(tl.uint32)
                infectious_source = source_lane & (
                    ((packed[:, None] >> bit[None, :]) & 1) != 0
                )
            else:
                source_state = tl.load(
                    state_ptr + source_offset,
                    mask=source_lane,
                    other=0,
                ).to(tl.int32)
                infectious_source = source_lane & (source_state == 2)
            if TRANSMISSION_AGE_DEPENDENT:
                source_age = tl.load(
                    age_ptr + source_offset,
                    mask=infectious_source,
                    other=0.0,
                ).to(tl.float32)
                source_value = tl.where(
                    infectious_source,
                    beta_value * _lognormal_hazard_triton(source_age, mu_ir_value, sig_ir_value),
                    0.0,
                )
            else:
                source_value = tl.where(
                    infectious_source,
                    beta_value,
                    0.0,
                )

            if HAS_WEIGHTS:
                # Shape [nodes, 1], hence one edge load per replica tile.
                weight = tl.load(
                    weights_ptr + edge,
                    mask=edge_mask,
                    other=0.0,
                ).to(tl.float32)[:, None]
                source_value *= weight
            pressure += source_value

            edge += 1
            any_edge = tl.max((active_row & (edge < row_end)).to(tl.int32), axis=0)

        # E and I share the log-normal formula, so choose their parameters per
        # lane and evaluate it once rather than computing two dense hazards.
        transition_mu = tl.where(exposed, mu_ei_value, mu_ir_value)
        transition_sigma = tl.where(exposed, sig_ei_value, sig_ir_value)
        transition_hazard = _lognormal_hazard_triton(age, transition_mu, transition_sigma)
        rate = tl.where(susceptible, pressure, 0.0)
        return tl.where(exposed | infected, transition_hazard, rate)

    @triton.jit
    def _ensemble_seir_renewal_rate_kernel(
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        state_ptr,
        infectious_mask_ptr,
        age_ptr,
        rates_ptr,
        min_rate_partials_ptr,
        max_rate_partials_ptr,
        beta_ptr,
        mu_ei_ptr,
        sig_ei_ptr,
        mu_ir_ptr,
        sig_ir_ptr,
        beta,
        mu_ei,
        sig_ei,
        mu_ir,
        sig_ir,
        N,
        R,
        MASK_WORDS,
        PARAMS_ON_DEVICE: tl.constexpr,
        HAS_WEIGHTS: tl.constexpr,
        USE_INFECTIOUS_MASK: tl.constexpr,
        TRANSMISSION_AGE_DEPENDENT: tl.constexpr,
        EMIT_RATE_BOUNDS: tl.constexpr,
        RATE_TILES_PER_PARTIAL: tl.constexpr,
        NODES_PER_PROGRAM: tl.constexpr,
        REPLICAS_PER_TILE: tl.constexpr,
    ):
        """Fuse incoming-CSR rates with optional compact min/max emission."""
        replica_tiles = tl.cdiv(R, REPLICAS_PER_TILE)
        program = tl.program_id(0)
        node_partial = program // replica_tiles
        replica_program = program - node_partial * replica_tiles
        replica = replica_program * REPLICAS_PER_TILE + tl.arange(0, REPLICAS_PER_TILE)
        replica_mask = replica < R

        if PARAMS_ON_DEVICE:
            beta_value = tl.load(beta_ptr).to(tl.float32)
            mu_ei_value = tl.load(mu_ei_ptr).to(tl.float32)
            sig_ei_value = tl.load(sig_ei_ptr).to(tl.float32)
            mu_ir_value = tl.load(mu_ir_ptr).to(tl.float32)
            sig_ir_value = tl.load(sig_ir_ptr).to(tl.float32)
        else:
            beta_value = beta
            mu_ei_value = mu_ei
            sig_ei_value = sig_ei
            mu_ir_value = mu_ir
            sig_ir_value = sig_ir

        partial_min = tl.full([REPLICAS_PER_TILE], float("inf"), tl.float32)
        partial_max = tl.full([REPLICAS_PER_TILE], -float("inf"), tl.float32)
        partial_nonfinite = tl.zeros([REPLICAS_PER_TILE], tl.int32)

        # Keep this as a compact loop rather than cloning the ragged CSR body.
        # At the default node tile, sixteen iterations emit one pair of bounds
        # for the same 128-node granularity used by transition event partials.
        for rate_tile in tl.range(
            0,
            RATE_TILES_PER_PARTIAL,
            loop_unroll_factor=1,
        ):
            node_program = (
                node_partial.to(tl.int64) * RATE_TILES_PER_PARTIAL + rate_tile
            )
            node = node_program * NODES_PER_PROGRAM + tl.arange(0, NODES_PER_PROGRAM)
            node_mask = node < N
            lane_mask = node_mask[:, None] & replica_mask[None, :]
            rate = _ensemble_seir_renewal_rate_tile(
                row_ptr_ptr,
                col_ind_ptr,
                weights_ptr,
                state_ptr,
                infectious_mask_ptr,
                age_ptr,
                node,
                replica,
                node_mask,
                replica_mask,
                beta_value,
                mu_ei_value,
                sig_ei_value,
                mu_ir_value,
                sig_ir_value,
                N,
                R,
                MASK_WORDS,
                replica_program,
                HAS_WEIGHTS=HAS_WEIGHTS,
                USE_INFECTIOUS_MASK=USE_INFECTIOUS_MASK,
                TRANSMISSION_AGE_DEPENDENT=TRANSMISSION_AGE_DEPENDENT,
                REPLICAS_PER_TILE=REPLICAS_PER_TILE,
            )
            offset = node.to(tl.int64)[:, None] * R + replica.to(tl.int64)[None, :]
            tl.store(rates_ptr + offset, rate, mask=lane_mask)

            if EMIT_RATE_BOUNDS:
                # Triton reductions need not propagate NaNs. Encode any NaN or
                # infinity as (-inf,+inf); the established finalizer assigns
                # severity 3 before negatives or invalid tau. Public rates keep
                # the original nonfinite value for diagnostics.
                finite = (
                    (rate == rate)
                    & (rate != float("inf"))
                    & (rate != -float("inf"))
                )
                safe_rate = tl.where(finite, rate, 0.0)
                tile_min = tl.min(
                    tl.where(lane_mask, safe_rate, float("inf")),
                    axis=0,
                )
                tile_max = tl.max(
                    tl.where(lane_mask, safe_rate, -float("inf")),
                    axis=0,
                )
                tile_nonfinite = tl.max(
                    (lane_mask & ~finite).to(tl.int32),
                    axis=0,
                )
                partial_min = tl.minimum(partial_min, tile_min)
                partial_max = tl.maximum(partial_max, tile_max)
                partial_nonfinite = tl.maximum(partial_nonfinite, tile_nonfinite)

        if EMIT_RATE_BOUNDS:
            partial_min = tl.where(
                partial_nonfinite != 0,
                -float("inf"),
                partial_min,
            )
            partial_max = tl.where(
                partial_nonfinite != 0,
                float("inf"),
                partial_max,
            )
            partial_offset = node_partial.to(tl.int64) * R + replica.to(tl.int64)
            tl.store(
                min_rate_partials_ptr + partial_offset,
                partial_min,
                mask=replica_mask,
            )
            tl.store(
                max_rate_partials_ptr + partial_offset,
                partial_max,
                mask=replica_mask,
            )


else:
    _pack_ensemble_infectious_mask_kernel = None
    _ensemble_csr_gather_kernel = None
    _ensemble_seir_renewal_rate_tile = None
    _ensemble_seir_renewal_rate_kernel = None


def _positive_power_of_two(name: str, value: int, *, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value <= 0 or value & (value - 1) or value > maximum:
        raise ValueError(f"{name} must be a positive power of two no larger than {maximum}")
    return value


def _default_replica_tile(replicas: int) -> int:
    return min(32, 1 << max(0, math.ceil(math.log2(replicas))))


def _rate_bound_partial_shape(num_nodes: int, replicas: int) -> tuple[int, int]:
    """Shape of the two compact production rate-bound arrays."""
    return (
        (num_nodes + _RATE_BOUND_NODES_PER_PARTIAL - 1)
        // _RATE_BOUND_NODES_PER_PARTIAL,
        replicas,
    )


def _flat_ensemble_grid(
    num_nodes: int,
    replicas: int,
    nodes_per_program: int,
    replicas_per_tile: int,
) -> tuple[int]:
    """Return a 1-D launch grid for the logical node-by-replica tile space."""
    node_tiles = (num_nodes + nodes_per_program - 1) // nodes_per_program
    replica_tiles = (replicas + replicas_per_tile - 1) // replicas_per_tile
    num_programs = node_tiles * replica_tiles
    if num_programs > (1 << 31) - 1:
        raise ValueError("the flattened ensemble grid exceeds CUDA grid-x capacity")
    return (num_programs,)


def _normalize_inducer_states(inducer_states) -> tuple[int, ...]:
    """Normalize one or more distinct non-negative int32 compartment ids."""
    if isinstance(inducer_states, bool):
        raise TypeError("inducer_states must contain integer compartment ids")
    try:
        states = (operator.index(inducer_states),)
    except TypeError:
        if isinstance(inducer_states, (str, bytes)):
            raise ValueError("inducer_states must contain integer ids") from None
        else:
            try:
                raw_states = tuple(inducer_states)
            except TypeError as exc:
                raise TypeError(
                    "inducer_states must be an integer or iterable of integers"
                ) from exc
        if not raw_states:
            raise ValueError("inducer_states must contain at least one state")
        if any(isinstance(state, bool) for state in raw_states):
            raise TypeError("inducer_states must contain integer compartment ids")
        try:
            states = tuple(operator.index(state) for state in raw_states)
        except TypeError as exc:
            raise TypeError("inducer_states must contain integer compartment ids") from exc
    if len(set(states)) != len(states):
        raise ValueError("inducer_states must not contain duplicates")
    if any(not 0 <= state <= torch.iinfo(torch.int32).max for state in states):
        raise ValueError("inducer_states must contain non-negative int32 values")
    return states


def _reject_output_alias(out: torch.Tensor, inputs: tuple[torch.Tensor, ...]) -> None:
    """Reject shared backing storage before an asynchronous gather launch."""
    out_storage = out.untyped_storage().data_ptr()
    if any(
        tensor.untyped_storage().data_ptr() == out_storage
        for tensor in inputs
        if tensor.untyped_storage().nbytes()
    ):
        raise ValueError("out must not share storage with an input or CSR tensor")


def _validate_gpu_inputs(
    graph,
    values: torch.Tensor,
    *,
    name: str,
    floating: bool,
    out: torch.Tensor | None,
) -> tuple[object, torch.Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError("Triton is required for ensemble GPU gathers") from (
            _TRITON_IMPORT_ERROR
        )
    graph = as_csr(graph)
    if graph.device.type != "cuda":
        raise ValueError("ensemble GPU gathers require a CUDA graph")
    if values.device != graph.device:
        raise ValueError(f"{name} and graph must be on the same CUDA device")
    if values.dim() != 2 or values.shape[0] != graph.num_nodes or values.shape[1] <= 0:
        raise ValueError(
            f"{name} must have shape [{graph.num_nodes}, replicas], got {tuple(values.shape)}"
        )
    if not values.is_contiguous():
        raise ValueError(f"{name} must be contiguous in node-major [N, R] layout")
    if floating:
        if not values.dtype.is_floating_point:
            raise TypeError(f"{name} must use a floating-point dtype")
    elif values.dtype == torch.bool or values.dtype.is_floating_point or values.dtype.is_complex:
        raise TypeError(f"{name} must use an integer dtype")

    shape = tuple(values.shape)
    if out is None:
        out = torch.empty(shape, device=values.device, dtype=torch.float32)
    elif (
        tuple(out.shape) != shape
        or out.device != values.device
        or out.dtype != torch.float32
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous float32 with the same [N, R] shape")
    # In payload mode `out=values` is shape/dtype compatible but incorrect:
    # targets may overwrite values that later rows still need as sources. A
    # shared backing storage is rejected even for apparently disjoint views so
    # the asynchronous kernel can never observe an aliasing race.
    _reject_output_alias(
        out,
        (
            values,
            graph.row_ptr,
            graph.col_ind,
            graph.weights_storage,
        ),
    )
    return graph, out


def _launch_gather(
    graph,
    values: torch.Tensor,
    out: torch.Tensor,
    *,
    payload_mode: bool,
    inducer_state: int,
    beta: float,
    accumulate: bool,
    nodes_per_program: int,
    replicas_per_tile: int | None,
) -> torch.Tensor:
    n, replicas = values.shape
    nodes_per_program = _positive_power_of_two("nodes_per_program", nodes_per_program, maximum=32)
    if replicas_per_tile is None:
        replicas_per_tile = _default_replica_tile(replicas)
    replicas_per_tile = _positive_power_of_two("replicas_per_tile", replicas_per_tile, maximum=32)
    if nodes_per_program * replicas_per_tile > 512:
        raise ValueError(
            "nodes_per_program * replicas_per_tile must be <= 512 to bound "
            "the fp32 accumulator tile"
        )
    if n == 0:
        return out

    grid = _flat_ensemble_grid(
        n,
        replicas,
        nodes_per_program,
        replicas_per_tile,
    )
    # The inactive pointer is compile-time dead. Reusing values avoids even a
    # scalar dummy allocation and keeps capture-time addresses static.
    with torch.cuda.device(graph.device):
        _ensemble_csr_gather_kernel[grid](
            row_ptr_ptr=graph.row_ptr,
            col_ind_ptr=graph.col_ind,
            weights_ptr=graph.weights_storage,
            state_ptr=values,
            payload_ptr=values,
            output_ptr=out,
            beta=beta,
            N=n,
            R=replicas,
            INDUCER_STATE=inducer_state,
            PAYLOAD_MODE=1 if payload_mode else 0,
            HAS_WEIGHTS=graph.has_weights,
            ACCUMULATE=1 if accumulate else 0,
            NODES_PER_PROGRAM=nodes_per_program,
            REPLICAS_PER_TILE=replicas_per_tile,
        )
    return out


def ensemble_influence_csr(
    graph,
    state: torch.Tensor,
    inducer_state,
    *,
    beta: float = 1.0,
    out: torch.Tensor | None = None,
    nodes_per_program: int = 8,
    replicas_per_tile: int | None = None,
) -> torch.Tensor:
    """GPU state-filtered influence for contiguous ``state[N, replicas]``.

    ``inducer_state`` may be one compartment id or an iterable of distinct ids.
    Multiple ids are accumulated through ordered launches into the same output,
    avoiding the reference path's ``[E, replicas]`` contribution temporary.
    """
    inducer_states = _normalize_inducer_states(inducer_state)
    beta = validate_fp32_control("beta", beta, nonnegative=True)
    graph, out = _validate_gpu_inputs(graph, state, name="state", floating=False, out=out)
    for index, state_id in enumerate(inducer_states):
        _launch_gather(
            graph,
            state,
            out,
            payload_mode=False,
            inducer_state=state_id,
            beta=beta,
            accumulate=index != 0,
            nodes_per_program=nodes_per_program,
            replicas_per_tile=replicas_per_tile,
        )
    return out


def ensemble_infectivity_csr(
    graph,
    infectivity: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    nodes_per_program: int = 8,
    replicas_per_tile: int | None = None,
) -> torch.Tensor:
    """GPU floating-payload influence for contiguous ``[node, replica]``.

    The wrapper deliberately does not reduce the payload to check finiteness:
    that would add a device synchronization to a hot path. Model construction
    and rate validation own that invariant; non-finite values remain visible in
    the output instead of being silently clamped.
    """
    graph, out = _validate_gpu_inputs(
        graph,
        infectivity,
        name="infectivity",
        floating=True,
        out=out,
    )
    return _launch_gather(
        graph,
        infectivity,
        out,
        payload_mode=True,
        inducer_state=0,
        beta=1.0,
        accumulate=False,
        nodes_per_program=nodes_per_program,
        replicas_per_tile=replicas_per_tile,
    )


def _normalize_seir_rate_parameters(
    device: torch.device,
    *,
    beta,
    mu_ei,
    sig_ei,
    mu_ir,
    sig_ir,
) -> tuple[bool, tuple[torch.Tensor, ...], tuple[float, ...]]:
    """Accept either five host controls or five prepared device scalars."""
    names = ("beta", "mu_ei", "sig_ei", "mu_ir", "sig_ir")
    values = (beta, mu_ei, sig_ei, mu_ir, sig_ir)
    tensor_flags = tuple(isinstance(value, torch.Tensor) for value in values)
    if any(tensor_flags) and not all(tensor_flags):
        raise TypeError(
            "beta, mu_ei, sig_ei, mu_ir, and sig_ir must be either all host "
            "scalars or all device scalar tensors"
        )
    if all(tensor_flags):
        tensors = values
        for name, value in zip(names, tensors, strict=True):
            if (
                value.dim() != 0
                or value.device != device
                or value.dtype != torch.float32
                or value.layout != torch.strided
            ):
                raise ValueError(f"{name} must be a scalar float32 strided tensor on {device}")
        # Prepared model tensors were validated before being copied to the
        # device. Avoiding `.item()` here is essential: it keeps rate evaluation
        # asynchronous and CUDA-Graph friendly. Direct callers that need value
        # validation should pass host controls instead.
        return True, tensors, (0.0,) * len(values)

    host_values = (
        validate_fp32_control("beta", beta, nonnegative=True),
        validate_fp32_control("mu_ei", mu_ei),
        validate_fp32_control("sig_ei", sig_ei, positive=True),
        validate_fp32_control("mu_ir", mu_ir),
        validate_fp32_control("sig_ir", sig_ir, positive=True),
    )
    return False, (), host_values


def pack_ensemble_infectious_mask(
    state: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    nodes_per_program: int = 16,
) -> torch.Tensor:
    """Pack built-in SEIR infectious lanes from ``state[N, R]``.

    The result is contiguous int32 with shape ``[N, ceil(R / 32)]``. Its
    signed elements are raw 32-bit patterns: bit ``r % 32`` is set exactly
    when replica ``r`` is in the built-in infectious compartment ``I=2``.
    Keeping the storage signed makes it directly representable by PyTorch;
    consumers interpret the bits as unsigned.

    The caller owns freshness. Re-run this helper after arbitrary external
    mutation of ``state`` before passing the mask to
    :func:`ensemble_seir_renewal_rates_csr`. A steady-state transition kernel
    may instead maintain the same bit pattern incrementally.
    """
    if not _HAS_TRITON or _pack_ensemble_infectious_mask_kernel is None:
        raise RuntimeError(
            "Triton renewal kernels are required to pack ensemble state"
        ) from _TRITON_IMPORT_ERROR
    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch.Tensor")
    if state.dim() != 2 or state.shape[1] <= 0:
        raise ValueError(f"state must have shape [nodes, replicas], got {tuple(state.shape)}")
    if state.device.type != "cuda":
        raise ValueError("packing ensemble state requires a CUDA tensor")
    if state.dtype != torch.int32:
        raise TypeError("state must use dtype torch.int32")
    if not state.is_contiguous():
        raise ValueError("state must be contiguous in node-major [N, R] layout")

    num_nodes, replicas = state.shape
    words = (replicas + 31) // 32
    shape = (num_nodes, words)
    if out is None:
        out = torch.empty(shape, device=state.device, dtype=torch.int32)
    elif not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor or None")
    elif (
        tuple(out.shape) != shape
        or out.device != state.device
        or out.dtype != torch.int32
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous int32 on the state device with shape "
            f"[{num_nodes}, {words}]"
        )
    _reject_output_alias(out, (state,))

    nodes_per_program = _positive_power_of_two(
        "nodes_per_program", nodes_per_program, maximum=32
    )
    if num_nodes == 0:
        return out
    grid = _flat_ensemble_grid(num_nodes, words, nodes_per_program, 1)
    with torch.cuda.device(state.device):
        _pack_ensemble_infectious_mask_kernel[grid](
            state_ptr=state,
            infectious_mask_ptr=out,
            N=num_nodes,
            R=replicas,
            NODES_PER_PROGRAM=nodes_per_program,
        )
    return out


def ensemble_seir_renewal_rates_csr(
    graph,
    state: torch.Tensor,
    age: torch.Tensor,
    *,
    beta,
    mu_ei,
    sig_ei,
    mu_ir,
    sig_ir,
    transmission_age_dependent: bool = False,
    infectious_mask: torch.Tensor | None = None,
    rate_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
    nodes_per_program: int = 8,
    replicas_per_tile: int | None = None,
) -> torch.Tensor:
    """Evaluate built-in SEIR renewal rates over ``[node, replica]`` tensors.

    Compartment ids are the built-in ``S=0, E=1, I=2, R=3``. Susceptible rates
    are an incoming weighted CSR sum. In constant mode each infectious source
    contributes ``beta``; in age-dependent mode it contributes
    ``beta * h_IR(source_age)``. Exposed and infectious targets receive their
    respective log-normal exit hazards, and all other target states receive
    zero.

    The five model parameters may be finite fp32-representable host scalars, or
    they may all be prepared scalar float32 tensors on the graph's CUDA device.
    Device tensors avoid a host synchronization and are therefore assumed to
    have been value-validated when prepared by the model.

    ``infectious_mask`` may be a fresh packed result from
    :func:`pack_ensemble_infectious_mask`. When supplied, source-compartment
    tests load one int32 bit pattern per edge and 32-replica word instead of
    one int32 state per source lane. The target states are still read from
    ``state``. This optimization applies to both constant and age-dependent
    transmission; the latter still gathers ages for infectious source lanes.

    ``rate_bounds=(minimum_partials, maximum_partials)`` is the production-step
    reduction path. Both buffers must be contiguous fp32 with shape
    ``[ceil(N / 128), R]``. The same kernel still writes every public rate, but
    also emits one min/max pair per 128-node group. A later reduction therefore
    reads only ``2 * ceil(N / 128) * R`` values instead of rereading ``N * R``
    rates. Any nonfinite lane is encoded as ``(-inf, +inf)`` in its partial so
    the established transactional finalizer retains severity precedence.
    """
    if not _HAS_TRITON or _ensemble_seir_renewal_rate_kernel is None:
        raise RuntimeError(
            "Triton renewal kernels are required for ensemble SEIR rates"
        ) from _TRITON_IMPORT_ERROR
    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch.Tensor")
    if not isinstance(age, torch.Tensor):
        raise TypeError("age must be a torch.Tensor")
    if not isinstance(transmission_age_dependent, bool):
        raise TypeError("transmission_age_dependent must be a bool")
    if infectious_mask is not None and not isinstance(infectious_mask, torch.Tensor):
        raise TypeError("infectious_mask must be a torch.Tensor or None")

    graph = as_csr(graph)
    if graph.device.type != "cuda":
        raise ValueError("ensemble SEIR rates require a CUDA graph")
    if state.device != graph.device or age.device != graph.device:
        raise ValueError("state, age, and graph must be on the same CUDA device")
    if state.dim() != 2 or state.shape[0] != graph.num_nodes or state.shape[1] <= 0:
        raise ValueError(
            f"state must have shape [{graph.num_nodes}, replicas], got {tuple(state.shape)}"
        )
    if tuple(age.shape) != tuple(state.shape):
        raise ValueError(
            "age must have the same node-major [N, R] shape as state, got "
            f"{tuple(age.shape)} and {tuple(state.shape)}"
        )
    if state.dtype != torch.int32:
        raise TypeError("state must use dtype torch.int32")
    if age.dtype != torch.float32:
        raise TypeError("age must use dtype torch.float32")
    if not state.is_contiguous() or not age.is_contiguous():
        raise ValueError("state and age must be contiguous in node-major [N, R] layout")

    replicas = state.shape[1]
    mask_words = (replicas + 31) // 32
    if infectious_mask is not None:
        expected_mask_shape = (graph.num_nodes, mask_words)
        if infectious_mask.device != graph.device:
            raise ValueError("infectious_mask and graph must be on the same CUDA device")
        if tuple(infectious_mask.shape) != expected_mask_shape:
            raise ValueError(
                "infectious_mask must have packed shape "
                f"{expected_mask_shape}, got {tuple(infectious_mask.shape)}"
            )
        if infectious_mask.dtype != torch.int32:
            raise TypeError("infectious_mask must use dtype torch.int32")
        if not infectious_mask.is_contiguous():
            raise ValueError("infectious_mask must be contiguous in [N, ceil(R / 32)] layout")

    shape = tuple(state.shape)
    if out is None:
        out = torch.empty(shape, device=graph.device, dtype=torch.float32)
    elif not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor or None")
    elif (
        tuple(out.shape) != shape
        or out.device != graph.device
        or out.dtype != torch.float32
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous float32 with the same [N, R] shape")

    nodes_per_program = _positive_power_of_two("nodes_per_program", nodes_per_program, maximum=32)
    if replicas_per_tile is None:
        replicas_per_tile = _default_replica_tile(replicas)
    replicas_per_tile = _positive_power_of_two("replicas_per_tile", replicas_per_tile, maximum=32)
    if nodes_per_program * replicas_per_tile > 512:
        raise ValueError(
            "nodes_per_program * replicas_per_tile must be <= 512 to bound "
            "the fp32 accumulator tile"
        )

    bound_outputs: tuple[torch.Tensor, ...]
    if rate_bounds is None:
        bound_outputs = ()
    else:
        if isinstance(rate_bounds, torch.Tensor):
            raise TypeError("rate_bounds must be a pair of tensors or None")
        try:
            bound_outputs = tuple(rate_bounds)
        except TypeError as exc:
            raise TypeError("rate_bounds must be a pair of tensors or None") from exc
        if len(bound_outputs) != 2 or not all(
            isinstance(bound, torch.Tensor) for bound in bound_outputs
        ):
            raise TypeError("rate_bounds must be a pair of tensors or None")
        expected_bound_shape = _rate_bound_partial_shape(graph.num_nodes, replicas)
        for name, bound in zip(("minimum", "maximum"), bound_outputs):
            if bound.device != graph.device:
                raise ValueError(f"{name} rate-bound partials must be on the graph device")
            if bound.dtype != torch.float32:
                raise TypeError(f"{name} rate-bound partials must use dtype torch.float32")
            if tuple(bound.shape) != expected_bound_shape or not bound.is_contiguous():
                raise ValueError(
                    f"{name} rate-bound partials must be contiguous with shape "
                    f"{expected_bound_shape}"
                )

    params_on_device, device_params, host_params = _normalize_seir_rate_parameters(
        graph.device,
        beta=beta,
        mu_ei=mu_ei,
        sig_ei=sig_ei,
        mu_ir=mu_ir,
        sig_ir=sig_ir,
    )
    parameter_inputs = device_params if params_on_device else ()
    input_tensors = (
        state,
        age,
        graph.row_ptr,
        graph.col_ind,
        graph.weights_storage,
        *((infectious_mask,) if infectious_mask is not None else ()),
        *parameter_inputs,
    )
    all_outputs = (out, *bound_outputs)
    for index, output in enumerate(all_outputs):
        _reject_output_alias(
            output,
            (*input_tensors, *all_outputs[:index], *all_outputs[index + 1 :]),
        )

    num_nodes = state.shape[0]
    if num_nodes == 0:
        return out
    parameter_ptrs = device_params if params_on_device else (age,) * 5
    infectious_mask_ptr = state if infectious_mask is None else infectious_mask
    beta_value, mu_ei_value, sig_ei_value, mu_ir_value, sig_ir_value = host_params
    emit_rate_bounds = bool(bound_outputs)
    rate_tiles_per_partial = (
        _RATE_BOUND_NODES_PER_PARTIAL // nodes_per_program
        if emit_rate_bounds
        else 1
    )
    min_rate_partials, max_rate_partials = (
        bound_outputs if emit_rate_bounds else (out, out)
    )
    grid = _flat_ensemble_grid(
        num_nodes,
        replicas,
        nodes_per_program * rate_tiles_per_partial,
        replicas_per_tile,
    )
    with torch.cuda.device(graph.device):
        _ensemble_seir_renewal_rate_kernel[grid](
            row_ptr_ptr=graph.row_ptr,
            col_ind_ptr=graph.col_ind,
            weights_ptr=graph.weights_storage,
            state_ptr=state,
            infectious_mask_ptr=infectious_mask_ptr,
            age_ptr=age,
            rates_ptr=out,
            min_rate_partials_ptr=min_rate_partials,
            max_rate_partials_ptr=max_rate_partials,
            beta_ptr=parameter_ptrs[0],
            mu_ei_ptr=parameter_ptrs[1],
            sig_ei_ptr=parameter_ptrs[2],
            mu_ir_ptr=parameter_ptrs[3],
            sig_ir_ptr=parameter_ptrs[4],
            beta=beta_value,
            mu_ei=mu_ei_value,
            sig_ei=sig_ei_value,
            mu_ir=mu_ir_value,
            sig_ir=sig_ir_value,
            N=num_nodes,
            R=replicas,
            MASK_WORDS=mask_words,
            PARAMS_ON_DEVICE=1 if params_on_device else 0,
            HAS_WEIGHTS=graph.has_weights,
            USE_INFECTIOUS_MASK=1 if infectious_mask is not None else 0,
            TRANSMISSION_AGE_DEPENDENT=(1 if transmission_age_dependent else 0),
            EMIT_RATE_BOUNDS=1 if emit_rate_bounds else 0,
            RATE_TILES_PER_PARTIAL=rate_tiles_per_partial,
            NODES_PER_PROGRAM=nodes_per_program,
            REPLICAS_PER_TILE=replicas_per_tile,
        )
    return out


__all__ = [
    "ensemble_influence_csr",
    "ensemble_infectivity_csr",
    "ensemble_seir_renewal_rates_csr",
    "pack_ensemble_infectious_mask",
    "reference_ensemble_influence_csr",
    "reference_ensemble_infectivity_csr",
]
