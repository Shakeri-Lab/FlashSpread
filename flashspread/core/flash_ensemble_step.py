"""Triton primitives for the built-in renewal-SEIR ensemble step tail.

The rate phase writes current-state rates in node-major ``[N, R]`` layout.
This module supplies the two ordered operations that follow it:

1. validate each current rate range and select a private ``tau`` candidate;
2. sample the fixed ``S -> E -> I -> R`` transitions and advance node ages.

Neither kernel owns a simulation clock or advances the RNG step id.  The
engine can therefore inspect the per-replica invalid flags and commit clocks,
the step id, and reduced event counts only after the whole ensemble is valid.
The transition kernel independently validates every tau so direct low-level
use cannot mutate an invalid replica.
"""

# NOTE: deliberately no `from __future__ import annotations`. Triton's
# interpreter resolves constexpr parameters with an exact
# `_normalize_ty(annotation) == "constexpr"` comparison, so a stringified
# `"tl.constexpr"` annotation stops being recognized and `tl.arange` then
# rejects its bounds. The compiled path uses a substring test and is
# unaffected, which is why this only ever broke interpreter-mode tests.

import math
import operator

import torch

from .flash_rng import (
    _event_probability,
    _mix_rng_key,
    _sample_bernoulli_counter,
)
from ..utils import validate_fp32_control

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

    @triton.jit
    def _ensemble_finalize_renewal_tau_kernel(
        min_rate_ptr,
        max_rate_ptr,
        tau_candidate_ptr,
        invalid_ptr,
        epsilon,
        tau_max,
        R,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Select independent renewal steps and expose invalid replicas."""
        replica = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = replica < R
        min_rate = tl.load(
            min_rate_ptr + replica,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        max_rate = tl.load(
            max_rate_ptr + replica,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        candidate = epsilon / max_rate
        tau = tl.minimum(candidate, tau_max)
        fp32_max: tl.constexpr = 3.4028234663852886e38
        nonfinite_rate = (
            (min_rate != min_rate)
            | (max_rate != max_rate)
            | (min_rate > fp32_max)
            | (min_rate < -fp32_max)
            | (max_rate > fp32_max)
            | (max_rate < -fp32_max)
        )
        finite_negative_rate = (~nonfinite_rate) & (min_rate < 0.0)
        invalid_candidate = (
            (max_rate < 0.0)
            | (min_rate > max_rate)
            | ((max_rate == 0.0) & (min_rate != 0.0))
            | ((max_rate > 0.0) & ((tau != tau) | (tau <= 0.0)))
        )
        status = tl.where(invalid_candidate, 1, 0)
        status = tl.where(finite_negative_rate, 2, status)
        status = tl.where(nonfinite_rate, 3, status)

        # An exact zero rate is the valid absorbing-state case. Invalid lanes
        # are poisoned as well as flagged, making the transition kernel safe
        # even when called without first checking ``invalid`` on the host.
        tau = tl.where(max_rate == 0.0, tau_max, tau)
        zero = (max_rate - max_rate) * 0.0
        nan_value = zero / zero
        tau = tl.where(status != 0, nan_value, tau)
        tl.store(tau_candidate_ptr + replica, tau, mask=mask)
        tl.store(invalid_ptr + replica, status.to(tl.int32), mask=mask)

    @triton.jit
    def _ensemble_seir_transition_kernel(
        state_ptr,
        age_ptr,
        rates_ptr,
        tau_ptr,
        rng_seed_ptr,
        step_id_ptr,
        event_partials_ptr,
        infectious_mask_ptr,
        N,
        R,
        NUM_REPLICA_TILES,
        UPDATE_INFECTIOUS_MASK: tl.constexpr,
        NODES_PER_PROGRAM: tl.constexpr,
        REPLICAS_PER_TILE: tl.constexpr,
    ):
        """Sample one in-place fixed-state SEIR step for a replica tile."""
        # CUDA grid-x is bounded by signed int32, so decode the flat tile id
        # with cheap 32-bit division and widen only before node/replica math.
        flat_program = tl.program_id(0)
        replica_tile = flat_program % NUM_REPLICA_TILES
        node_block = flat_program // NUM_REPLICA_TILES
        node = node_block.to(tl.int64) * NODES_PER_PROGRAM + tl.arange(0, NODES_PER_PROGRAM).to(
            tl.int64
        )
        replica = replica_tile.to(tl.int64) * REPLICAS_PER_TILE + tl.arange(
            0, REPLICAS_PER_TILE
        ).to(tl.int64)
        node_mask = node < N
        replica_mask = replica < R
        lane_mask = node_mask[:, None] & replica_mask[None, :]
        offset = node.to(tl.int64)[:, None] * R + replica.to(tl.int64)[None, :]

        state = tl.load(
            state_ptr + offset,
            mask=lane_mask,
            other=3,
        ).to(tl.int32)
        age = tl.load(
            age_ptr + offset,
            mask=lane_mask,
            other=0.0,
        ).to(tl.float32)
        rate = tl.load(
            rates_ptr + offset,
            mask=lane_mask,
            other=0.0,
        ).to(tl.float32)
        tau = tl.load(
            tau_ptr + replica,
            mask=replica_mask,
            other=0.0,
        ).to(tl.float32)
        fp32_max: tl.constexpr = 3.4028234663852886e38
        valid_tau = (tau == tau) & (tau > 0.0) & (tau <= fp32_max)
        safe_tau = tl.where(valid_tau, tau, 0.0)

        probability = _event_probability(rate * safe_tau[None, :])
        base_seed = tl.load(rng_seed_ptr).to(tl.uint64)
        step_id = tl.load(step_id_ptr).to(tl.uint64)

        # One full-width key identifies the event seed and accepted simulation
        # step. Node and replica identities occupy two disjoint uint32 *counter
        # words*, hence every [N, R] lane has a unique stream position
        # independent of launch tiling. They must not be packed into one 64-bit
        # word: Philox picks its word width from the counter dtype, so a uint64
        # counter returns uint64 random words that no longer compare against the
        # uint32 threshold (see _bernoulli_from_words).
        key = _mix_rng_key(base_seed, step_id)
        node_counter = tl.broadcast_to(node.to(tl.uint32)[:, None], probability.shape)
        replica_counter = tl.broadcast_to(
            replica.to(tl.uint32)[None, :], probability.shape
        )
        event = _sample_bernoulli_counter(
            probability, key, node_counter, replica_counter
        )
        event &= lane_mask & valid_tau[None, :]

        is_s = state == 0
        is_e = state == 1
        is_i = state == 2
        new_state = tl.where(event & is_s, 1, state)
        new_state = tl.where(event & is_e, 2, new_state)
        new_state = tl.where(event & is_i, 3, new_state)
        changed = lane_mask & (new_state != state)
        new_age = tl.where(changed, 0.0, age + safe_tau[None, :])

        # Invalid tau lanes are a failed transaction: not even an idempotent
        # store reaches state or age. Their event partial is explicitly zero.
        mutation_mask = lane_mask & valid_tau[None, :]
        # State is unchanged for the overwhelmingly common no-event lane, so
        # avoid a redundant global write. Age is different: every accepted
        # replica advances unchanged lanes by tau and must therefore retain the
        # dense valid-lane store below.
        tl.store(state_ptr + offset, new_state, mask=changed & mutation_mask)
        tl.store(age_ptr + offset, new_age, mask=mutation_mask)

        if UPDATE_INFECTIOUS_MASK:
            # The persistent bitmap is node-major [N, ceil(R/32)]. Each lane
            # owns one bit. Sparse transition atomics avoid rereading all NR
            # states before the next graph phase; updates to distinct bits
            # commute even when several replica lanes share one word.
            mask_words = tl.cdiv(R, 32)
            mask_offset = node[:, None] * mask_words + (replica[None, :] >> 5)
            bit_u32 = (1 << (replica[None, :] & 31)).to(tl.uint32)
            bit_i32 = bit_u32.to(tl.int32, bitcast=True)
            entered_infectious = lane_mask & valid_tau[None, :] & event & is_e
            left_infectious = lane_mask & valid_tau[None, :] & event & is_i
            tl.atomic_or(
                infectious_mask_ptr + mask_offset,
                bit_i32,
                mask=entered_infectious,
            )
            tl.atomic_and(
                infectious_mask_ptr + mask_offset,
                ~bit_i32,
                mask=left_infectious,
            )

        block_events = tl.sum(changed.to(tl.int32), axis=0).to(tl.int32)
        block_events = tl.where(valid_tau, block_events, 0)
        partial_offset = node_block.to(tl.int64) * R + replica.to(tl.int64)
        tl.store(
            event_partials_ptr + partial_offset,
            block_events,
            mask=replica_mask,
        )


else:
    _ensemble_finalize_renewal_tau_kernel = None
    _ensemble_seir_transition_kernel = None


def _positive_power_of_two(name: str, value: int, *, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0 or result & (result - 1) or result > maximum:
        raise ValueError(f"{name} must be a positive power of two no larger than {maximum}")
    return result


def _default_replica_tile(replicas: int) -> int:
    return min(32, 1 << max(0, math.ceil(math.log2(replicas))))


def _require_tensor(name: str, value) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_cuda_triton() -> None:
    if not _HAS_TRITON:
        raise RuntimeError(
            "Triton is required for ensemble renewal step kernels"
        ) from _TRITON_IMPORT_ERROR


def _require_same_cuda_device(
    named_tensors: tuple[tuple[str, torch.Tensor], ...],
) -> torch.device:
    device = named_tensors[0][1].device
    if device.type != "cuda":
        raise ValueError("ensemble renewal step kernels require CUDA tensors")
    for name, tensor in named_tensors[1:]:
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")
    return device


def _require_contiguous(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must use dtype {dtype}")
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous strided tensor")


def _reject_storage_aliases(
    named_tensors: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    seen: dict[int, str] = {}
    for name, tensor in named_tensors:
        if tensor.untyped_storage().nbytes() == 0:
            continue
        pointer = tensor.untyped_storage().data_ptr()
        previous = seen.get(pointer)
        if previous is not None:
            raise ValueError(f"{name} must not share storage with {previous}")
        seen[pointer] = name


def finalize_ensemble_renewal_tau(
    min_rate: torch.Tensor,
    max_rate: torch.Tensor,
    tau_candidate: torch.Tensor,
    invalid: torch.Tensor,
    *,
    epsilon: float,
    tau_max: float,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write private ``tau_candidate[R]`` and int32 validity flags.

    Inputs and outputs are distinct persistent one-dimensional CUDA tensors
    with equal nonzero length. A nonfinite or negative rate bound, inconsistent
    min/max pair, or positive maximum whose selected step underflows or is not
    positive finite fp32 poisons ``tau_candidate``. ``invalid`` is a severity
    status: ``3`` for a nonfinite min/max, ``2`` for a finite negative minimum,
    ``1`` for an inconsistent range or invalid/underflowed tau candidate, and
    ``0`` for valid input. An exact all-zero range writes the valid absorbing
    fallback ``tau_max``. The caller can commit candidates to its public tau
    only after checking every status. No clock or RNG counter is touched.

    The wrapper checks only tensor metadata and host controls; it performs no
    device reduction, scalar extraction, or synchronization.
    """
    _require_cuda_triton()
    min_rate = _require_tensor("min_rate", min_rate)
    max_rate = _require_tensor("max_rate", max_rate)
    tau_candidate = _require_tensor("tau_candidate", tau_candidate)
    invalid = _require_tensor("invalid", invalid)
    device = _require_same_cuda_device(
        (
            ("min_rate", min_rate),
            ("max_rate", max_rate),
            ("tau_candidate", tau_candidate),
            ("invalid", invalid),
        )
    )
    _require_contiguous("min_rate", min_rate, dtype=torch.float32)
    _require_contiguous("max_rate", max_rate, dtype=torch.float32)
    _require_contiguous("tau_candidate", tau_candidate, dtype=torch.float32)
    _require_contiguous("invalid", invalid, dtype=torch.int32)
    if min_rate.dim() != 1 or min_rate.numel() == 0:
        raise ValueError("min_rate must be a non-empty one-dimensional tensor")
    if tuple(max_rate.shape) != tuple(min_rate.shape):
        raise ValueError("max_rate must have the same shape as min_rate")
    if tuple(tau_candidate.shape) != tuple(min_rate.shape):
        raise ValueError("tau_candidate must have the same shape as min_rate")
    if tuple(invalid.shape) != tuple(max_rate.shape):
        raise ValueError("invalid must have the same shape as min_rate")
    _reject_storage_aliases(
        (
            ("min_rate", min_rate),
            ("max_rate", max_rate),
            ("tau_candidate", tau_candidate),
            ("invalid", invalid),
        )
    )

    epsilon = validate_fp32_control("epsilon", epsilon, positive=True)
    tau_max = validate_fp32_control("tau_max", tau_max, positive=True)
    block_size = _positive_power_of_two("block_size", block_size, maximum=1024)
    replicas = min_rate.numel()
    with torch.cuda.device(device):
        _ensemble_finalize_renewal_tau_kernel[(triton.cdiv(replicas, block_size),)](
            min_rate_ptr=min_rate,
            max_rate_ptr=max_rate,
            tau_candidate_ptr=tau_candidate,
            invalid_ptr=invalid,
            epsilon=epsilon,
            tau_max=tau_max,
            R=replicas,
            BLOCK_SIZE=block_size,
        )
    return tau_candidate, invalid


def transition_ensemble_seir(
    state: torch.Tensor,
    age: torch.Tensor,
    rates: torch.Tensor,
    tau: torch.Tensor,
    rng_seed: torch.Tensor,
    step_id: torch.Tensor,
    event_partials: torch.Tensor,
    *,
    infectious_mask: torch.Tensor | None = None,
    nodes_per_program: int = 8,
    replicas_per_tile: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one in-place built-in renewal-SEIR ensemble transition.

    Stateful tensors use contiguous node-major ``[N, R]`` storage. ``rng_seed``
    and ``step_id`` are scalar int64 CUDA tensors and are read-only. They form
    one full-width event-seed/accepted-step key. Each lane uses the unique
    two-word uint32 Philox counter ``(node, replica)``; the two ids stay in
    separate counter words because a packed 64-bit counter changes Philox's
    word width. ``event_partials`` must have
    shape ``[ceil(N / node_tile), R]`` and receives int32 changed-node counts
    without final-count atomics. When ``infectious_mask`` is supplied, it is a
    persistent int32 bit-pattern tensor with shape ``[N, ceil(R/32)]``;
    accepted ``E -> I`` and ``I -> R`` events update their bits atomically.
    Logical node and replica tiles are flattened onto CUDA grid-x, so large
    replica counts do not encounter grid-y's 65,535 block ceiling. A caller
    should perform the final node-block reduction with an int64 accumulator.

    Valid lanes always store their advanced age. State stores are sparse: only
    accepted events that change ``S -> E``, ``E -> I``, or ``I -> R`` write the
    state tensor. Recovered/no-event lanes retain their existing state bytes.

    A non-positive, NaN, or infinite tau prevents every state and age store for
    that replica and writes zero to all of its event partials. The wrapper does
    not inspect device values or synchronize.
    """
    _require_cuda_triton()
    state = _require_tensor("state", state)
    age = _require_tensor("age", age)
    rates = _require_tensor("rates", rates)
    tau = _require_tensor("tau", tau)
    rng_seed = _require_tensor("rng_seed", rng_seed)
    step_id = _require_tensor("step_id", step_id)
    event_partials = _require_tensor("event_partials", event_partials)
    if infectious_mask is not None:
        infectious_mask = _require_tensor("infectious_mask", infectious_mask)
    named_tensors = (
        ("state", state),
        ("age", age),
        ("rates", rates),
        ("tau", tau),
        ("rng_seed", rng_seed),
        ("step_id", step_id),
        ("event_partials", event_partials),
    )
    if infectious_mask is not None:
        named_tensors += (("infectious_mask", infectious_mask),)
    device = _require_same_cuda_device(named_tensors)
    _require_contiguous("state", state, dtype=torch.int32)
    _require_contiguous("age", age, dtype=torch.float32)
    _require_contiguous("rates", rates, dtype=torch.float32)
    _require_contiguous("tau", tau, dtype=torch.float32)
    _require_contiguous("rng_seed", rng_seed, dtype=torch.int64)
    _require_contiguous("step_id", step_id, dtype=torch.int64)
    _require_contiguous("event_partials", event_partials, dtype=torch.int32)
    if infectious_mask is not None:
        _require_contiguous("infectious_mask", infectious_mask, dtype=torch.int32)

    if state.dim() != 2 or state.shape[0] <= 0 or state.shape[1] <= 0:
        raise ValueError("state must have non-empty node-major shape [N, R]")
    if tuple(age.shape) != tuple(state.shape):
        raise ValueError("age must have the same [N, R] shape as state")
    if tuple(rates.shape) != tuple(state.shape):
        raise ValueError("rates must have the same [N, R] shape as state")
    replicas = state.shape[1]
    if tau.dim() != 1 or tau.shape[0] != replicas:
        raise ValueError("tau must have shape [R]")
    if rng_seed.dim() != 0:
        raise ValueError("rng_seed must be a scalar tensor")
    if step_id.dim() != 0:
        raise ValueError("step_id must be a scalar tensor")
    expected_mask_shape = (state.shape[0], (replicas + 31) // 32)
    if infectious_mask is not None and tuple(infectious_mask.shape) != expected_mask_shape:
        raise ValueError(
            "infectious_mask must have shape "
            f"{expected_mask_shape}, got {tuple(infectious_mask.shape)}"
        )

    uint32_cardinality = 1 << 32
    if state.shape[0] > uint32_cardinality:
        raise ValueError("the node dimension must fit in uint32 counter ids")
    if replicas > uint32_cardinality:
        raise ValueError("the replica dimension must fit in uint32 counter ids")

    nodes_per_program = _positive_power_of_two("nodes_per_program", nodes_per_program, maximum=128)
    if replicas_per_tile is None:
        replicas_per_tile = _default_replica_tile(replicas)
    replicas_per_tile = _positive_power_of_two("replicas_per_tile", replicas_per_tile, maximum=32)
    if nodes_per_program * replicas_per_tile > 512:
        raise ValueError("nodes_per_program * replicas_per_tile must be <= 512")

    num_nodes = state.shape[0]
    num_node_blocks = triton.cdiv(num_nodes, nodes_per_program)
    expected_partials = (num_node_blocks, replicas)
    if tuple(event_partials.shape) != expected_partials:
        raise ValueError(
            f"event_partials must have shape {expected_partials}, got {tuple(event_partials.shape)}"
        )
    _reject_storage_aliases(named_tensors)

    num_replica_tiles = triton.cdiv(replicas, replicas_per_tile)
    num_programs = num_node_blocks * num_replica_tiles
    if num_programs > (1 << 31) - 1:
        raise ValueError("the flattened transition grid exceeds CUDA grid-x capacity")
    grid = (num_programs,)
    with torch.cuda.device(device):
        _ensemble_seir_transition_kernel[grid](
            state_ptr=state,
            age_ptr=age,
            rates_ptr=rates,
            tau_ptr=tau,
            rng_seed_ptr=rng_seed,
            step_id_ptr=step_id,
            event_partials_ptr=event_partials,
            infectious_mask_ptr=(infectious_mask if infectious_mask is not None else state),
            N=num_nodes,
            R=replicas,
            NUM_REPLICA_TILES=num_replica_tiles,
            UPDATE_INFECTIOUS_MASK=infectious_mask is not None,
            NODES_PER_PROGRAM=nodes_per_program,
            REPLICAS_PER_TILE=replicas_per_tile,
        )
    return state, age, event_partials


__all__ = [
    "finalize_ensemble_renewal_tau",
    "transition_ensemble_seir",
]
