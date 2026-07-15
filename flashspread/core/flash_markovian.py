"""Sparse GPU primitives for Markovian frontier updates."""

from __future__ import annotations

import torch

from .flash_rng import _event_probability, _mix_rng_key, _sample_bernoulli

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
    _TRITON_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on optional GPU stack
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = exc


if _HAS_TRITON:

    @triton.jit
    def _markov_rate_reduce_kernel(
        state_ptr,
        influence_ptr,
        rates_ptr,
        total_rate_ptr,
        max_rate_ptr,
        beta,
        recovery_rate,
        N,
        STATE_S: tl.constexpr,
        STATE_I: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Evaluate built-in SIS/SIR rates and reduce tau statistics."""
        idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        state = tl.load(state_ptr + idx, mask=mask, other=-1).to(tl.int32)
        susceptible = mask & (state == STATE_S)
        influence = tl.load(
            influence_ptr + idx,
            mask=susceptible,
            other=0.0,
        ).to(tl.float32)
        rate = tl.where(
            state == STATE_S,
            tl.maximum(beta * influence, 0.0),
            0.0,
        )
        rate = tl.where(state == STATE_I, recovery_rate, rate)
        rate = tl.where(mask, rate, 0.0)
        tl.store(rates_ptr + idx, rate, mask=mask)

        # One private partial per program avoids hundreds of thousands of
        # same-address scalar atomics at large N. Fixed-shape hierarchy kernels
        # below reduce these arrays before tau finalization.
        program = tl.program_id(0)
        tl.store(total_rate_ptr + program, tl.sum(rate, axis=0))
        tl.store(max_rate_ptr + program, tl.max(rate, axis=0))

    @triton.jit
    def _markov_reduce_rate_partials_kernel(
        input_sum_ptr,
        input_max_ptr,
        output_sum_ptr,
        output_max_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Reduce one fixed level of rate sum/max partials."""
        program = tl.program_id(0)
        idx = program * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        partial_sum = tl.load(input_sum_ptr + idx, mask=mask, other=0.0).to(
            tl.float32
        )
        partial_max = tl.load(input_max_ptr + idx, mask=mask, other=0.0).to(
            tl.float32
        )
        tl.store(output_sum_ptr + program, tl.sum(partial_sum, axis=0))
        tl.store(output_max_ptr + program, tl.max(partial_max, axis=0))

    @triton.jit
    def _markov_reduce_event_partials_kernel(
        input_ptr,
        output_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Reduce one fixed level of int64 event-count partials."""
        program = tl.program_id(0)
        idx = program * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        partial = tl.load(input_ptr + idx, mask=mask, other=0).to(tl.int64)
        tl.store(output_ptr + program, tl.sum(partial, axis=0))

    @triton.jit
    def _markov_finalize_tau_kernel(
        total_rate_ptr,
        max_rate_ptr,
        tau_ptr,
        elapsed_ptr,
        step_id_ptr,
        target_events,
        probability_scale,
        tau_min,
        tau_max,
        ACCUMULATE_TIME: tl.constexpr,
    ):
        """Finalize Markov tau and reset the next step's reduction scalars."""
        total_rate = tl.load(total_rate_ptr).to(tl.float32)
        max_rate = tl.load(max_rate_ptr).to(tl.float32)
        expected_bound = tl.maximum(target_events / total_rate, tau_min)
        probability_bound = probability_scale / max_rate
        tau = tl.minimum(tl.minimum(expected_bound, probability_bound), tau_max)
        fp32_max: tl.constexpr = 3.4028234663852886e38
        invalid = (
            (total_rate != total_rate)
            | (max_rate != max_rate)
            | (total_rate > fp32_max)
            | (max_rate > fp32_max)
            | (total_rate < 0.0)
            | (max_rate < 0.0)
            | ((total_rate == 0.0) & (max_rate != 0.0))
            | ((total_rate != 0.0) & (max_rate == 0.0))
            | (
                (total_rate > 0.0)
                & ((tau != tau) | (tau <= 0.0))
            )
        )
        # 0/0 produces NaN for finite negatives as well as NaN/infinity. The
        # host boundary turns that poison value into an actionable error, and
        # a captured multi-step elapsed accumulator remains NaN thereafter.
        zero = (total_rate - total_rate) * 0.0
        nan_value = zero / zero
        tau = tl.where(total_rate <= 0.0, tau_max, tau)
        # Apply invalid last so the absorbing-state fallback cannot hide -inf
        # or a logically impossible negative aggregate.
        tau = tl.where(invalid, nan_value, tau)

        if ACCUMULATE_TIME:
            elapsed = tl.load(elapsed_ptr).to(tl.float64)
            # Once one internal replay step fails, gate every later transition
            # in the same captured window as well.
            tau = tl.where(elapsed != elapsed, nan_value, tau)
            tl.store(tau_ptr, tau)
            tl.store(elapsed_ptr, elapsed + tau.to(tl.float64))
        else:
            tl.store(tau_ptr, tau)

        step_id = tl.load(step_id_ptr).to(tl.int64) + 1
        tl.store(step_id_ptr, step_id)
        # A following rate kernel can reuse these fixed-address scalars without
        # a separate zeroing launch.
        tl.store(total_rate_ptr, 0.0)
        tl.store(max_rate_ptr, 0.0)

    @triton.jit
    def _markov_transition_frontier_kernel(
        state_ptr,
        rates_ptr,
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        influence_ptr,
        tau_ptr,
        rng_seed_ptr,
        step_id_ptr,
        event_count_ptr,
        N,
        STATE_S: tl.constexpr,
        STATE_I: tl.constexpr,
        STATE_R: tl.constexpr,
        MODEL_SIS: tl.constexpr,
        HAS_WEIGHTS: tl.constexpr,
        ACCUMULATE_EVENTS: tl.constexpr,
        PROPAGATE_INFLUENCE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Sample built-in transitions and propagate their outgoing frontier."""
        idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        state = tl.load(state_ptr + idx, mask=mask, other=STATE_R).to(tl.int32)
        can_transition = mask & ((state == STATE_S) | (state == STATE_I))
        rate = tl.load(
            rates_ptr + idx,
            mask=can_transition,
            other=0.0,
        ).to(tl.float32)
        tau = tl.load(tau_ptr).to(tl.float32)
        valid_tau = (
            (tau == tau)
            & (tau > 0.0)
            & (tau <= 3.4028234663852886e38)
        )
        safe_tau = tl.where(valid_tau, tau, 0.0)
        probability = _event_probability(rate * safe_tau)
        base_seed = tl.load(rng_seed_ptr)
        step_id = tl.load(step_id_ptr)
        key = _mix_rng_key(base_seed, step_id)
        event = tl.where(
            valid_tau,
            _sample_bernoulli(probability, key, idx),
            False,
        ) & mask

        is_s = state == STATE_S
        is_i = state == STATE_I
        new_state = tl.where(event & is_s, STATE_I, state)
        if MODEL_SIS:
            new_state = tl.where(event & is_i, STATE_S, new_state)
        else:
            new_state = tl.where(event & is_i, STATE_R, new_state)

        changed = mask & (new_state != state)
        block_events = tl.sum(changed.to(tl.int32), axis=0).to(tl.int64)
        program = tl.program_id(0)
        if ACCUMULATE_EVENTS:
            block_events += tl.load(event_count_ptr + program).to(tl.int64)
        tl.store(event_count_ptr + program, block_events)
        tl.store(state_ptr + idx, new_state, mask=mask)

        if PROPAGATE_INFLUENCE:
            old_inducer = state == STATE_I
            new_inducer = new_state == STATE_I
            delta = new_inducer.to(tl.float32) - old_inducer.to(tl.float32)
            changed_inducer = mask & (delta != 0.0)
            row_start = tl.load(
                row_ptr_ptr + idx,
                mask=changed_inducer,
                other=0,
            )
            row_end = tl.load(
                row_ptr_ptr + idx + 1,
                mask=changed_inducer,
                other=0,
            )
            edge = row_start
            active = changed_inducer & (edge < row_end)
            any_active = tl.max(active.to(tl.int32), axis=0)
            while any_active != 0:
                target = tl.load(col_ind_ptr + edge, mask=active, other=0)
                if HAS_WEIGHTS:
                    weight = tl.load(
                        weights_ptr + edge,
                        mask=active,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    weight = tl.full([BLOCK_SIZE], 1.0, tl.float32)
                tl.atomic_add(
                    influence_ptr + target,
                    delta * weight,
                    mask=active,
                )
                edge += 1
                active = changed_inducer & (edge < row_end)
                any_active = tl.max(active.to(tl.int32), axis=0)

    @triton.jit
    def _propagate_frontier_kernel(
        changed_ptr,
        delta_ptr,
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        influence_ptr,
        K,
        HAS_WEIGHTS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program owns one changed source and walks its outgoing row."""
        frontier_idx = tl.program_id(0)
        valid_source = frontier_idx < K
        source = tl.load(changed_ptr + frontier_idx, mask=valid_source, other=0)
        delta = tl.load(delta_ptr + frontier_idx, mask=valid_source, other=0.0)

        row_start = tl.load(row_ptr_ptr + source, mask=valid_source, other=0)
        row_end = tl.load(row_ptr_ptr + source + 1, mask=valid_source, other=0)
        lane = tl.arange(0, BLOCK_SIZE)
        edge = row_start + lane
        active = tl.max((edge < row_end).to(tl.int32), axis=0)

        while active != 0:
            mask = valid_source & (edge < row_end)
            target = tl.load(col_ind_ptr + edge, mask=mask, other=0)
            if HAS_WEIGHTS:
                weight = tl.load(weights_ptr + edge, mask=mask, other=0.0).to(
                    tl.float32
                )
            else:
                weight = tl.full([BLOCK_SIZE], 1.0, tl.float32)
            tl.atomic_add(influence_ptr + target, delta * weight, mask=mask)
            edge += BLOCK_SIZE
            active = tl.max((edge < row_end).to(tl.int32), axis=0)


else:
    _markov_rate_reduce_kernel = None
    _markov_reduce_rate_partials_kernel = None
    _markov_reduce_event_partials_kernel = None
    _markov_finalize_tau_kernel = None
    _markov_transition_frontier_kernel = None
    _propagate_frontier_kernel = None


def propagate_frontier(
    outgoing_graph,
    changed: torch.Tensor,
    delta: torch.Tensor,
    influence: torch.Tensor,
) -> None:
    """Propagate inducer deltas from a compact changed-node frontier."""
    if not _HAS_TRITON:
        raise RuntimeError("Triton is required for GPU frontier propagation") from (
            _TRITON_IMPORT_ERROR
        )
    if outgoing_graph.device.type != "cuda":
        raise ValueError("frontier propagation requires a CUDA graph")
    if getattr(outgoing_graph, "incoming", True):
        raise ValueError("frontier propagation requires outgoing CSR orientation")
    if (
        changed.device != outgoing_graph.device
        or delta.device != outgoing_graph.device
        or influence.device != outgoing_graph.device
    ):
        raise ValueError("frontier, delta, influence, and graph must share one device")
    if changed.dim() != 1 or delta.shape != changed.shape:
        raise ValueError("changed and delta must be one-dimensional with equal shape")
    if influence.shape != (outgoing_graph.num_nodes,):
        raise ValueError(f"influence must have shape [{outgoing_graph.num_nodes}]")
    if influence.dtype != torch.float32:
        raise TypeError("influence must use float32 storage")
    if changed.numel() == 0:
        return
    if changed.dtype != torch.int32:
        changed = changed.to(torch.int32)
    if delta.dtype != torch.float32:
        delta = delta.to(torch.float32)

    with torch.cuda.device(outgoing_graph.device):
        _propagate_frontier_kernel[(changed.numel(),)](
            changed_ptr=changed,
            delta_ptr=delta,
            row_ptr_ptr=outgoing_graph.row_ptr,
            col_ind_ptr=outgoing_graph.col_ind,
            weights_ptr=outgoing_graph.weights_storage,
            influence_ptr=influence,
            K=changed.numel(),
            HAS_WEIGHTS=outgoing_graph.has_weights,
            BLOCK_SIZE=128,
        )
