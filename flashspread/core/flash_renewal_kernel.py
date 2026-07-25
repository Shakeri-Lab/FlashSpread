"""Triton kernels for the renewal fast path.

One simulation step has two globally ordered phases:

1. traverse the incoming CSR and evaluate *current-state* rates;
2. reduce those rates to the current adaptive ``tau``, then sample and update.

The barrier between the phases is essential: sampling with the previous step's
``tau`` does not enforce the package's epsilon bound when rates rise abruptly.
Graph traversal and hazard evaluation remain fused, and all traversal strategies
share one transition kernel so their stochastic semantics cannot drift apart.
"""

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the optional GPU stack
    triton = None
    tl = None
    _HAS_TRITON = False

from .flash_rng import _event_probability, _mix_rng_key, _sample_bernoulli


if _HAS_TRITON:

    @triton.jit
    def _erfcx_approx(z):
        """Piecewise fp32 approximation of ``exp(z**2) * erfc(z)``."""
        az = tl.abs(z)
        az_sq = az * az
        small = tl.exp(az_sq) * (1.0 - tl.math.erf(az))

        inv_z = 1.0 / (az + 1e-30)
        inv_z2 = inv_z * inv_z
        rsqrt_pi: tl.constexpr = 0.5641895835477563
        large = rsqrt_pi * inv_z * (
            1.0
            - 0.5 * inv_z2
            + 0.75 * inv_z2 * inv_z2
            - 1.875 * inv_z2 * inv_z2 * inv_z2
        )
        result_pos = tl.where(az <= 3.5, small, large)
        result_pos = tl.where(az > 9.0, rsqrt_pi * inv_z, result_pos)

        # For large negative z this intentionally tends to +inf, making the
        # log-normal hazard tend to zero at age zero.
        result = tl.where(
            z >= 0.0,
            result_pos,
            2.0 * tl.exp(z * z) - result_pos,
        )
        return tl.maximum(result, 1e-30)

    @triton.jit
    def _lognormal_hazard_triton(age, mu, sigma):
        """Numerically stable log-normal hazard in fp32 registers."""
        sqrt_2_over_pi: tl.constexpr = 0.7978845608028654
        sqrt_2: tl.constexpr = 1.4142135623730951
        t = tl.maximum(age, 1e-10)
        z = (tl.log(t) - mu) / (sigma * sqrt_2)
        return sqrt_2_over_pi / (t * sigma * _erfcx_approx(z))

    @triton.jit
    def _seir_rate(
        state,
        age,
        pressure,
        mu_ei,
        sig_ei,
        mu_ir,
        sig_ir,
        STATE_S: tl.constexpr,
        STATE_E: tl.constexpr,
        STATE_I: tl.constexpr,
    ):
        """Evaluate one block/tile of current-state SEIR exit rates."""
        is_s = state == STATE_S
        is_e = state == STATE_E
        is_i = state == STATE_I

        hazard_e = tl.zeros(state.shape, dtype=tl.float32)
        hazard_i = tl.zeros(state.shape, dtype=tl.float32)
        if tl.sum(is_e.to(tl.int32), axis=0) > 0:
            hazard_e = tl.where(
                is_e, _lognormal_hazard_triton(age, mu_ei, sig_ei), 0.0
            )
        if tl.sum(is_i.to(tl.int32), axis=0) > 0:
            hazard_i = tl.where(
                is_i, _lognormal_hazard_triton(age, mu_ir, sig_ir), 0.0
            )

        rate = tl.where(is_s, pressure, 0.0)
        rate = tl.where(is_e, hazard_e, rate)
        return tl.where(is_i, hazard_i, rate)

    @triton.jit
    def _store_rate_max_partial(rate, mask, max_rate_partials_ptr):
        """Emit one finite-aware maximum for the calling rate program."""
        finite = (
            (rate == rate)
            & (rate != float("inf"))
            & (rate != -float("inf"))
        )
        candidate = tl.where(mask, rate, 0.0)
        # Preserve the full-rate reduction's failure behavior without asking
        # the next phase to reread every public rate: any nonfinite lane makes
        # the compact maximum nonfinite as well. The tau finalizer then poisons
        # the transaction before transition sampling.
        candidate = tl.where(mask & ~finite, float("inf"), candidate)
        tl.store(
            max_rate_partials_ptr + tl.program_id(0),
            tl.max(candidate, axis=0),
        )

    @triton.jit
    def _flash_renewal_rate_kernel(
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        infectivity_ptr,
        age_ptr,
        state_ptr,
        beta,
        mu_ei,
        sig_ei,
        mu_ir,
        sig_ir,
        rates_ptr,
        max_rate_partials_ptr,
        active_nodes_ptr,
        num_active_ptr,
        N,
        STATE_S: tl.constexpr,
        STATE_E: tl.constexpr,
        STATE_I: tl.constexpr,
        HAS_WEIGHTS: tl.constexpr,
        TRANSMISSION_AGE_DEPENDENT: tl.constexpr,
        USE_COMPACTION: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Thread-per-target CSR gather fused with current-rate evaluation."""
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if USE_COMPACTION:
            num_active = tl.load(num_active_ptr)
            mask = offsets < num_active
            idx = tl.load(active_nodes_ptr + offsets, mask=mask, other=0)
        else:
            mask = offsets < N
            idx = offsets

        state = tl.load(state_ptr + idx, mask=mask, other=0).to(tl.int32)
        needs_age = mask & ((state == STATE_E) | (state == STATE_I))
        age = tl.load(age_ptr + idx, mask=needs_age, other=0.0).to(tl.float32)

        # Only susceptible targets consume graph pressure. This removes all
        # edge/index/source loads for E, I and R targets.
        susceptible = mask & (state == STATE_S)
        row_start = tl.load(row_ptr_ptr + idx, mask=susceptible, other=0)
        row_end = tl.load(row_ptr_ptr + idx + 1, mask=susceptible, other=0)
        edge = row_start
        pressure = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        any_edge = tl.max((edge < row_end).to(tl.int32), axis=0)
        while any_edge != 0:
            edge_mask = susceptible & (edge < row_end)
            neighbor = tl.load(col_ind_ptr + edge, mask=edge_mask, other=0)
            if TRANSMISSION_AGE_DEPENDENT:
                source_inf = tl.load(
                    infectivity_ptr + neighbor, mask=edge_mask, other=0.0
                ).to(tl.float32)
            else:
                source_state = tl.load(
                    state_ptr + neighbor, mask=edge_mask, other=STATE_S
                ).to(tl.int32)
                source_inf = tl.where(source_state == STATE_I, beta, 0.0)
            if HAS_WEIGHTS:
                weight = tl.load(
                    weights_ptr + edge, mask=edge_mask, other=0.0
                ).to(tl.float32)
            else:
                weight = tl.full([BLOCK_SIZE], 1.0, tl.float32)
            pressure += tl.where(edge_mask, source_inf * weight, 0.0)
            edge += 1
            any_edge = tl.max((edge < row_end).to(tl.int32), axis=0)

        rate = _seir_rate(
            state,
            age,
            pressure,
            mu_ei,
            sig_ei,
            mu_ir,
            sig_ir,
            STATE_S,
            STATE_E,
            STATE_I,
        )
        tl.store(rates_ptr + idx, rate, mask=mask)
        _store_rate_max_partial(rate, mask, max_rate_partials_ptr)

    @triton.jit
    def _flash_renewal_rate_wc_kernel(
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        infectivity_ptr,
        age_ptr,
        state_ptr,
        beta,
        mu_ei,
        sig_ei,
        mu_ir,
        sig_ir,
        rates_ptr,
        max_rate_partials_ptr,
        N,
        STATE_S: tl.constexpr,
        STATE_E: tl.constexpr,
        STATE_I: tl.constexpr,
        HAS_WEIGHTS: tl.constexpr,
        TRANSMISSION_AGE_DEPENDENT: tl.constexpr,
        NODES_PER_BLOCK: tl.constexpr,
        LANES_PER_NODE: tl.constexpr,
    ):
        """Warp-per-target CSR gather fused with current-rate evaluation."""
        node = tl.program_id(0) * NODES_PER_BLOCK + tl.arange(0, NODES_PER_BLOCK)
        lane = tl.arange(0, LANES_PER_NODE)
        node_mask = node < N
        state = tl.load(state_ptr + node, mask=node_mask, other=0).to(tl.int32)
        needs_age = node_mask & ((state == STATE_E) | (state == STATE_I))
        age = tl.load(age_ptr + node, mask=needs_age, other=0.0).to(tl.float32)
        susceptible = node_mask & (state == STATE_S)

        row_start = tl.load(row_ptr_ptr + node, mask=susceptible, other=0)
        row_end = tl.load(row_ptr_ptr + node + 1, mask=susceptible, other=0)
        row_start_2d = row_start[:, None]
        row_end_2d = row_end[:, None]
        susceptible_2d = susceptible[:, None]
        lane_2d = lane[None, :]
        pressure_2d = tl.zeros(
            [NODES_PER_BLOCK, LANES_PER_NODE], dtype=tl.float32
        )

        chunk = 0
        edge = row_start_2d + lane_2d
        any_edge = tl.max((edge < row_end_2d).to(tl.int32))
        while any_edge != 0:
            edge_mask = susceptible_2d & (edge < row_end_2d)
            neighbor = tl.load(col_ind_ptr + edge, mask=edge_mask, other=0)
            if TRANSMISSION_AGE_DEPENDENT:
                source_inf = tl.load(
                    infectivity_ptr + neighbor, mask=edge_mask, other=0.0
                ).to(tl.float32)
            else:
                source_state = tl.load(
                    state_ptr + neighbor, mask=edge_mask, other=STATE_S
                ).to(tl.int32)
                source_inf = tl.where(source_state == STATE_I, beta, 0.0)
            if HAS_WEIGHTS:
                weight = tl.load(
                    weights_ptr + edge, mask=edge_mask, other=0.0
                ).to(tl.float32)
            else:
                weight = tl.full(
                    [NODES_PER_BLOCK, LANES_PER_NODE], 1.0, tl.float32
                )
            pressure_2d += tl.where(edge_mask, source_inf * weight, 0.0)
            chunk += 1
            edge = row_start_2d + chunk * LANES_PER_NODE + lane_2d
            any_edge = tl.max((edge < row_end_2d).to(tl.int32))

        pressure = tl.sum(pressure_2d, axis=1)
        rate = _seir_rate(
            state,
            age,
            pressure,
            mu_ei,
            sig_ei,
            mu_ir,
            sig_ir,
            STATE_S,
            STATE_E,
            STATE_I,
        )
        tl.store(rates_ptr + node, rate, mask=node_mask)
        _store_rate_max_partial(rate, node_mask, max_rate_partials_ptr)

    @triton.jit
    def _pressure_merge_kernel(
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        infectivity_ptr,
        state_ptr,
        pressure_ptr,
        beta,
        N,
        E,
        STATE_S: tl.constexpr,
        STATE_I: tl.constexpr,
        HAS_WEIGHTS: tl.constexpr,
        TRANSMISSION_AGE_DEPENDENT: tl.constexpr,
        EDGES_PER_BLOCK: tl.constexpr,
        BSEARCH_ITERS: tl.constexpr,
    ):
        """Edge-partitioned pressure accumulation for heavy-tailed rows."""
        edge = tl.program_id(0) * EDGES_PER_BLOCK + tl.arange(0, EDGES_PER_BLOCK)
        edge_mask = edge < E
        edge32 = edge.to(tl.int32)

        # Upper-bound search in row_ptr; node is the row owning this edge.
        lo = tl.zeros([EDGES_PER_BLOCK], dtype=tl.int32)
        hi = tl.zeros([EDGES_PER_BLOCK], dtype=tl.int32) + (N + 1)
        for _ in tl.static_range(BSEARCH_ITERS):
            # Overflow-safe midpoint. ``(lo + hi) // 2`` wraps to a negative
            # int32 once lo + hi exceeds 2**31 - 1, i.e. for N > 2**30: the
            # clamp below then forces below=True, drives lo negative, and the
            # search converges on a fixed *wrong* row, silently attributing
            # every affected edge's pressure to the wrong target.
            mid = lo + ((hi - lo) >> 1)
            row_mid = tl.load(row_ptr_ptr + tl.minimum(tl.maximum(mid, 0), N))
            row_mid = tl.where(mid > N, E + 1, row_mid)
            below = row_mid <= edge32
            lo = tl.where(below, mid + 1, lo)
            hi = tl.where(below, hi, mid)
        node = tl.maximum(lo - 1, 0)

        # Pressure is irrelevant outside S. Filtering payload/atomic traffic is
        # especially valuable for hubs, even though row ownership is still found.
        target_state = tl.load(state_ptr + node, mask=edge_mask, other=0)
        payload_mask = edge_mask & (target_state == STATE_S)
        neighbor = tl.load(col_ind_ptr + edge, mask=payload_mask, other=0)
        if TRANSMISSION_AGE_DEPENDENT:
            source_inf = tl.load(
                infectivity_ptr + neighbor, mask=payload_mask, other=0.0
            ).to(tl.float32)
        else:
            source_state = tl.load(
                state_ptr + neighbor, mask=payload_mask, other=STATE_S
            ).to(tl.int32)
            source_inf = tl.where(source_state == STATE_I, beta, 0.0)
        if HAS_WEIGHTS:
            weight = tl.load(
                weights_ptr + edge, mask=payload_mask, other=0.0
            ).to(tl.float32)
        else:
            weight = tl.full([EDGES_PER_BLOCK], 1.0, tl.float32)
        tl.atomic_add(
            pressure_ptr + node,
            source_inf * weight,
            mask=payload_mask,
        )

    @triton.jit
    def _flash_renewal_rate_from_pressure_kernel(
        pressure_ptr,
        age_ptr,
        state_ptr,
        mu_ei,
        sig_ei,
        mu_ir,
        sig_ir,
        rates_ptr,
        max_rate_partials_ptr,
        N,
        STATE_S: tl.constexpr,
        STATE_E: tl.constexpr,
        STATE_I: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Rate tail for edge-partitioned pressure accumulation."""
        idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        state = tl.load(state_ptr + idx, mask=mask, other=0).to(tl.int32)
        needs_age = mask & ((state == STATE_E) | (state == STATE_I))
        age = tl.load(age_ptr + idx, mask=needs_age, other=0.0).to(tl.float32)
        pressure = tl.load(
            pressure_ptr + idx,
            mask=mask & (state == STATE_S),
            other=0.0,
        )
        rate = _seir_rate(
            state,
            age,
            pressure,
            mu_ei,
            sig_ei,
            mu_ir,
            sig_ir,
            STATE_S,
            STATE_E,
            STATE_I,
        )
        tl.store(rates_ptr + idx, rate, mask=mask)
        _store_rate_max_partial(rate, mask, max_rate_partials_ptr)

    @triton.jit
    def _flash_renewal_finalize_tau_kernel(
        max_rate_ptr,
        tau_ptr,
        elapsed_ptr,
        step_id_ptr,
        epsilon,
        tau_max,
        ACCUMULATE_TIME: tl.constexpr,
    ):
        """Finalize adaptive tau, the RNG step id, and optional replay time.

        The global maximum is produced by PyTorch's optimized reduction. This
        one-program kernel replaces the former chain of scalar div/min/compare/
        where/copy/add launches while preserving the global ordering point
        before transition sampling.
        """
        max_rate = tl.load(max_rate_ptr).to(tl.float32)
        tau_candidate = epsilon / max_rate
        tau = tl.minimum(tau_candidate, tau_max)
        fp32_max: tl.constexpr = 3.4028234663852886e38
        invalid = (
            (max_rate != max_rate)
            | (max_rate > fp32_max)
            | (max_rate < 0.0)
            | ((max_rate > 0.0) & ((tau != tau) | (tau <= 0.0)))
        )
        zero = (max_rate - max_rate) * 0.0
        nan_value = zero / zero
        # Only an exact all-zero rate is a valid absorbing-state fallback.
        tau = tl.where(max_rate == 0.0, tau_max, tau)
        # Poison invalid internal steps so a multi-step elapsed accumulator
        # remains invalid until the host boundary checks it.
        tau = tl.where(invalid, nan_value, tau)

        if ACCUMULATE_TIME:
            elapsed = tl.load(elapsed_ptr).to(tl.float64)
            # Keep all later steps in the captured replay gated after the first
            # invalid tau instead of allowing subsequent state mutation.
            tau = tl.where(elapsed != elapsed, nan_value, tau)
            tl.store(tau_ptr, tau)
            tl.store(elapsed_ptr, elapsed + tau.to(tl.float64))
        else:
            tl.store(tau_ptr, tau)

        step_id = tl.load(step_id_ptr).to(tl.int64)
        tl.store(step_id_ptr, step_id + 1)

    @triton.jit
    def _flash_renewal_transition_kernel(
        age_ptr,
        state_ptr,
        rates_ptr,
        beta,
        mu_ir,
        sig_ir,
        tau_ptr,
        rng_seed_ptr,
        step_id_ptr,
        next_state_ptr,
        next_age_ptr,
        next_infectivity_ptr,
        N,
        STATE_S: tl.constexpr,
        STATE_E: tl.constexpr,
        STATE_I: tl.constexpr,
        STATE_R: tl.constexpr,
        TRANSMISSION_AGE_DEPENDENT: tl.constexpr,
        MIXED_PRECISION: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Shared Bernoulli transition, renewal reset and shedding write."""
        idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        state = tl.load(state_ptr + idx, mask=mask, other=STATE_R).to(tl.int32)
        age = tl.load(age_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        rate = tl.load(rates_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        tau = tl.load(tau_ptr)
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
        )

        is_s = state == STATE_S
        is_e = state == STATE_E
        is_i = state == STATE_I
        new_state = state
        new_state = tl.where(event & is_s, STATE_E, new_state)
        new_state = tl.where(event & is_e, STATE_I, new_state)
        new_state = tl.where(event & is_i, STATE_R, new_state)
        new_age = tl.where(new_state != state, 0.0, age + safe_tau)

        is_i_next = new_state == STATE_I
        if TRANSMISSION_AGE_DEPENDENT:
            next_infectivity = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            if tl.sum(is_i_next.to(tl.int32), axis=0) > 0:
                shedding = _lognormal_hazard_triton(new_age, mu_ir, sig_ir)
                next_infectivity = tl.where(is_i_next, beta * shedding, 0.0)

        # This phase deliberately writes every node, even when rate evaluation
        # was compacted. Both ping-pong buffers therefore agree on absorbing R
        # nodes and stale infectivity can never leak into a later gather.
        if MIXED_PRECISION:
            tl.store(next_state_ptr + idx, new_state.to(tl.int8), mask=mask)
            # Age remains fp32 even in mixed storage. An fp16 clock can stop
            # advancing when adaptive tau falls below half an age ULP.
            tl.store(next_age_ptr + idx, new_age, mask=mask)
            if TRANSMISSION_AGE_DEPENDENT:
                tl.store(
                    next_infectivity_ptr + idx,
                    next_infectivity.to(tl.bfloat16),
                    mask=mask,
                )
        else:
            tl.store(next_state_ptr + idx, new_state, mask=mask)
            tl.store(next_age_ptr + idx, new_age, mask=mask)
            if TRANSMISSION_AGE_DEPENDENT:
                tl.store(next_infectivity_ptr + idx, next_infectivity, mask=mask)


else:
    _erfcx_approx = None
    _lognormal_hazard_triton = None
    _flash_renewal_rate_kernel = None
    _flash_renewal_rate_wc_kernel = None
    _pressure_merge_kernel = None
    _flash_renewal_rate_from_pressure_kernel = None
    _flash_renewal_finalize_tau_kernel = None
    _flash_renewal_transition_kernel = None
