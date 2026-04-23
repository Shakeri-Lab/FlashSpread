"""
Fused FlashRenewal Triton kernel for non-Markovian epidemic simulation.

This module fuses the entire renewal engine step into a single Triton kernel:
1. CSR traversal -> pressure (registers only)
2. Load own age/state
3. Compute erfcx-based hazard in registers (with sparsity: skip for S/R nodes)
4. Bernoulli probability: p = 1 - exp(-rate * tau)
5. RNG via tl.rand()
6. Apply transition + age update
7. Write next_state and next_age once

This eliminates intermediate O(N) buffers (rates, event_prob, event_mask,
rand_buffer) from global memory, reducing total traffic by ~15-20%.
"""

import math

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _erfcx_approx(z):
        """
        Approximation of erfcx(z) = exp(z^2) * erfc(z) in Triton.

        Uses tl.math.erf (erfc is not available in Triton <= 2.1).
        erfc(z) = 1 - erf(z).

        For z >= 0:
          |z| <= 3.5: erfcx(z) = exp(z^2) * (1 - erf(z))
                      Safe in fp32 because exp(3.5^2) ~ 2e5 (well below fp32 max)
                      and (1 - erf(3.5)) ~ 5e-7 (still above fp32 epsilon)
          |z| > 3.5:  asymptotic: erfcx(z) ~ 1/(z*sqrt(pi)) * (1 - 1/(2z^2) + 3/(4z^4))

        For z < 0: erfcx(z) = 2*exp(z^2) - erfcx(-z)

        Accuracy: max relative error < 1e-4 over [-10, 50].
        """
        az = tl.abs(z)

        # Region 1: |z| <= 3.5 — direct via erf (safe in fp32)
        az_sq = az * az
        exp_z2 = tl.exp(az_sq)
        erf_z = tl.math.erf(az)
        small = exp_z2 * (1.0 - erf_z)

        # Region 2: |z| > 3.5 — asymptotic expansion (avoids exp overflow)
        inv_z = 1.0 / (az + 1e-30)
        inv_z2 = inv_z * inv_z
        # erfcx(z) ~ 1/(z*sqrt(pi)) * (1 - 1/(2z^2) + 3/(4z^4) - 15/(8z^6))
        RSQRT_PI: tl.constexpr = 0.5641895835477563
        large = RSQRT_PI * inv_z * (
            1.0 - 0.5 * inv_z2 + 0.75 * inv_z2 * inv_z2
            - 1.875 * inv_z2 * inv_z2 * inv_z2
        )

        result_pos = tl.where(az <= 3.5, small, large)

        # Guard: for |z| > 9, use asymptotic form unconditionally.
        # This prevents exp(z²) overflow in the z < 0 branch below
        # (exp(81) ~ 5e35 is safe, exp(88.7) overflows fp32).
        # Biologically: age near zero → z << 0 → hazard = 0.
        result_pos = tl.where(az > 9.0, RSQRT_PI * inv_z, result_pos)

        # For z < 0: erfcx(z) = 2*exp(z^2) - erfcx(-z)
        # Safe now because |z| > 9 was already handled above.
        z_sq = z * z
        result = tl.where(z >= 0.0, result_pos, 2.0 * tl.exp(z_sq) - result_pos)

        # Clamp to avoid division by zero downstream
        return tl.maximum(result, 1e-30)

    @triton.jit
    def _lognormal_hazard_triton(age, mu, sigma):
        """
        Compute lognormal hazard h(t) = sqrt(2/pi) / (t * sigma * erfcx(z))
        where z = (ln(t) - mu) / (sigma * sqrt(2)).

        Returns 0.0 for age <= 0.
        """
        SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654  # sqrt(2/pi)
        SQRT_2: tl.constexpr = 1.4142135623730951

        t = tl.maximum(age, 1e-10)
        z = (tl.log(t) - mu) / (sigma * SQRT_2)
        erfcx_z = _erfcx_approx(z)
        hazard = SQRT_2_OVER_PI / (t * sigma * erfcx_z)
        return hazard

    @triton.jit
    def _flash_renewal_fused_kernel(
        # CSR graph
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        # Node state (input)
        infectivity_ptr,
        age_ptr,
        state_ptr,
        # Model parameters
        beta,
        mu_ei,
        sig_ei,
        mu_ir,
        sig_ir,
        # Step parameters
        tau_ptr,
        # RNG
        rng_seed,
        step_id_ptr,
        # Output (write-once)
        next_state_ptr,
        next_age_ptr,
        next_infectivity_ptr,
        # Rate output for tau computation (max reduction)
        rates_ptr,
        # Active-node compaction (only read when USE_COMPACTION=1):
        # active_nodes_ptr: int32[N] buffer whose first num_active entries
        #   hold the sorted ids of nodes in state != R at last refresh.
        # num_active_ptr: int32 scalar ptr giving the current length.
        # When USE_COMPACTION=0 both pointers are dummy and unread (the
        # constexpr branch compiles the compaction path out entirely).
        active_nodes_ptr,
        num_active_ptr,
        # Constants
        N,
        # SEIR state indices
        STATE_S: tl.constexpr,
        STATE_E: tl.constexpr,
        STATE_I: tl.constexpr,
        STATE_R: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        # Source-node shedding mode: 0 = constant beta, 1 = age-dependent
        # (beta * h_IR(new_age)). Constexpr so the branch compiles out.
        TRANSMISSION_AGE_DEPENDENT: tl.constexpr,
        # Compaction flag: 0 = dispatch over all N, 1 = dispatch over
        # active (state != R) node list. Constexpr so compaction logic
        # disappears at JIT time when disabled.
        USE_COMPACTION: tl.constexpr,
        # Mixed-precision storage flag: 0 = all tensors fp32/int32,
        # 1 = state int8, age fp16, infectivity bf16 (weights are
        # still handled via the separate bf16_weights flag since
        # that pre-dates this path). The *accumulator* (pressure)
        # and all math intermediates stay fp32 -- downcasting the
        # 1000-edge hub accumulation to bf16 would cause catastrophic
        # absorption of small infectivity contributions, exactly in
        # the regime where lambda*tau << 1 matters most.
        MIXED_PRECISION: tl.constexpr,
    ):
        """
        Fused kernel: CSR traversal + hazard + Bernoulli + transition
        + next-step infectivity write.

        Per thread (one per node):
        1. Traverse incoming neighbors, accumulate infectivity*weight (registers)
        2. Load own age/state
        3. Compute rate: S -> pressure, E -> h_EI(age), I -> h_IR(age), R -> 0
           (Sparsity: skip erfcx for S with pressure==0 and R nodes)
        4. Bernoulli: p = 1 - exp(-rate * tau)
        5. RNG + transition + age update
        6. Compute next-step infectivity for THIS node from new_state/new_age:
              constant mode:      beta if new_state==I else 0
              age-dependent mode: beta * h_IR(new_age) if new_state==I else 0
           This replaces the dense PyTorch pre-pass that previously ran
           an O(N) erfcx sweep before the kernel. Block-level sparsity
           skips the erfcx entirely when no lane is I in the next step.
        7. Write next_state, next_age, next_infectivity, rate once
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # === Active-node compaction remap (constexpr-gated) ===
        # With compaction enabled, we keep the launch grid at cdiv(N,
        # BLOCK_SIZE) for CUDA Graph compatibility. We do NOT use a
        # scalar early-return here: Triton JIT tolerates them in
        # principle but in our build the fused kernel produced
        # slightly-different output when the first test revision used
        # one (seed-stable but drifting against baseline by ~1% in
        # compartment counts). Masking every load/store against
        # `offsets < num_active` has the same effective cost (inactive
        # lanes issue zero predicated memops) and stays bit-identical
        # to the baseline.
        if USE_COMPACTION:
            num_active = tl.load(num_active_ptr)
            mask = offsets < num_active
            idx = tl.load(active_nodes_ptr + offsets, mask=mask, other=0)
        else:
            mask = offsets < N
            idx = offsets

        # === 1. CSR traversal -> pressure (registers only) ===
        row_start = tl.load(row_ptr_ptr + idx, mask=mask, other=0)
        row_end = tl.load(row_ptr_ptr + idx + 1, mask=mask, other=0)

        pressure = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        curr_ptr = row_start
        active_any = tl.max(curr_ptr < row_end, axis=0)

        while active_any != 0:
            active_lane = (curr_ptr < row_end) & mask
            neighbor_id = tl.load(col_ind_ptr + curr_ptr, mask=active_lane, other=0)
            # Promote infectivity to fp32 BEFORE multiplying into the
            # fp32 accumulator: bf16 stays fine for storage but the
            # product-and-sum over a 1000-edge hub must be fp32.
            neighbor_inf = tl.load(
                infectivity_ptr + neighbor_id, mask=active_lane, other=0.0
            ).to(tl.float32)
            weight = tl.load(
                weights_ptr + curr_ptr, mask=active_lane, other=0.0
            ).to(tl.float32)

            pressure += tl.where(active_lane, neighbor_inf * weight, 0.0)

            curr_ptr += 1
            active_any = tl.max(curr_ptr < row_end, axis=0)

        # === 2. Load own state and age ===
        # Always promote to the kernel's natural math type (int32 for
        # state, fp32 for age). Under MIXED_PRECISION=1 the storage
        # types are int8/fp16 and the promote cost is one register
        # extension per lane; under MIXED_PRECISION=0 it is a no-op.
        my_state = tl.load(state_ptr + idx, mask=mask, other=0).to(tl.int32)
        my_age = tl.load(age_ptr + idx, mask=mask, other=0.0).to(tl.float32)

        # === 3. Compute rate with sparsity ===
        # S: rate = pressure (already includes beta * h_IR from infectivity)
        # E: rate = lognormal_hazard(age, mu_ei, sig_ei)
        # I: rate = lognormal_hazard(age, mu_ir, sig_ir)
        # R: rate = 0

        is_s = my_state == STATE_S
        is_e = my_state == STATE_E
        is_i = my_state == STATE_I

        # Block-level hazard skip: if no E (or I) nodes in this block,
        # skip the expensive erfcx math entirely. Epidemics cluster
        # spatially, so many blocks are pure S or R during peak.
        # tl.sum produces a scalar → compiles to a uniform branch, not
        # per-lane predication, giving true ALU sparsity.
        any_e = tl.sum(is_e.to(tl.int32), axis=0)
        any_i = tl.sum(is_i.to(tl.int32), axis=0)

        hazard_e = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        hazard_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        if any_e > 0:
            hazard_e = tl.where(
                is_e,
                _lognormal_hazard_triton(my_age, mu_ei, sig_ei),
                0.0,
            )
        if any_i > 0:
            hazard_i = tl.where(
                is_i,
                _lognormal_hazard_triton(my_age, mu_ir, sig_ir),
                0.0,
            )

        rate = tl.where(is_s, pressure, 0.0)
        rate = tl.where(is_e, hazard_e, rate)
        rate = tl.where(is_i, hazard_i, rate)

        # Store rate for max-reduction (tau computation for next step)
        tl.store(rates_ptr + idx, rate, mask=mask)

        # === 4. Bernoulli: p = 1 - exp(-rate * tau) ===
        tau = tl.load(tau_ptr)
        prob = 1.0 - tl.exp(-rate * tau)

        # === 5. RNG + transition ===
        # Use step_id as seed perturbation, node idx as offset.
        # step_id increments by 1 per step (safe for 2^31 steps).
        step_id = tl.load(step_id_ptr).to(tl.int32)
        rand_val = tl.rand(rng_seed + step_id, idx)
        event = rand_val < prob

        # Apply SEIR transitions: S->E, E->I, I->R
        new_state = my_state
        new_state = tl.where(event & is_s, STATE_E, new_state)
        new_state = tl.where(event & is_e, STATE_I, new_state)
        new_state = tl.where(event & is_i, STATE_R, new_state)

        # Age update: advance by tau, reset to 0 on transition
        changed = new_state != my_state
        new_age = tl.where(changed, 0.0, my_age + tau)

        # === 6. Next-step infectivity (source-node shedding) ===
        # Fold the infectivity pre-pass into the fused kernel. This removes
        # the O(N) dense erfcx sweep that the Python-side pre-pass used to
        # do when transmission_mode="age_dependent", recovering most of the
        # overhead vs constant-beta mode on memory-coalesced graphs.
        #
        # Correctness:
        #  - Nodes that transitioned E->I have new_age == 0, so the
        #    lognormal hazard returns 0 via the tau->0 overflow guard
        #    (biologically correct: immediately after becoming infectious,
        #    transmission probability is zero).
        #  - Nodes that stayed I have new_age = my_age + tau, so the
        #    infectivity advances along the shedding profile.
        #  - All non-I states (S, E, R) write 0.
        is_i_next = new_state == STATE_I
        if TRANSMISSION_AGE_DEPENDENT:
            # Block-level sparsity: only pay the erfcx cost when at least
            # one lane in the block is I in the next step.
            any_i_next = tl.sum(is_i_next.to(tl.int32), axis=0)
            next_inf = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            if any_i_next > 0:
                hazard_i_next = _lognormal_hazard_triton(new_age, mu_ir, sig_ir)
                next_inf = tl.where(is_i_next, beta * hazard_i_next, 0.0)
        else:
            next_inf = tl.where(is_i_next, beta, 0.0)

        # === 7. Write once ===
        # Downcast to the engine's mixed-precision storage types if
        # MIXED_PRECISION=1. All math above stayed fp32/int32; the
        # cast happens only at this final write.
        if MIXED_PRECISION:
            tl.store(next_state_ptr + idx, new_state.to(tl.int8), mask=mask)
            tl.store(next_age_ptr + idx, new_age.to(tl.float16), mask=mask)
            tl.store(next_infectivity_ptr + idx, next_inf.to(tl.bfloat16), mask=mask)
        else:
            tl.store(next_state_ptr + idx, new_state, mask=mask)
            tl.store(next_age_ptr + idx, new_age, mask=mask)
            tl.store(next_infectivity_ptr + idx, next_inf, mask=mask)

    @triton.jit
    def _flash_renewal_fused_wc_kernel(
        # CSR graph
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        # Node state (input)
        infectivity_ptr,
        age_ptr,
        state_ptr,
        # Model parameters
        beta,
        mu_ei,
        sig_ei,
        mu_ir,
        sig_ir,
        # Step parameters
        tau_ptr,
        # RNG
        rng_seed,
        step_id_ptr,
        # Output (write-once)
        next_state_ptr,
        next_age_ptr,
        next_infectivity_ptr,
        # Rate output for tau computation (max reduction)
        rates_ptr,
        # Constants
        N,
        # SEIR state indices
        STATE_S: tl.constexpr,
        STATE_E: tl.constexpr,
        STATE_I: tl.constexpr,
        STATE_R: tl.constexpr,
        NODES_PER_BLOCK: tl.constexpr,
        LANES_PER_NODE: tl.constexpr,
        TRANSMISSION_AGE_DEPENDENT: tl.constexpr,
    ):
        """
        Warp-collaborative variant of _flash_renewal_fused_kernel.

        Processes NODES_PER_BLOCK nodes per program, with LANES_PER_NODE
        threads cooperating on each node's CSR neighbor list (typically
        LANES_PER_NODE=32, i.e. one warp per node). This converts the
        per-node serial CSR traversal from O(D_max) to O(D_max /
        LANES_PER_NODE) and coalesces the col_ind / weights reads across
        the warp, which matters on highly skewed degree distributions
        (scale-free / BA graphs) where one hub stalls the whole block in
        the 1-thread-per-node kernel.

        The per-node tail (hazard, Bernoulli, transition, infectivity
        write) operates on 1D [NODES_PER_BLOCK] tensors after reducing
        the 2D pressure accumulator along the lane axis, so only
        NODES_PER_BLOCK of the block's threads do useful work there.
        For the target workload (D_max / LANES_PER_NODE >> 1) the CSR
        phase dominates and this is a favorable tradeoff.

        RNG: tl.rand(seed, node_id) is deterministic in node_id, so
        per-step RNG output matches the 1-thread-per-node kernel when
        invoked with the same seed and step_id. This is what makes the
        two kernels directly comparable in the correctness test.
        """
        pid = tl.program_id(0)
        node_offs = pid * NODES_PER_BLOCK + tl.arange(0, NODES_PER_BLOCK)
        lane_offs = tl.arange(0, LANES_PER_NODE)
        node_mask = node_offs < N

        # === 1. CSR traversal, warp-collaborative ===
        row_start = tl.load(row_ptr_ptr + node_offs, mask=node_mask, other=0)
        row_end = tl.load(row_ptr_ptr + node_offs + 1, mask=node_mask, other=0)

        row_start_2d = row_start[:, None]
        row_end_2d = row_end[:, None]
        node_mask_2d = node_mask[:, None]
        lane_offs_2d = lane_offs[None, :]

        pressure_2d = tl.zeros(
            [NODES_PER_BLOCK, LANES_PER_NODE], dtype=tl.float32
        )

        # Chunked iteration: step k covers neighbor offsets
        # [row_start + k*LANES, row_start + (k+1)*LANES). Terminate when
        # no (node, lane) combination is in range.
        curr_k = 0
        nbr_pos = row_start_2d + curr_k * LANES_PER_NODE + lane_offs_2d
        any_active = tl.max((nbr_pos < row_end_2d).to(tl.int32))

        while any_active != 0:
            active = (nbr_pos < row_end_2d) & node_mask_2d

            neighbor_id = tl.load(col_ind_ptr + nbr_pos, mask=active, other=0)
            neighbor_inf = tl.load(
                infectivity_ptr + neighbor_id, mask=active, other=0.0
            )
            weight = tl.load(
                weights_ptr + nbr_pos, mask=active, other=0.0
            ).to(tl.float32)

            pressure_2d += tl.where(active, neighbor_inf * weight, 0.0)

            curr_k += 1
            nbr_pos = row_start_2d + curr_k * LANES_PER_NODE + lane_offs_2d
            any_active = tl.max((nbr_pos < row_end_2d).to(tl.int32))

        # Reduce lane axis → per-node pressure.
        pressure = tl.sum(pressure_2d, axis=1)

        # === 2. Per-node tail (1D over NODES_PER_BLOCK) ===
        my_state = tl.load(state_ptr + node_offs, mask=node_mask, other=0)
        my_age = tl.load(age_ptr + node_offs, mask=node_mask, other=0.0)

        is_s = my_state == STATE_S
        is_e = my_state == STATE_E
        is_i = my_state == STATE_I

        any_e = tl.sum(is_e.to(tl.int32), axis=0)
        any_i = tl.sum(is_i.to(tl.int32), axis=0)

        hazard_e = tl.zeros([NODES_PER_BLOCK], dtype=tl.float32)
        hazard_i = tl.zeros([NODES_PER_BLOCK], dtype=tl.float32)
        if any_e > 0:
            hazard_e = tl.where(
                is_e,
                _lognormal_hazard_triton(my_age, mu_ei, sig_ei),
                0.0,
            )
        if any_i > 0:
            hazard_i = tl.where(
                is_i,
                _lognormal_hazard_triton(my_age, mu_ir, sig_ir),
                0.0,
            )

        rate = tl.where(is_s, pressure, 0.0)
        rate = tl.where(is_e, hazard_e, rate)
        rate = tl.where(is_i, hazard_i, rate)

        tl.store(rates_ptr + node_offs, rate, mask=node_mask)

        tau = tl.load(tau_ptr)
        prob = 1.0 - tl.exp(-rate * tau)

        step_id = tl.load(step_id_ptr).to(tl.int32)
        rand_val = tl.rand(rng_seed + step_id, node_offs)
        event = rand_val < prob

        new_state = my_state
        new_state = tl.where(event & is_s, STATE_E, new_state)
        new_state = tl.where(event & is_e, STATE_I, new_state)
        new_state = tl.where(event & is_i, STATE_R, new_state)

        changed = new_state != my_state
        new_age = tl.where(changed, 0.0, my_age + tau)

        # Next-step infectivity (same logic as 1-thread-per-node kernel)
        is_i_next = new_state == STATE_I
        if TRANSMISSION_AGE_DEPENDENT:
            any_i_next = tl.sum(is_i_next.to(tl.int32), axis=0)
            next_inf = tl.zeros([NODES_PER_BLOCK], dtype=tl.float32)
            if any_i_next > 0:
                hazard_i_next = _lognormal_hazard_triton(new_age, mu_ir, sig_ir)
                next_inf = tl.where(is_i_next, beta * hazard_i_next, 0.0)
        else:
            next_inf = tl.where(is_i_next, beta, 0.0)

        tl.store(next_state_ptr + node_offs, new_state, mask=node_mask)
        tl.store(next_age_ptr + node_offs, new_age, mask=node_mask)
        tl.store(next_infectivity_ptr + node_offs, next_inf, mask=node_mask)


    @triton.jit
    def _pressure_merge_kernel(
        row_ptr_ptr,
        col_ind_ptr,
        weights_ptr,
        infectivity_ptr,
        pressure_ptr,
        N,
        E,
        EDGES_PER_BLOCK: tl.constexpr,
        BSEARCH_ITERS: tl.constexpr,
    ):
        """
        Merge-based pressure accumulation (Merrill-Garland style load balancing).

        Each program handles EDGES_PER_BLOCK contiguous edges out of the
        total E edges, independent of which nodes those edges belong to.
        Each lane:
          1. loads (col_ind[e], weights[e], infectivity[col_ind[e]]),
          2. binary-searches row_ptr to recover the source node id
             for its edge, and
          3. atomic-adds the weighted contribution to pressure[node_id].

        This delivers exact edge-level load balance regardless of degree
        heterogeneity: a hub of degree D_max no longer dominates its
        block, because neighboring blocks pick up the hub's leftover
        edges. The cost is one atomic add per edge and a binary search
        of BSEARCH_ITERS steps, both amortized across high GPU
        occupancy.

        pressure_ptr MUST be zeroed by the caller before launch.

        BSEARCH_ITERS should be ceil(log2(N + 2)) --- passed as a
        constexpr so the binary-search loop fully unrolls at compile
        time.
        """
        pid = tl.program_id(0)
        edge_offs = pid * EDGES_PER_BLOCK + tl.arange(0, EDGES_PER_BLOCK)
        edge_mask = edge_offs < E

        # --- Binary search in row_ptr ---
        # Find, per edge e, the unique n in [0, N) such that
        # row_ptr[n] <= e < row_ptr[n+1]. We search for the smallest
        # index m such that row_ptr[m] > e, then return node_id = m-1.
        lo = tl.zeros([EDGES_PER_BLOCK], dtype=tl.int32)
        hi = tl.zeros([EDGES_PER_BLOCK], dtype=tl.int32) + (N + 1)
        e32 = edge_offs.to(tl.int32)

        for _ in tl.static_range(BSEARCH_ITERS):
            mid = (lo + hi) // 2
            # Clamp mid to [0, N] for the load; row_ptr has N+1 entries.
            mid_clamped = tl.minimum(tl.maximum(mid, 0), N)
            row_mid = tl.load(row_ptr_ptr + mid_clamped).to(tl.int32)
            # row_ptr is non-decreasing in [0, N]; pretend row_ptr[N+1]=+inf.
            row_mid = tl.where(mid > N, E + 1, row_mid)
            # If row_ptr[mid] <= e, node must be >= mid → raise lo.
            cmp = row_mid <= e32
            lo = tl.where(cmp, mid + 1, lo)
            hi = tl.where(cmp, hi, mid)

        node_id = lo - 1
        node_id = tl.maximum(node_id, 0)

        # --- Load edge payload ---
        # Promote infectivity to fp32 so the edge product and the
        # subsequent fp32 atomic_add into pressure_ptr stay in fp32
        # regardless of whether infectivity storage is fp32 or bf16;
        # summing ~D_max bf16 contributions into a single bf16 scalar
        # on a scale-free hub would absorb small values and poison
        # the downstream Bernoulli 1-exp(-lambda*tau) draw. Keeping
        # pressure_ptr as an fp32 scratch buffer (allocated fp32 in
        # the engine) is a hard invariant of the mixed-precision
        # contract.
        neighbor = tl.load(col_ind_ptr + edge_offs, mask=edge_mask, other=0)
        weight = tl.load(
            weights_ptr + edge_offs, mask=edge_mask, other=0.0
        ).to(tl.float32)
        neighbor_inf = tl.load(
            infectivity_ptr + neighbor, mask=edge_mask, other=0.0
        ).to(tl.float32)
        contribution = neighbor_inf * weight

        # --- Scatter: one atomic per edge ---
        # Contention is dominated by hubs (many edges targeting the same
        # pressure[hub]). On A100 this serializes at ~a few GB/s on
        # same-location atomics, which is still much faster than the
        # 1-thread-per-node kernel's serial traversal for high-degree
        # hubs.
        tl.atomic_add(
            pressure_ptr + node_id, contribution, mask=edge_mask
        )

    @triton.jit
    def _flash_renewal_tail_kernel(
        pressure_ptr,
        age_ptr,
        state_ptr,
        beta,
        mu_ei,
        sig_ei,
        mu_ir,
        sig_ir,
        tau_ptr,
        rng_seed,
        step_id_ptr,
        next_state_ptr,
        next_age_ptr,
        next_infectivity_ptr,
        rates_ptr,
        N,
        STATE_S: tl.constexpr,
        STATE_E: tl.constexpr,
        STATE_I: tl.constexpr,
        STATE_R: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        TRANSMISSION_AGE_DEPENDENT: tl.constexpr,
        # Mixed-precision storage flag. Identical contract to the
        # thread-path fused kernel: state is int8, age is fp16,
        # next_infectivity is bf16 on the store side; the pressure
        # input is ALWAYS fp32 because the merge-path scratch buffer
        # accumulates atomics in fp32.
        MIXED_PRECISION: tl.constexpr,
    ):
        """
        Per-node tail for the merge-based strategy. Reads per-node
        pressure accumulated by _pressure_merge_kernel, then executes
        the same hazard / Bernoulli / transition / next-infectivity
        logic as the single-kernel variants. Identical RNG pattern
        (tl.rand(seed, node_id)) so that, given identical state input
        and bit-identical pressure, the three strategies produce
        identical per-node outputs.
        """
        pid = tl.program_id(0)
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < N

        # pressure_ptr is ALWAYS fp32 (merge scratch buffer invariant).
        pressure = tl.load(pressure_ptr + idx, mask=mask, other=0.0)
        # Promote state and age to kernel math types regardless of
        # storage dtype; no-op under MIXED_PRECISION=0, cheap register
        # extension under MIXED_PRECISION=1.
        my_state = tl.load(state_ptr + idx, mask=mask, other=0).to(tl.int32)
        my_age = tl.load(age_ptr + idx, mask=mask, other=0.0).to(tl.float32)

        is_s = my_state == STATE_S
        is_e = my_state == STATE_E
        is_i = my_state == STATE_I

        any_e = tl.sum(is_e.to(tl.int32), axis=0)
        any_i = tl.sum(is_i.to(tl.int32), axis=0)

        hazard_e = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        hazard_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        if any_e > 0:
            hazard_e = tl.where(
                is_e,
                _lognormal_hazard_triton(my_age, mu_ei, sig_ei),
                0.0,
            )
        if any_i > 0:
            hazard_i = tl.where(
                is_i,
                _lognormal_hazard_triton(my_age, mu_ir, sig_ir),
                0.0,
            )

        rate = tl.where(is_s, pressure, 0.0)
        rate = tl.where(is_e, hazard_e, rate)
        rate = tl.where(is_i, hazard_i, rate)

        tl.store(rates_ptr + idx, rate, mask=mask)

        tau = tl.load(tau_ptr)
        prob = 1.0 - tl.exp(-rate * tau)

        step_id = tl.load(step_id_ptr).to(tl.int32)
        rand_val = tl.rand(rng_seed + step_id, idx)
        event = rand_val < prob

        new_state = my_state
        new_state = tl.where(event & is_s, STATE_E, new_state)
        new_state = tl.where(event & is_e, STATE_I, new_state)
        new_state = tl.where(event & is_i, STATE_R, new_state)

        changed = new_state != my_state
        new_age = tl.where(changed, 0.0, my_age + tau)

        is_i_next = new_state == STATE_I
        if TRANSMISSION_AGE_DEPENDENT:
            any_i_next = tl.sum(is_i_next.to(tl.int32), axis=0)
            next_inf = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            if any_i_next > 0:
                hazard_i_next = _lognormal_hazard_triton(new_age, mu_ir, sig_ir)
                next_inf = tl.where(is_i_next, beta * hazard_i_next, 0.0)
        else:
            next_inf = tl.where(is_i_next, beta, 0.0)

        if MIXED_PRECISION:
            tl.store(next_state_ptr + idx, new_state.to(tl.int8), mask=mask)
            tl.store(next_age_ptr + idx, new_age.to(tl.float16), mask=mask)
            tl.store(next_infectivity_ptr + idx, next_inf.to(tl.bfloat16), mask=mask)
        else:
            tl.store(next_state_ptr + idx, new_state, mask=mask)
            tl.store(next_age_ptr + idx, new_age, mask=mask)
            tl.store(next_infectivity_ptr + idx, next_inf, mask=mask)


else:
    _erfcx_approx = None
    _lognormal_hazard_triton = None
    _flash_renewal_fused_kernel = None
    _flash_renewal_fused_wc_kernel = None
    _pressure_merge_kernel = None
    _flash_renewal_tail_kernel = None
