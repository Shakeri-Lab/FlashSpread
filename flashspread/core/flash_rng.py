"""Shared counter-based Triton RNG primitives for fused simulation kernels.

The helpers are deliberately stateless: a kernel derives one Philox stream
from ``(base_seed, step_id)`` and indexes it by node/replica offset. Keeping
this contract in one module prevents the Markovian and renewal fast paths from
silently drifting to different rare-event or seed-collision semantics.
"""

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the optional GPU stack
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _event_probability(rate_tau):
        """Stable ``1 - exp(-rate*tau)`` for Bernoulli tau-leaping."""
        x = tl.maximum(rate_tau, 0.0)
        probability_small = x * (
            1.0 + x * (-0.5 + x * (1.0 / 6.0 - x / 24.0))
        )
        return tl.where(
            x <= 0.1,
            probability_small,
            1.0 - tl.exp(-x),
        )

    @triton.jit
    def _mix_rng_key(base_seed, step_id):
        """Avalanche ``(base_seed, step_id)`` into Philox's full 64-bit key.

        Every operation is bijective in one 64-bit argument while the other is
        fixed, so neither distinct 64-bit seeds nor distinct 64-bit step ids are
        collapsed to the old 32-bit key family.
        """
        seed64 = base_seed.to(tl.uint64)
        step64 = step_id.to(tl.uint64)
        x = seed64 ^ (step64 * 0x9E3779B97F4A7C15)
        x = x ^ (x >> 30)
        x = x * 0xBF58476D1CE4E5B9
        x = x ^ (x >> 27)
        x = x * 0x94D049BB133111EB
        return (x ^ (x >> 31)).to(tl.uint64)

    @triton.jit
    def _sample_bernoulli(probability, key, offset):
        """Sample with a 64-bit threshold assembled from one Philox call.

        ``tl.rand`` exposes only about 31 useful bits and maps exact zero to an
        event for every positive probability. Two uint32 words retain
        probabilities down to 2**-64 without slow fp64 kernel arithmetic.
        """
        random_hi, random_lo, _, _ = tl.randint4x(key, offset)
        two32: tl.constexpr = 4294967296.0
        safe_probability = tl.minimum(
            tl.maximum(probability, 0.0),
            0.99999994,
        )
        scaled_hi = safe_probability * two32
        threshold_hi_float = tl.floor(scaled_hi)
        threshold_hi = threshold_hi_float.to(tl.uint32)
        threshold_lo = tl.floor(
            (scaled_hi - threshold_hi_float) * two32
        ).to(tl.uint32)
        below_hi = random_hi < threshold_hi
        equal_hi_below_lo = (random_hi == threshold_hi) & (
            random_lo < threshold_lo
        )
        return (probability >= 1.0) | below_hi | equal_hi_below_lo


else:
    _event_probability = None
    _mix_rng_key = None
    _sample_bernoulli = None


__all__ = ["_event_probability", "_mix_rng_key", "_sample_bernoulli"]
