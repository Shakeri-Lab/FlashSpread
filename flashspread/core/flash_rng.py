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
    def _bernoulli_from_words(probability, random_hi, random_lo):
        """Compare a probability against a 64-bit threshold in two uint32 words.

        Both random words MUST be uint32. Triton's ``philox`` selects its word
        width from the *counter* dtype (``triton/language/random.py``: a 64-bit
        counter takes the ``uint64`` branch), so a widened counter silently
        returns uint64 words. Comparing those against a zero-extended uint32
        threshold scales every probability by 2**-32 and suppresses every event.
        The assertions below make that failure a compile error instead.
        """
        tl.static_assert(
            random_hi.dtype == tl.uint32,
            "Philox high word must be uint32; a 64-bit counter widens it and "
            "silently scales every event probability by 2**-32",
        )
        tl.static_assert(
            random_lo.dtype == tl.uint32,
            "Philox low word must be uint32; a 64-bit counter widens it and "
            "silently scales every event probability by 2**-32",
        )
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

    @triton.jit
    def _sample_bernoulli(probability, key, offset):
        """Sample with a 64-bit threshold assembled from one Philox call.

        ``tl.rand`` exposes only about 31 useful bits and maps exact zero to an
        event for every positive probability. Two uint32 words retain
        probabilities down to 2**-64 without slow fp64 kernel arithmetic.

        ``offset`` must be a 32-bit lane index so Philox stays in its uint32
        word family. Callers whose lane identity needs more than 32 bits use
        :func:`_sample_bernoulli_counter` instead of widening this argument.
        """
        random_hi, random_lo, _, _ = tl.randint4x(key, offset)
        return _bernoulli_from_words(probability, random_hi, random_lo)

    @triton.jit
    def _sample_bernoulli_counter(probability, key, counter_lo, counter_hi):
        """Sample from a two-word uint32 Philox counter.

        Philox4x32 is natively counter-addressed, so a lane identity that does
        not fit one uint32 word belongs in a second *counter word* rather than
        in a widened first word. Both words stay uint32, which keeps the cheap
        32-bit round constants and the shared threshold contract above.
        """
        random_hi, random_lo, _, _ = tl.philox(
            key,
            counter_lo.to(tl.uint32),
            counter_hi.to(tl.uint32),
            tl.zeros_like(counter_lo).to(tl.uint32),
            tl.zeros_like(counter_lo).to(tl.uint32),
        )
        return _bernoulli_from_words(probability, random_hi, random_lo)


else:
    _event_probability = None
    _mix_rng_key = None
    _bernoulli_from_words = None
    _sample_bernoulli = None
    _sample_bernoulli_counter = None


__all__ = [
    "_event_probability",
    "_mix_rng_key",
    "_sample_bernoulli",
    "_sample_bernoulli_counter",
]
