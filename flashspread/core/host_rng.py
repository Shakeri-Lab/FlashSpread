"""Triton-free 64-bit seed helpers shared by host/reference engines."""

from __future__ import annotations

import operator

import torch


UINT64_MASK = (1 << 64) - 1
SPLITMIX_INCREMENT = -7046029254386353131  # 0x9E3779B97F4A7C15
SPLITMIX_MULTIPLIER_1 = -4658895280553007687  # 0xBF58476D1CE4E5B9
SPLITMIX_MULTIPLIER_2 = -7723592293110705685  # 0x94D049BB133111EB
STREAM_STRIDE = -3335678366873096957  # 0xD1B54A32D192ED03
GLOBAL_GENERATOR_DOMAIN = 0xA0761D6478BD642F
INITIAL_CONDITION_DOMAIN = 0xE7037ED1A0B428DB
MARKOV_GENERATOR_DOMAIN = 0x8EBC6AF09C88C6E3


def normalize_seed(seed: int, *, name: str = "seed") -> int:
    """Validate a PyTorch seed and return its unsigned 64-bit word."""
    if isinstance(seed, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        seed = operator.index(seed)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not -(1 << 63) <= seed <= UINT64_MASK:
        raise ValueError(
            f"{name} must lie in PyTorch's 64-bit seed range "
            f"[-2**63, 2**64-1], got {seed}"
        )
    return seed & UINT64_MASK


def offset_seed(base_seed: int, offset: int, *, name: str = "offset") -> int:
    """Add an episode/stream offset with explicit uint64 wraparound."""
    if isinstance(offset, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        offset = operator.index(offset)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    return (base_seed + offset) & UINT64_MASK


def signed_int64(word: int) -> int:
    """Interpret the low 64 bits as signed two's-complement storage."""
    word &= UINT64_MASK
    return word - (1 << 64) if word >= (1 << 63) else word


def splitmix64_word(value: int) -> int:
    """Pure-Python SplitMix64 round used to decorrelate host seeds."""
    word = (value + (SPLITMIX_INCREMENT & UINT64_MASK)) & UINT64_MASK
    word = ((word ^ (word >> 30)) * (SPLITMIX_MULTIPLIER_1 & UINT64_MASK))
    word &= UINT64_MASK
    word = ((word ^ (word >> 27)) * (SPLITMIX_MULTIPLIER_2 & UINT64_MASK))
    word &= UINT64_MASK
    return (word ^ (word >> 31)) & UINT64_MASK


def project_seed(base_seed: int, domain: int) -> int:
    """Mix all seed bits before handing a word to a backend generator."""
    return splitmix64_word((base_seed ^ domain) & UINT64_MASK)


def _fill_splitmix_counter_(counter: torch.Tensor, seed: int) -> torch.Tensor:
    """Fill contiguous int64 storage with seed-scrambled lane counters."""
    flat = counter.view(-1)
    torch.arange(
        flat.numel(),
        device=counter.device,
        dtype=torch.int64,
        out=flat,
    )
    flat.mul_(STREAM_STRIDE).add_(signed_int64(splitmix64_word(seed)))
    return counter


def _logical_right_shift(value: torch.Tensor, bits: int) -> torch.Tensor:
    """Logical uint64 shift expressed with supported signed int64 ops."""
    return (value >> bits) & ((1 << (64 - bits)) - 1)


def _splitmix_uniform_(
    counter: torch.Tensor,
    out: torch.Tensor,
    *,
    advance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance counters and write exact open 52-bit midpoint uniforms."""
    if advance is None:
        counter.add_(SPLITMIX_INCREMENT)
    else:
        counter.add_(advance.to(torch.int64) * SPLITMIX_INCREMENT)

    # Reuse fp64 output storage as int64 mixing scratch. The final conversion
    # overwrites it with exact, equally weighted binary64 midpoints in (0, 1).
    mixed = out.view(torch.int64)
    mixed.copy_(counter)
    mixed.bitwise_xor_(_logical_right_shift(mixed, 30))
    mixed.mul_(SPLITMIX_MULTIPLIER_1)
    mixed.bitwise_xor_(_logical_right_shift(mixed, 27))
    mixed.mul_(SPLITMIX_MULTIPLIER_2)
    mixed.bitwise_xor_(_logical_right_shift(mixed, 31))
    mantissa = _logical_right_shift(mixed, 12)
    out.copy_(mantissa).add_(0.5).mul_(2.0**-52)
    return out


__all__ = [
    "SPLITMIX_INCREMENT",
    "SPLITMIX_MULTIPLIER_1",
    "SPLITMIX_MULTIPLIER_2",
    "STREAM_STRIDE",
    "GLOBAL_GENERATOR_DOMAIN",
    "INITIAL_CONDITION_DOMAIN",
    "MARKOV_GENERATOR_DOMAIN",
    "UINT64_MASK",
    "normalize_seed",
    "offset_seed",
    "project_seed",
    "signed_int64",
    "splitmix64_word",
]
