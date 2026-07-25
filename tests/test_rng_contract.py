"""Compile-time contract tests for the shared counter-based Triton RNG.

Triton's ``philox`` selects its word width from the *counter* dtype
(``triton/language/random.py``). ``_bernoulli_from_words`` builds its threshold in
two uint32 words, so 64-bit random words silently compare a 2**64-range sample
against a 2**32-range threshold, scaling every event probability by 2**-32 -- the
ensemble then advances every clock and samples nothing.

Whether a 64-bit *counter* actually produces 64-bit words is version-dependent,
which is what made this dangerous. Measured on an A100: Triton 3.1.0, 3.2.0 and
3.3.1 return uint64 words and the historical packed counter accepted zero
transitions in every replica; 3.6.0 and 3.7.1 return uint32 words and it worked.
The package therefore does not rely on the observed behaviour -- it keeps counters
in two uint32 words -- and the tests below assert the *safety property* rather
than any one version's dtype choice.

The failure is invisible to a throughput benchmark, and on some Triton versions
also invisible to the interpreter (its kernel stores do not reach host tensors).
The ``tl.static_assert`` guards in ``flash_rng`` therefore turn it into a
*trace-time* error, which the interpreter does report through the subprocess exit
status. These tests pin that without needing a GPU; empirical sampling rates are
asserted on hardware in ``test_ensemble_gpu.py``.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytest.importorskip("triton", reason="the counter-RNG contract needs Triton")


_PREAMBLE = """
import torch
import triton
import triton.language as tl

from flashspread.core.flash_rng import (
    _bernoulli_from_words,
    _mix_rng_key,
    _sample_bernoulli,
    _sample_bernoulli_counter,
)
"""


def _run_under_interpreter(script: str) -> subprocess.CompletedProcess:
    """Trace ``script`` with the Triton interpreter and return the result."""
    repository = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    return subprocess.run(
        [sys.executable, "-c", _PREAMBLE + script],
        cwd=repository,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )


def test_widened_random_words_are_rejected_at_trace_time():
    """The guard must reject 64-bit random words on every Triton version.

    This constructs the widened words directly instead of going through a
    widened *counter*, because whether a uint64 counter actually widens the
    words is version-dependent (see the next test). The invariant the guard
    protects -- ``_bernoulli_from_words`` never sees anything but uint32 -- is
    not version-dependent, so this is the assertion worth pinning.
    """
    result = _run_under_interpreter(
        """
@triton.jit
def kernel(out, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    # Words as they arrive from a 64-bit Philox: uniform over 2**64, compared
    # against a threshold built in two uint32 words.
    high = idx.to(tl.uint64)
    low = idx.to(tl.uint64)
    event = _bernoulli_from_words(tl.full([BLOCK], 0.5, tl.float32), high, low)
    tl.store(out + idx, event.to(tl.int32))


kernel[(1,)](torch.zeros(64, dtype=torch.int32), BLOCK=64)
"""
    )
    assert result.returncode != 0, (
        "64-bit random words must not trace successfully: compared against a "
        "uint32 threshold they scale every event probability by 2**-32\n"
        + result.stdout
        + result.stderr
    )
    assert "uint32" in (result.stdout + result.stderr)


def test_packed_counter_is_safe_or_rejected_on_this_triton():
    """Whichever width this Triton returns, the composition must be safe.

    ``philox`` selects its word width from the counter dtype, but what it does
    for a *64-bit* counter has changed across releases: Triton 3.1-3.3 return
    uint64 words (measured), while 3.6-3.7 return uint32. So a packed 64-bit
    counter is silently wrong on some versions and harmless on others -- which
    is exactly why the package keeps its counters in two uint32 words and does
    not rely on the observed behaviour.

    This test asserts the safety property rather than a version: either the
    words come back narrow and tracing succeeds, or they come back wide and the
    guard refuses to compile. It must never trace successfully *with* wide words.
    """
    result = _run_under_interpreter(
        """
import sys

@triton.jit
def width_kernel(out, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    high, _, _, _ = tl.randint4x(tl.full((), 99, tl.uint64), idx.to(tl.uint64))
    tl.store(out + idx, tl.full([BLOCK], high.dtype.primitive_bitwidth, tl.int32))


@triton.jit
def sample_kernel(out, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    key = _mix_rng_key(tl.full((), 12345, tl.int64), tl.full((), 1, tl.int64))
    event = _sample_bernoulli(
        tl.full([BLOCK], 0.5, tl.float32), key, idx.to(tl.uint64)
    )
    tl.store(out + idx, event.to(tl.int32))


# Trace-time only: the interpreter need not deliver stores for this to be
# meaningful, because a rejected dtype raises while tracing.
try:
    width_kernel[(1,)](torch.zeros(64, dtype=torch.int32), BLOCK=64)
    width_traced = True
except Exception:
    width_traced = False

try:
    sample_kernel[(1,)](torch.zeros(64, dtype=torch.int32), BLOCK=64)
    sampled = True
    detail = ""
except Exception as exc:
    sampled = False
    detail = repr(exc)

if sampled and "uint32" in detail:
    raise SystemExit("inconsistent: traced yet reported a dtype rejection")
print("SAMPLED" if sampled else "REJECTED", detail[:200])
"""
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout
    assert ("SAMPLED" in output) or ("REJECTED" in output), output
    if "REJECTED" in output:
        # Wide words: the guard must be what refused it.
        assert "uint32" in output, (
            "a packed counter was rejected for some reason other than the word "
            "width guard: " + output
        )


def test_int32_offset_and_two_word_counter_both_trace():
    """The two supported counter shapes must keep Philox in its uint32 family."""
    result = _run_under_interpreter(
        """
@triton.jit
def scalar_kernel(out, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    key = _mix_rng_key(tl.full((), 12345, tl.int64), tl.full((), 1, tl.int64))
    event = _sample_bernoulli(tl.full([BLOCK], 0.5, tl.float32), key, idx)
    tl.store(out + idx, event.to(tl.int32))


@triton.jit
def ensemble_kernel(out, R: tl.constexpr, NODES: tl.constexpr):
    node = tl.arange(0, NODES)[:, None]
    replica = tl.arange(0, R)[None, :]
    probability = tl.full([NODES, R], 0.5, tl.float32)
    key = _mix_rng_key(tl.full((), 12345, tl.int64), tl.full((), 1, tl.int64))
    event = _sample_bernoulli_counter(
        probability,
        key,
        tl.broadcast_to(node.to(tl.uint32), probability.shape),
        tl.broadcast_to(replica.to(tl.uint32), probability.shape),
    )
    tl.store(out + node * R + replica, event.to(tl.int32))


scalar_kernel[(1,)](torch.zeros(64, dtype=torch.int32), BLOCK=64)
ensemble_kernel[(1,)](torch.zeros(16 * 4, dtype=torch.int32), R=4, NODES=16)
"""
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ensemble_transition_kernel_uses_two_word_counter():
    """Pin the call shape so a future edit cannot repack the counter."""
    source = (
        Path(__file__).resolve().parents[1]
        / "flashspread"
        / "core"
        / "flash_ensemble_step.py"
    ).read_text()
    assert "_sample_bernoulli_counter(" in source
    assert "<< 32) | node" not in source, (
        "the ensemble counter must stay in two uint32 words; packing node and "
        "replica into one 64-bit word changes Philox's word width"
    )
