"""Compile-time contract tests for the shared counter-based Triton RNG.

Triton's ``philox`` selects its word width from the *counter* dtype
(``triton/language/random.py``): a 64-bit counter takes the ``uint64`` branch and
returns uint64 random words. ``_bernoulli_from_words`` builds its threshold in
two uint32 words, so a widened counter silently compares a 2**64-range sample
against a 2**32-range threshold and scales every event probability by 2**-32 --
the ensemble then advances every clock and samples nothing.

That failure is invisible to a throughput benchmark and, on Triton 3.1, also
invisible to the interpreter (its kernel stores do not reach host tensors). The
``tl.static_assert`` guards in ``flash_rng`` therefore turn it into a *trace-time*
error, which the interpreter does report through the subprocess exit status.
These tests pin that behaviour without needing a GPU; the empirical sampling
rates are asserted on hardware in ``test_ensemble_gpu.py``.
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


def test_uint64_counter_is_rejected_at_trace_time():
    """A widened Philox counter must fail loudly instead of suppressing events."""
    result = _run_under_interpreter(
        """
@triton.jit
def kernel(out, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    key = _mix_rng_key(tl.full((), 12345, tl.int64), tl.full((), 1, tl.int64))
    # The historical ensemble counter: two ids packed into one 64-bit word.
    event = _sample_bernoulli(
        tl.full([BLOCK], 0.5, tl.float32), key, idx.to(tl.uint64)
    )
    tl.store(out + idx, event.to(tl.int32))


kernel[(1,)](torch.zeros(64, dtype=torch.int32), BLOCK=64)
"""
    )
    assert result.returncode != 0, (
        "a uint64 Philox counter must not trace successfully: it silently "
        "scales every event probability by 2**-32\n" + result.stdout + result.stderr
    )
    assert "uint32" in (result.stdout + result.stderr)


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
