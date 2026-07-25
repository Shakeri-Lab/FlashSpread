"""Shared pytest configuration for FlashSpread.

Tests that require a CUDA device are marked ``gpu`` (see the ``gpu`` marker
registered in pyproject.toml). On a machine without CUDA those tests are
skipped automatically here, while pure-CPU tests still run — so CI on a CPU
runner exercises real logic instead of silently collecting zero tests.

Run only CPU tests explicitly with:  pytest -m "not gpu"

A second class of tests runs real Triton kernels through ``TRITON_INTERPRET=1``
to get device-free kernel coverage. That interpreter is not uniformly capable
across the Triton versions this package supports, and when it is incapable it
fails in ways that look like package bugs, or — worse — succeeds while its
stores never reach the host tensors, making every assertion vacuous.
``triton_interpreter_skip_reason`` probes both properties directly instead of
hard-coding a version range, so the gate stays correct as Triton evolves.
"""

import pytest
import torch

from ._triton_support import triton_interpreter_skip_reason


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


#: Ready-made decorator for tests that execute kernels under the interpreter.
requires_triton_interpreter = pytest.mark.skipif(
    triton_interpreter_skip_reason() is not None,
    reason=triton_interpreter_skip_reason() or "",
)
