"""Shared pytest configuration for FlashSpread.

Tests that require a CUDA device are marked ``gpu`` (see the ``gpu`` marker
registered in pyproject.toml). On a machine without CUDA those tests are
skipped automatically here, while pure-CPU tests still run — so CI on a CPU
runner exercises real logic instead of silently collecting zero tests.

Run only CPU tests explicitly with:  pytest -m "not gpu"
"""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
