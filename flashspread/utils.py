"""
Small cross-cutting utilities for FlashSpread.

These helpers intentionally avoid importing the optional dependencies
(triton / networkx / scipy) at module load so that ``import flashspread``
stays lightweight and CPU-importable.
"""

from __future__ import annotations

import importlib.util

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a device, defaulting to CUDA when available else CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _has(module: str) -> bool:
    """True if ``module`` is importable, without importing it.

    ``find_spec`` can raise (e.g. ModuleNotFoundError for a missing parent,
    or a broken/partly-installed package); treat any failure as "absent".
    """
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def check_env(verbose: bool = True) -> dict:
    """Report the runtime environment and whether GPU engines are usable.

    Returns a dict describing torch / CUDA / optional-dependency
    availability. ``gpu_ready`` is True only when a CUDA device *and*
    Triton are both present (the requirement for the fused engines).

    Example:
        >>> import flashspread
        >>> flashspread.check_env()            # doctest: +SKIP
    """
    from . import __version__

    cuda = torch.cuda.is_available()
    info = {
        "flashspread": __version__,
        "torch": torch.__version__,
        "cuda_available": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "triton": _has("triton"),
        "networkx": _has("networkx"),
        "scipy": _has("scipy"),
    }
    info["gpu_ready"] = bool(cuda and info["triton"])

    if verbose:
        width = max(len(k) for k in info)
        for key, value in info.items():
            print(f"{key:<{width}} : {value}")
        if not info["gpu_ready"]:
            print(
                "\nNOTE: the fused GPU engines need a CUDA device and Triton "
                "(`pip install flashspread[gpu]`). The reference/CPU path "
                "still runs without them."
            )
        if not info["networkx"]:
            print(
                "NOTE: graph generators need NetworkX "
                "(`pip install flashspread[graph]`)."
            )
    return info
