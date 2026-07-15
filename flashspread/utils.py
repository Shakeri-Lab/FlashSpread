"""
Small cross-cutting utilities for FlashSpread.

These helpers intentionally avoid importing the optional dependencies
(triton / networkx / scipy) at module load so that ``import flashspread``
stays lightweight and CPU-importable.
"""

from __future__ import annotations

import importlib.util
import math
import numbers
import operator
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from torch import Tensor as _Tensor
    from torch import device as _DeviceLike
else:
    # Keep runtime type-hint resolution meaningful without importing Torch;
    # static analysis still sees the concrete Torch types above.
    _Tensor = Any

    class _DeviceLike(Protocol):
        """Minimal runtime protocol used by import-light configuration code."""

        type: str


_FP32_MAX = 3.4028234663852886e38
_FP32_MIN_SUBNORMAL = 2.0**-149


def validate_fp32_control(
    name: str,
    value,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Validate a host scalar that will be passed/stored as fp32 on device."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    if abs(result) > _FP32_MAX or (
        result != 0.0 and abs(result) < _FP32_MIN_SUBNORMAL
    ):
        raise ValueError(
            f"{name}={result} is not representable as a nonzero finite fp32 scalar"
        )
    return result


def resolve_device(device: str | _DeviceLike | None = None) -> _DeviceLike:
    """Resolve a device, defaulting to CUDA when available else CPU."""
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch from one PyTorch-range 64-bit word."""
    import random

    import numpy as np
    import torch

    from .core.host_rng import (
        GLOBAL_GENERATOR_DOMAIN,
        normalize_seed,
        project_seed,
    )

    seed_word = normalize_seed(seed)
    generator_seed = project_seed(seed_word, GLOBAL_GENERATOR_DOMAIN)
    random.seed(generator_seed)
    # NumPy's legacy process-global RandomState accepts only uint32 seeds.
    # Engines keep their own full-width streams; this deterministic projection
    # exists solely for user model/setup code that consumes global NumPy RNG.
    np.random.seed(generator_seed & ((1 << 32) - 1))
    torch.manual_seed(generator_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(generator_seed)


def is_markovian(model) -> bool:
    """Whether a compartmental model is Markovian (SIS/SIR) vs renewal (SEIR).

    Prefers an explicit ``is_markovian`` class attribute; falls back to a
    capability check (renewal models expose ``compute_infectivity``).
    """
    if hasattr(model, "is_markovian"):
        if not isinstance(model.is_markovian, bool):
            raise TypeError("model.is_markovian must be a bool")
        return model.is_markovian
    return not hasattr(model, "compute_infectivity")


def state_names(model) -> list[str]:
    """Return single-letter state names in index order, e.g. ('S','E','I','R').

    Falls back to ``state{i}`` for any index without a known attribute.
    """
    idx_to_name = {}
    for attr in ("susceptible", "exposed", "infected", "recovered"):
        if hasattr(model, attr):
            idx_to_name[getattr(model, attr)] = attr[0].upper()
    return [idx_to_name.get(i, f"state{i}") for i in range(model.num_states)]


def validate_model_contract(
    model,
    *,
    markovian: bool,
    methods: tuple[str, ...],
) -> tuple[int, tuple[int, ...]]:
    """Validate the small model protocol shared by engine constructors."""
    if hasattr(model, "is_markovian"):
        if not isinstance(model.is_markovian, bool):
            raise TypeError("model.is_markovian must be a bool")
        if model.is_markovian != markovian:
            family = "Markovian" if markovian else "renewal"
            raise TypeError(f"{family} engine received the wrong model family")
    missing = [name for name in methods if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(f"model is missing callable methods: {', '.join(missing)}")
    if isinstance(getattr(model, "num_states", None), bool):
        raise TypeError("model.num_states must be a positive integer")
    try:
        num_states = operator.index(model.num_states)
    except (AttributeError, TypeError) as exc:
        raise TypeError("model.num_states must be a positive integer") from exc
    if num_states <= 0:
        raise ValueError("model.num_states must be positive")
    try:
        raw_inducers = tuple(model.inducer_states)
    except (AttributeError, TypeError) as exc:
        raise TypeError("model.inducer_states must be an iterable of integers") from exc
    if not raw_inducers:
        raise ValueError("model.inducer_states must contain at least one state")
    if any(isinstance(value, bool) for value in raw_inducers):
        raise TypeError("model.inducer_states must contain integers")
    try:
        inducers = tuple(operator.index(value) for value in raw_inducers)
    except TypeError as exc:
        raise TypeError("model.inducer_states must contain integers") from exc
    if any(value < 0 or value >= num_states for value in inducers):
        raise ValueError("model.inducer_states values must lie within num_states")
    if len(set(inducers)) != len(inducers):
        raise ValueError("model.inducer_states must not contain duplicates")
    return num_states, inducers


def validate_compartment(state: int, num_states: int) -> int:
    """Validate and normalize one compartment index."""
    if isinstance(state, bool):
        raise TypeError("state must be an integer compartment index, not bool")
    try:
        state = operator.index(state)
    except TypeError as exc:
        raise TypeError("state must be an integer compartment index") from exc
    if not 0 <= state < int(num_states):
        raise ValueError(f"state must be in [0, {num_states}), got {state}")
    return state


def validate_population_count(value: int, num_nodes: int) -> int:
    """Validate a node count used by initial-condition samplers."""
    if isinstance(value, bool):
        raise TypeError("num_infected must be an integer count, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError("num_infected must be an integer count") from exc
    if not 0 <= value <= num_nodes:
        raise ValueError(
            f"num_infected must be in [0, {num_nodes}], got {value}"
        )
    return value


def validate_initial_tensors(
    initial_state,
    *,
    num_nodes: int,
    num_states: int,
    device: _DeviceLike,
    initial_age=None,
) -> tuple[_Tensor, _Tensor | None]:
    """Normalize and validate public initial state/age arrays."""
    import torch

    state = torch.as_tensor(initial_state, device=device)
    if state.dim() != 1 or state.numel() != num_nodes:
        raise ValueError(
            f"initial_state must have shape [{num_nodes}], got {tuple(state.shape)}"
        )
    if state.dtype == torch.bool or state.dtype.is_floating_point or state.dtype.is_complex:
        raise TypeError("initial_state must use an integer dtype")
    if state.numel() and (int(state.min()) < 0 or int(state.max()) >= num_states):
        raise ValueError(f"initial_state values must lie in [0, {num_states})")
    # Validate before narrowing: values such as 2**32 otherwise wrap to zero
    # in int32 and masquerade as a valid susceptible compartment.
    state = state.to(torch.int32)

    age = None
    if initial_age is not None:
        age = torch.as_tensor(initial_age, device=device, dtype=torch.float32)
        if age.dim() != 1 or age.numel() != num_nodes:
            raise ValueError(
                f"initial_age must have shape [{num_nodes}], got {tuple(age.shape)}"
            )
        if not bool(torch.isfinite(age).all()) or bool((age < 0).any()):
            raise ValueError("initial_age must be finite and non-negative")
    return state, age


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
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")

    import torch

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
