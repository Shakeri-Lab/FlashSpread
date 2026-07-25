"""Validated construction-time configuration for simulation backends."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Literal

from .utils import _DeviceLike, validate_fp32_control


Backend = Literal["auto", "reference", "fused"]
Execution = Literal["auto", "eager", "cuda_graph"]
Traversal = Literal["auto", "thread", "warp", "merge"]
Transmission = Literal["model", "constant", "age_dependent"]
Precision = Literal["fp32", "bf16_weights", "mixed"]
_EngineFamily = Literal["markovian", "renewal"]


def supports_fused_renewal(model) -> bool:
    """Whether ``model`` has the exact built-in SEIR GPU semantics.

    The fused scalar kernels hard-code the built-in log-normal rates and SEIR
    transition map; they do not call the model's Python rate/transition hooks.
    Attribute compatibility is therefore insufficient: a subclass or an
    instance-shadowed method must retain the reference path so its behavior is
    not silently replaced by the built-in equations.

    A cheap type-origin gate comes first so even a protocol-complete custom
    model resolves to the reference backend without importing the Torch-backed
    model stack. A canonical candidate has necessarily loaded that stack
    already, at which point the exact identity check adds no eager-import cost.
    """
    model_type = type(model)
    if (
        model_type.__module__ != "flashspread.models.compartmental"
        or model_type.__qualname__ != "SEIRModel"
    ):
        return False

    required = (
        "susceptible",
        "exposed",
        "infected",
        "recovered",
        "num_states",
        "beta",
        "_mu_ei",
        "_sig_ei",
        "_mu_ir",
        "_sig_ir",
        "prepare",
        "compute_rates",
        "apply_transitions",
    )
    if not all(hasattr(model, name) for name in required):
        return False

    from .models.compartmental import SEIRModel, _SEIR_FUSED_BUILTIN_HOOKS

    if type(model) is not SEIRModel:
        return False

    state_ids = tuple(
        getattr(model, name, None)
        for name in ("susceptible", "exposed", "infected", "recovered")
    )
    try:
        inducer_states = tuple(model.inducer_states)
    except TypeError:
        return False
    if (
        getattr(model, "is_markovian", None) is not False
        or getattr(model, "transmission_mode", None)
        not in ("constant", "age_dependent")
        or type(getattr(model, "num_states", None)) is not int
        or model.num_states != 4
        or not all(type(state_id) is int for state_id in state_ids)
        or state_ids != (0, 1, 2, 3)
        or inducer_states != (2,)
    ):
        return False

    # The kernel bypasses all of these hooks. Reject an instance-level shadow
    # even if it happens to wrap the base function: the exact built-in contract
    # must be explicit and stable at dispatch time.
    instance_attributes = getattr(model, "__dict__", {})
    for name, original_hook in _SEIR_FUSED_BUILTIN_HOOKS:
        if name in instance_attributes:
            return False
        if getattr(getattr(model, name, None), "__func__", None) is not original_hook:
            return False
    return True


def supports_builtin_markovian(model) -> str | None:
    """Return ``"sis"``/``"sir"`` when ``model`` has exact built-in GPU semantics.

    This is the Markovian counterpart of :func:`supports_fused_renewal`, and it
    exists for the same reason: the Triton SIS/SIR pipeline hard-codes the
    built-in rate expressions and the compartment transition map, so it never
    calls the model's Python hooks. A bare ``type(model) is SISModel`` check is
    therefore not sufficient — an instance whose ``apply_transitions`` has been
    shadowed still passes it, and the kernel would silently substitute the
    built-in equations for the user's.

    Returning ``None`` selects the generic PyTorch path, which honours every
    hook, so a rejected model loses performance rather than correctness.
    """
    model_type = type(model)
    if model_type.__module__ != "flashspread.models.compartmental":
        return None
    kind = {"SISModel": "sis", "SIRModel": "sir"}.get(model_type.__qualname__)
    if kind is None:
        return None

    from .models.compartmental import (
        SIRModel,
        SISModel,
        _SIR_FUSED_BUILTIN_HOOKS,
        _SIS_FUSED_BUILTIN_HOOKS,
    )

    if kind == "sis":
        expected_type, hooks, num_states = SISModel, _SIS_FUSED_BUILTIN_HOOKS, 2
        rate_attribute = "delta"
    else:
        expected_type, hooks, num_states = SIRModel, _SIR_FUSED_BUILTIN_HOOKS, 3
        rate_attribute = "gamma"
    if model_type is not expected_type:
        return None

    required = ("susceptible", "infected", "num_states", "beta", rate_attribute)
    if not all(hasattr(model, name) for name in required):
        return None
    if kind == "sir" and not hasattr(model, "recovered"):
        return None

    state_names = ("susceptible", "infected")
    if kind == "sir":
        state_names += ("recovered",)
    state_ids = tuple(getattr(model, name, None) for name in state_names)
    try:
        inducer_states = tuple(model.inducer_states)
    except TypeError:
        return None
    if (
        getattr(model, "is_markovian", None) is not True
        or type(getattr(model, "num_states", None)) is not int
        or model.num_states != num_states
        or not all(type(state_id) is int for state_id in state_ids)
        or state_ids != tuple(range(num_states))
        # The frontier kernel treats infected as the sole inducer. The generic
        # rebuild honours a wider set, so the two would silently disagree.
        or inducer_states != (1,)
    ):
        return None

    # Reject an instance-level shadow even when it happens to wrap the base
    # function: the exact built-in contract must be explicit at dispatch time.
    instance_attributes = getattr(model, "__dict__", {})
    for name, original_hook in hooks:
        if name in instance_attributes:
            return None
        if getattr(getattr(model, name, None), "__func__", None) is not original_hook:
            return None
    return kind


@dataclass(frozen=True, slots=True)
class _ResolvedEnginePlan:
    """Private typed boundary between policy resolution and construction."""

    family: _EngineFamily
    device: str | _DeviceLike
    seed: int
    options: dict[str, object]

    def factory_kwargs(self) -> dict:
        """Return the historical keyword mapping consumed by public adapters."""
        return dict(self.options)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """One explicit configuration replacing interacting engine booleans.

    The legacy keyword flags remain supported by :class:`Simulator`; new code
    can use this immutable object so invalid combinations fail before an engine
    allocates memory or captures a CUDA Graph.
    """

    backend: Backend = "auto"
    execution: Execution = "auto"
    traversal: Traversal = "auto"
    transmission: Transmission = "model"
    precision: Precision = "fp32"
    compact: bool = False
    batch_steps: int = 50
    epsilon: float = 0.03
    tau_max: float = 1.0
    max_prob: float = 0.1
    theta: float = 0.01
    tau_min: float = 1e-6
    nodes_per_block: int = 8
    lanes_per_node: int = 32
    edges_per_merge_block: int = 1024

    def __post_init__(self) -> None:
        choices = {
            "backend": (self.backend, {"auto", "reference", "fused"}),
            "execution": (self.execution, {"auto", "eager", "cuda_graph"}),
            "traversal": (self.traversal, {"auto", "thread", "warp", "merge"}),
            "transmission": (
                self.transmission,
                {"model", "constant", "age_dependent"},
            ),
            "precision": (
                self.precision,
                {"fp32", "bf16_weights", "mixed"},
            ),
        }
        for name, (value, allowed) in choices.items():
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string, got {type(value).__name__}")
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")

        if not isinstance(self.compact, bool):
            raise TypeError("compact must be a bool")

        integer_fields = {}
        for name in (
            "batch_steps",
            "nodes_per_block",
            "lanes_per_node",
            "edges_per_merge_block",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, not bool")
            try:
                integer_fields[name] = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be an integer") from exc

        if integer_fields["batch_steps"] <= 0:
            raise ValueError("batch_steps must be positive")

        for name in ("epsilon", "tau_max", "max_prob", "theta", "tau_min"):
            validate_fp32_control(name, getattr(self, name), positive=True)

        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if self.tau_max <= 0.0:
            raise ValueError("tau_max must be finite and positive")
        if not 0.0 < self.max_prob < 1.0:
            raise ValueError("max_prob must be finite and in (0, 1)")
        if self.theta <= 0.0:
            raise ValueError("theta must be finite and positive")
        if self.theta > 1.0:
            raise ValueError("theta is a target fraction and must be <= 1")
        if self.tau_min <= 0.0:
            raise ValueError("tau_min must be finite and positive")
        for name in ("nodes_per_block", "lanes_per_node", "edges_per_merge_block"):
            value = integer_fields[name]
            if value <= 0 or value & (value - 1):
                raise ValueError(f"{name} must be a positive power of two")
        if integer_fields["lanes_per_node"] > 32:
            raise ValueError("lanes_per_node must be <= 32")
        if integer_fields["nodes_per_block"] * integer_fields["lanes_per_node"] > 1024:
            raise ValueError("nodes_per_block * lanes_per_node must be <= 1024")

    def _resolve_plan(
        self,
        device: _DeviceLike,
        *,
        markovian: bool,
        model,
        seed: int = 12345,
    ) -> _ResolvedEnginePlan:
        """Resolve declarative policy once into a typed construction plan."""
        backend = self.backend
        if backend == "auto":
            backend = (
                "fused"
                if device.type == "cuda"
                and not markovian
                and supports_fused_renewal(model)
                else "reference"
            )

        execution = self.execution
        if execution == "auto":
            execution = (
                "cuda_graph"
                if device.type == "cuda" and backend == "fused" and not markovian
                else "eager"
            )

        if backend == "fused" and device.type != "cuda":
            raise ValueError("backend='fused' requires a CUDA device")
        if execution == "cuda_graph" and device.type != "cuda":
            raise ValueError("execution='cuda_graph' requires a CUDA device")
        if markovian and self.backend != "auto":
            raise ValueError(
                "backend is not configurable for Markovian models; their one "
                "engine selects its CPU reference or CUDA kernels from device"
            )
        if markovian and self.traversal != "auto":
            raise ValueError("traversal is only configurable for fused renewal models")
        if markovian and self.transmission != "model":
            raise ValueError("transmission is not configurable for Markovian models")
        if markovian and self.precision != "fp32":
            raise ValueError("precision is not configurable for Markovian models")
        if markovian and self.tau_min > self.tau_max:
            raise ValueError("Markovian models require tau_min <= tau_max")

        traversal = self.traversal
        if self.compact and traversal == "auto":
            traversal = "thread"
        if self.compact and (backend != "fused" or execution != "cuda_graph"):
            raise ValueError("compact=True requires fused CUDA-Graph execution")
        if self.compact and traversal != "thread":
            raise ValueError("compact=True requires traversal='thread'")
        if self.precision == "mixed" and backend != "fused":
            raise ValueError("precision='mixed' requires the fused backend")
        if not markovian and backend != "fused" and traversal != "auto":
            raise ValueError("traversal is only configurable for the fused backend")

        if markovian:
            return _ResolvedEnginePlan(
                family="markovian",
                device=device,
                seed=seed,
                options={
                    "use_cuda_graph": execution == "cuda_graph",
                    "steps_per_launch": self.batch_steps,
                    "max_prob": self.max_prob,
                    "theta": self.theta,
                    "tau_min": self.tau_min,
                    "tau_max": self.tau_max,
                },
            )

        transmission = self.transmission
        if transmission == "model":
            transmission = getattr(model, "transmission_mode", "constant")
        nonmark_edges = backend == "fused" or transmission == "age_dependent"
        return _ResolvedEnginePlan(
            family="renewal",
            device=device,
            seed=seed,
            options={
                "use_fused": backend == "fused",
                "use_cuda_graph": execution == "cuda_graph",
                "nonmarkov_edges": nonmark_edges,
                "transmission_mode": transmission,
                "bf16_weights": self.precision in {"bf16_weights", "mixed"},
                "use_mixed_precision": self.precision == "mixed",
                "use_active_compaction": self.compact,
                "csr_strategy": traversal,
                "steps_per_launch": self.batch_steps,
                "epsilon": self.epsilon,
                "tau_max": self.tau_max,
                "nodes_per_block": self.nodes_per_block,
                "lanes_per_node": self.lanes_per_node,
                "edges_per_merge_block": self.edges_per_merge_block,
            },
        )

    def resolve(self, device: _DeviceLike, *, markovian: bool, model) -> dict:
        """Translate declarative choices into legacy factory arguments."""
        return self._resolve_plan(
            device,
            markovian=markovian,
            model=model,
        ).factory_kwargs()
