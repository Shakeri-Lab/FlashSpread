"""
Simulation engines for FlashSpread.

Factories in this module resolve the memoryless Markovian path versus the
age-dependent renewal path. Engine implementations are imported lazily so a
reference simulation does not load benchmark-only or fused GPU modules.

Prefer :class:`flashspread.Simulator` with :class:`flashspread.EngineConfig`;
the concrete engine names remain available for advanced use and compatibility.
"""

from __future__ import annotations

import copy
import importlib
import operator
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    import torch

    from ..config import EngineConfig, _ResolvedEnginePlan


_HISTORICAL_EXPORTS = {
    "RenewalEngineTunable",
    "RenewalEngineTunableCUDAGraph",
    "estimate_flops_per_step",
    "estimate_memory_bytes_per_step",
}

_LAZY_EXPORTS = {
    "MarkovianEngine": ("flashspread.engines.markovian", "MarkovianEngine"),
    "MarkovianEngineCUDAGraph": (
        "flashspread.engines.markovian",
        "MarkovianEngineCUDAGraph",
    ),
    "ReferenceEnsembleEngine": (
        "flashspread.engines.ensemble",
        "ReferenceEnsembleEngine",
    ),
    "EnsembleEngine": (
        "flashspread.engines.ensemble",
        "EnsembleEngine",
    ),
    "RenewalEngine": ("flashspread.engines.renewal", "RenewalEngine"),
    "RenewalEngineCUDAGraph": (
        "flashspread.engines.renewal",
        "RenewalEngineCUDAGraph",
    ),
    "RenewalEngineNonMarkov": (
        "flashspread.engines.renewal",
        "RenewalEngineNonMarkov",
    ),
    "RenewalEngineNonMarkovCUDAGraph": (
        "flashspread.engines.renewal",
        "RenewalEngineNonMarkovCUDAGraph",
    ),
    "RenewalEngineFused": (
        "flashspread.engines.renewal_fused",
        "RenewalEngineFused",
    ),
    "RenewalEngineFusedCUDAGraph": (
        "flashspread.engines.renewal_fused",
        "RenewalEngineFusedCUDAGraph",
    ),
    "RenewalEngineTunable": (
        "flashspread.engines.renewal_tunable",
        "RenewalEngineTunable",
    ),
    "RenewalEngineTunableCUDAGraph": (
        "flashspread.engines.renewal_tunable",
        "RenewalEngineTunableCUDAGraph",
    ),
    "estimate_flops_per_step": (
        "flashspread.engines.renewal_tunable",
        "estimate_flops_per_step",
    ),
    "estimate_memory_bytes_per_step": (
        "flashspread.engines.renewal_tunable",
        "estimate_memory_bytes_per_step",
    ),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'flashspread.engines' has no attribute {name!r}")
    if name in _HISTORICAL_EXPORTS:
        warnings.warn(
            f"{name} is a historical synthetic-benchmark API; use the production "
            "engines and experiments/perf_model.py instead",
            DeprecationWarning,
            stacklevel=2,
        )
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


# Recommended default configurations
RENEWAL_DEFAULTS = {
    "epsilon": 0.03,        # Accuracy parameter
    "tau_max": 1.0,         # Maximum time step
    "steps_per_launch": 50, # CUDA Graph batch size (for CUDAGraph variant)
}

MARKOVIAN_DEFAULTS = {
    "max_prob": 0.1,   # Maximum transition probability per step
    "theta": 0.01,     # Target fraction of nodes transitioning
    "tau_max": 1.0,    # Maximum time step
    "steps_per_launch": 50,  # Explicit CUDA Graph batch size
}


def _validate_engine_plan(plan: _ResolvedEnginePlan, model) -> _ResolvedEnginePlan:
    """Validate one plan with the historical public-factory error contract."""
    from dataclasses import replace

    options = plan.factory_kwargs()
    if plan.family == "markovian":
        if not isinstance(options["use_cuda_graph"], bool):
            raise TypeError("use_cuda_graph must be a bool")
        if isinstance(options["steps_per_launch"], bool):
            raise TypeError("steps_per_launch must be an integer, not bool")
        try:
            steps_per_launch = operator.index(options["steps_per_launch"])
        except TypeError as exc:
            raise TypeError("steps_per_launch must be an integer") from exc
        if steps_per_launch <= 0:
            raise ValueError(
                f"steps_per_launch must be positive, got {steps_per_launch}"
            )
        if options["use_cuda_graph"] and steps_per_launch > 4096:
            raise ValueError(
                "steps_per_launch must be <= 4096 for Markov CUDA Graph capture"
            )
        options["steps_per_launch"] = steps_per_launch
        return replace(plan, options=options)

    if plan.family != "renewal":
        raise TypeError("plan options must describe a scalar simulation engine")
    options.setdefault("warp_collaborative", False)
    boolean_options = {
        name: options[name]
        for name in (
            "use_cuda_graph",
            "nonmarkov_edges",
            "use_fused",
            "bf16_weights",
            "use_mixed_precision",
            "warp_collaborative",
            "use_active_compaction",
        )
    }
    for name, value in boolean_options.items():
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
    transmission_mode = options["transmission_mode"]
    if transmission_mode is None:
        transmission_mode = getattr(model, "transmission_mode", "constant")
        options["transmission_mode"] = transmission_mode
    if transmission_mode not in ("constant", "age_dependent"):
        raise ValueError(
            "transmission_mode must be 'constant' or 'age_dependent', got "
            f"{transmission_mode!r}"
        )
    if options["use_fused"] and not options["nonmarkov_edges"]:
        raise ValueError("use_fused=True requires nonmarkov_edges=True")
    if options["use_mixed_precision"] and not options["use_fused"]:
        raise ValueError("use_mixed_precision=True requires use_fused=True")
    if not options["use_fused"] and (
        options["csr_strategy"] != "auto"
        or options["warp_collaborative"]
        or options["nodes_per_block"] != 8
        or options["lanes_per_node"] != 32
        or options["edges_per_merge_block"] != 1024
    ):
        raise ValueError("CSR traversal tuning options require use_fused=True")
    if options["use_active_compaction"] and not (
        options["use_fused"] and options["use_cuda_graph"]
    ):
        raise ValueError(
            "use_active_compaction=True requires the fused CUDA-Graph engine"
        )
    if not options["nonmarkov_edges"] and transmission_mode != "constant":
        raise ValueError("age-dependent transmission requires nonmarkov_edges=True")
    if isinstance(options["steps_per_launch"], bool):
        raise TypeError("steps_per_launch must be an integer, not bool")
    try:
        steps_per_launch = operator.index(options["steps_per_launch"])
    except TypeError as exc:
        raise TypeError("steps_per_launch must be an integer") from exc
    if steps_per_launch <= 0:
        raise ValueError(
            f"steps_per_launch must be positive, got {steps_per_launch}"
        )
    options["steps_per_launch"] = steps_per_launch
    return replace(plan, options=options)


def _create_from_plan(graph, model, plan: _ResolvedEnginePlan):
    """Construct exactly one engine from the canonical validated plan."""
    plan = _validate_engine_plan(plan, model)
    options = plan.factory_kwargs()
    if plan.family == "markovian":
        module = importlib.import_module("flashspread.engines.markovian")
        engine_class = getattr(
            module,
            "MarkovianEngineCUDAGraph"
            if options["use_cuda_graph"]
            else "MarkovianEngine",
        )
        engine_kwargs = {
            "device": plan.device,
            "max_prob": options["max_prob"],
            "theta": options["theta"],
            "tau_min": options["tau_min"],
            "tau_max": options["tau_max"],
            "seed": plan.seed,
        }
        if options["use_cuda_graph"]:
            engine_kwargs["steps_per_launch"] = options["steps_per_launch"]
        return engine_class(graph, copy.copy(model), **engine_kwargs)

    engine_model = copy.copy(model)
    if hasattr(engine_model, "transmission_mode"):
        engine_model.transmission_mode = options["transmission_mode"]
    engine_kwargs = {
        "device": plan.device,
        "epsilon": options["epsilon"],
        "tau_max": options["tau_max"],
        "seed": plan.seed,
        "bf16_weights": options["bf16_weights"],
    }
    use_graph = options["use_cuda_graph"]
    if options["use_fused"]:
        from ..config import supports_fused_renewal

        if not supports_fused_renewal(engine_model):
            raise TypeError(
                "the fused renewal backend requires the exact, unmodified "
                "built-in SEIRModel; custom models must use the reference backend"
            )
        module = importlib.import_module("flashspread.engines.renewal_fused")
        engine_class = getattr(
            module,
            "RenewalEngineFusedCUDAGraph" if use_graph else "RenewalEngineFused",
        )
        engine_kwargs.update(
            use_mixed_precision=options["use_mixed_precision"],
            csr_strategy=options["csr_strategy"],
            nodes_per_block=options["nodes_per_block"],
            lanes_per_node=options["lanes_per_node"],
            edges_per_merge_block=options["edges_per_merge_block"],
            warp_collaborative=options["warp_collaborative"],
        )
        if use_graph:
            engine_kwargs.update(
                steps_per_launch=options["steps_per_launch"],
                use_active_compaction=options["use_active_compaction"],
            )
    else:
        module = importlib.import_module("flashspread.engines.renewal")
        prefix = (
            "RenewalEngineNonMarkov"
            if options["nonmarkov_edges"]
            else "RenewalEngine"
        )
        engine_class = getattr(module, prefix + ("CUDAGraph" if use_graph else ""))
        if use_graph:
            engine_kwargs["steps_per_launch"] = options["steps_per_launch"]
    return engine_class(graph, engine_model, **engine_kwargs)


def create_renewal_engine(
    graph,
    model,
    device: str = "cuda",
    use_cuda_graph: bool = True,
    nonmarkov_edges: bool = True,
    use_fused: bool = True,
    bf16_weights: bool = False,
    transmission_mode: str | None = None,
    epsilon: float = RENEWAL_DEFAULTS["epsilon"],
    tau_max: float = RENEWAL_DEFAULTS["tau_max"],
    steps_per_launch: int = RENEWAL_DEFAULTS["steps_per_launch"],
    seed: int = 12345,
    use_mixed_precision: bool = False,
    csr_strategy: str = "auto",
    nodes_per_block: int = 8,
    lanes_per_node: int = 32,
    edges_per_merge_block: int = 1024,
    warp_collaborative: bool = False,
    use_active_compaction: bool = False,
):
    """
    Factory function to create the recommended Renewal engine.

    Default configuration: RenewalEngineFusedCUDAGraph (fastest).

    Args:
        graph: Network object with edge_index and csr attributes
        model: Non-Markovian compartmental model (e.g., SEIRModel)
        device: PyTorch device (default: "cuda")
        use_cuda_graph: Use CUDA Graph batching (default: True)
        nonmarkov_edges: Use infectivity-based edge kernel (default: True)
        use_fused: Use fused Triton kernel (default: True). Requires
                  nonmarkov_edges=True. Falls back to unfused if False.
        bf16_weights: Downcast edge weights to bfloat16 (default: False)
        transmission_mode: "constant" (default, Markovian-equivalent) or
                          "age_dependent" (source-node compromise with h_IR)
        epsilon: Maximum per-step hazard scale used for adaptive tau.
        tau_max: Maximum simulated time advanced by one internal step.
        steps_per_launch: Internal steps in a CUDA Graph replay.
        seed: Base random seed.
        use_mixed_precision: Compact fused state/infectivity and edge weights;
            age and all arithmetic remain fp32 to preserve renewal-clock progress.
        csr_strategy: Fused traversal strategy: auto/thread/warp/merge.
        nodes_per_block: Warp strategy nodes per Triton program.
        lanes_per_node: Warp strategy lanes collaborating on each node.
        edges_per_merge_block: Merge strategy edges per Triton program.
        warp_collaborative: Deprecated alias that forces the warp strategy.
        use_active_compaction: Skip recovered nodes between CUDA Graph windows.

    Returns:
        Appropriate engine variant based on options.
    """
    from ..config import _ResolvedEnginePlan

    plan = _ResolvedEnginePlan(
        family="renewal",
        device=device,
        seed=seed,
        options={
            "use_cuda_graph": use_cuda_graph,
            "nonmarkov_edges": nonmarkov_edges,
            "use_fused": use_fused,
            "bf16_weights": bf16_weights,
            "transmission_mode": transmission_mode,
            "epsilon": epsilon,
            "tau_max": tau_max,
            "steps_per_launch": steps_per_launch,
            "use_mixed_precision": use_mixed_precision,
            "csr_strategy": csr_strategy,
            "nodes_per_block": nodes_per_block,
            "lanes_per_node": lanes_per_node,
            "edges_per_merge_block": edges_per_merge_block,
            "warp_collaborative": warp_collaborative,
            "use_active_compaction": use_active_compaction,
        },
    )
    return _create_from_plan(graph, model, plan)


def create_markovian_engine(
    graph,
    model,
    device: str = "cuda",
    max_prob: float = MARKOVIAN_DEFAULTS["max_prob"],
    theta: float = MARKOVIAN_DEFAULTS["theta"],
    tau_min: float = 1e-6,
    tau_max: float = MARKOVIAN_DEFAULTS["tau_max"],
    seed: int = 12345,
    use_cuda_graph: bool = False,
    steps_per_launch: int = MARKOVIAN_DEFAULTS["steps_per_launch"],
):
    """
    Factory function to create the recommended Markovian engine.

    Args:
        graph: Network object with edge_index and csr attributes
        model: Markovian compartmental model (e.g., SISModel, SIRModel)
        device: PyTorch device (default: "cuda")
        max_prob: Maximum allowed transition probability per step.
        theta: Target fraction of nodes transitioning in a step.
        tau_min: Preferred time-step floor; ``max_prob`` takes precedence.
        tau_max: Maximum time step.
        seed: Base random seed.
        use_cuda_graph: Batch fixed-shape built-in SIS/SIR steps in a captured
            CUDA Graph. Disabled by default to preserve one-step granularity.
        steps_per_launch: Internal steps advanced by one CUDA Graph replay.

    Returns:
        Eager or CUDA Graph Markovian engine with recommended defaults.
    """
    from ..config import _ResolvedEnginePlan

    plan = _ResolvedEnginePlan(
        family="markovian",
        device=device,
        seed=seed,
        options={
            "max_prob": max_prob,
            "theta": theta,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "use_cuda_graph": use_cuda_graph,
            "steps_per_launch": steps_per_launch,
        },
    )
    return _create_from_plan(graph, model, plan)


def create_engine(
    graph,
    model,
    device: str | torch.device = "cuda",
    *,
    config: EngineConfig | None = None,
    seed: int | None = None,
    **engine_kwargs,
):
    """Create one scalar engine through the canonical family dispatch.

    ``config`` selects the declarative :class:`flashspread.EngineConfig` path.
    Without it, legacy factory keywords retain the historical
    :class:`flashspread.Simulator` auto policy: Markovian models use their one
    device-adaptive engine; renewal models prefer supported fused CUDA execution
    and otherwise select the appropriate eager/CUDA-Graph reference variant.

    The family-specific factory functions remain compatible adapters around the
    same typed plan and constructor. Passing both ``config`` and legacy engine
    keywords is an error.
    """
    import torch

    from ..config import EngineConfig
    from ..utils import is_markovian

    if config is not None and engine_kwargs:
        raise ValueError(
            "pass either config=EngineConfig(...) or legacy engine keyword "
            "arguments, not both"
        )
    markovian = is_markovian(model)
    if config is not None:
        resolved_device = torch.device(device)
        # Simulator historically accepted a duck-typed ``resolve`` object even
        # though its public annotation says EngineConfig. Preserve that escape
        # hatch (and subclass overrides) through the legacy adapter path.
        if type(config) is not EngineConfig:
            resolved_kwargs = dict(
                config.resolve(
                    resolved_device,
                    markovian=markovian,
                    model=model,
                )
            )
            return create_engine(
                graph,
                model,
                device=device,
                seed=seed,
                **resolved_kwargs,
            )

        from dataclasses import replace

        plan = config._resolve_plan(
            resolved_device,
            markovian=markovian,
            model=model,
            seed=12345 if seed is None else seed,
        )
        # Policy resolution uses a normalized torch.device, while constructors
        # retain the caller's historical device argument representation.
        return _create_from_plan(graph, model, replace(plan, device=device))

    kwargs = dict(engine_kwargs)
    if seed is not None:
        kwargs.setdefault("seed", seed)
    if markovian:
        return create_markovian_engine(graph, model, device=device, **kwargs)

    device_type = torch.device(device).type
    if device_type == "cuda":
        from ..config import supports_fused_renewal

        fused_capable = supports_fused_renewal(model)
        explicit_use_fused = "use_fused" in kwargs
        use_fused = kwargs.get("use_fused", fused_capable)
        if not isinstance(use_fused, bool):
            raise TypeError("use_fused must be a bool")
        kwargs.setdefault("use_fused", use_fused)
        if not fused_capable and not explicit_use_fused:
            kwargs.setdefault("use_cuda_graph", False)
        kwargs.setdefault(
            "nonmarkov_edges",
            use_fused
            or getattr(model, "transmission_mode", "constant")
            == "age_dependent",
        )
    else:
        # Fused and CUDA-Graph execution are GPU-only. Both constant and
        # age-dependent transmission have CSR-native CPU references.
        kwargs.setdefault("use_cuda_graph", False)
        kwargs.setdefault("use_fused", False)
        kwargs.setdefault(
            "nonmarkov_edges",
            getattr(model, "transmission_mode", "constant") == "age_dependent",
        )
    return create_renewal_engine(graph, model, device=device, **kwargs)


def create_ensemble_engine(
    graph,
    model,
    replicas: int,
    *,
    device: str | torch.device | None = None,
    backend: str = "auto",
    seed: int = 12345,
    epsilon: float = RENEWAL_DEFAULTS["epsilon"],
    max_prob: float = MARKOVIAN_DEFAULTS["max_prob"],
    theta: float = MARKOVIAN_DEFAULTS["theta"],
    tau_min: float = 1e-6,
    tau_max: float = RENEWAL_DEFAULTS["tau_max"],
    nodes_per_program: int = 8,
    replicas_per_tile: int | None = None,
):
    """Create independent trajectories that retain one shared graph.

    Ensemble tensors are node-major ``[N, replicas]`` and each replica keeps
    an independent adaptive clock and random stream. ``backend='auto'`` uses
    the tiled Triton graph phase on CUDA and the PyTorch reference on CPU.
    The tiled backend removes the reference implementation's ``[E, replicas]``
    temporary. For the exact built-in non-Markovian SEIR model with constant
    transmission, it also uses a packed infectious-state bitmap, fused CSR/rate
    evaluation, device reduction and transactional time-step finalization, and
    a tiled Triton transition phase that maintains changed bitmap bits. Other
    models retain the reference rate/transition phases after the tiled gather.
    The specialized path remains a multi-phase eager step with one host
    validation read; it is not one monolithic kernel or a captured CUDA Graph.

    The scalar :class:`flashspread.Simulator` intentionally does not dispatch
    here: its single-clock :class:`flashspread.Trajectory` contract is
    incompatible with independently adaptive replica clocks.
    """
    import torch

    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    if backend not in {"auto", "reference", "tiled"}:
        raise ValueError("backend must be one of ['auto', 'reference', 'tiled']")

    graph_csr = graph.csr if hasattr(graph, "csr") else graph
    if device is None:
        if not hasattr(graph_csr, "device"):
            raise TypeError("graph must expose a device directly or through .csr")
        resolved_device = torch.device(graph_csr.device)
    else:
        resolved_device = torch.device(device)
    resolved_backend = backend
    if resolved_backend == "auto":
        resolved_backend = "tiled" if resolved_device.type == "cuda" else "reference"
    if resolved_backend == "tiled" and resolved_device.type != "cuda":
        raise ValueError("backend='tiled' requires a CUDA device")
    if resolved_backend == "reference" and (
        nodes_per_program != 8 or replicas_per_tile is not None
    ):
        raise ValueError("ensemble tile controls require backend='tiled'")

    common = dict(
        device=resolved_device,
        seed=seed,
        epsilon=epsilon,
        max_prob=max_prob,
        theta=theta,
        tau_min=tau_min,
        tau_max=tau_max,
    )
    from .ensemble import EnsembleEngine, ReferenceEnsembleEngine

    if resolved_backend == "reference":
        return ReferenceEnsembleEngine(graph, model, replicas, **common)
    return EnsembleEngine(
        graph,
        model,
        replicas,
        nodes_per_program=nodes_per_program,
        replicas_per_tile=replicas_per_tile,
        **common,
    )


__all__ = [
    # Core engines
    "MarkovianEngine",
    "MarkovianEngineCUDAGraph",
    "RenewalEngine",
    "RenewalEngineCUDAGraph",
    # Non-Markovian edge engines
    "RenewalEngineNonMarkov",
    "RenewalEngineNonMarkovCUDAGraph",
    # Fused Triton kernel engines
    "RenewalEngineFused",
    "RenewalEngineFusedCUDAGraph",
    # Graph-reusing independent trajectories
    "EnsembleEngine",
    # Factory functions (recommended)
    "create_engine",
    "create_renewal_engine",
    "create_markovian_engine",
    "create_ensemble_engine",
    # Default configurations
    "RENEWAL_DEFAULTS",
    "MARKOVIAN_DEFAULTS",
]
