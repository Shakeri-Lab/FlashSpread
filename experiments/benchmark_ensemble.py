#!/usr/bin/env python
"""Honest wall-time and profiling harness for the production ensemble path.

The measured target is exactly one ``EnsembleEngine.step()`` for the built-in
constant-transmission renewal SEIR model on an exact circulant graph. Reports
contain observations and provenance, not hardware ceilings, inferred FLOPs,
performance thresholds, or claims that the eager multi-kernel step is one
fused kernel.

``print-ncu-command`` emits a shell-safe command whose per-NVTX summary covers
every kernel launched by that one production step.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import benchmark_acceptance as _acceptance  # noqa: E402
from experiments.ensemble_perf_model import (  # noqa: E402
    EnsembleActivity,
    ensemble_renewal_logical_traffic,
)
from experiments.perf_model import unique_storage_usage  # noqa: E402
from flashspread.core.network import _circulant_memory_plan  # noqa: E402


SCHEMA_VERSION = "flashspread.ensemble_acceptance.v3"
CHECKPOINT_FRACTIONS = _acceptance.CHECKPOINT_FRACTIONS
WARMUP_MINIMUM_CALLS = _acceptance.WARMUP_MINIMUM_CALLS
WARMUP_MINIMUM_DURATION_SECONDS = _acceptance.WARMUP_MINIMUM_DURATION_SECONDS
_INT32_MAX = torch.iinfo(torch.int32).max
_UINT32_CARDINALITY = 1 << 32

# These aliases deliberately reuse small, CPU-only acceptance helpers. Importing
# this module does not import the optional Triton ensemble kernels; construction
# of the tiled engine remains inside ``_make_engine``.
_optional_nvtx = _acceptance._optional_nvtx
_quantile = _acceptance._quantile


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not (0.0 <= value < float("inf")):
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nodes", type=_positive_int, default=1_000_000)
    parser.add_argument("--degree", type=_positive_int, default=8)
    parser.add_argument("--replicas", type=_positive_int, default=32)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--checkpoint", choices=CHECKPOINT_FRACTIONS)
    parser.add_argument("--device", default="cuda")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)

    wall = modes.add_parser(
        "walltime", help="time one production step at early/peak/late checkpoints"
    )
    _common(wall)
    wall.add_argument("--repetitions", type=_positive_int, default=10)
    wall.add_argument("--min-duration", type=_nonnegative_float, default=1.0)
    wall.add_argument("--output", help="versioned JSON path; '-' means stdout")
    wall.add_argument("--dry-run", action="store_true")

    profile = modes.add_parser("profile", help="profile exactly one production ensemble step")
    _common(profile)
    profile.add_argument("--output", help="versioned JSON path; '-' means stdout")
    profile.add_argument("--dry-run", action="store_true")

    ncu = modes.add_parser("print-ncu-command", help="print a shell-safe per-NVTX ncu command")
    _common(ncu)
    ncu.add_argument("--ncu-output", default="results/flashspread_ensemble_profile")
    ncu.add_argument("--json-output", default="results/flashspread_ensemble_profile.json")
    ncu.add_argument("--ncu-bin", default="ncu")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.degree >= args.nodes:
        raise ValueError("degree must be smaller than nodes")
    if args.nodes > _INT32_MAX:
        raise ValueError("nodes exceeds FlashSpread's int32 CSR node limit")
    directed_edges = args.nodes * args.degree
    if directed_edges > _INT32_MAX:
        raise ValueError(
            "requested graph exceeds FlashSpread's int32 CSR edge limit: "
            f"{directed_edges} > {_INT32_MAX}"
        )
    if directed_edges % 2:
        raise ValueError("nodes * degree must be even")
    if args.replicas > _UINT32_CARDINALITY:
        raise ValueError(
            "replicas exceeds the uint32 counter-id cardinality used by the "
            "production transition kernel"
        )
    try:
        selected = torch.device(args.device)
    except RuntimeError as exc:
        raise ValueError(f"invalid device {args.device!r}") from exc
    if selected.type != "cuda":
        raise ValueError("ensemble acceptance measurements require a CUDA device")


def _checkpoint_names(args: argparse.Namespace) -> tuple[str, ...]:
    if args.checkpoint is not None:
        return (args.checkpoint,)
    if args.mode in {"profile", "print-ncu-command"}:
        return ("peak",)
    return tuple(CHECKPOINT_FRACTIONS)


def _counts(num_nodes: int, fractions: Sequence[float]) -> list[int]:
    raw = [num_nodes * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    order = sorted(range(4), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[: num_nodes - sum(counts)]:
        counts[index] += 1
    return counts


def checkpoint_count_definition(num_nodes: int, checkpoint: str) -> dict[str, Any]:
    """Return the tensor-free population definition used by a checkpoint."""
    if checkpoint not in CHECKPOINT_FRACTIONS:
        raise ValueError(f"unknown checkpoint {checkpoint!r}")
    fractions = CHECKPOINT_FRACTIONS[checkpoint]
    labels = ("S", "E", "I", "R")
    counts = _counts(num_nodes, fractions)
    return {
        "fractions_requested": dict(zip(labels, fractions)),
        "counts": dict(zip(labels, counts)),
    }


def build_checkpoints(
    num_nodes: int,
    seed: int,
    *,
    exposed_age: float = 2.0,
    infected_age: float = 1.5,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]:
    """Build deterministic shared ``[N]`` checkpoints without CUDA or Triton."""
    return _acceptance.build_checkpoints(
        num_nodes,
        seed,
        exposed_age=exposed_age,
        infected_age=infected_age,
    )


def iter_checkpoints(
    num_nodes: int,
    seed: int,
    names: Sequence[str],
    *,
    exposed_age: float = 2.0,
    infected_age: float = 1.5,
):
    """Stream deterministic shared ``[N]`` checkpoints one phase at a time."""
    yield from _acceptance.iter_checkpoints(
        num_nodes,
        seed,
        names,
        exposed_age=exposed_age,
        infected_age=infected_age,
    )


def default_replica_tile(replicas: int) -> int:
    """Mirror the production default without importing the Triton module."""
    if isinstance(replicas, bool) or not isinstance(replicas, int):
        raise TypeError("replicas must be an integer")
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    return min(32, 1 << (replicas - 1).bit_length())


def checkpoint_activity(
    *,
    num_nodes: int,
    degree: int,
    replicas: int,
    replicas_per_tile: int,
    counts: dict[str, int],
) -> EnsembleActivity:
    """Construct exact activity for one shared checkpoint on a regular CSR.

    Every replica receives the same one-dimensional state and age tensors via
    ``set_initial_state``. Consequently each replica and each execution-tile
    union has the same susceptible rows on the initial measured step.
    """
    required = ("S", "E", "I", "R")
    if len(counts) != len(required) or set(counts) != set(required):
        raise ValueError("counts must contain exactly S, E, I, R")
    values = tuple(counts[name] for name in required)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise TypeError("checkpoint counts must be non-negative integers")
    if sum(values) != num_nodes:
        raise ValueError("checkpoint counts must sum to num_nodes")
    if degree < 0 or degree >= num_nodes:
        raise ValueError("degree must satisfy 0 <= degree < num_nodes")
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    if (
        replicas_per_tile <= 0
        or replicas_per_tile > 32
        or replicas_per_tile & (replicas_per_tile - 1)
    ):
        raise ValueError("replicas_per_tile must be a power of two no larger than 32")

    tile_count = math.ceil(replicas / replicas_per_tile)
    susceptible_nodes = values[0]
    susceptible_edges = susceptible_nodes * degree
    hazard_nodes = values[1] + values[2]
    return EnsembleActivity(
        num_nodes=num_nodes,
        num_edges=num_nodes * degree,
        replicas=replicas,
        replicas_per_tile=replicas_per_tile,
        tile_count=tile_count,
        replica_susceptible_nodes=(susceptible_nodes,) * replicas,
        replica_susceptible_edges=(susceptible_edges,) * replicas,
        replica_hazard_nodes=(hazard_nodes,) * replicas,
        tile_susceptible_union_nodes=(susceptible_nodes,) * tile_count,
        tile_susceptible_union_edges=(susceptible_edges,) * tile_count,
    )


def logical_traffic_reference(
    *,
    num_nodes: int,
    degree: int,
    replicas: int,
    replicas_per_tile: int,
    counts: dict[str, int],
    state_bytes: int = 4,
    age_bytes: int = 4,
    rate_bytes: int = 4,
    index_bytes: int = 4,
    weight_bytes: int = 4,
    packed_word_bytes: int = 4,
    has_weights: bool = False,
    rate_bound_nodes_per_partial: int = 128,
    transition_changed_events: int | None = None,
) -> dict[str, Any]:
    """Build a logical-byte reference for the initial production step.

    This is an auditable request model, not an HBM counter. It intentionally
    omits state-dependent bitmap atomic traffic because its E->I/I->R subset is
    not retained by the engine. Before execution, leave
    ``transition_changed_events`` unset and state-write bytes are a dense upper
    bound. After execution, pass the engine's observed changed-event count to
    make the sparse state-write component exact. Checkpoint restoration has
    already refreshed the packed bitmap and is outside the measured target.
    """
    activity = checkpoint_activity(
        num_nodes=num_nodes,
        degree=degree,
        replicas=replicas,
        replicas_per_tile=replicas_per_tile,
        counts=counts,
    )
    traffic = ensemble_renewal_logical_traffic(
        num_nodes=num_nodes,
        activity=activity,
        transmission="constant",
        has_weights=has_weights,
        state_bytes=state_bytes,
        age_bytes=age_bytes,
        # Constant transmission does not read infectivity, but pass the actual
        # fp32 width rather than relying on the compact-target default.
        infectivity_bytes=rate_bytes,
        weight_bytes=weight_bytes,
        index_bytes=index_bytes,
        rate_bytes=rate_bytes,
        constant_source_encoding="packed_bitmap",
        packed_word_bytes=packed_word_bytes,
        rate_bound_nodes_per_partial=rate_bound_nodes_per_partial,
        bitmap_refresh=False,
        bitmap_atomic_updates=None,
        transition_changed_events=transition_changed_events,
    )
    return {
        "scope": "shared checkpoint state before exactly one EnsembleEngine.step()",
        "counter_kind": "logical byte requests, not measured HBM traffic",
        "constant_source_encoding": "packed_bitmap",
        "rate_bound_nodes_per_partial": rate_bound_nodes_per_partial,
        "bitmap_refresh_in_target": False,
        "bitmap_atomic_updates": None,
        "bitmap_atomic_traffic_note": (
            "omitted because the aggregate changed-event counter does not retain "
            "the E->I plus I->R subset"
        ),
        "transition_changed_events": transition_changed_events,
        "transition_state_write_accounting": (
            "dense all-lanes conservative upper bound"
            if transition_changed_events is None
            else "exact observed sparse changed-lane writes"
        ),
        "storage_width_bytes": {
            "state": state_bytes,
            "age": age_bytes,
            "rate": rate_bytes,
            "rate_bound_partial": 4,
            "event_partial": 4,
            "csr_index": index_bytes,
            "weight_storage": weight_bytes,
            "packed_word": packed_word_bytes,
        },
        "broadcast_activity": {
            "replicas": replicas,
            "replicas_per_tile": replicas_per_tile,
            "tile_count": activity.tile_count,
            "susceptible_nodes_per_replica": activity.replica_susceptible_nodes[0],
            "susceptible_edges_per_replica": activity.replica_susceptible_edges[0],
            "hazard_nodes_per_replica": activity.replica_hazard_nodes[0],
            "susceptible_union_nodes_per_tile": activity.tile_susceptible_union_nodes[0],
            "susceptible_union_edges_per_tile": activity.tile_susceptible_union_edges[0],
        },
        "bytes": asdict(traffic),
    }


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ("git", "-C", str(ROOT), *args),
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    status_scope = (
        "flashspread",
        "experiments/benchmark_ensemble.py",
        "experiments/benchmark_acceptance.py",
        "experiments/ensemble_perf_model.py",
        "experiments/perf_model.py",
        "pyproject.toml",
    )
    status = run("status", "--porcelain", "--untracked-files=all", "--", *status_scope)
    source_paths = sorted((ROOT / "flashspread").rglob("*.py"))
    source_paths.extend(ROOT / relative for relative in status_scope[1:])
    digest = hashlib.sha256()
    source_files = 0
    for path in source_paths:
        try:
            relative = path.relative_to(ROOT).as_posix().encode()
            payload = path.read_bytes()
        except OSError:
            continue
        digest.update(relative)
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
        source_files += 1
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": None if status is None else status.splitlines(),
        "status_scope": list(status_scope),
        "measured_source_sha256": digest.hexdigest(),
        "measured_source_files": source_files,
    }


def collect_metadata(device: str | None = None) -> dict[str, Any]:
    """Reuse host/GPU metadata collection but fingerprint this target's sources."""
    result = _acceptance.collect_metadata(device)
    result["git"] = _git()
    return result


def _shallow_storage(engine, graph) -> dict[str, Any]:
    canonical = graph.csr if hasattr(graph, "csr") else graph
    owners = (engine, engine.graph, graph, canonical)
    tensors = [
        value
        for owner in owners
        if owner is not None and hasattr(owner, "__dict__")
        for value in vars(owner).values()
        if isinstance(value, torch.Tensor)
    ]
    return {
        **asdict(unique_storage_usage(tensors)),
        "scope": (
            "Unique backing storage for tensors directly owned by the engine, "
            "its canonical graph, and graph wrapper at construction time. "
            "Excludes checkpoint staging, tensors nested in containers/model "
            "objects, compiler workspaces, allocator reservation, and caches."
        ),
    }


def _make_engine(args: argparse.Namespace):
    # Keep optional GPU imports behind the non-dry execution boundary.
    import flashspread as fs
    from flashspread.engines import create_ensemble_engine

    graph = fs.regular_graph(
        args.nodes,
        args.degree,
        seed=args.seed,
        device=args.device,
        algorithm="circulant",
    )
    model = fs.SEIRModel(transmission_mode="constant")
    engine = create_ensemble_engine(
        graph,
        model,
        args.replicas,
        device=args.device,
        backend="tiled",
        seed=args.seed,
    )
    if type(engine).__name__ != "EnsembleEngine":
        raise RuntimeError("backend='tiled' did not create EnsembleEngine")
    if engine.storage_profile != "fused_seir":
        raise RuntimeError(
            "the requested built-in constant SEIR model did not select the "
            "production fused_seir storage path"
        )

    csr = graph.csr if hasattr(graph, "csr") else graph
    actual = {
        "nodes": int(csr.num_nodes),
        "edges": int(csr.num_edges),
        "replicas": engine.replicas,
        "graph_construction_algorithm": getattr(graph, "construction_algorithm", None),
        "circulant_offsets": list(getattr(graph, "circulant_offsets", ())),
        "circulant_component_count": getattr(graph, "circulant_component_count", None),
        "graph_construction_memory_plan": dict(getattr(graph, "construction_memory_plan", {})),
        "engine": type(engine).__name__,
        "backend": "tiled",
        "storage_profile": engine.storage_profile,
        "nodes_per_program": engine.nodes_per_program,
        "replicas_per_tile": engine.replicas_per_tile,
        "rate_bound_nodes_per_partial": engine._rate_bound_nodes_per_partial,
        "rate_bound_partial_shape": list(engine._min_rate_partials.shape),
        "rate_event_partial_temporal_alias": (
            engine._min_rate_partials.untyped_storage().data_ptr()
            == engine._event_partials.untyped_storage().data_ptr()
        ),
        "rate_event_temporally_shared_bytes": (
            engine._min_rate_partials.untyped_storage().nbytes()
        ),
        "epsilon": engine.epsilon,
        "tau_max": engine.tau_max,
        "tensor_width_bytes": {
            "state": engine.state.element_size(),
            "age": engine.age.element_size(),
            "rate": engine.rates.element_size(),
            "csr_row_pointer": csr.row_ptr.element_size(),
            "csr_column_index": csr.col_ind.element_size(),
            "weight_storage": csr.weights_storage.element_size(),
            "packed_word": engine._infectious_mask.element_size(),
        },
        "shallow_engine_graph_tensor_storage": _shallow_storage(engine, graph),
    }
    return engine, graph, actual


def _restore(engine, state: torch.Tensor, age: torch.Tensor) -> None:
    engine.reset(episode=0)
    # Both inputs remain one-dimensional. The engine broadcasts them into its
    # node-major [N, R] public tensors; no [N, R] checkpoint copy is retained.
    engine.set_initial_state(state, age)


def _warmup(
    engine,
    state: torch.Tensor,
    age: torch.Tensor,
    *,
    minimum_calls: int = WARMUP_MINIMUM_CALLS,
    minimum_duration: float = WARMUP_MINIMUM_DURATION_SECONDS,
) -> dict[str, int | float]:
    """Prime fixed specializations, then sustain synchronized warmup work."""
    torch.cuda.synchronize(engine.device)
    priming_start = time.perf_counter()
    for _ in range(minimum_calls):
        _restore(engine, state, age)
        engine.step()
        torch.cuda.synchronize(engine.device)
    priming_duration = time.perf_counter() - priming_start

    start = time.perf_counter()
    duration_calls = 0
    duration = 0.0
    while duration < minimum_duration:
        _restore(engine, state, age)
        engine.step()
        torch.cuda.synchronize(engine.device)
        duration_calls += 1
        duration = time.perf_counter() - start
    return {
        "total_calls": minimum_calls + duration_calls,
        "priming_calls": minimum_calls,
        "priming_phase_seconds": priming_duration,
        "duration_phase_calls": duration_calls,
        "duration_phase_seconds": duration,
    }


def summarize_tau(tau: torch.Tensor) -> dict[str, Any]:
    """Return a compact, deterministic summary of one fp32 replica vector."""
    values_tensor = tau.detach().to(device="cpu", copy=True).contiguous()
    values = values_tensor.to(torch.float64).tolist()
    digest = hashlib.sha256(memoryview(values_tensor.numpy()).cast("B")).hexdigest()
    return {
        "replicas": len(values),
        "dtype": str(tau.dtype),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
        "sha256": digest,
    }


def time_target(
    engine,
    state: torch.Tensor,
    age: torch.Tensor,
    *,
    repetitions: int,
    min_duration: float,
) -> tuple[list[float], list[dict[str, Any]], list[int]]:
    elapsed: list[float] = []
    tau_summaries: list[dict[str, Any]] = []
    changed_events: list[int] = []
    total = 0.0
    while len(elapsed) < repetitions or total < min_duration:
        _restore(engine, state, age)
        torch.cuda.synchronize(engine.device)
        start = time.perf_counter()
        tau, _ = engine.step()  # exactly one measured production target
        torch.cuda.synchronize(engine.device)
        duration = time.perf_counter() - start
        elapsed.append(duration)
        tau_summaries.append(summarize_tau(tau))
        # Restore zeroed total_events before the target. This read happens
        # after the elapsed-time sample and is therefore the exact number of
        # sparse state stores issued by that one step without entering timing.
        changed_events.append(int(engine.total_events.sum().item()))
        total += duration
    return elapsed, tau_summaries, changed_events


def nvtx_range_name(checkpoint: str) -> str:
    return f"flashspread_ensemble_acceptance_{checkpoint}"


def profile_target(
    engine,
    state: torch.Tensor,
    age: torch.Tensor,
    name: str,
) -> dict[str, Any]:
    _restore(engine, state, age)
    torch.cuda.synchronize(engine.device)
    start = time.perf_counter()
    with _optional_nvtx(name) as enabled:
        tau, _ = engine.step()  # exactly one profiled production target
    torch.cuda.synchronize(engine.device)
    elapsed = time.perf_counter() - start
    return {
        "nvtx_range": name,
        "nvtx_enabled": enabled,
        "elapsed_seconds": elapsed,
        "tau_vector_summary": summarize_tau(tau),
        # The scalar extraction is after both the NVTX range and timing sample.
        "transition_changed_events": int(engine.total_events.sum().item()),
        "node_replica_updates_per_second": (engine.num_nodes * engine.replicas / elapsed),
    }


def _workload(args: argparse.Namespace) -> dict[str, Any]:
    checkpoints = {
        name: checkpoint_count_definition(args.nodes, name) for name in _checkpoint_names(args)
    }
    result = {
        "graph": "circulant",
        "nodes_requested": args.nodes,
        "degree": args.degree,
        "directed_edges_requested": args.nodes * args.degree,
        "replicas_requested": args.replicas,
        "seed": args.seed,
        "device": args.device,
        "backend_requested": "tiled",
        "replicas_per_tile_requested": None,
        "replicas_per_tile_expected_default": default_replica_tile(args.replicas),
        "checkpoint_ages": {"E": 2.0, "I": 1.5},
        "checkpoint_definitions": checkpoints,
        "checkpoint_replica_semantics": (
            "one deterministic shared int32 state[N] and fp32 age[N] pair is "
            "broadcast by EnsembleEngine.set_initial_state into [N, R]"
        ),
        "warmup_policy": {
            "priming_calls": WARMUP_MINIMUM_CALLS,
            "minimum_duration_seconds_after_priming": (
                WARMUP_MINIMUM_DURATION_SECONDS
            ),
            "phase_order": "finish priming calls, then start a fresh duration clock",
            "duration_threshold_scope": "additional post-priming calls only",
            "synchronization": "before priming and after every target call",
        },
        "graph_semantics": (
            "seeded exact-simple undirected degree-regular circulant built "
            "directly in int32 CSR; not a uniform random-regular sample"
        ),
        "graph_construction_memory_plan_model": _circulant_memory_plan(args.nodes, args.degree),
        "model": {
            "kind": "built-in SEIR",
            "transmission": "constant",
            "is_markovian": False,
            "beta": 0.3,
            "mean_ei": 5.0,
            "median_ei": 4.0,
            "mean_ir": 3.9,
            "median_ir": 1.5,
        },
    }
    if args.mode == "walltime":
        result.update(
            repetitions_requested=args.repetitions,
            minimum_target_seconds=args.min_duration,
        )
    return result


def _document(
    args: argparse.Namespace,
    *,
    dry: bool,
    invocation_args: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "flashspread-production-ensemble-acceptance",
        "mode": args.mode,
        "invocation": [
            sys.executable,
            str(Path(__file__).resolve()),
            *invocation_args,
        ],
        "metadata": collect_metadata(None if dry else args.device),
        "workload": _workload(args),
        "profiling_contract": {
            "target": "exactly one EnsembleEngine.step() call",
            "execution": "eager multi-kernel production path",
            "nvtx_aggregation": "per-nvtx across every kernel in the range",
            "single_kernel_claim": False,
            "logical_traffic_scope": (
                "checkpoint state before the target; logical requests are a "
                "model rather than measured cache/HBM traffic"
            ),
        },
    }


def _traffic_for_engine(
    engine,
    definition: dict[str, Any],
    *,
    transition_changed_events: int | None = None,
) -> dict[str, Any]:
    graph = engine.graph
    if graph.row_ptr.element_size() != graph.col_ind.element_size():
        raise RuntimeError("traffic model requires one common CSR index width")
    return logical_traffic_reference(
        num_nodes=engine.num_nodes,
        degree=graph.num_edges // graph.num_nodes,
        replicas=engine.replicas,
        replicas_per_tile=engine.replicas_per_tile,
        counts=definition["counts"],
        state_bytes=engine.state.element_size(),
        age_bytes=engine.age.element_size(),
        rate_bytes=engine.rates.element_size(),
        index_bytes=graph.row_ptr.element_size(),
        weight_bytes=graph.weights_storage.element_size(),
        packed_word_bytes=engine._infectious_mask.element_size(),
        has_weights=graph.has_weights,
        rate_bound_nodes_per_partial=engine._rate_bound_nodes_per_partial,
        transition_changed_events=transition_changed_events,
    )


def _run(args: argparse.Namespace, document: dict[str, Any]) -> None:
    engine, graph, actual = _make_engine(args)
    document["workload"]["actual"] = actual
    results = []
    for name, state_cpu, age_cpu, definition in iter_checkpoints(
        args.nodes,
        args.seed,
        _checkpoint_names(args),
    ):
        # One-dimensional device staging preserves shared checkpoint semantics;
        # set_initial_state performs the replica broadcast on every restore.
        state = state_cpu.to(engine.device)
        age = age_cpu.to(engine.device)
        definition["replica_initialization"] = "shared [N] broadcast to [N, R]"
        definition["initial_step_logical_traffic_reference"] = _traffic_for_engine(
            engine, definition
        )
        warmup = _warmup(engine, state, age)

        if args.mode == "profile":
            profile = profile_target(engine, state, age, nvtx_range_name(name))
            definition["observed_initial_step_logical_traffic_reference"] = _traffic_for_engine(
                engine,
                definition,
                transition_changed_events=profile["transition_changed_events"],
            )
            results.append(
                {
                    "checkpoint": name,
                    "definition": definition,
                    "warmup": warmup,
                    **profile,
                }
            )
            del state_cpu, age_cpu, state, age
            continue

        elapsed, tau_summaries, changed_events = time_target(
            engine,
            state,
            age,
            repetitions=args.repetitions,
            min_duration=args.min_duration,
        )
        if len(set(changed_events)) != 1:
            raise RuntimeError("restored deterministic targets produced different event counts")
        definition["observed_initial_step_logical_traffic_reference"] = _traffic_for_engine(
            engine,
            definition,
            transition_changed_events=changed_events[0],
        )
        median = statistics.median(elapsed)
        mean = statistics.fmean(elapsed)
        mad = statistics.median(abs(value - median) for value in elapsed)
        results.append(
            {
                "checkpoint": name,
                "definition": definition,
                "warmup": warmup,
                "target_calls": len(elapsed),
                "production_steps_timed": len(elapsed),
                "elapsed_seconds": elapsed,
                "elapsed_seconds_summary": {
                    "min": min(elapsed),
                    "median": median,
                    "p95": _quantile(elapsed, 0.95),
                    "mean": mean,
                    "max": max(elapsed),
                    "mad": mad,
                    "coefficient_of_variation": (
                        statistics.pstdev(elapsed) / mean if mean else 0.0
                    ),
                },
                "node_replica_updates_per_second_median": (args.nodes * args.replicas / median),
                "tau_vector_summaries": tau_summaries,
                "transition_changed_events": changed_events,
            }
        )
        del state_cpu, age_cpu, state, age
    document["results"] = results


def ncu_command(args: argparse.Namespace) -> list[str]:
    checkpoint = _checkpoint_names(args)[0]
    profile_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "profile",
        "--nodes",
        str(args.nodes),
        "--degree",
        str(args.degree),
        "--replicas",
        str(args.replicas),
        "--seed",
        str(args.seed),
        "--checkpoint",
        checkpoint,
        "--device",
        args.device,
        "--output",
        args.json_output,
    ]
    return [
        args.ncu_bin,
        "--replay-mode",
        "application",
        "--cache-control",
        "none",
        "--clock-control",
        "boost",
        "--target-processes",
        "all",
        "--section",
        "SpeedOfLight",
        "--section",
        "SpeedOfLight_RooflineChart",
        "--section",
        "ComputeWorkloadAnalysis",
        "--section",
        "MemoryWorkloadAnalysis",
        "--section",
        "LaunchStats",
        "--section",
        "Occupancy",
        "--print-summary",
        "per-nvtx",
        "--force-overwrite",
        "--nvtx",
        "--nvtx-include",
        f"{nvtx_range_name(checkpoint)}/",
        "--export",
        args.ncu_output,
        *profile_args,
    ]


def _write(document: dict[str, Any], output: str | None, mode: str) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output == "-":
        sys.stdout.write(payload)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(output) if output else Path("results") / f"ensemble_acceptance_{mode}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    print(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    invocation_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(invocation_args)
    try:
        _validate(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.mode == "print-ncu-command":
        print(shlex.join(ncu_command(args)))
        return 0
    if args.dry_run:
        document = _document(args, dry=True, invocation_args=invocation_args)
        document.update(status="dry_run", checkpoints=list(_checkpoint_names(args)))
        _write(document, args.output or "-", args.mode)
        return 0
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --dry-run for CPU-side validation")

    try:
        torch.empty(1, device=args.device)
        document = _document(args, dry=False, invocation_args=invocation_args)
        _run(args, document)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    _write(document, args.output, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
