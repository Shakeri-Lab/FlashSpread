#!/usr/bin/env python
"""Honest production-path wall-time and profiling acceptance harness.

Reports measurements and provenance, never inferred FLOPs, hardware ceilings,
or threshold claims. ``print-ncu-command`` targets the single production call
inside the profile mode's NVTX range.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, Iterator, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flashspread as fs  # noqa: E402
from flashspread.core.network import _circulant_memory_plan  # noqa: E402
from experiments.perf_model import (  # noqa: E402
    renewal_logical_traffic,
    unique_storage_usage,
)


SCHEMA_VERSION = "flashspread.acceptance.v4"
CHECKPOINT_FRACTIONS = {
    "early": (0.98, 0.01, 0.01, 0.00),
    "peak": (0.45, 0.15, 0.25, 0.15),
    "late": (0.05, 0.02, 0.03, 0.90),
}
# Presets, rather than free-form engine switches, keep comparisons meaningful.
PRESETS = {
    "regular-constant": ("circulant", "thread", "constant", "fp32", False),
    "regular-age": ("circulant", "thread", "age_dependent", "fp32", False),
    "regular-mixed": ("circulant", "thread", "constant", "mixed", False),
    "regular-late-compact": ("circulant", "thread", "constant", "fp32", True),
    "ba-auto": ("ba", "auto", "constant", "fp32", False),
    "ba-thread": ("ba", "thread", "constant", "fp32", False),
    "ba-warp": ("ba", "warp", "constant", "fp32", False),
    "ba-merge": ("ba", "merge", "constant", "fp32", False),
}
WARMUP_MINIMUM_CALLS = 5
WARMUP_MINIMUM_DURATION_SECONDS = 0.25
_INT32_MAX = torch.iinfo(torch.int32).max


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
    parser.add_argument("--case", choices=PRESETS, default="regular-constant")
    parser.add_argument("--nodes", type=_positive_int, default=1_000_000)
    parser.add_argument(
        "--degree", type=_positive_int, default=8, help="circulant graph degree"
    )
    parser.add_argument("--m", type=_positive_int, default=4, help="BA attachments")
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--batch-steps", type=_positive_int, default=50)
    parser.add_argument("--checkpoint", choices=CHECKPOINT_FRACTIONS)
    parser.add_argument("--device", default="cuda")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    wall = modes.add_parser("walltime", help="time early/peak/late production calls")
    _common(wall)
    wall.add_argument("--repetitions", type=_positive_int, default=10)
    wall.add_argument("--min-duration", type=_nonnegative_float, default=1.0)
    wall.add_argument("--output", help="versioned JSON path; '-' means stdout")
    wall.add_argument("--dry-run", action="store_true")
    profile = modes.add_parser("profile", help="profile exactly one production call")
    _common(profile)
    profile.add_argument("--output", help="versioned JSON path; '-' means stdout")
    profile.add_argument("--dry-run", action="store_true")
    ncu = modes.add_parser("print-ncu-command", help="print a shell-safe ncu command")
    _common(ncu)
    ncu.add_argument("--ncu-output", default="results/flashspread_acceptance_profile")
    ncu.add_argument("--json-output", default="results/flashspread_acceptance_profile.json")
    ncu.add_argument("--ncu-bin", default="ncu")
    return parser


def _counts(n: int, fractions: Sequence[float]) -> list[int]:
    raw = [n * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    order = sorted(range(4), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for index in order[: n - sum(counts)]:
        counts[index] += 1
    return counts


def build_checkpoints(
    num_nodes: int, seed: int, *, exposed_age: float = 2.0, infected_age: float = 1.5
) -> dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]:
    """Materialize all phase states from one seeded permutation (test helper)."""
    return {
        name: (state, age, definition)
        for name, state, age, definition in iter_checkpoints(
            num_nodes,
            seed,
            CHECKPOINT_FRACTIONS,
            exposed_age=exposed_age,
            infected_age=infected_age,
        )
    }


def iter_checkpoints(
    num_nodes: int,
    seed: int,
    names: Sequence[str],
    *,
    exposed_age: float = 2.0,
    infected_age: float = 1.5,
):
    """Yield one deterministic phase state at a time.

    Streaming keeps the acceptance harness at one state/age pair instead of
    retaining early, peak, and late arrays simultaneously at N=10^8.
    """
    generator = torch.Generator().manual_seed(seed)
    # Population sizes are already constrained by the int32 CSR contract.
    # Keeping the one shared checkpoint permutation int32 saves 4N host bytes
    # (400 MB at N=10^8) without changing its ordering or seed semantics.
    permutation = torch.randperm(
        num_nodes, generator=generator, dtype=torch.int32
    )
    labels = ("S", "E", "I", "R")
    for name in names:
        fractions = CHECKPOINT_FRACTIONS[name]
        counts = _counts(num_nodes, fractions)
        state = torch.empty(num_nodes, dtype=torch.int32)
        start = 0
        for compartment, count in enumerate(counts):
            state[permutation[start : start + count]] = compartment
            start += count
        age = torch.zeros(num_nodes, dtype=torch.float32)
        e_start = counts[0]
        i_start = e_start + counts[1]
        age[permutation[e_start:i_start]] = exposed_age
        age[permutation[i_start : i_start + counts[2]]] = infected_age
        digest = hashlib.sha256()
        digest.update(memoryview(state.numpy()).cast("B"))
        digest.update(memoryview(age.numpy()).cast("B"))
        definition = {
            "fractions_requested": dict(zip(labels, fractions)),
            "counts": dict(zip(labels, counts)),
            "state_age_sha256": digest.hexdigest(),
        }
        yield name, state, age, definition


def _git() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ("git", "-C", str(ROOT), *args), capture_output=True, text=True,
                check=True, timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    status_scope = (
        "flashspread",
        "experiments/benchmark_acceptance.py",
        "experiments/perf_model.py",
        "pyproject.toml",
    )
    status = run(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *status_scope,
    )
    # Hash the exact measured implementation, independent of generated JSON or
    # NCU files under results/. This fingerprints uncommitted and untracked
    # source without allowing application replay outputs to contaminate it.
    source_paths = sorted((ROOT / "flashspread").rglob("*.py"))
    source_paths.extend(
        (ROOT / relative)
        for relative in (
            "experiments/benchmark_acceptance.py",
            "experiments/perf_model.py",
            "pyproject.toml",
        )
    )
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
    def version(name: str) -> str | None:
        try:
            return package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            return None
    uname = platform.uname()
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "system": uname.system, "release": uname.release, "machine": uname.machine,
            "cpu_count": os.cpu_count(),
        },
        "software": {
            "python": platform.python_version(), "executable": sys.executable,
            "flashspread": fs.__version__, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "triton": version("triton"),
            "numpy": version("numpy"), "networkx": version("networkx"),
        },
        "git": _git(),
        "environment": {key: os.environ[key] for key in (
            "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_NODELIST"
        ) if key in os.environ},
        "gpu": None,
    }
    if device is not None:
        selected = torch.device(device)
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        try:
            driver_query = subprocess.run(
                (
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ),
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            driver_versions = sorted(
                {line.strip() for line in driver_query.stdout.splitlines() if line.strip()}
            )
        except (OSError, subprocess.SubprocessError):
            driver_versions = []
        result["gpu"] = {
            "logical_index": index, "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "uuid": str(getattr(properties, "uuid", "")) or None,
            "driver_versions_reported": driver_versions or None,
        }
    return result


def _validate(args: argparse.Namespace) -> None:
    graph_kind = PRESETS[args.case][0]
    graph_parameter = args.degree if graph_kind == "circulant" else args.m
    if graph_parameter >= args.nodes:
        raise ValueError("degree/m must be smaller than nodes")
    if args.nodes > _INT32_MAX:
        raise ValueError("nodes exceeds FlashSpread's int32 CSR node limit")
    directed_edges = (
        args.nodes * args.degree
        if graph_kind == "circulant"
        else 2 * args.m * (args.nodes - args.m)
    )
    if directed_edges > _INT32_MAX:
        raise ValueError(
            "requested graph exceeds FlashSpread's int32 CSR edge limit: "
            f"{directed_edges} > {_INT32_MAX}"
        )
    if graph_kind == "circulant" and (args.nodes * args.degree) % 2:
        raise ValueError("nodes * degree must be even")
    if torch.device(args.device).type != "cuda":
        raise ValueError("acceptance measurements require a CUDA device")


def _checkpoint_names(args: argparse.Namespace) -> tuple[str, ...]:
    if args.checkpoint:
        return (args.checkpoint,)
    if args.case == "regular-late-compact" or args.mode in {"profile", "print-ncu-command"}:
        return ("late" if args.case == "regular-late-compact" else "peak",)
    return tuple(CHECKPOINT_FRACTIONS)


def _make_simulator(args: argparse.Namespace):
    graph_kind, traversal, transmission, precision, compact = PRESETS[args.case]
    graph = (
        fs.regular_graph(
            args.nodes,
            args.degree,
            seed=args.seed,
            device=args.device,
            algorithm="circulant",
        )
        if graph_kind == "circulant"
        else fs.barabasi_albert(args.nodes, args.m, seed=args.seed, device=args.device)
    )
    model = fs.SEIRModel(transmission_mode=transmission)
    config = fs.EngineConfig(
        backend="fused", execution="cuda_graph", traversal=traversal,
        transmission=transmission, precision=precision, compact=compact,
        batch_steps=args.batch_steps,
    )
    simulator = fs.Simulator(graph, model, device=args.device, seed=args.seed, config=config)
    csr = graph.csr if hasattr(graph, "csr") else graph
    actual = {
        "nodes": int(csr.num_nodes), "edges": int(csr.num_edges),
        "graph_construction_algorithm": getattr(
            graph, "construction_algorithm", None
        ),
        "engine": type(simulator.engine).__name__,
        "batch_steps_effective": simulator.steps_per_launch,
        "traversal_effective": getattr(simulator.engine, "csr_strategy", traversal),
        "epsilon_effective": simulator.engine.epsilon,
        "tau_max_effective": simulator.engine.tau_max,
        "nodes_per_block_effective": simulator.engine.nodes_per_block,
        "lanes_per_node_effective": simulator.engine.lanes_per_node,
        "edges_per_merge_block_effective": simulator.engine.edges_per_merge_block,
        "rate_max_partial_shape": list(
            simulator.engine._max_rate_partials.shape
        ),
        "rate_max_partial_bytes": (
            simulator.engine._max_rate_partials.numel()
            * simulator.engine._max_rate_partials.element_size()
        ),
        "rate_reduction_source": "per-rate-program maximum partials",
    }
    if hasattr(graph, "circulant_offsets"):
        actual["circulant_offsets"] = list(graph.circulant_offsets)
        actual["circulant_component_count"] = graph.circulant_component_count
        actual["graph_construction_memory_plan"] = dict(
            graph.construction_memory_plan
        )
    tensor_owners = (
        simulator.engine,
        simulator.engine.graph,
        simulator.graph,
        getattr(simulator.graph, "csr", None),
    )
    tensors = [
        value
        for owner in tensor_owners
        if owner is not None and hasattr(owner, "__dict__")
        for value in vars(owner).values()
        if isinstance(value, torch.Tensor)
    ]
    actual["shallow_engine_graph_tensor_storage"] = {
        **asdict(unique_storage_usage(tensors)),
        "scope": (
            "Unique backing storage for tensors directly owned by the engine, "
            "its canonical graph, and the graph wrapper at construction time. "
            "Excludes checkpoint state/age staging, tensors nested in containers "
            "or model objects, CUDA Graph/private pools, compiler workspaces, and "
            "allocator reservation; this is not total GPU residency."
        ),
    }
    return simulator, actual


def _restore(simulator, state: torch.Tensor, age: torch.Tensor) -> None:
    simulator.reset(episode=0)
    simulator.set_initial_state(state, age)


def _warmup(
    simulator,
    state: torch.Tensor,
    age: torch.Tensor,
    *,
    minimum_calls: int = WARMUP_MINIMUM_CALLS,
    minimum_duration: float = WARMUP_MINIMUM_DURATION_SECONDS,
) -> dict[str, int | float]:
    """Prime fixed specializations, then sustain synchronized warmup work."""
    torch.cuda.synchronize(simulator.device)
    priming_start = time.perf_counter()
    for _ in range(minimum_calls):
        _restore(simulator, state, age)
        simulator.step()
        torch.cuda.synchronize(simulator.device)
    priming_duration = time.perf_counter() - priming_start

    start = time.perf_counter()
    duration_calls = 0
    duration = 0.0
    while duration < minimum_duration:
        _restore(simulator, state, age)
        simulator.step()
        torch.cuda.synchronize(simulator.device)
        duration_calls += 1
        duration = time.perf_counter() - start
    return {
        "total_calls": minimum_calls + duration_calls,
        "priming_calls": minimum_calls,
        "priming_phase_seconds": priming_duration,
        "duration_phase_calls": duration_calls,
        "duration_phase_seconds": duration,
    }


def time_target(simulator, state, age, *, repetitions: int, min_duration: float):
    elapsed, simulated = [], []
    total = 0.0
    while len(elapsed) < repetitions or total < min_duration:
        _restore(simulator, state, age)
        torch.cuda.synchronize(simulator.device)
        start = time.perf_counter()
        simulated.append(float(simulator.step()))
        torch.cuda.synchronize(simulator.device)
        elapsed.append(time.perf_counter() - start)
        total += elapsed[-1]
    return elapsed, simulated


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def nvtx_range_name(checkpoint: str) -> str:
    return f"flashspread_acceptance_{checkpoint}"


@contextmanager
def _optional_nvtx(name: str) -> Iterator[bool]:
    try:
        torch.cuda.nvtx.range_push(name)
    except (AttributeError, RuntimeError):
        yield False
    else:
        try:
            yield True
        finally:
            torch.cuda.nvtx.range_pop()


def profile_target(simulator, state, age, name: str) -> dict[str, Any]:
    _restore(simulator, state, age)
    torch.cuda.synchronize(simulator.device)
    start = time.perf_counter()
    with _optional_nvtx(name) as enabled:
        tau = simulator.step()  # exactly one profiled production target
    torch.cuda.synchronize(simulator.device)
    return {
        "nvtx_range": name, "nvtx_enabled": enabled,
        "elapsed_seconds": time.perf_counter() - start,
        "simulated_time_advanced": float(tau),
        "internal_steps": simulator.steps_per_launch,
    }


def _workload(args: argparse.Namespace) -> dict[str, Any]:
    graph, traversal, transmission, precision, compact = PRESETS[args.case]
    result = {
        "case": args.case, "graph": graph, "nodes_requested": args.nodes,
        "degree": args.degree if graph == "circulant" else None,
        "m": args.m if graph == "ba" else None, "seed": args.seed,
        "engine_config": {
            "backend": "fused", "execution": "cuda_graph", "traversal": traversal,
            "transmission": transmission, "precision": precision, "compact": compact,
            "batch_steps_requested": args.batch_steps,
        },
        "checkpoint_ages": {"E": 2.0, "I": 1.5},
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
            "seeded exact-simple undirected circulant built directly in CSR; "
            "not a uniform random-regular sample; shared row offsets give it "
            "more regular/coalesced adjacency access than a randomly labelled "
            "regular graph"
            if graph == "circulant"
            else "NetworkX Barabasi-Albert graph converted to incoming CSR"
        ),
        "model": {
            "kind": "SEIR", "beta": 0.3, "mean_ei": 5.0, "median_ei": 4.0,
            "mean_ir": 3.9, "median_ir": 1.5,
        },
    }
    if args.mode == "walltime":
        result.update(repetitions_requested=args.repetitions,
                      minimum_target_seconds=args.min_duration)
    if graph == "circulant":
        result["graph_construction_memory_plan_model"] = _circulant_memory_plan(
            args.nodes, args.degree
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
        "benchmark": "flashspread-production-acceptance", "mode": args.mode,
        "invocation": [
            sys.executable,
            str(Path(__file__).resolve()),
            *invocation_args,
        ],
        "metadata": collect_metadata(None if dry else args.device),
        "workload": _workload(args),
        "profiling_contract": {
            "target": "one Simulator.step() production call",
            "cuda_graph_node_mode": True,
            "aggregation_required": (
                "Aggregate every CUDA Graph node in the NVTX range; a single "
                "kernel RooflineChart does not represent the split pipeline."
            ),
            "logical_traffic_scope": (
                "Each checkpoint reference models only its first internal step "
                "and is not an aggregate-byte estimate for the full replay."
            ),
        },
    }


def _run(args: argparse.Namespace, document: dict[str, Any]) -> None:
    simulator, actual = _make_simulator(args)
    document["workload"]["actual"] = actual
    results = []
    checkpoint_names = _checkpoint_names(args)
    for name, state_cpu, age_cpu, definition in iter_checkpoints(
        args.nodes,
        args.seed,
        checkpoint_names,
    ):
        state, age = state_cpu.to(simulator.device), age_cpu.to(simulator.device)
        degrees = simulator.engine.graph.row_ptr[1:] - simulator.engine.graph.row_ptr[:-1]
        definition["susceptible_edges"] = int(degrees[state == 0].sum().item())
        # Do not carry this 4*N accounting temporary through warmup or timing.
        del degrees
        definition["hazard_nodes"] = int(((state == 1) | (state == 2)).sum().item())
        _, traversal, transmission, _, compact = PRESETS[args.case]
        effective_traversal = getattr(simulator.engine, "csr_strategy", traversal)
        if effective_traversal in {"thread", "warp"} and not compact:
            graph = simulator.engine.graph
            definition["initial_internal_step_logical_traffic_reference"] = {
                "scope": "checkpoint state before internal step 1 only",
                "aggregate_replay_comparable": False,
                "bytes": asdict(renewal_logical_traffic(
                    num_nodes=graph.num_nodes,
                    num_edges=graph.num_edges,
                    susceptible_nodes=definition["counts"]["S"],
                    susceptible_edges=definition["susceptible_edges"],
                    hazard_nodes=definition["hazard_nodes"],
                    transmission=transmission,
                    has_weights=graph.has_weights,
                    state_bytes=simulator.engine.state.element_size(),
                    age_bytes=simulator.engine.age.element_size(),
                    infectivity_bytes=simulator.engine._inf_dtype.itemsize,
                    weight_bytes=graph.weights_storage.element_size(),
                    rate_nodes_per_partial=(
                        simulator.engine.nodes_per_block
                        if effective_traversal == "warp"
                        else 128
                    ),
                )),
            }
        warmup = _warmup(simulator, state, age)
        if args.mode == "profile":
            results.append({
                "checkpoint": name,
                "definition": definition,
                "warmup": warmup,
                **profile_target(simulator, state, age, nvtx_range_name(name)),
            })
            del state_cpu, age_cpu, state, age
            continue
        elapsed, simulated = time_target(
            simulator, state, age, repetitions=args.repetitions,
            min_duration=args.min_duration,
        )
        steps = simulator.steps_per_launch
        median = statistics.median(elapsed)
        mean = statistics.fmean(elapsed)
        mad = statistics.median(abs(value - median) for value in elapsed)
        results.append({
            "checkpoint": name, "definition": definition, "warmup": warmup,
            "target_calls": len(elapsed),
            "internal_steps_per_target": steps, "internal_steps_timed": len(elapsed) * steps,
            "elapsed_seconds": elapsed,
            "elapsed_seconds_summary": {
                "min": min(elapsed), "median": median, "p95": _quantile(elapsed, 0.95),
                "mean": mean, "max": max(elapsed), "mad": mad,
                "coefficient_of_variation": (
                    statistics.pstdev(elapsed) / mean if mean else 0.0
                ),
            },
            "internal_steps_per_second_median": steps / median,
            "node_updates_per_second_median": args.nodes * steps / median,
            "simulated_time_per_wall_second_median": (
                statistics.median(simulated) / median
            ),
            "simulated_time_advanced": simulated,
        })
        # Release this phase before the generator constructs the next pair.
        del state_cpu, age_cpu, state, age
    document["results"] = results


def ncu_command(args: argparse.Namespace) -> list[str]:
    checkpoint = _checkpoint_names(args)[0]
    profile_args = [
        sys.executable, str(Path(__file__).resolve()), "profile", "--case", args.case,
        "--nodes", str(args.nodes), "--degree", str(args.degree), "--m", str(args.m),
        "--seed", str(args.seed), "--batch-steps", str(args.batch_steps),
        "--checkpoint", checkpoint, "--device", args.device, "--output", args.json_output,
    ]
    return [
        args.ncu_bin,
        "--replay-mode", "application",
        "--graph-profiling", "node",
        "--cache-control", "none",
        "--clock-control", "boost",
        "--target-processes", "all",
        "--section", "SpeedOfLight",
        "--section", "SpeedOfLight_RooflineChart",
        "--section", "ComputeWorkloadAnalysis",
        "--section", "MemoryWorkloadAnalysis",
        "--section", "LaunchStats",
        "--section", "Occupancy",
        "--print-summary", "per-nvtx",
        "--force-overwrite",
        "--nvtx",
        "--nvtx-include", f"{nvtx_range_name(checkpoint)}/",
        "--export", args.ncu_output, *profile_args,
    ]


def _write(document: dict[str, Any], output: str | None, mode: str) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output == "-":
        sys.stdout.write(payload)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(output) if output else Path("results") / f"acceptance_{mode}_{stamp}.json"
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
    except (RuntimeError, ImportError, ValueError) as exc:
        parser.error(str(exc))
    _write(document, args.output, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
