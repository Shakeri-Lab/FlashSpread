#!/usr/bin/env python
"""Production-path acceptance harness for batched Markovian SIS simulation.

The timed target is one ``Simulator.step()`` backed by
``MarkovianEngineCUDAGraph``. Early, peak, and late are deterministic synthetic
prevalence checkpoints, not phases inferred from one epidemic trajectory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
import flashspread as fs  # noqa: E402
from flashspread.core.network import _circulant_memory_plan  # noqa: E402


SCHEMA_VERSION = "flashspread.markovian_acceptance.v2"
CHECKPOINT_INFECTED_FRACTIONS = {
    "early": 0.01,
    "peak": 0.25,
    "late": 0.03,
}
WARMUP_MINIMUM_CALLS = 5
WARMUP_MINIMUM_DURATION_SECONDS = 0.25
_INT32_MAX = torch.iinfo(torch.int32).max
_MAX_CAPTURED_STEPS = 4096
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
    parser.add_argument("--batch-steps", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--checkpoint", choices=CHECKPOINT_INFECTED_FRACTIONS)
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
    ncu.add_argument("--ncu-output", default="results/flashspread_markovian_profile")
    ncu.add_argument(
        "--json-output",
        default="results/flashspread_markovian_profile.json",
    )
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
    if args.batch_steps > _MAX_CAPTURED_STEPS:
        raise ValueError(
            f"batch-steps must be <= {_MAX_CAPTURED_STEPS} for Markov CUDA Graph capture"
        )
    try:
        device = torch.device(args.device)
    except RuntimeError as exc:
        raise ValueError(f"invalid device {args.device!r}") from exc
    if device.type != "cuda":
        raise ValueError("Markovian acceptance measurements require a CUDA device")


def _checkpoint_names(args: argparse.Namespace) -> tuple[str, ...]:
    if args.checkpoint is not None:
        return (args.checkpoint,)
    if args.mode in {"profile", "print-ncu-command"}:
        return ("peak",)
    return tuple(CHECKPOINT_INFECTED_FRACTIONS)


def build_checkpoints(
    num_nodes: int,
    seed: int,
) -> dict[str, tuple[torch.Tensor, dict[str, Any]]]:
    """Build deterministic, nested SIS prevalence checkpoints on the CPU."""
    return {
        name: (state, definition)
        for name, state, definition in iter_checkpoints(
            num_nodes,
            seed,
            tuple(CHECKPOINT_INFECTED_FRACTIONS),
        )
    }


def iter_checkpoints(
    num_nodes: int,
    seed: int,
    names: Sequence[str],
):
    """Stream SIS checkpoints while retaining one shared int32 permutation."""
    permutation = torch.randperm(
        num_nodes,
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.int32,
    )
    for name in names:
        infected_fraction = CHECKPOINT_INFECTED_FRACTIONS[name]
        infected = int(num_nodes * infected_fraction)
        state = torch.zeros(num_nodes, dtype=torch.int32)
        state[permutation[:infected]] = 1
        digest = hashlib.sha256(memoryview(state.numpy()).cast("B")).hexdigest()
        yield (
            name,
            state,
            {
                "fractions_requested": {
                    "S": 1.0 - infected_fraction,
                    "I": infected_fraction,
                },
                "counts": {"S": num_nodes - infected, "I": infected},
                "state_sha256": digest,
            },
        )


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
        "experiments/benchmark_markovian.py",
        "experiments/benchmark_acceptance.py",
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
    """Reuse host/GPU discovery while fingerprinting this exact target."""
    metadata = _acceptance.collect_metadata(device)
    metadata["git"] = _git()
    return metadata


def _make_simulator(args: argparse.Namespace):
    graph = fs.regular_graph(
        args.nodes,
        args.degree,
        seed=args.seed,
        device=args.device,
        algorithm="circulant",
    )
    model = fs.SISModel(beta=0.5, delta=1.0)
    config = fs.EngineConfig(
        execution="cuda_graph",
        batch_steps=args.batch_steps,
    )
    simulator = fs.Simulator(
        graph,
        model,
        device=args.device,
        seed=args.seed,
        config=config,
    )
    engine = simulator.engine
    if type(engine).__name__ != "MarkovianEngineCUDAGraph":
        raise RuntimeError(
            "public Simulator factory did not select MarkovianEngineCUDAGraph"
        )
    csr = graph.csr if hasattr(graph, "csr") else graph
    actual = {
        "nodes": int(csr.num_nodes),
        "edges": int(csr.num_edges),
        "graph_construction_algorithm": getattr(graph, "construction_algorithm", None),
        "circulant_offsets": list(getattr(graph, "circulant_offsets", ())),
        "circulant_component_count": getattr(graph, "circulant_component_count", None),
        "graph_construction_memory_plan": dict(
            getattr(graph, "construction_memory_plan", {})
        ),
        "engine": type(engine).__name__,
        "batch_steps_effective": simulator.steps_per_launch,
        "max_prob_effective": engine.max_prob,
        "theta_effective": engine.theta,
        "tau_min_effective": engine.tau_min,
        "tau_max_effective": engine.tau_max,
        "outgoing_csr_shared": bool(engine._shares_outgoing_csr),
        "rate_reduction_level_sizes": [
            int(level.numel()) for level in engine._rate_sum_levels
        ],
        "event_reduction_level_sizes": [
            int(level.numel()) for level in engine._event_count_levels
        ],
    }
    return simulator, actual


def _restore(simulator, state: torch.Tensor) -> None:
    simulator.reset(episode=0)
    simulator.set_initial_state(state)


def _warmup(
    simulator,
    state: torch.Tensor,
    *,
    minimum_calls: int = WARMUP_MINIMUM_CALLS,
    minimum_duration: float = WARMUP_MINIMUM_DURATION_SECONDS,
) -> dict[str, int | float]:
    torch.cuda.synchronize(simulator.device)
    priming_start = time.perf_counter()
    for _ in range(minimum_calls):
        _restore(simulator, state)
        simulator.step()
        torch.cuda.synchronize(simulator.device)
    priming_duration = time.perf_counter() - priming_start

    start = time.perf_counter()
    duration_calls = 0
    duration = 0.0
    while duration < minimum_duration:
        _restore(simulator, state)
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


def time_target(
    simulator,
    state: torch.Tensor,
    *,
    repetitions: int,
    min_duration: float,
) -> tuple[list[float], list[float], list[int]]:
    elapsed: list[float] = []
    simulated: list[float] = []
    events: list[int] = []
    total = 0.0
    while len(elapsed) < repetitions or total < min_duration:
        _restore(simulator, state)
        torch.cuda.synchronize(simulator.device)
        start = time.perf_counter()
        simulated.append(float(simulator.step()))
        torch.cuda.synchronize(simulator.device)
        duration = time.perf_counter() - start
        elapsed.append(duration)
        events.append(int(simulator.engine.total_events))
        total += duration
    return elapsed, simulated, events


def nvtx_range_name(checkpoint: str) -> str:
    return f"flashspread_markovian_acceptance_{checkpoint}"


def profile_target(
    simulator,
    state: torch.Tensor,
    name: str,
) -> dict[str, Any]:
    _restore(simulator, state)
    torch.cuda.synchronize(simulator.device)
    start = time.perf_counter()
    with _optional_nvtx(name) as enabled:
        tau = simulator.step()
    torch.cuda.synchronize(simulator.device)
    return {
        "nvtx_range": name,
        "nvtx_enabled": enabled,
        "elapsed_seconds": time.perf_counter() - start,
        "simulated_time_advanced": float(tau),
        "transition_events": int(simulator.engine.total_events),
        "internal_steps": simulator.steps_per_launch,
    }


def _workload(args: argparse.Namespace) -> dict[str, Any]:
    result = {
        "graph": "circulant",
        "nodes_requested": args.nodes,
        "degree": args.degree,
        "seed": args.seed,
        "engine_config": {
            "execution": "cuda_graph",
            "batch_steps_requested": args.batch_steps,
            "max_prob": 0.1,
            "theta": 0.01,
            "tau_min": 1e-6,
            "tau_max": 1.0,
        },
        "model": {"kind": "SIS", "beta": 0.5, "delta": 1.0},
        "checkpoint_infected_fractions": dict(CHECKPOINT_INFECTED_FRACTIONS),
        "checkpoint_semantics": (
            "deterministic synthetic prevalence labels restored before every "
            "target; not phases observed from one trajectory"
        ),
        "graph_semantics": (
            "seeded exact-simple undirected circulant built directly in CSR; "
            "not a uniform random-regular sample"
        ),
        "graph_construction_memory_plan_model": _circulant_memory_plan(
            args.nodes, args.degree
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
        "node_updates_per_second_semantics": (
            "N * effective internal tau-leaps divided by median target wall time; "
            "a normalized node-step rate, not realized state transitions, frontier "
            "edges, or unique nodes changed"
        ),
        "timing_scope": (
            "one synchronized Simulator.step() production call; excludes graph/model "
            "construction, checkpoint restoration, and warmup"
        ),
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
        "benchmark": "flashspread-production-markovian-acceptance",
        "mode": args.mode,
        "invocation": [sys.executable, str(Path(__file__).resolve()), *invocation_args],
        "metadata": collect_metadata(None if dry else args.device),
        "workload": _workload(args),
        "profiling_contract": {
            "target": "exactly one Simulator.step() production call",
            "cuda_graph_node_mode": True,
            "aggregation_required": (
                "Aggregate every CUDA Graph node in the NVTX range; no single "
                "kernel represents the production pipeline."
            ),
        },
    }


def _run(args: argparse.Namespace, document: dict[str, Any]) -> None:
    simulator, actual = _make_simulator(args)
    document["workload"]["actual"] = actual
    results = []
    for name, state_cpu, definition in iter_checkpoints(
        args.nodes,
        args.seed,
        _checkpoint_names(args),
    ):
        state = state_cpu.to(simulator.device)
        definition["directed_csr_entries_in_infected_rows"] = (
            definition["counts"]["I"] * args.degree
        )
        warmup = _warmup(simulator, state)
        if args.mode == "profile":
            results.append(
                {
                    "checkpoint": name,
                    "definition": definition,
                    "warmup": warmup,
                    **profile_target(simulator, state, nvtx_range_name(name)),
                }
            )
            del state_cpu, state
            continue

        elapsed, simulated, events = time_target(
            simulator,
            state,
            repetitions=args.repetitions,
            min_duration=args.min_duration,
        )
        median = statistics.median(elapsed)
        mean = statistics.fmean(elapsed)
        mad = statistics.median(abs(value - median) for value in elapsed)
        steps = simulator.steps_per_launch
        results.append(
            {
                "checkpoint": name,
                "definition": definition,
                "warmup": warmup,
                "target_calls": len(elapsed),
                "internal_steps_per_target": steps,
                "internal_steps_timed": len(elapsed) * steps,
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
                "internal_steps_per_second_median": steps / median,
                "node_updates_per_second_median": args.nodes * steps / median,
                "simulated_time_per_wall_second_median": (
                    statistics.median(simulated) / median
                ),
                "simulated_time_advanced": simulated,
                "transition_events": events,
                "restored_target_outputs_identical": (
                    len(set(simulated)) == 1 and len(set(events)) == 1
                ),
            }
        )
        del state_cpu, state
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
        "--batch-steps",
        str(args.batch_steps),
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
        "--graph-profiling",
        "node",
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
    path = (
        Path(output)
        if output
        else Path("results") / f"markovian_acceptance_{mode}_{stamp}.json"
    )
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
