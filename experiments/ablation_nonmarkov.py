#!/usr/bin/env python
"""
Ablation Study for Non-Markovian Edge Transmission & Kernel Fusion.

Tests each Phase 1-3 optimization independently and in combination,
measuring performance (ms/step, speedup) and accuracy (infected count
vs baseline) to justify each design choice.

Usage:
    python experiments/ablation_nonmarkov.py --config CONFIG_ID [--output-dir DIR]

Configs:
    0: baseline           — RenewalEngine (Markovian edges, fp32)
    1: bf16_only          — RenewalEngine + bf16 weights
    2: cudagraph_baseline — RenewalEngineCUDAGraph (existing best)
    3: bf16_cudagraph     — RenewalEngineCUDAGraph + bf16 weights
    4: nonmarkov          — RenewalEngineNonMarkov (source-node compromise)
    5: nonmarkov_bf16     — RenewalEngineNonMarkov + bf16 weights
    6: nonmarkov_cg       — RenewalEngineNonMarkovCUDAGraph
    7: nonmarkov_cg_bf16  — RenewalEngineNonMarkovCUDAGraph + bf16
    8: fused              — RenewalEngineFused (Triton-fused kernel)
    9: fused_bf16         — RenewalEngineFused + bf16
   10: fused_cg           — RenewalEngineFusedCUDAGraph
   11: fused_cg_bf16      — RenewalEngineFusedCUDAGraph + bf16
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread import FixedDegreeGraph, SEIRModel
from flashspread.engines.renewal import (
    RenewalEngine,
    RenewalEngineCUDAGraph,
    RenewalEngineNonMarkov,
    RenewalEngineNonMarkovCUDAGraph,
)
from flashspread.engines.renewal_fused import (
    RenewalEngineFused,
    RenewalEngineFusedCUDAGraph,
)


@dataclass
class Config:
    name: str
    config_id: int
    # Engine selection
    engine_type: str  # "baseline", "cudagraph", "nonmarkov", "nonmarkov_cg", "fused", "fused_cg"
    bf16_weights: bool = False
    steps_per_launch: int = 50
    # Simulation parameters
    num_nodes: int = 100_000
    degree: int = 15
    epsilon: float = 0.03
    tau_max: float = 1.0
    seed: int = 42
    # Benchmark parameters
    warmup_steps: int = 50
    benchmark_steps: int = 200
    accuracy_runs: int = 5


@dataclass
class Result:
    config_name: str
    config_id: int
    engine_type: str
    bf16_weights: bool
    steps_per_launch: int
    num_nodes: int
    num_edges: int
    total_wall_time_s: float
    steps_executed: int
    time_per_step_ms: float
    steps_per_second: float
    speedup_vs_baseline: float
    final_infected_mean: float
    final_infected_std: float
    accuracy_error_percent: float
    accuracy_passed: bool


def create_engine(graph, model, config: Config, device: str, seed: int):
    """Create the appropriate engine variant."""
    kwargs = dict(
        device=device,
        epsilon=config.epsilon,
        tau_max=config.tau_max,
        seed=seed,
        bf16_weights=config.bf16_weights,
    )

    if config.engine_type == "baseline":
        return RenewalEngine(graph, model, **kwargs)

    elif config.engine_type == "cudagraph":
        return RenewalEngineCUDAGraph(
            graph, model,
            steps_per_launch=config.steps_per_launch,
            **kwargs,
        )

    elif config.engine_type == "nonmarkov":
        return RenewalEngineNonMarkov(graph, model, **kwargs)

    elif config.engine_type == "nonmarkov_cg":
        return RenewalEngineNonMarkovCUDAGraph(
            graph, model,
            steps_per_launch=config.steps_per_launch,
            **kwargs,
        )

    elif config.engine_type == "fused":
        return RenewalEngineFused(graph, model, **kwargs)

    elif config.engine_type == "fused_cg":
        return RenewalEngineFusedCUDAGraph(
            graph, model,
            steps_per_launch=config.steps_per_launch,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown engine_type: {config.engine_type}")


def run_simulation(graph, model, config: Config, device: str) -> Tuple[float, int, float]:
    """Run one simulation. Returns (wall_time, steps_done, final_infected)."""
    engine = create_engine(graph, model, config, device, config.seed)
    engine.seed_infection(max(100, config.num_nodes // 100), state=model.exposed)

    # Steps per engine.step() call
    is_batched = config.engine_type in ("cudagraph", "nonmarkov_cg", "fused_cg")
    steps_per_call = config.steps_per_launch if is_batched else 1

    # Warmup
    warmup_calls = max(1, config.warmup_steps // steps_per_call)
    for _ in range(warmup_calls):
        engine.step()
    torch.cuda.synchronize()

    # Benchmark
    target_steps = config.benchmark_steps
    num_calls = max(1, target_steps // steps_per_call)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(num_calls):
        engine.step()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    steps_done = num_calls * steps_per_call
    wall_time = t1 - t0

    final_infected = engine.count_by_state()[2].item()  # state 2 = Infected
    return wall_time, steps_done, final_infected


def get_configs() -> List[Config]:
    configs = []

    # 0: baseline
    configs.append(Config(name="baseline", config_id=0, engine_type="baseline"))

    # 1: bf16 only
    configs.append(Config(name="bf16_only", config_id=1, engine_type="baseline", bf16_weights=True))

    # 2: cudagraph baseline (existing best)
    configs.append(Config(name="cudagraph_baseline", config_id=2, engine_type="cudagraph"))

    # 3: cudagraph + bf16
    configs.append(Config(name="bf16_cudagraph", config_id=3, engine_type="cudagraph", bf16_weights=True))

    # 4: nonmarkov edges (source-node compromise)
    configs.append(Config(name="nonmarkov", config_id=4, engine_type="nonmarkov"))

    # 5: nonmarkov + bf16
    configs.append(Config(name="nonmarkov_bf16", config_id=5, engine_type="nonmarkov", bf16_weights=True))

    # 6: nonmarkov + CUDA Graph
    configs.append(Config(name="nonmarkov_cg", config_id=6, engine_type="nonmarkov_cg"))

    # 7: nonmarkov + CUDA Graph + bf16
    configs.append(Config(name="nonmarkov_cg_bf16", config_id=7, engine_type="nonmarkov_cg", bf16_weights=True))

    # 8: fused Triton kernel
    configs.append(Config(name="fused", config_id=8, engine_type="fused"))

    # 9: fused + bf16
    configs.append(Config(name="fused_bf16", config_id=9, engine_type="fused", bf16_weights=True))

    # 10: fused + CUDA Graph
    configs.append(Config(name="fused_cg", config_id=10, engine_type="fused_cg"))

    # 11: fused + CUDA Graph + bf16
    configs.append(Config(name="fused_cg_bf16", config_id=11, engine_type="fused_cg", bf16_weights=True))

    return configs


def run_ablation(config: Config, device: str, baseline_result: Optional[Result]) -> Result:
    print(f"\n{'='*60}")
    print(f"Config {config.config_id}: {config.name}")
    print(f"  engine={config.engine_type}, bf16={config.bf16_weights}")
    print(f"{'='*60}")

    graph = FixedDegreeGraph(config.num_nodes, config.degree, device=device)
    model = SEIRModel(
        beta=0.3, mean_ei=5.0, median_ei=4.0, mean_ir=3.9, median_ir=1.5,
    )

    wall_times = []
    final_infected_list = []

    for run_idx in range(config.accuracy_runs):
        run_config = Config(**asdict(config))
        run_config.seed = config.seed + run_idx * 1000

        print(f"  Run {run_idx+1}/{config.accuracy_runs}...", end=" ", flush=True)
        try:
            wt, steps_done, final_inf = run_simulation(graph, model, run_config, device)
            wall_times.append(wt)
            final_infected_list.append(final_inf)
            print(f"OK ({wt*1000/steps_done:.3f} ms/step, infected={final_inf:.0f})")
        except Exception as e:
            print(f"FAILED: {e}")
            # Record failure as very slow + zero infected
            wall_times.append(999.0)
            final_infected_list.append(0.0)
            steps_done = config.benchmark_steps

    mean_wt = float(np.mean(wall_times))
    mean_inf = float(np.mean(final_infected_list))
    std_inf = float(np.std(final_infected_list))
    time_per_step = float(mean_wt * 1000 / steps_done)

    speedup = 1.0
    acc_err_pct = 0.0
    if baseline_result is not None:
        speedup = baseline_result.time_per_step_ms / time_per_step
        acc_err_pct = abs(mean_inf - baseline_result.final_infected_mean) / max(1, baseline_result.final_infected_mean) * 100

    result = Result(
        config_name=config.name,
        config_id=config.config_id,
        engine_type=config.engine_type,
        bf16_weights=config.bf16_weights,
        steps_per_launch=config.steps_per_launch,
        num_nodes=config.num_nodes,
        num_edges=graph.num_edges,
        total_wall_time_s=mean_wt,
        steps_executed=steps_done,
        time_per_step_ms=time_per_step,
        steps_per_second=float(steps_done / mean_wt),
        speedup_vs_baseline=float(speedup),
        final_infected_mean=mean_inf,
        final_infected_std=std_inf,
        accuracy_error_percent=float(acc_err_pct),
        accuracy_passed=bool(acc_err_pct < 15.0),
    )

    print(f"\n  Results:")
    print(f"    ms/step:  {result.time_per_step_ms:.3f}")
    print(f"    Speedup:  {result.speedup_vs_baseline:.2f}x")
    print(f"    Infected: {result.final_infected_mean:.0f} ± {result.final_infected_std:.0f}")
    print(f"    AccErr%:  {result.accuracy_error_percent:.1f}%  {'PASS' if result.accuracy_passed else 'FAIL'}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Non-Markovian Edge Ablation Study")
    parser.add_argument("--config", type=int, required=True, help="Config ID (0-11)")
    parser.add_argument("--output-dir", type=str, default="results/ablation_nonmarkov")
    parser.add_argument("--num-nodes", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configs = get_configs()
    if args.config < 0 or args.config >= len(configs):
        print(f"Error: config must be 0-{len(configs)-1}")
        sys.exit(1)

    config = configs[args.config]
    config.num_nodes = args.num_nodes

    print("FlashSpread Non-Markovian Edge Ablation")
    print("=" * 60)
    print(f"Config:  {config.name} (ID: {config.config_id})")
    print(f"Nodes:   {config.num_nodes:,}")
    print(f"Device:  {args.device}")
    if args.device == "cuda":
        print(f"GPU:     {torch.cuda.get_device_name()}")

    # Load baseline if available
    baseline_result = None
    baseline_path = os.path.join(args.output_dir, "ablation_baseline.json")
    if config.config_id != 0 and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_result = Result(**json.load(f))
        print(f"Baseline: {baseline_result.time_per_step_ms:.3f} ms/step")

    result = run_ablation(config, args.device, baseline_result)

    # Save
    out_path = os.path.join(args.output_dir, f"ablation_{config.name}.json")
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\nSaved: {out_path}")

    # If baseline, also save as ablation_baseline.json
    if config.config_id == 0:
        with open(baseline_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"Saved baseline: {baseline_path}")


if __name__ == "__main__":
    main()
