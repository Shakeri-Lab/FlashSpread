#!/usr/bin/env python
"""Historical synthetic roofline exploration (not an acceptance benchmark).

This script exercises ``RenewalEngineTunable`` and artificial compute
multipliers. Its fixed A100-80GB ceiling and pre-two-phase byte formulas do not
describe the current production engine. Use ``benchmark_acceptance.py`` plus
Nsight Compute for publishable production-path measurements.

This script runs systematic experiments to characterize the compute vs
memory-bound behavior of the Renewal and Markovian engines under various
parameter configurations.

Usage:
    python experiments/benchmark_roofline.py [--config CONFIG] [--output-dir DIR]

Output:
    - results/roofline_data.csv: Raw benchmark data
    - results/roofline_plot.png: Roofline visualization
    - results/roofline_summary.md: Analysis summary
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, List
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread import FixedDegreeGraph, SEIRModel, SISModel
from flashspread.engines import MarkovianEngine
from flashspread.engines.renewal_tunable import (
    RenewalEngineTunable,
    RenewalEngineTunableCUDAGraph,
    estimate_flops_per_step,
    estimate_memory_bytes_per_step,
)


# A100 GPU specifications
A100_PEAK_FLOPS = 19.5e12  # 19.5 TFLOPS FP32
A100_PEAK_BANDWIDTH = 2039e9  # 2039 GB/s HBM2e
A100_RIDGE_POINT = A100_PEAK_FLOPS / A100_PEAK_BANDWIDTH  # ~9.6 FLOPs/byte


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    engine_type: str  # "renewal", "renewal_cuda_graph", "markovian"
    num_nodes: int = 1_000_000
    degree: int = 15
    # Renewal parameters
    epsilon: float = 0.03
    tau_max: float = 1.0
    steps_per_launch: int = 1
    sparse_hazard: bool = True
    compute_multiplier: int = 1
    dense_pressure: bool = False
    # Markovian parameters
    max_prob: float = 0.1
    theta: float = 0.01
    # Simulation parameters
    warmup_steps: int = 100
    benchmark_steps: int = 500
    target_time: float = 10.0  # For renewal engines
    seed: int = 12345


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    engine_type: str
    num_nodes: int
    num_edges: int
    epsilon: float
    steps_per_launch: int
    sparse_hazard: bool
    compute_multiplier: int
    # Timing results
    total_wall_time_s: float
    steps_executed: int
    time_per_step_ms: float
    steps_per_second: float
    # FLOP estimates
    estimated_flops_per_step: int
    estimated_bytes_per_step: int
    arithmetic_intensity: float  # FLOPs/byte
    achieved_gflops: float
    # Roofline position
    memory_bound_ceiling_gflops: float
    compute_bound_ceiling_gflops: float
    efficiency_percent: float


def create_graph(num_nodes: int, degree: int, device: str) -> Any:
    """Create a FixedDegreeGraph for benchmarking."""
    print(f"  Creating graph: {num_nodes:,} nodes, degree {degree}...")
    start = time.time()
    graph = FixedDegreeGraph(num_nodes, degree, device=device)
    elapsed = time.time() - start
    print(f"  Graph created in {elapsed:.2f}s, {graph.num_edges:,} edges")
    return graph


def run_renewal_benchmark(
    config: BenchmarkConfig,
    graph: Any,
    device: str,
) -> BenchmarkResult:
    """Run benchmark for Renewal engine."""
    # Create model
    model = SEIRModel(
        beta=0.3,
        mean_ei=5.0,
        median_ei=4.0,
        mean_ir=3.9,
        median_ir=1.5,
    )
    model.sparse_hazard = config.sparse_hazard

    # Create engine based on type
    if config.engine_type == "renewal_cuda_graph":
        engine = RenewalEngineTunableCUDAGraph(
            graph, model, device=device,
            epsilon=config.epsilon,
            tau_max=config.tau_max,
            seed=config.seed,
            steps_per_launch=config.steps_per_launch,
            compute_multiplier=config.compute_multiplier,
            dense_pressure=config.dense_pressure,
        )
    else:
        engine = RenewalEngineTunable(
            graph, model, device=device,
            epsilon=config.epsilon,
            tau_max=config.tau_max,
            seed=config.seed,
            compute_multiplier=config.compute_multiplier,
            dense_pressure=config.dense_pressure,
            timing_enabled=False,  # Disable for benchmark speed
        )

    # Seed infection
    engine.seed_infection(max(100, config.num_nodes // 100), state=model.exposed)

    # Warmup
    print(f"  Warming up ({config.warmup_steps} steps)...")
    for _ in range(config.warmup_steps):
        engine.step()
    torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking ({config.benchmark_steps} steps)...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    steps_done = 0
    for _ in range(config.benchmark_steps):
        engine.step()
        steps_done += config.steps_per_launch if config.engine_type == "renewal_cuda_graph" else 1

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_wall_time = end_time - start_time

    # Compute estimates
    num_edges = graph.num_edges
    flops_estimate = estimate_flops_per_step(
        config.num_nodes, num_edges,
        config.compute_multiplier,
        not config.sparse_hazard,
    )
    bytes_estimate = estimate_memory_bytes_per_step(
        config.num_nodes, num_edges,
        config.dense_pressure,
    )

    flops_per_step = flops_estimate["total"]
    bytes_per_step = bytes_estimate["total"]
    arithmetic_intensity = flops_per_step / bytes_per_step

    time_per_step_s = total_wall_time / steps_done
    achieved_gflops = (flops_per_step / time_per_step_s) / 1e9

    # Roofline ceilings
    memory_ceiling = A100_PEAK_BANDWIDTH * arithmetic_intensity / 1e9
    compute_ceiling = A100_PEAK_FLOPS / 1e9
    theoretical_max = min(memory_ceiling, compute_ceiling)
    efficiency = (achieved_gflops / theoretical_max) * 100 if theoretical_max > 0 else 0

    return BenchmarkResult(
        config_name=config.name,
        engine_type=config.engine_type,
        num_nodes=config.num_nodes,
        num_edges=num_edges,
        epsilon=config.epsilon,
        steps_per_launch=config.steps_per_launch,
        sparse_hazard=config.sparse_hazard,
        compute_multiplier=config.compute_multiplier,
        total_wall_time_s=total_wall_time,
        steps_executed=steps_done,
        time_per_step_ms=time_per_step_s * 1000,
        steps_per_second=steps_done / total_wall_time,
        estimated_flops_per_step=flops_per_step,
        estimated_bytes_per_step=bytes_per_step,
        arithmetic_intensity=arithmetic_intensity,
        achieved_gflops=achieved_gflops,
        memory_bound_ceiling_gflops=memory_ceiling,
        compute_bound_ceiling_gflops=compute_ceiling,
        efficiency_percent=efficiency,
    )


def run_markovian_benchmark(
    config: BenchmarkConfig,
    graph: Any,
    device: str,
) -> BenchmarkResult:
    """Run benchmark for Markovian engine."""
    # Create SIS model for Markovian engine
    model = SISModel(beta=0.5, delta=1.0)

    engine = MarkovianEngine(
        graph, model, device=device,
        max_prob=config.max_prob,
        theta=config.theta,
        tau_min=1e-6,
        tau_max=config.tau_max,
        seed=config.seed,
    )

    # Seed infection
    engine.seed_infection(max(100, config.num_nodes // 100))

    # Warmup
    print(f"  Warming up ({config.warmup_steps} steps)...")
    for _ in range(config.warmup_steps):
        engine.step()
    torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking ({config.benchmark_steps} steps)...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(config.benchmark_steps):
        engine.step()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_wall_time = end_time - start_time
    steps_done = config.benchmark_steps

    # Markovian FLOP estimates (simpler than renewal)
    num_edges = graph.num_edges
    # FlashNeighbor + rate computation (simpler, no erfcx)
    flops_per_step = num_edges * 3 + config.num_nodes * 15
    bytes_per_step = (
        (config.num_nodes + 1) * 4 +  # row_ptr
        num_edges * 8 +               # col_ind + weights
        config.num_nodes * 4 * 4      # states, rates, influence, etc
    )

    arithmetic_intensity = flops_per_step / bytes_per_step
    time_per_step_s = total_wall_time / steps_done
    achieved_gflops = (flops_per_step / time_per_step_s) / 1e9

    memory_ceiling = A100_PEAK_BANDWIDTH * arithmetic_intensity / 1e9
    compute_ceiling = A100_PEAK_FLOPS / 1e9
    theoretical_max = min(memory_ceiling, compute_ceiling)
    efficiency = (achieved_gflops / theoretical_max) * 100 if theoretical_max > 0 else 0

    return BenchmarkResult(
        config_name=config.name,
        engine_type=config.engine_type,
        num_nodes=config.num_nodes,
        num_edges=num_edges,
        epsilon=0,  # Not applicable
        steps_per_launch=1,
        sparse_hazard=True,  # Markovian is always sparse
        compute_multiplier=1,
        total_wall_time_s=total_wall_time,
        steps_executed=steps_done,
        time_per_step_ms=time_per_step_s * 1000,
        steps_per_second=steps_done / total_wall_time,
        estimated_flops_per_step=flops_per_step,
        estimated_bytes_per_step=bytes_per_step,
        arithmetic_intensity=arithmetic_intensity,
        achieved_gflops=achieved_gflops,
        memory_bound_ceiling_gflops=memory_ceiling,
        compute_bound_ceiling_gflops=compute_ceiling,
        efficiency_percent=efficiency,
    )


def get_default_configs(num_nodes: int = 1_000_000) -> List[BenchmarkConfig]:
    """Return default benchmark configurations."""
    configs = []

    # Renewal engine configurations
    # Baseline: current default
    configs.append(BenchmarkConfig(
        name="renewal_baseline",
        engine_type="renewal",
        num_nodes=num_nodes,
        epsilon=0.03,
        tau_max=1.0,
        sparse_hazard=True,
        compute_multiplier=1,
    ))

    # Dense hazard computation
    configs.append(BenchmarkConfig(
        name="renewal_dense",
        engine_type="renewal",
        num_nodes=num_nodes,
        epsilon=0.03,
        tau_max=1.0,
        sparse_hazard=False,
        compute_multiplier=1,
    ))

    # CUDA Graph batched versions
    for steps in [10, 50, 100]:
        configs.append(BenchmarkConfig(
            name=f"renewal_batched_{steps}",
            engine_type="renewal_cuda_graph",
            num_nodes=num_nodes,
            epsilon=0.03,
            tau_max=1.0,
            steps_per_launch=steps,
            sparse_hazard=True,
            compute_multiplier=1,
        ))

    # Small tau (more steps, finer granularity)
    configs.append(BenchmarkConfig(
        name="renewal_small_tau",
        engine_type="renewal",
        num_nodes=num_nodes,
        epsilon=0.01,
        tau_max=0.5,
        sparse_hazard=True,
        compute_multiplier=1,
    ))

    # Large tau (fewer steps)
    configs.append(BenchmarkConfig(
        name="renewal_large_tau",
        engine_type="renewal",
        num_nodes=num_nodes,
        epsilon=0.1,
        tau_max=2.0,
        sparse_hazard=True,
        compute_multiplier=1,
    ))

    # Compute-heavy: dense + batched + multiplier
    configs.append(BenchmarkConfig(
        name="renewal_compute_heavy",
        engine_type="renewal_cuda_graph",
        num_nodes=num_nodes,
        epsilon=0.03,
        tau_max=1.0,
        steps_per_launch=50,
        sparse_hazard=False,
        compute_multiplier=2,
    ))

    # Ridge-crossing configurations (AI approaching/exceeding 9.6 FLOPs/byte)
    configs.append(BenchmarkConfig(
        name="renewal_ridge_8",
        engine_type="renewal_cuda_graph",
        num_nodes=num_nodes,
        epsilon=0.03,
        tau_max=1.0,
        steps_per_launch=50,
        sparse_hazard=False,
        compute_multiplier=8,
    ))

    configs.append(BenchmarkConfig(
        name="renewal_ridge_16",
        engine_type="renewal_cuda_graph",
        num_nodes=num_nodes,
        epsilon=0.03,
        tau_max=1.0,
        steps_per_launch=50,
        sparse_hazard=False,
        compute_multiplier=16,
    ))

    configs.append(BenchmarkConfig(
        name="renewal_compute_bound",
        engine_type="renewal_cuda_graph",
        num_nodes=num_nodes,
        epsilon=0.03,
        tau_max=1.0,
        steps_per_launch=50,
        sparse_hazard=False,
        compute_multiplier=20,
    ))

    # Markovian engine configurations
    configs.append(BenchmarkConfig(
        name="markov_baseline",
        engine_type="markovian",
        num_nodes=num_nodes,
        max_prob=0.1,
        theta=0.01,
        tau_max=1.0,
    ))

    configs.append(BenchmarkConfig(
        name="markov_aggressive",
        engine_type="markovian",
        num_nodes=num_nodes,
        max_prob=0.2,
        theta=0.05,
        tau_max=2.0,
    ))

    configs.append(BenchmarkConfig(
        name="markov_conservative",
        engine_type="markovian",
        num_nodes=num_nodes,
        max_prob=0.05,
        theta=0.005,
        tau_max=0.5,
    ))

    return configs


def run_benchmarks(
    configs: List[BenchmarkConfig],
    device: str = "cuda",
    output_dir: str = "results",
) -> List[BenchmarkResult]:
    """Run all benchmark configurations."""
    results = []

    # Group configs by num_nodes to reuse graphs
    nodes_configs = {}
    for config in configs:
        key = (config.num_nodes, config.degree)
        if key not in nodes_configs:
            nodes_configs[key] = []
        nodes_configs[key].append(config)

    for (num_nodes, degree), group_configs in nodes_configs.items():
        print(f"\n{'='*60}")
        print(f"Network: {num_nodes:,} nodes, degree {degree}")
        print(f"{'='*60}")

        graph = create_graph(num_nodes, degree, device)

        for config in group_configs:
            print(f"\nRunning: {config.name}")
            print(f"  Engine: {config.engine_type}")

            try:
                if config.engine_type == "markovian":
                    result = run_markovian_benchmark(config, graph, device)
                else:
                    result = run_renewal_benchmark(config, graph, device)

                results.append(result)

                print("  Results:")
                print(f"    Time/step: {result.time_per_step_ms:.3f} ms")
                print(f"    Steps/sec: {result.steps_per_second:.1f}")
                print(f"    Arithmetic Intensity: {result.arithmetic_intensity:.2f} FLOPs/byte")
                print(f"    Achieved: {result.achieved_gflops:.2f} GFLOPS")
                print(f"    Efficiency: {result.efficiency_percent:.1f}%")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Clear graph to free memory
        del graph
        torch.cuda.empty_cache()

    return results


def save_results(
    results: List[BenchmarkResult],
    output_dir: str,
) -> None:
    """Save benchmark results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(output_dir, "roofline_data.csv")
    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
    print(f"\nSaved CSV: {csv_path}")

    # Save JSON (for programmatic access)
    json_path = os.path.join(output_dir, "roofline_data.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Saved JSON: {json_path}")


def generate_summary(
    results: List[BenchmarkResult],
    output_dir: str,
) -> None:
    """Generate markdown summary of results."""
    if not results:
        return

    summary_path = os.path.join(output_dir, "roofline_summary.md")

    # Find best performers
    renewal_results = [r for r in results if "renewal" in r.engine_type]
    markov_results = [r for r in results if "markov" in r.engine_type]

    best_renewal = max(renewal_results, key=lambda r: r.achieved_gflops) if renewal_results else None
    best_markov = max(markov_results, key=lambda r: r.achieved_gflops) if markov_results else None
    highest_intensity = max(results, key=lambda r: r.arithmetic_intensity)

    with open(summary_path, "w") as f:
        f.write("# Roofline Benchmark Summary\n\n")

        f.write("## GPU Specifications\n\n")
        f.write(f"- **Peak FP32 Performance**: {A100_PEAK_FLOPS/1e12:.1f} TFLOPS\n")
        f.write(f"- **Memory Bandwidth**: {A100_PEAK_BANDWIDTH/1e9:.0f} GB/s\n")
        f.write(f"- **Ridge Point**: {A100_RIDGE_POINT:.1f} FLOPs/byte\n\n")

        f.write("## Best Performers\n\n")

        if best_renewal:
            f.write(f"### Best Renewal Engine: `{best_renewal.config_name}`\n")
            f.write(f"- Achieved: **{best_renewal.achieved_gflops:.2f} GFLOPS**\n")
            f.write(f"- Efficiency: {best_renewal.efficiency_percent:.1f}%\n")
            f.write(f"- Arithmetic Intensity: {best_renewal.arithmetic_intensity:.2f} FLOPs/byte\n")
            f.write(f"- Time per step: {best_renewal.time_per_step_ms:.3f} ms\n\n")

        if best_markov:
            f.write(f"### Best Markovian Engine: `{best_markov.config_name}`\n")
            f.write(f"- Achieved: **{best_markov.achieved_gflops:.2f} GFLOPS**\n")
            f.write(f"- Efficiency: {best_markov.efficiency_percent:.1f}%\n")
            f.write(f"- Arithmetic Intensity: {best_markov.arithmetic_intensity:.2f} FLOPs/byte\n")
            f.write(f"- Time per step: {best_markov.time_per_step_ms:.3f} ms\n\n")

        f.write(f"### Highest Arithmetic Intensity: `{highest_intensity.config_name}`\n")
        f.write(f"- Arithmetic Intensity: **{highest_intensity.arithmetic_intensity:.2f} FLOPs/byte**\n")
        f.write(f"- Achieved: {highest_intensity.achieved_gflops:.2f} GFLOPS\n\n")

        # Determine compute vs memory bound
        f.write("## Compute vs Memory Bound Analysis\n\n")
        f.write("| Configuration | AI (FLOPs/byte) | Bound | Achieved GFLOPS | Efficiency |\n")
        f.write("|---------------|-----------------|-------|-----------------|------------|\n")

        for r in sorted(results, key=lambda x: x.arithmetic_intensity):
            bound = "Memory" if r.arithmetic_intensity < A100_RIDGE_POINT else "Compute"
            f.write(f"| {r.config_name} | {r.arithmetic_intensity:.2f} | {bound} | "
                    f"{r.achieved_gflops:.2f} | {r.efficiency_percent:.1f}% |\n")

        f.write("\n## Recommendations\n\n")

        # Check if any config is compute-bound
        compute_bound = [r for r in results if r.arithmetic_intensity >= A100_RIDGE_POINT]
        if compute_bound:
            f.write(f"- **{len(compute_bound)} configuration(s) achieved compute-bound behavior**\n")
            for r in compute_bound:
                f.write(f"  - `{r.config_name}`: AI = {r.arithmetic_intensity:.2f}\n")
        else:
            f.write("- All configurations are **memory-bound**\n")
            f.write(f"- To achieve compute-bound (AI > {A100_RIDGE_POINT:.1f}), try:\n")
            f.write("  - Increase `compute_multiplier` (repeats hazard computation)\n")
            f.write("  - Use `sparse_hazard=False` (dense hazard for all nodes)\n")
            f.write("  - Larger `steps_per_launch` with CUDA Graphs\n")

    print(f"Saved summary: {summary_path}")


def main():
    print(
        "WARNING: benchmark_roofline.py is a historical synthetic experiment; "
        "use experiments/benchmark_acceptance.py for production measurements.",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser(description="FlashSpread Roofline Benchmark")
    parser.add_argument(
        "--num-nodes", type=int, default=1_000_000,
        help="Number of nodes in the network (default: 1,000,000)"
    )
    parser.add_argument(
        "--degree", type=int, default=8,
        help="Mean degree of the (symmetric) benchmark graph (default: 8, "
             "matching the manuscript's d=8 setup)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=100,
        help="Warmup steps before timing (default: 100)"
    )
    parser.add_argument(
        "--benchmark-steps", type=int, default=500,
        help="Steps to time (default: 500)"
    )

    args = parser.parse_args()

    print("FlashSpread Roofline Benchmark")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Network size: {args.num_nodes:,} nodes")
    print(f"Output directory: {args.output_dir}")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available")
            sys.exit(1)
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Get configs with updated parameters
    configs = get_default_configs(args.num_nodes)
    for config in configs:
        config.degree = args.degree
        config.warmup_steps = args.warmup_steps
        config.benchmark_steps = args.benchmark_steps

    print(f"\nRunning {len(configs)} benchmark configurations...")

    results = run_benchmarks(configs, args.device, args.output_dir)

    print("\n" + "=" * 60)
    print("Saving results...")
    save_results(results, args.output_dir)
    generate_summary(results, args.output_dir)

    print("\nBenchmark complete!")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
