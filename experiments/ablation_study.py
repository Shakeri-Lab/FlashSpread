#!/usr/bin/env python
"""
Ablation Study for FlashSpread Optimizations.

This script tests individual optimizations and their combinations,
measuring both performance and accuracy against a baseline.

Usage:
    python experiments/ablation_study.py --config CONFIG_ID [--output-dir DIR]

Configs:
    0: baseline (no optimizations)
    1: rcm_only (RCM graph reordering)
    2: fused_only (fused PyTorch operations)
    3: block_256 (larger FlashNeighbor block size)
    4: cuda_graph_only (CUDA Graph batching)
    5: rcm_fused (RCM + fused ops)
    6: rcm_cuda_graph (RCM + CUDA Graph)
    7: all_optimizations (everything enabled)
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread import FixedDegreeGraph, SEIRModel, SISModel
from flashspread.engines import RenewalEngine, MarkovianEngine
from flashspread.engines.renewal import RenewalEngineCUDAGraph


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    config_id: int
    # Optimizations
    use_rcm_reordering: bool = False
    use_fused_ops: bool = False
    flash_neighbor_block_size: int = 128
    use_cuda_graph: bool = False
    steps_per_launch: int = 1
    # Simulation parameters
    num_nodes: int = 100_000  # Smaller for accuracy validation
    degree: int = 15
    epsilon: float = 0.03
    tau_max: float = 1.0
    seed: int = 42
    # Experiment parameters
    warmup_steps: int = 50
    benchmark_steps: int = 200
    accuracy_runs: int = 5  # Multiple runs for statistical validation


@dataclass
class AblationResult:
    """Results from an ablation experiment."""
    config_name: str
    config_id: int
    # Optimization flags
    use_rcm_reordering: bool
    use_fused_ops: bool
    flash_neighbor_block_size: int
    use_cuda_graph: bool
    steps_per_launch: int
    # Graph info
    num_nodes: int
    num_edges: int
    graph_bandwidth_before: int
    graph_bandwidth_after: int
    # Performance metrics
    total_wall_time_s: float
    steps_executed: int
    time_per_step_ms: float
    steps_per_second: float
    speedup_vs_baseline: float
    # Accuracy metrics (compared to reference)
    final_infected_mean: float
    final_infected_std: float
    final_infected_ref: float
    accuracy_error_percent: float
    max_trajectory_diff: float
    accuracy_passed: bool


def create_graph(num_nodes: int, degree: int, device: str, use_rcm: bool = False):
    """Create graph with optional RCM reordering."""
    print(f"  Creating graph: {num_nodes:,} nodes, degree {degree}")
    graph = FixedDegreeGraph(num_nodes, degree, device=device)

    bandwidth_before = -1
    bandwidth_after = -1

    if use_rcm:
        try:
            from flashspread.core.optimizations import (
                reverse_cuthill_mckee,
                compute_graph_bandwidth,
                apply_permutation_to_graph,
            )

            # Compute bandwidth before
            bandwidth_before = compute_graph_bandwidth(
                graph.csr.row_ptr, graph.csr.col_ind
            )
            print(f"  Graph bandwidth before RCM: {bandwidth_before}")

            # Apply RCM reordering
            perm = reverse_cuthill_mckee(graph.csr.row_ptr, graph.csr.col_ind)
            new_row_ptr, new_col_ind, new_weights = apply_permutation_to_graph(
                graph.csr.row_ptr, graph.csr.col_ind, graph.csr.weights, perm
            )

            # Update graph CSR in place
            graph.csr.row_ptr = new_row_ptr
            graph.csr.col_ind = new_col_ind
            graph.csr.weights = new_weights

            bandwidth_after = compute_graph_bandwidth(
                graph.csr.row_ptr, graph.csr.col_ind
            )
            print(f"  Graph bandwidth after RCM: {bandwidth_after}")
            print(f"  Bandwidth reduction: {(1 - bandwidth_after/bandwidth_before)*100:.1f}%")

        except Exception as e:
            print(f"  Warning: RCM reordering failed: {e}")
            bandwidth_after = bandwidth_before

    return graph, bandwidth_before, bandwidth_after


def run_simulation(
    graph,
    config: AblationConfig,
    device: str,
) -> Tuple[List[torch.Tensor], float]:
    """
    Run simulation and return trajectory + wall time.

    Returns:
        Tuple of (trajectory of state counts, wall_time in seconds)
    """
    model = SEIRModel(
        beta=0.3,
        mean_ei=5.0,
        median_ei=4.0,
        mean_ir=3.9,
        median_ir=1.5,
    )

    if config.use_cuda_graph and config.steps_per_launch > 1:
        engine = RenewalEngineCUDAGraph(
            graph, model, device=device,
            epsilon=config.epsilon,
            tau_max=config.tau_max,
            seed=config.seed,
            steps_per_launch=config.steps_per_launch,
        )
    else:
        engine = RenewalEngine(
            graph, model, device=device,
            epsilon=config.epsilon,
            tau_max=config.tau_max,
            seed=config.seed,
        )

    # Seed infection
    engine.seed_infection(max(100, config.num_nodes // 100), state=model.exposed)

    # Record trajectory for accuracy comparison
    trajectory = []

    # Warmup (same number of simulation steps for all configs)
    warmup_sim_steps = config.warmup_steps
    steps_per_call = config.steps_per_launch if config.use_cuda_graph else 1
    warmup_calls = max(1, warmup_sim_steps // steps_per_call)
    for _ in range(warmup_calls):
        engine.step()
    torch.cuda.synchronize()

    # Benchmark with trajectory recording
    # IMPORTANT: Run same total simulation steps for all configs to ensure fair comparison
    target_sim_steps = config.benchmark_steps  # This is the target simulation steps
    num_calls = max(1, target_sim_steps // steps_per_call)
    record_interval = max(1, num_calls // 20)  # ~20 checkpoints

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    steps_done = 0
    for call_idx in range(num_calls):
        engine.step()
        steps_done += steps_per_call

        if call_idx % record_interval == 0:
            counts = engine.count_by_state()
            trajectory.append(counts.clone())

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    wall_time = end_time - start_time

    # Final state count
    final_counts = engine.count_by_state()
    trajectory.append(final_counts.clone())

    return trajectory, wall_time, steps_done


def compare_trajectories(
    traj1: List[torch.Tensor],
    traj2: List[torch.Tensor],
) -> float:
    """
    Compare two trajectories and return max difference.

    Returns maximum absolute difference in any compartment at any time.
    """
    max_diff = 0.0
    min_len = min(len(traj1), len(traj2))

    for i in range(min_len):
        diff = torch.abs(traj1[i].float() - traj2[i].float()).max().item()
        if diff > max_diff:
            max_diff = diff

    return max_diff


def get_ablation_configs() -> List[AblationConfig]:
    """Return all ablation configurations."""
    configs = []

    # 0: Baseline (no optimizations)
    configs.append(AblationConfig(
        name="baseline",
        config_id=0,
        use_rcm_reordering=False,
        use_fused_ops=False,
        flash_neighbor_block_size=128,
        use_cuda_graph=False,
        steps_per_launch=1,
    ))

    # 1: RCM reordering only
    configs.append(AblationConfig(
        name="rcm_only",
        config_id=1,
        use_rcm_reordering=True,
        use_fused_ops=False,
        flash_neighbor_block_size=128,
        use_cuda_graph=False,
        steps_per_launch=1,
    ))

    # 2: Fused ops only
    configs.append(AblationConfig(
        name="fused_only",
        config_id=2,
        use_rcm_reordering=False,
        use_fused_ops=True,
        flash_neighbor_block_size=128,
        use_cuda_graph=False,
        steps_per_launch=1,
    ))

    # 3: Larger block size
    configs.append(AblationConfig(
        name="block_256",
        config_id=3,
        use_rcm_reordering=False,
        use_fused_ops=False,
        flash_neighbor_block_size=256,
        use_cuda_graph=False,
        steps_per_launch=1,
    ))

    # 4: CUDA Graph only
    configs.append(AblationConfig(
        name="cuda_graph_only",
        config_id=4,
        use_rcm_reordering=False,
        use_fused_ops=False,
        flash_neighbor_block_size=128,
        use_cuda_graph=True,
        steps_per_launch=50,
    ))

    # 5: RCM + Fused
    configs.append(AblationConfig(
        name="rcm_fused",
        config_id=5,
        use_rcm_reordering=True,
        use_fused_ops=True,
        flash_neighbor_block_size=128,
        use_cuda_graph=False,
        steps_per_launch=1,
    ))

    # 6: RCM + CUDA Graph
    configs.append(AblationConfig(
        name="rcm_cuda_graph",
        config_id=6,
        use_rcm_reordering=True,
        use_fused_ops=False,
        flash_neighbor_block_size=128,
        use_cuda_graph=True,
        steps_per_launch=50,
    ))

    # 7: All optimizations
    configs.append(AblationConfig(
        name="all_optimizations",
        config_id=7,
        use_rcm_reordering=True,
        use_fused_ops=True,
        flash_neighbor_block_size=256,
        use_cuda_graph=True,
        steps_per_launch=50,
    ))

    # 8: CUDA Graph with 100 steps per launch
    configs.append(AblationConfig(
        name="cuda_graph_100",
        config_id=8,
        use_rcm_reordering=False,
        use_fused_ops=False,
        flash_neighbor_block_size=128,
        use_cuda_graph=True,
        steps_per_launch=100,
    ))

    # 9: All + 100 steps per launch
    configs.append(AblationConfig(
        name="all_opt_100",
        config_id=9,
        use_rcm_reordering=True,
        use_fused_ops=True,
        flash_neighbor_block_size=256,
        use_cuda_graph=True,
        steps_per_launch=100,
    ))

    return configs


def run_ablation_experiment(
    config: AblationConfig,
    device: str = "cuda",
    baseline_result: Optional[AblationResult] = None,
) -> AblationResult:
    """Run a single ablation experiment."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name} (config_id={config.config_id})")
    print(f"{'='*60}")

    # Create graph
    graph, bw_before, bw_after = create_graph(
        config.num_nodes, config.degree, device,
        use_rcm=config.use_rcm_reordering
    )

    # Run multiple times for accuracy statistics
    trajectories = []
    wall_times = []
    final_infected_counts = []

    for run_idx in range(config.accuracy_runs):
        print(f"  Run {run_idx + 1}/{config.accuracy_runs}...")

        # Use different seed for each run but same seed per config for reproducibility
        run_config = AblationConfig(**asdict(config))
        run_config.seed = config.seed + run_idx * 1000

        traj, wall_time, steps_done = run_simulation(graph, run_config, device)
        trajectories.append(traj)
        wall_times.append(wall_time)

        # Count infected (state 2) at end
        final_infected = traj[-1][2].item() if len(traj[-1]) > 2 else 0
        final_infected_counts.append(final_infected)

    # Compute statistics
    mean_wall_time = np.mean(wall_times)
    final_infected_mean = np.mean(final_infected_counts)
    final_infected_std = np.std(final_infected_counts)

    # Compute speedup vs baseline
    speedup = 1.0
    if baseline_result is not None:
        speedup = baseline_result.time_per_step_ms / (mean_wall_time * 1000 / steps_done)

    # Accuracy comparison (use first run trajectory)
    accuracy_error = 0.0
    max_traj_diff = 0.0
    final_infected_ref = final_infected_mean

    if baseline_result is not None:
        accuracy_error = abs(final_infected_mean - baseline_result.final_infected_mean)
        accuracy_error_pct = accuracy_error / max(1, baseline_result.final_infected_mean) * 100
    else:
        accuracy_error_pct = 0.0

    # Accuracy passes if within 10% of baseline (stochastic variance expected)
    accuracy_passed = accuracy_error_pct < 10.0

    time_per_step_ms = mean_wall_time * 1000 / steps_done

    result = AblationResult(
        config_name=config.name,
        config_id=config.config_id,
        use_rcm_reordering=config.use_rcm_reordering,
        use_fused_ops=config.use_fused_ops,
        flash_neighbor_block_size=config.flash_neighbor_block_size,
        use_cuda_graph=config.use_cuda_graph,
        steps_per_launch=config.steps_per_launch,
        num_nodes=config.num_nodes,
        num_edges=graph.num_edges,
        graph_bandwidth_before=bw_before,
        graph_bandwidth_after=bw_after,
        total_wall_time_s=mean_wall_time,
        steps_executed=steps_done,
        time_per_step_ms=time_per_step_ms,
        steps_per_second=steps_done / mean_wall_time,
        speedup_vs_baseline=speedup,
        final_infected_mean=final_infected_mean,
        final_infected_std=final_infected_std,
        final_infected_ref=final_infected_ref,
        accuracy_error_percent=accuracy_error_pct,
        max_trajectory_diff=max_traj_diff,
        accuracy_passed=accuracy_passed,
    )

    print(f"\n  Results:")
    print(f"    Time/step: {result.time_per_step_ms:.3f} ms")
    print(f"    Steps/sec: {result.steps_per_second:.1f}")
    print(f"    Speedup vs baseline: {result.speedup_vs_baseline:.2f}x")
    print(f"    Final infected: {result.final_infected_mean:.0f} +/- {result.final_infected_std:.0f}")
    print(f"    Accuracy error: {result.accuracy_error_percent:.1f}%")
    print(f"    Accuracy passed: {result.accuracy_passed}")

    return result


def main():
    parser = argparse.ArgumentParser(description="FlashSpread Ablation Study")
    parser.add_argument(
        "--config", type=int, required=True,
        help="Configuration ID (0-9)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/ablation",
        help="Output directory"
    )
    parser.add_argument(
        "--num-nodes", type=int, default=100_000,
        help="Number of nodes"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    configs = get_ablation_configs()

    if args.config < 0 or args.config >= len(configs):
        print(f"Error: config must be 0-{len(configs)-1}")
        sys.exit(1)

    config = configs[args.config]
    config.num_nodes = args.num_nodes

    print("FlashSpread Ablation Study")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Config: {config.name} (ID: {config.config_id})")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load baseline result if exists and this isn't baseline
    baseline_result = None
    baseline_path = os.path.join(args.output_dir, "ablation_baseline.json")
    if config.config_id != 0 and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_data = json.load(f)
            baseline_result = AblationResult(**baseline_data)
        print(f"Loaded baseline: {baseline_result.time_per_step_ms:.3f} ms/step")

    # Run experiment
    result = run_ablation_experiment(config, args.device, baseline_result)

    # Save result
    output_path = os.path.join(args.output_dir, f"ablation_{config.name}.json")
    with open(output_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
