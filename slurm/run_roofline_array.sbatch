#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J flashspread_roofline
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-12

set -euo pipefail

# Configuration
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

mkdir -p logs results

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Configuration array (13 configurations, index 0-12)
# Format: name:engine_type:epsilon:tau_max:steps_per_launch:sparse_hazard:compute_mult
CONFIGS=(
    "renewal_baseline:renewal:0.03:1.0:1:true:1"
    "renewal_dense:renewal:0.03:1.0:1:false:1"
    "renewal_batched_10:renewal_cuda_graph:0.03:1.0:10:true:1"
    "renewal_batched_50:renewal_cuda_graph:0.03:1.0:50:true:1"
    "renewal_batched_100:renewal_cuda_graph:0.03:1.0:100:true:1"
    "renewal_small_tau:renewal:0.01:0.5:1:true:1"
    "renewal_large_tau:renewal:0.1:2.0:1:true:1"
    "renewal_compute_heavy:renewal_cuda_graph:0.03:1.0:50:false:2"
    "markov_baseline:markovian:0.1:1.0:1:true:1"
    "markov_aggressive:markovian:0.2:2.0:1:true:1"
    "renewal_ridge_8:renewal_cuda_graph:0.03:1.0:50:false:8"
    "renewal_ridge_16:renewal_cuda_graph:0.03:1.0:50:false:16"
    "renewal_compute_bound:renewal_cuda_graph:0.03:1.0:50:false:20"
)

# Parse config for this array task
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
IFS=':' read -r NAME ENGINE_TYPE EPSILON TAU_MAX STEPS_PER_LAUNCH SPARSE_HAZARD COMPUTE_MULT <<< "$CONFIG"

echo "=============================================="
echo "FlashSpread Roofline Benchmark - Array Task"
echo "=============================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Config: ${NAME}"
echo "Engine: ${ENGINE_TYPE}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

# Run single configuration benchmark
python -c "
import sys
sys.path.insert(0, '.')

import torch
import json
import time
from dataclasses import asdict

from flashspread import FixedDegreeGraph, SEIRModel, SISModel
from flashspread.engines import RenewalEngine, MarkovianEngine
from flashspread.engines.renewal import RenewalEngineCUDAGraph
from flashspread.engines.renewal_tunable import (
    RenewalEngineTunable,
    RenewalEngineTunableCUDAGraph,
    estimate_flops_per_step,
    estimate_memory_bytes_per_step,
)

# A100 specs
A100_PEAK_FLOPS = 19.5e12
A100_PEAK_BANDWIDTH = 2039e9

# Config from environment
name = '${NAME}'
engine_type = '${ENGINE_TYPE}'
epsilon = float('${EPSILON}')
tau_max = float('${TAU_MAX}')
steps_per_launch = int('${STEPS_PER_LAUNCH}')
sparse_hazard = '${SPARSE_HAZARD}'.lower() == 'true'
compute_mult = int('${COMPUTE_MULT}')

num_nodes = 1_000_000
degree = 15
warmup_steps = 100
benchmark_steps = 500
device = 'cuda'

print(f'Creating graph: {num_nodes:,} nodes, degree {degree}')
graph = FixedDegreeGraph(num_nodes, degree, device=device)
num_edges = graph.num_edges
print(f'Edges: {num_edges:,}')

if engine_type == 'markovian':
    model = SISModel(beta=0.5, delta=1.0)
    engine = MarkovianEngine(graph, model, device=device, max_prob=epsilon, tau_max=tau_max)
    engine.seed_infection(num_nodes // 100)
else:
    model = SEIRModel(beta=0.3, mean_ei=5.0, median_ei=4.0, mean_ir=3.9, median_ir=1.5)
    model.sparse_hazard = sparse_hazard

    if engine_type == 'renewal_cuda_graph':
        engine = RenewalEngineTunableCUDAGraph(
            graph, model, device=device, epsilon=epsilon, tau_max=tau_max,
            steps_per_launch=steps_per_launch, compute_multiplier=compute_mult,
        )
    else:
        engine = RenewalEngineTunable(
            graph, model, device=device, epsilon=epsilon, tau_max=tau_max,
            compute_multiplier=compute_mult,
        )
    engine.seed_infection(num_nodes // 100, state=model.exposed)

# Warmup
print(f'Warmup: {warmup_steps} steps')
for _ in range(warmup_steps):
    engine.step()
torch.cuda.synchronize()

# Benchmark
print(f'Benchmark: {benchmark_steps} steps')
torch.cuda.synchronize()
start = time.perf_counter()

steps_done = 0
for _ in range(benchmark_steps):
    engine.step()
    steps_done += steps_per_launch if engine_type == 'renewal_cuda_graph' else 1

torch.cuda.synchronize()
total_time = time.perf_counter() - start

# Compute metrics
if engine_type == 'markovian':
    flops_per_step = num_edges * 3 + num_nodes * 15
    bytes_per_step = (num_nodes + 1) * 4 + num_edges * 8 + num_nodes * 16
else:
    flops_est = estimate_flops_per_step(num_nodes, num_edges, compute_mult, not sparse_hazard)
    bytes_est = estimate_memory_bytes_per_step(num_nodes, num_edges, False)
    flops_per_step = flops_est['total']
    bytes_per_step = bytes_est['total']

ai = flops_per_step / bytes_per_step
time_per_step = total_time / steps_done
achieved_gflops = (flops_per_step / time_per_step) / 1e9
mem_ceiling = A100_PEAK_BANDWIDTH * ai / 1e9
compute_ceiling = A100_PEAK_FLOPS / 1e9
efficiency = (achieved_gflops / min(mem_ceiling, compute_ceiling)) * 100

result = {
    'config_name': name,
    'engine_type': engine_type,
    'num_nodes': num_nodes,
    'num_edges': num_edges,
    'epsilon': epsilon,
    'steps_per_launch': steps_per_launch,
    'sparse_hazard': sparse_hazard,
    'compute_multiplier': compute_mult,
    'total_wall_time_s': total_time,
    'steps_executed': steps_done,
    'time_per_step_ms': time_per_step * 1000,
    'steps_per_second': steps_done / total_time,
    'estimated_flops_per_step': flops_per_step,
    'estimated_bytes_per_step': bytes_per_step,
    'arithmetic_intensity': ai,
    'achieved_gflops': achieved_gflops,
    'memory_bound_ceiling_gflops': mem_ceiling,
    'compute_bound_ceiling_gflops': compute_ceiling,
    'efficiency_percent': efficiency,
}

print()
print('Results:')
print(f'  Time/step: {result[\"time_per_step_ms\"]:.3f} ms')
print(f'  Steps/sec: {result[\"steps_per_second\"]:.1f}')
print(f'  Arithmetic Intensity: {ai:.2f} FLOPs/byte')
print(f'  Achieved: {achieved_gflops:.2f} GFLOPS')
print(f'  Efficiency: {efficiency:.1f}%')

# Save individual result
with open(f'results/roofline_{name}.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f'Saved: results/roofline_{name}.json')
"

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} complete: $(date)"
