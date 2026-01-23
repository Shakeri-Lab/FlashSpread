#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J flashspread_ablation
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-9

set -euo pipefail

# Configuration
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

mkdir -p logs results/ablation

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "=============================================="
echo "FlashSpread Ablation Study - Task ${SLURM_ARRAY_TASK_ID}"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo ""

# Configuration names for reference
CONFIG_NAMES=(
    "baseline"
    "rcm_only"
    "fused_only"
    "block_256"
    "cuda_graph_only"
    "rcm_fused"
    "rcm_cuda_graph"
    "all_optimizations"
    "cuda_graph_100"
    "all_opt_100"
)

echo "Config: ${CONFIG_NAMES[$SLURM_ARRAY_TASK_ID]}"
echo ""

# Run ablation experiment
# Use 100K nodes for quick testing, 1M for final results
python experiments/ablation_study.py \
    --config "${SLURM_ARRAY_TASK_ID}" \
    --num-nodes 100000 \
    --output-dir results/ablation \
    --device cuda

echo ""
echo "Task ${SLURM_ARRAY_TASK_ID} complete: $(date)"
