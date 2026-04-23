#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J flashspread_roofline
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Configuration
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

# Create directories
mkdir -p logs results

# Activate environment if needed
# source /path/to/venv/bin/activate

# Ensure package is importable
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "=============================================="
echo "FlashSpread Roofline Benchmark"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Started: $(date)"
echo ""

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Python info
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Run benchmark
echo "Running benchmark..."
echo "=============================================="

python experiments/benchmark_roofline.py \
    --num-nodes 1000000 \
    --output-dir results \
    --warmup-steps 100 \
    --benchmark-steps 500

echo ""
echo "=============================================="
echo "Generating plots..."

python experiments/roofline_utils.py results

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "Results saved to: results/"
echo "Finished: $(date)"
