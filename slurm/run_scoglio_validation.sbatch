#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J fs_scoglio
#SBATCH -o logs/scoglio-%j.out
#SBATCH -e logs/scoglio-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

mkdir -p logs results

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "=============================================="
echo "Scoglio Validation"
echo "=============================================="
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURM_NODELIST:-$(hostname)}"
echo "Started: $(date)"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

python experiments/validate_scoglio.py \
    --num-nodes 50000 \
    --num-seeds 20 \
    --target-time 20.0 \
    --output results/validation_scoglio.png

echo ""
echo "Finished: $(date)"
