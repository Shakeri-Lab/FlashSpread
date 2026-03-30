#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J fs_scoglio_f
#SBATCH -o logs/scoglio_focused-%j.out
#SBATCH -e logs/scoglio_focused-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Scoglio Focused Validation: NonMarkov vs Fused"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

python experiments/validate_scoglio_focused.py \
    --num-nodes 50000 \
    --num-seeds 20 \
    --initial-exposed 100 \
    --target-time 30.0 \
    --output results/validation_scoglio_focused.png

echo "Finished: $(date)"
