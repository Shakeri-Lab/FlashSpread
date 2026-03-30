#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J fs_scoglio_sc
#SBATCH -o logs/scoglio_scenarios-%j.out
#SBATCH -e logs/scoglio_scenarios-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Scoglio Scenarios: Py vs NonMarkov vs Fused"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

python experiments/validate_scoglio_scenarios.py \
    --num-nodes 1000 \
    --degree 8 \
    --num-runs 100 \
    --target-time 50.0 \
    --output results/validation_scoglio_scenarios.png

echo "Finished: $(date)"
