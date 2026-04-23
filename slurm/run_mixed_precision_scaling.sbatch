#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 00:45:00
#SBATCH -J fs_mixsc
#SBATCH -o logs/mixsc-%j.out
#SBATCH -e logs/mixsc-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Mixed-precision throughput scaling (for Fig 6)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="

python experiments/bench_mixed_precision_scaling.py \
    --trials 10 \
    --max-n 10000000 \
    --output results/mixed_precision_scaling.csv

echo "Done."
ls -la results/mixed_precision_scaling.csv
