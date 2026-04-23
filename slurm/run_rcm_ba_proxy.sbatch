#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -J fs_rcmba
#SBATCH -o logs/rcmba-%j.out
#SBATCH -e logs/rcmba-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "RCM-on-BA proxy (Step 1 Rabbit/Gorder go/no-go)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=============================================="

python experiments/bench_rcm_ba_proxy.py \
    --num-nodes 1000000 \
    --trials 5 \
    --output results/rcm_ba_proxy_summary.csv

echo "Done."
