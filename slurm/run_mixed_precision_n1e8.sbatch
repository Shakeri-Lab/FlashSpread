#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -J fs_mix1e8
#SBATCH -o logs/mix1e8-%j.out
#SBATCH -e logs/mix1e8-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Mixed-precision throughput at N = 1e8 (zoom-panel endpoint)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="

# N=1e8 fits in 40 GB: ~9.6 GB baseline, ~6.6 GB mixed.
# Use fewer trials (3 + 1 warmup) because each trial is ~30-60 s
# post-L2-cliff and the dominant noise source is single-run jitter,
# not replicate variance.
python experiments/bench_mixed_precision_scaling.py \
    --trials 3 \
    --min-n 100000000 --max-n 100000000 \
    --output results/mixed_precision_scaling_n1e8.csv

echo "Done."
ls -la results/mixed_precision_scaling_n1e8.csv
