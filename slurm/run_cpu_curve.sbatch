#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J fs_cpu_curve
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "${ROOT}"; mkdir -p logs results/speed_check
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY="${FLASHSPREAD_PY:-/home/hs9hd/.conda/envs/hpo_edu/bin/python}"
"${PY}" experiments/bench_cpu_tauleaping.py --sizes 1000,10000,100000,1000000 --trials 2 --tf 50.0 \
    --output results/speed_check/cpu_tauleaping_full.csv
cat results/speed_check/cpu_tauleaping_full.csv
