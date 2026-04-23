#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -J fs_fidelity
#SBATCH -o logs/fidelity-%j.out
#SBATCH -e logs/fidelity-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Tau-leaping fidelity sweep"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=============================================="

# Phase A: data
python experiments/fidelity_sweep.py \
    --num-runs 100 \
    --output-csv results/fidelity_summary.csv \
    --output-npz results/fidelity_trajectories.npz

# Phase B: plots (login-capable matplotlib)
python experiments/plot_fidelity.py \
    --summary results/fidelity_summary.csv \
    --trajectories results/fidelity_trajectories.npz \
    --outdir docs/jocs/figures

echo "Done. See docs/jocs/figures/fig_fidelity_*.png"
