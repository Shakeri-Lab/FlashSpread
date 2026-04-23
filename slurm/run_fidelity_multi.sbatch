#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J fs_fidm
#SBATCH -o logs/fidm-%j.out
#SBATCH -e logs/fidm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Multi-topology/size fidelity sweep"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=============================================="

python experiments/fidelity_multi_graph.py \
    --graphs er,ba,fixed \
    --sizes 1000,10000,100000 \
    --epsilons 0.005,0.03,0.1 \
    --trials 20 \
    --output-npz results/fidelity_multi.npz \
    --output-csv results/fidelity_multi_summary.csv

python experiments/plot_fidelity_multi.py \
    --npz results/fidelity_multi.npz \
    --out docs/jocs/figures/fig_fidelity_multi.png

echo "Done. See docs/jocs/figures/fig_fidelity_multi.png"
