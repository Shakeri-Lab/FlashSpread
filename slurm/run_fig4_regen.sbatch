#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J fs_fig4
#SBATCH -o logs/fig4-%j.out
#SBATCH -e logs/fig4-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=== Fig 4 regen (short title) ==="
nvidia-smi --query-gpu=name --format=csv,noheader

python experiments/plot_scoglio_fused_vs_matlab.py \
    --output docs/jocs/figures/validation_scoglio_exact.png

echo "Done."
ls -la docs/jocs/figures/validation_scoglio_exact.png
