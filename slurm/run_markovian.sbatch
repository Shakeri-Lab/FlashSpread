#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J flashspread_markov
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Configuration
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

# Create logs directory
mkdir -p logs

# Activate environment if needed
# source /path/to/venv/bin/activate

# Ensure package is importable
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Run example
echo "Running FlashSpread Markovian SIS Example"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

python examples/markovian_sis.py

echo ""
echo "Done."
