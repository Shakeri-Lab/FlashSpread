#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -J flashspread_tests
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Configuration
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

# Create logs directory
mkdir -p logs

# Ensure package is importable
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Run smoke tests
echo "Running FlashSpread Smoke Tests"
echo "================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

python -m pytest tests/smoke_test.py -v

echo ""
echo "All tests passed."
