#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J fs_nm_ablation
#SBATCH -o logs/nm_ablation-%A-%a.out
#SBATCH -e logs/nm_ablation-%A-%a.err
#SBATCH --array=0-11

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"

mkdir -p logs results/ablation_nonmarkov

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

CONFIG_ID=${SLURM_ARRAY_TASK_ID}

echo "=============================================="
echo "FlashSpread Non-Markovian Ablation"
echo "=============================================="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID:-$CONFIG_ID}"
echo "Node:      ${SLURM_NODELIST:-$(hostname)}"
echo "Started:   $(date)"
echo ""

# GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Run smoke tests first (only on config 0 to avoid redundancy)
if [ "$CONFIG_ID" -eq 0 ]; then
    echo "Running smoke tests..."
    python -m pytest tests/smoke_test.py -v --tb=short 2>&1 | tail -30 || true
    echo ""
fi

# Run ablation
echo "Running config ${CONFIG_ID}..."
echo "=============================================="

python experiments/ablation_nonmarkov.py \
    --config "${CONFIG_ID}" \
    --num-nodes 100000 \
    --output-dir results/ablation_nonmarkov

echo ""
echo "=============================================="
echo "Config ${CONFIG_ID} complete!"
echo "Finished: $(date)"
