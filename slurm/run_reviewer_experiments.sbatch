#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -J fs_reviewer
#SBATCH -o logs/reviewer_exp-%j.out
#SBATCH -e logs/reviewer_exp-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Reviewer Experiments"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Triton: $(python -c 'import triton; print(triton.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "Python: $(python --version)"
echo "=============================================="
echo ""

# --- Experiment 1: CPU tau-leaping baseline ---
echo "=== Experiment 1: CPU Tau-Leaping Baseline ==="
python experiments/bench_cpu_tauleaping.py \
    --sizes "1000,10000,100000,1000000" \
    --trials 3 \
    --tf 50.0 \
    --output results/cpu_tauleaping_nups.csv

echo ""
echo "=== Experiment 2: Epsilon Convergence Sweep ==="
python experiments/validate_epsilon_sweep.py \
    --device cuda \
    --output results/epsilon_sweep.csv

echo ""
echo "=== Experiment 3: BA Scale-Free Throughput ==="
python experiments/bench_throughput_ba.py \
    --num-nodes 1000000 \
    --m 4 \
    --trials 5 \
    --output results/ba_throughput.csv \
    --device cuda

echo ""
echo "=============================================="
echo "All reviewer experiments complete!"
echo "Finished: $(date)"
