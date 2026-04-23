#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -J fs_merge_v1
#SBATCH -o logs/merge_v1-%j.out
#SBATCH -e logs/merge_v1-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Merge-based Kernel - Step 2 Verification"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="
echo ""

echo "=== [A] Parity test (thread / warp / merge) on small graphs ==="
python experiments/verify_warp_collab.py \
    --num-nodes 10000 --num-steps 50 --seed 4242 || {
    echo "!!! Parity check FAILED — continuing to throughput phase anyway"
}
echo ""

echo "=== [B] Smoke tests ==="
python -m pytest \
    tests/smoke_test.py::TestRenewalEngineFused \
    tests/smoke_test.py::TestRenewalEngineNonMarkov \
    -v -s || echo "!!! Smoke tests FAILED"
echo ""

echo "=== [C] Throughput benchmark on ER + BA ==="
python experiments/bench_warp_collab_ba.py \
    --num-nodes 1000000 \
    --trials 5 \
    --graphs er,ba \
    --sweep 4,8 \
    --output results/merge_throughput.csv
echo ""

echo "Done. See results/merge_throughput.csv"
