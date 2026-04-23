#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:30:00
#SBATCH -J fs_wc_v1
#SBATCH -o logs/wc_v1-%j.out
#SBATCH -e logs/wc_v1-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Warp-Collaborative Kernel - Step 1 Verification"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="
echo ""

# --- Phase A: correctness (original vs warp-collab) on small N, ER + BA ---
echo "=== [A] Correctness parity test ==="
python experiments/verify_warp_collab.py \
    --num-nodes 10000 --num-steps 50 --seed 4242 || {
    echo "!!! Correctness check FAILED (exit $?) — aborting Phase B"
    exit 1
}
echo ""

# --- Phase B: smoke tests pass with the new kernel plumbing ---
echo "=== [B] Smoke tests (fused engine still passes) ==="
python -m pytest \
    tests/smoke_test.py::TestRenewalEngineFused \
    tests/smoke_test.py::TestRenewalEngineNonMarkov \
    -v -s || echo "!!! Smoke tests FAILED — continuing to Phase C"
echo ""

# --- Phase C: throughput comparison on ER + BA at N=10^6 ---
echo "=== [C] Throughput benchmark (warp-collab vs baseline) ==="
python experiments/bench_warp_collab_ba.py \
    --num-nodes 1000000 \
    --trials 5 \
    --graphs er,ba \
    --sweep 2,4,8 \
    --output results/warp_collab_throughput.csv
echo ""

echo "Done. See results/warp_collab_throughput.csv"
