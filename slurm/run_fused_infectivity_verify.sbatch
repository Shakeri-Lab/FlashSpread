#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -J fs_fused_v2
#SBATCH -o logs/fused_v2-%j.out
#SBATCH -e logs/fused_v2-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Fused-Infectivity Optimization Verification"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="
echo ""

# --- Phase A: regression smoke tests (fused engine conservation, erfcx) ---
echo "=== [A] Smoke tests (fused engine + erfcx) ==="
python -m pytest \
    tests/smoke_test.py::TestRenewalEngineFused \
    tests/smoke_test.py::TestRenewalEngineNonMarkovCUDAGraph \
    tests/smoke_test.py::TestTritonErfcxAccuracy \
    -v -s || {
    echo "!!! Regression tests FAILED (exit $?) — continuing to Phase B"
}
echo ""

# --- Phase B: age-dependent vs constant throughput post-optimization ---
echo "=== [B] Age-dependent benchmark (post-optimization) ==="
python experiments/bench_throughput_age_dependent.py \
    --num-nodes 1000000 \
    --trials 5 \
    --graphs er,ba \
    --output results/age_dependent_throughput_v2.csv
echo ""

echo "Done. See results/age_dependent_throughput_v2.csv"
