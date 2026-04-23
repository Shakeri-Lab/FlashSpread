#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH -J fs_age_dep
#SBATCH -o logs/age_dep-%j.out
#SBATCH -e logs/age_dep-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Age-Dependent Infectivity Throughput Benchmark"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Triton: $(python -c 'import triton; print(triton.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "Python:  $(python --version)"
echo "=============================================="
echo ""

# --- Phase A: verify Triton erfcx accuracy (Fix 4) ---
# Run with '|| true' so a test regression does not abort Phase B; the
# recorded exit code and captured test output are still visible in the log.
echo "=== [A] Triton erfcx accuracy test ==="
python -m pytest tests/smoke_test.py::TestTritonErfcxAccuracy -v || {
    echo "!!! erfcx accuracy test FAILED (exit $?) — continuing to Phase B"
}
echo ""

# --- Phase B: age-dependent vs constant throughput (Fix 1) ---
echo "=== [B] Age-dependent benchmark ==="
python experiments/bench_throughput_age_dependent.py \
    --num-nodes 1000000 \
    --trials 5 \
    --graphs er,ba \
    --output results/age_dependent_throughput.csv
echo ""

echo "Done. See results/age_dependent_throughput.csv"
