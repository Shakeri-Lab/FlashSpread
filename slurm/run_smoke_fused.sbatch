#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -J fs_smoke
#SBATCH -o logs/smoke-%j.out
#SBATCH -e logs/smoke-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=== Fused engine regression smoke tests ==="
nvidia-smi --query-gpu=name --format=csv,noheader

# Run the tests that exercise RenewalEngineFused{,CUDAGraph} + erfcx.
python -m pytest \
    tests/smoke_test.py::TestRenewalEngineFused \
    tests/smoke_test.py::TestRenewalEngineNonMarkov \
    tests/smoke_test.py::TestTritonErfcxAccuracy \
    tests/smoke_test.py::TestFactoryFunction \
    -v -s
