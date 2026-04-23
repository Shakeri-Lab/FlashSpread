#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -J fs_erfcx
#SBATCH -o logs/erfcx_test-%j.out
#SBATCH -e logs/erfcx_test-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=== Triton erfcx accuracy test (stand-alone) ==="
nvidia-smi --query-gpu=name --format=csv,noheader
python -m pytest tests/smoke_test.py::TestTritonErfcxAccuracy -v -s
