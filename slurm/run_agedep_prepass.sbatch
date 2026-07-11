#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J fs_agedep_prepass
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Closes tab:agedep_overhead on the corrected symmetric d=8 graphs: the naive
# dense-erfcx pre-pass vs the fused in-kernel infectivity write, for both
# transmission modes, on Regular (d=8) and BA (m=4). Uses the `thread` CSR
# strategy to match the manuscript's harness.

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "${ROOT}"
mkdir -p logs results/speed_check
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY="${FLASHSPREAD_PY:-/home/hs9hd/.conda/envs/hpo_edu/bin/python}"

"${PY}" -c "import torch, flashspread; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

"${PY}" experiments/bench_agedep_prepass.py \
    --num-nodes 1000000 --trials 3 --device cuda \
    --output results/speed_check/agedep_prepass.csv

echo "--- agedep_prepass.csv ---"
cat results/speed_check/agedep_prepass.csv
