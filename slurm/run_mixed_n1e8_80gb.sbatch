#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -J fs_mix_n1e8
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Re-measure the N=1e8 endpoint of Figure 3 (the zoom-panel asymptote) on the
# corrected code, on an 80 GB A100. The earlier full-sweep run OOM'd on a 40 GB
# card because memory accumulated through N=1e7; here we run ONLY N=1e8 (as the
# original run_mixed_precision_n1e8.sbatch did) and give it the 80 GB card.

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "${ROOT}"
mkdir -p logs results/speed_check
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation
PY="${FLASHSPREAD_PY:-/home/hs9hd/.conda/envs/hpo_edu/bin/python}"

echo "=== environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import torch; print('GPU mem (GiB):', round(torch.cuda.get_device_properties(0).total_memory/2**30,1))"

"${PY}" experiments/bench_mixed_precision_scaling.py \
    --min-n 100000000 --max-n 100000000 \
    --trials 5 \
    --output results/speed_check/mixed_precision_scaling_n1e8.csv

echo "--- mixed_precision_scaling_n1e8.csv ---"
cat results/speed_check/mixed_precision_scaling_n1e8.csv
