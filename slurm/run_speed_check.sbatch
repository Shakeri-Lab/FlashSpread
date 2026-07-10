#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:30:00
#SBATCH -J fs_speed_check
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Speed / roofline re-measurement on the CURRENT (Phase 0/1) code.
#
# All benchmarks build their graphs with the PUBLIC FixedDegreeGraph, so the
# C1 symmetrization fix DOUBLES the edge count vs the published numbers -- this
# job reports the honest throughput on the corrected (full-degree d=8) graphs.
# Order: fast NUPS headline first, then CPU baseline (for the 217x ratio), then
# the roofline (now d=8 to match the manuscript and to build fast). Outputs go
# to results/speed_check/ so the published results/ are untouched.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"
mkdir -p logs results/speed_check
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PY="${FLASHSPREAD_PY:-/home/hs9hd/.conda/envs/hpo_edu/bin/python}"
OUT="results/speed_check"

echo "=== environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import torch, flashspread; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

# 1. Fused-engine NUPS (headline). 'thread' on FixedDegree d=8 is the
#    8.09-Giga-NUPS config -- now on the corrected degree-8 graph.
echo "=== [1/3] fused-engine NUPS throughput (FixedDegree d=8, BA m=4) ==="
"${PY}" experiments/bench_warp_collab_ba.py \
    --num-nodes 1000000 --graphs er,ba --trials 5 \
    --output "${OUT}/nups.csv" --device cuda

# 2. CPU tau-leaping baseline on the SAME corrected d=8 graph (both sides of
#    the 217x ratio were previously measured on the buggy half-degree graph).
echo "=== [2/3] CPU tau-leaping baseline (for the 217x ratio) ==="
"${PY}" experiments/bench_cpu_tauleaping.py \
    --sizes 1000000 --trials 3 --tf 50.0 \
    --output "${OUT}/cpu_tauleaping_nups.csv"

# 3. Roofline analysis at d=8 (matches the manuscript; builds fast).
echo "=== [3/3] roofline benchmark (13 configs, d=8) ==="
"${PY}" experiments/benchmark_roofline.py \
    --num-nodes 1000000 --degree 8 \
    --output-dir "${OUT}" --device cuda

echo "=== SUMMARY ==="
echo "--- nups.csv ---"; cat "${OUT}/nups.csv" || true
echo "--- cpu_tauleaping_nups.csv ---"; cat "${OUT}/cpu_tauleaping_nups.csv" || true
echo "--- roofline_summary.md ---"; sed -n '1,80p' "${OUT}/roofline_summary.md" || true
echo "=== speed check complete: ${OUT}/ ==="
