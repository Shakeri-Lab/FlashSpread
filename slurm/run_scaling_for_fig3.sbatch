#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -J fs_scaling_fig3
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Scaling data for regenerating fig3-renewal-throughput.tex (N=1e2..1e8).
# scoglio + mixed-precision-scaling both use the PRIVATE on-device builder
# (directed degree-8), so their fused/mixed curves are C1-UNAFFECTED; this run
# gives the authoritative current events/s at each N (and resolves whether the
# fig3 headline "5.86 M ev/s" is really the scoglio curve).

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "${ROOT}"
mkdir -p logs results/speed_check
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY="${FLASHSPREAD_PY:-/home/hs9hd/.conda/envs/hpo_edu/bin/python}"
OUT="results/speed_check"

"${PY}" -c "import torch, flashspread; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# 1. Throughput scaling (events/s) across N=1e2..1e6 for Fused/Fused-CG/unfused-CG.
echo "=== [1/2] scoglio throughput scaling (events/s) ==="
"${PY}" experiments/bench_throughput_scoglio.py \
    --device cuda \
    --output "${OUT}/throughput_scoglio_overlay.png"   # also writes ...overlay.csv

# 2. Mixed-precision scaling to N=1e8 (base vs mixed; the L2-cliff + asymptote).
echo "=== [2/2] mixed-precision scaling to N=1e8 ==="
"${PY}" experiments/bench_mixed_precision_scaling.py \
    --output "${OUT}/mixed_precision_scaling.csv" \
    --max-n 100000000 --trials 5 || {
    echo "1e8 failed (likely OOM); retry capped at 1e7"
    "${PY}" experiments/bench_mixed_precision_scaling.py \
        --output "${OUT}/mixed_precision_scaling.csv" --max-n 10000000 --trials 5
}

echo "=== SUMMARY ==="
echo "--- throughput_scoglio_overlay.csv ---"; cat "${OUT}/throughput_scoglio_overlay.csv" 2>/dev/null || true
echo "--- mixed_precision_scaling.csv ---"; cat "${OUT}/mixed_precision_scaling.csv" 2>/dev/null || true
