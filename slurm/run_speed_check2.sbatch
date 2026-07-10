#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J fs_speed_check2
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Speed re-measurement, part 2: the remaining C1-affected (public-generator)
# benches at N=1e6 on the corrected symmetric d=8 graphs. The N=1e7/1e8 scaling,
# the L2-cliff (2.32x/1.15x), and the mixed-precision *scaling* curves use the
# PRIVATE on-device builder and are UNAFFECTED, so they are intentionally not
# re-run here.

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "${ROOT}"
mkdir -p logs results/speed_check
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY="${FLASHSPREAD_PY:-/home/hs9hd/.conda/envs/hpo_edu/bin/python}"
OUT="results/speed_check"

echo "=== environment ==="
"${PY}" -c "import torch, flashspread; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# 1. Mixed-precision at N=1e6 (er=regular d=8 + ba). Corrects 1.14x / 1.06x.
echo "=== [1/3] mixed precision (N=1e6) ==="
"${PY}" experiments/bench_mixed_precision.py \
    --num-nodes 1000000 --trials 3 --graphs er,ba \
    --output "${OUT}/mixed_precision_summary.csv"

# 2. Active-node compaction (er + ba). Corrects 1.53x whole-run + attack rates.
echo "=== [2/3] active-node compaction (N=1e6) ==="
"${PY}" experiments/bench_active_compaction.py \
    --num-nodes 1000000 --trials 3 --graphs er,ba \
    --output-csv "${OUT}/active_compaction_summary.csv" \
    --output-buckets "${OUT}/active_compaction_buckets.npz"

# 3. Direct events/s probe on the corrected FixedDegree d=8 fused engine
#    (the manuscript's 5.86 M events/s <-> 8.09 GNUPS; measure it, don't derive).
echo "=== [3/3] events/s + fused time/step probe (FixedDegree d=8, N=1e6) ==="
"${PY}" - "${OUT}" <<'PYEOF'
import sys, time, torch
from flashspread import FixedDegreeGraph, SEIRModel
from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph
out = sys.argv[1]
N, d, beta = 1_000_000, 8, 0.25
g = FixedDegreeGraph(N, d, device="cuda", seed=0)
m = SEIRModel(beta=beta, mean_ei=5.0, median_ei=4.0, mean_ir=7.5, median_ir=5.0)
eng = RenewalEngineFusedCUDAGraph(g, m, device="cuda", epsilon=0.03, tau_max=0.1,
                                  steps_per_launch=50, seed=0)
eng.seed_infection(max(10, int(0.01*N)), state=m.exposed)
# count transitions by tracking state changes across a fixed number of steps
prev = eng.state.clone()
torch.cuda.synchronize(); t0 = time.time()
total_events, total_steps = 0, 0
for _ in range(40):                       # 40 replay windows * 50 steps = 2000 steps
    eng.step()
    cur = eng.state
    total_events += int((cur != prev).sum().item())
    prev = cur.clone()
    total_steps += eng.steps_per_launch
torch.cuda.synchronize(); dt = time.time() - t0
nups = N * total_steps / dt
events_per_s = total_events / dt
edges = int(g.num_edges)
with open(f"{out}/events_probe.csv", "w") as f:
    f.write("N,degree,edges,total_steps,total_events,wall_s,nups,events_per_s,events_per_nups\n")
    f.write(f"{N},{d},{edges},{total_steps},{total_events},{dt:.4f},{nups:.1f},{events_per_s:.1f},{events_per_s/nups:.3e}\n")
print(f"corrected: NUPS={nups/1e9:.3f}G  events/s={events_per_s/1e6:.3f}M  "
      f"events/NUPS={events_per_s/nups:.3e}  edges={edges}")
PYEOF

echo "=== SUMMARY ==="
echo "--- mixed_precision_summary.csv ---"; cat "${OUT}/mixed_precision_summary.csv" || true
echo "--- active_compaction_summary.csv ---"; cat "${OUT}/active_compaction_summary.csv" || true
echo "--- events_probe.csv ---"; cat "${OUT}/events_probe.csv" || true
echo "=== part 2 complete ==="
