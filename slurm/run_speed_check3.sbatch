#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J fs_speed_check3
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Speed re-measurement, part 3: close the flagged items on the corrected d=8
# graphs -- (1) age-dependent overhead table, (2) full-epidemic events/s
# calibration, (3) fused-kernel roofline point (GFLOPS + AI via the same
# estimators benchmark_roofline uses for the unfused rows).

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "${ROOT}"
mkdir -p logs results/speed_check
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY="${FLASHSPREAD_PY:-/home/hs9hd/.conda/envs/hpo_edu/bin/python}"
OUT="results/speed_check"

"${PY}" -c "import torch, flashspread; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# 1. Age-dependent transmission overhead (pre-pass vs in-kernel) on er,ba.
echo "=== [1/2] age-dependent overhead ==="
"${PY}" experiments/bench_throughput_age_dependent.py \
    --num-nodes 1000000 --trials 3 --graphs er,ba \
    --output "${OUT}/agedep_overhead.csv"

# 2. Full-epidemic events/s calibration + fused-kernel roofline point.
echo "=== [2/2] full-run events/s + fused roofline (FixedDegree d=8) ==="
"${PY}" - "${OUT}" <<'PYEOF'
import sys, time, torch
from flashspread import FixedDegreeGraph, SEIRModel
from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph
from flashspread.engines.renewal_tunable import (
    estimate_flops_per_step, estimate_memory_bytes_per_step)
out = sys.argv[1]
N, d, beta = 1_000_000, 8, 0.25
g = FixedDegreeGraph(N, d, device="cuda", seed=0)
edges = int(g.num_edges)
m = SEIRModel(beta=beta, mean_ei=5.0, median_ei=4.0, mean_ir=7.5, median_ir=5.0)

# (a) clean time/step (no per-step sync) for the roofline point
eng = RenewalEngineFusedCUDAGraph(g, m, device="cuda", epsilon=0.03, tau_max=0.1,
                                  steps_per_launch=50, seed=0)
eng.seed_infection(max(10, int(0.01*N)), state=m.exposed)
for _ in range(3): eng.step()                      # warm up
torch.cuda.synchronize(); t0 = time.time()
NW = 20
for _ in range(NW): eng.step()
torch.cuda.synchronize()
dt = time.time() - t0
steps = NW * eng.steps_per_launch
time_per_step = dt / steps
nups = N / time_per_step
flops = estimate_flops_per_step(N, edges, dense_hazard=False)["total"]
byts  = estimate_memory_bytes_per_step(N, edges)["total"]
gflops = flops / time_per_step / 1e9
ai = flops / byts

# (b) full-epidemic events/s: run a fresh engine to t=50, counting transitions
eng2 = RenewalEngineFusedCUDAGraph(g, m, device="cuda", epsilon=0.03, tau_max=0.1,
                                   steps_per_launch=50, seed=1)
eng2.seed_infection(max(10, int(0.01*N)), state=m.exposed)
prev = eng2.state.clone(); total_events = 0; total_steps = 0
torch.cuda.synchronize(); e0 = time.time()
while eng2.current_time < 50.0 and total_steps < 20000:
    eng2.step()
    total_events += int((eng2.state != prev).sum().item()); prev = eng2.state.clone()
    total_steps += eng2.steps_per_launch
torch.cuda.synchronize(); edt = time.time() - e0
events_per_nups = total_events / (N * total_steps)
events_per_s_at_headline = nups * events_per_nups   # events/s at the clean NUPS

with open(f"{out}/roofline_fused_events.csv", "w") as f:
    f.write("N,degree,edges,fused_nups,fused_time_per_step_ms,fused_gflops,fused_ai,"
            "flops_per_step,bytes_per_step,full_run_steps,full_run_events,"
            "events_per_nups,events_per_s_at_headline,final_attack_rate\n")
    fr = float((eng2.state == m.recovered).float().mean())
    f.write(f"{N},{d},{edges},{nups:.1f},{time_per_step*1e3:.4f},{gflops:.1f},{ai:.3f},"
            f"{flops},{byts},{total_steps},{total_events},{events_per_nups:.3e},"
            f"{events_per_s_at_headline:.1f},{fr:.3f}\n")
print(f"FUSED roofline: NUPS={nups/1e9:.3f}G time/step={time_per_step*1e3:.3f}ms "
      f"GFLOPS={gflops:.1f} AI={ai:.3f} (standard estimator; paper's fused AI=0.42 "
      f"uses a fused-specific byte model)")
print(f"events/s: full-run events/NUPS={events_per_nups:.3e} -> "
      f"events/s@headline={events_per_s_at_headline/1e6:.3f}M  attack={fr:.3f}")
PYEOF

echo "=== SUMMARY ==="
echo "--- agedep_overhead.csv ---"; cat "${OUT}/agedep_overhead.csv" || true
echo "--- roofline_fused_events.csv ---"; cat "${OUT}/roofline_fused_events.csv" || true
echo "=== part 3 complete ==="
