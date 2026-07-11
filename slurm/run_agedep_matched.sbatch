#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J fs_agedep_matched
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Matched re-measurement of tab:agedep_overhead on the corrected d=8 graphs.
# Matches the manuscript's harness: the fused In-kernel rows use the THREAD
# CSR strategy (not auto-dispatch), and the Pre-pass rows use the NonMarkov
# CUDA-Graph engine (separate dense infectivity pre-pass). Regular = FixedDegree
# d=8, BA = m=4. Slowdown = constant/age_dependent - 1 within each implementation.

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "${ROOT}"
mkdir -p logs results/speed_check
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY="${FLASHSPREAD_PY:-/home/hs9hd/.conda/envs/hpo_edu/bin/python}"
OUT="results/speed_check"

"${PY}" -c "import torch, flashspread; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

"${PY}" - "${OUT}" <<'PYEOF'
import sys, time, torch
from flashspread import SEIRModel
from flashspread.core.network import FixedDegreeGraph, BarabasiAlbertGraph
from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph
from flashspread.engines.renewal import RenewalEngineNonMarkovCUDAGraph
out = sys.argv[1]
N, TF, SPL = 1_000_000, 50.0, 50

def bench(EngCls, g, mode, kw, trials=3):
    m = SEIRModel(beta=0.25, mean_ei=5.0, median_ei=4.0, mean_ir=7.5, median_ir=5.0)
    m.transmission_mode = mode
    tot_nups = tot_t = 0.0
    for tr in range(2 + trials):
        eng = EngCls(g, m, device="cuda", epsilon=0.03, tau_max=0.1,
                     steps_per_launch=SPL, seed=12345 + tr*100, **kw)
        eng.state[0] = m.infected; eng.age[0] = 0.0
        torch.cuda.synchronize(); t0 = time.perf_counter(); steps = 0
        while eng.current_time < TF:
            eng.step(); steps += eng.steps_per_launch
        torch.cuda.synchronize(); t1 = time.perf_counter()
        if tr >= 2:
            tot_nups += N * steps; tot_t += (t1 - t0)
    return tot_nups / tot_t

graphs = [("Regular (d=8)", FixedDegreeGraph(N, 8, device="cuda", seed=0)),
          ("BA (m=4)",      BarabasiAlbertGraph(N, 4, device="cuda", seed=0))]
rows = []
for gname, g in graphs:
    # Pre-pass = NonMarkov CG (dense infectivity pre-pass).
    c_pp = bench(RenewalEngineNonMarkovCUDAGraph, g, "constant", {})
    a_pp = bench(RenewalEngineNonMarkovCUDAGraph, g, "age_dependent", {})
    # In-kernel = Fused CG, thread strategy (matches the paper).
    c_ik = bench(RenewalEngineFusedCUDAGraph, g, "constant", {"csr_strategy": "thread"})
    a_ik = bench(RenewalEngineFusedCUDAGraph, g, "age_dependent", {"csr_strategy": "thread"})
    rows.append((gname, "Pre-pass",  c_pp, a_pp, c_pp/a_pp - 1))
    rows.append((gname, "In-kernel", c_ik, a_ik, c_ik/a_ik - 1))
    print(f"{gname}: pre-pass const={c_pp/1e9:.3f}G age={a_pp/1e9:.3f}G ({(c_pp/a_pp-1)*100:+.1f}%); "
          f"in-kernel const={c_ik/1e9:.3f}G age={a_ik/1e9:.3f}G ({(c_ik/a_ik-1)*100:+.1f}%)")

with open(f"{out}/agedep_matched.csv", "w") as f:
    f.write("graph,implementation,constant_nups,age_dependent_nups,slowdown\n")
    for gname, impl, c, a, s in rows:
        f.write(f"{gname},{impl},{c:.1f},{a:.1f},{s:.4f}\n")
print("wrote", f"{out}/agedep_matched.csv")
PYEOF

echo "--- agedep_matched.csv ---"; cat "${OUT}/agedep_matched.csv"
