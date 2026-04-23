#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J fs_mixed
#SBATCH -o logs/mixed-%j.out
#SBATCH -e logs/mixed-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Mixed-precision ablation (Step 3: int8/fp16/bf16 storage)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=============================================="
echo ""

# --- Phase A: small-N correctness check (mixed precision is NOT
# bit-identical to baseline by design -- we only assert final R is
# within 10% and compartment counts are sane). ---
echo "=== [A] Small-N fidelity bound (N=1e4) ==="
python - <<'PY'
import torch
from flashspread import SEIRModel, FixedDegreeGraph
from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph

N = 10_000
g = FixedDegreeGraph(N, 8, device="cuda")

def run(use_mixed, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = SEIRModel(beta=0.25, mean_ei=5.0, median_ei=4.0,
                      mean_ir=7.5, median_ir=5.0)
    e = RenewalEngineFusedCUDAGraph(
        g, model, device="cuda",
        epsilon=0.03, tau_max=0.1,
        seed=seed, csr_strategy="thread",
        steps_per_launch=50,
        use_mixed_precision=use_mixed,
    )
    e.seed_infection(100, state=model.exposed)
    while e.current_time < 50.0:
        e.step()
    return torch.bincount(e.state.to(torch.int32), minlength=4).tolist()

for seed in (1, 2, 3):
    b = run(False, seed)
    m = run(True, seed)
    rel_R = abs(m[3] - b[3]) / max(b[3], 1)
    print(f"  seed={seed} baseline={b} mixed={m} relR={rel_R*100:.2f}%")
    assert rel_R < 0.10, f"Mixed precision R-err too large at seed={seed}"
print("Fidelity bound PASSED (|R_mixed - R_base| < 10% at N=1e4).")
PY
echo ""

# --- Phase B: full ablation at N=1e6 ---
echo "=== [B] Throughput ablation: ER + BA, 5 trials each ==="
python experiments/bench_mixed_precision.py \
    --num-nodes 1000000 \
    --trials 5 \
    --graphs er,ba \
    --output results/mixed_precision_summary.csv
echo ""

echo "Done."
ls -la results/mixed_precision_summary.csv
