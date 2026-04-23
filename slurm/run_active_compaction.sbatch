#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH -J fs_compact
#SBATCH -o logs/compact-%j.out
#SBATCH -e logs/compact-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Active-Node Compaction Ablation (Step 2: ER + BA, N=1e6)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=============================================="
echo ""

# --- Phase A: compaction-aware smoke test ---
# Quick correctness sanity at N=10^4 before committing to the full
# bench. We check bitwise state equality at several checkpoints along
# the trajectory, not just final compartment counts; if compaction
# has a latent bug the divergence usually starts at the first R
# refresh (around t=15 for this SEIR parametrisation) so an early
# checkpoint catches it cheaply.
echo "=== [A] Fidelity sanity check at N=1e4 ==="
python - <<'PY'
import torch
from flashspread import SEIRModel, FixedDegreeGraph
from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph

N = 10_000
g = FixedDegreeGraph(N, 8, device="cuda")
model = SEIRModel(beta=0.25, mean_ei=5.0, median_ei=4.0,
                  mean_ir=7.5, median_ir=5.0)

CHECKPOINTS = [1.0, 5.0, 15.0, 30.0, 50.0]

def run(use_comp, seed):
    # Seed torch's GLOBAL generator before each run -- seed_infection
    # uses torch.randperm which pulls from the default PRNG, so
    # back-to-back run(False, seed) + run(True, seed) would otherwise
    # see different initial infected sets and diverge from step 1.
    # The engine's own kernel-side RNG is independent (seeded via the
    # `seed=` constructor arg) but that doesn't cover the Python-side
    # initial-seed selection.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    e = RenewalEngineFusedCUDAGraph(
        g, model, device="cuda",
        epsilon=0.03, tau_max=0.1,
        seed=seed,
        csr_strategy="thread",
        steps_per_launch=50,
        use_active_compaction=use_comp,
    )
    e.seed_infection(100, state=model.exposed)
    snaps = {}
    for t_stop in CHECKPOINTS:
        while e.current_time < t_stop:
            e.step()
        snaps[t_stop] = e.state.detach().clone()
    return snaps

fail = False
for seed in (1, 2, 3):
    base = run(False, seed)
    comp = run(True,  seed)
    for t_stop in CHECKPOINTS:
        b = base[t_stop]
        c = comp[t_stop]
        diff = int((b != c).sum().item())
        b_cts = torch.bincount(b, minlength=4).tolist()
        c_cts = torch.bincount(c, minlength=4).tolist()
        ok = "OK " if diff == 0 else "DIFF"
        print(f"  seed={seed} t={t_stop:>4.1f} {ok} diff={diff:>5d} "
              f"baseline={b_cts} compact={c_cts}")
        if diff != 0:
            fail = True

if fail:
    raise AssertionError("Compaction diverged from baseline")
print("Fidelity sanity PASSED (bit-identical states at all checkpoints).")
PY
echo ""

# --- Phase B: full compaction ablation with NUPS(t) bucketing ---
echo "=== [B] Full ablation: ER + BA, 5 trials each, NUPS(t) buckets ==="
python experiments/bench_active_compaction.py \
    --num-nodes 1000000 \
    --trials 5 \
    --graphs er,ba \
    --output-csv results/active_compaction_summary.csv \
    --output-buckets results/active_compaction_buckets.npz
echo ""

# --- Phase C: RCM-BA proxy gate (same queue, opportunistic) ---
echo "=== [C] RCM-BA proxy (Step 1 go/no-go gate for Rabbit/Gorder) ==="
python experiments/bench_rcm_ba_proxy.py \
    --num-nodes 1000000 \
    --trials 5 \
    --output results/rcm_ba_proxy_summary.csv
echo ""

echo "Done."
echo "Artefacts:"
ls -la results/active_compaction_summary.csv \
       results/active_compaction_buckets.npz \
       results/rcm_ba_proxy_summary.csv
