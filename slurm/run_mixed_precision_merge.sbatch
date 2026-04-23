#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -J fs_mixmrg
#SBATCH -o logs/mixmrg-%j.out
#SBATCH -e logs/mixmrg-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "BA merge-path mixed-precision (code-freeze final)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=============================================="
echo ""

# Quick correctness sanity at N=1e4, production auto-dispatch path.
echo "=== [A] Fidelity sanity at N=1e4 (strategy=auto -> merge) ==="
python - <<'PY'
import torch
from flashspread import SEIRModel
from flashspread.core.graph import GraphCSR
import networkx as nx

N = 10_000
G = nx.barabasi_albert_graph(N, 4, seed=42)
edges = []
for u, v in G.edges():
    edges.append([u, v]); edges.append([v, u])
ei = torch.tensor(edges, dtype=torch.long).t().contiguous().cuda()
csr = GraphCSR(ei, N, incoming=True)
class GW: pass
gw = GW(); gw.csr = csr; gw.edge_index = ei
gw.num_nodes = N; gw.num_edges = ei.size(1)

from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph

def run(use_mixed, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    m = SEIRModel(beta=0.25, mean_ei=5.0, median_ei=4.0,
                  mean_ir=7.5, median_ir=5.0)
    # Force merge at N=1e4 because BA's D_max/D_mean ratio at small
    # N falls inside the warp dispatch window (ratio ~25 vs merge
    # cutoff 50), which would reject mixed precision on the warp
    # path. At N=1e6 auto-dispatch correctly selects merge; the
    # Phase B bench uses csr_strategy="auto" unchanged.
    e = RenewalEngineFusedCUDAGraph(
        gw, m, device="cuda",
        epsilon=0.03, tau_max=0.1,
        seed=seed, csr_strategy="merge",
        steps_per_launch=50,
        use_mixed_precision=use_mixed,
    )
    e.seed_infection(100, state=m.exposed)
    while e.current_time < 50.0:
        e.step()
    return torch.bincount(e.state.to(torch.int32), minlength=4).tolist()

for seed in (1, 2, 3):
    b = run(False, seed); m = run(True, seed)
    rR = abs(m[3] - b[3]) / max(b[3], 1)
    print(f"  seed={seed} baseline={b} mixed={m} relR={rR*100:.2f}%")
    assert rR < 0.10, f"Merge+mixed R-err too large at seed={seed}"
print("Fidelity bound PASSED on merge path.")
PY
echo ""

# Throughput at N=1e6, production csr_strategy="auto" (-> merge on BA).
echo "=== [B] Throughput on BA (auto-dispatch merge kernel) ==="
python experiments/bench_mixed_precision_merge.py \
    --num-nodes 1000000 --trials 5 \
    --output results/mixed_precision_merge_summary.csv

echo ""
echo "Done."
ls -la results/mixed_precision_merge_summary.csv
