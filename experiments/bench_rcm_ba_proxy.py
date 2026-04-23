#!/usr/bin/env python
"""
BA + RCM proxy benchmark (go/no-go gate for Rabbit/Gorder, Step 1 of
the effective-bandwidth plan).

The existing ablation (CLAUDE.md: job 7379061) showed RCM alone gives
only 1.01x on ER d=8 -- regular graphs are already near-optimally
laid out. The open question is whether RCM delivers on BA. This
benchmark answers that with a minimal experiment:

  * BA m=4 at N=1e6, fused CG, 5 trials each.
  * Baseline: raw BA CSR (NetworkX edge order).
  * RCM: CSR re-ordered with the existing Reverse Cuthill-McKee
    pass in flashspread.core.optimizations, nodes remapped
    accordingly.

Go/no-go:
  - RCM on BA >= 1.20x  : basic matrix-bandwidth minimisation is
    "good enough" locality, Rabbit/Gorder's marginal win not worth
    the C++ build-system bloat.
  - RCM on BA <= 1.05x  : concrete proof that bandwidth-minimising
    reorders fail on scale-free graphs, justifying integration of a
    cache-line-aware algorithm (Rabbit Order / Gorder) for a future
    revision.
  - 1.05x < RCM < 1.20x : grey zone, flag for a longer-form A/B.

No CPU dependency beyond scipy (already used). Pure Python bench;
all simulation stays on GPU.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread import SEIRModel
from flashspread.core.graph import GraphCSR
from flashspread.core.optimizations import reorder_graph_rcm
from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph

NLINK = 8
BETA = 2.0 / NLINK
MEAN_EI, MEDIAN_EI = 5.0, 4.0
MEAN_IR, MEDIAN_IR = 7.5, 5.0
EPSILON = 0.03
TAU_MAX = 0.1
TF = 50.0
STEPS_PER_LAUNCH = 50


def _build_ba(N, m, device):
    import networkx as nx
    G = nx.barabasi_albert_graph(N, m, seed=42)
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(ei, N, incoming=True)
    return csr, ei


class _GraphWrapper:
    def __init__(self, csr, num_nodes):
        self.csr = csr
        self.num_nodes = num_nodes
        # Engine also sometimes inspects edge_index; rebuild lazily
        # only if needed. For fused-CG engine on thread/merge strategy,
        # the CSR alone suffices.
        self.num_edges = csr.col_ind.numel()
        self.edge_index = None


def _run_trial(gw, model, seed, device):
    engine = RenewalEngineFusedCUDAGraph(
        gw, model, device=device,
        epsilon=EPSILON, tau_max=TAU_MAX,
        seed=seed,
        csr_strategy="auto",      # let it pick merge on BA as production would
        steps_per_launch=STEPS_PER_LAUNCH,
        use_active_compaction=False,
    )
    n_seed = max(10, int(0.01 * gw.num_nodes))
    engine.seed_infection(n_seed, state=model.exposed)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while engine.current_time < TF:
        engine.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    total_nups = gw.num_nodes * engine.total_steps
    final_cts = torch.bincount(engine.state, minlength=4).tolist()
    return {
        "wall_s": wall,
        "nups_per_s": total_nups / wall,
        "total_nups": total_nups,
        "total_steps": engine.total_steps,
        "final_counts": final_cts,
    }


def bench_cell(label, gw, model, trials, device):
    per_trial = []
    for t in range(trials):
        seed = 20260101 + 100 * t
        out = _run_trial(gw, model, seed, device)
        per_trial.append(out)
        print(
            f"  [{label}] trial {t}: {out['nups_per_s']:.3e} NUPS, "
            f"wall {out['wall_s']:.2f}s, counts={out['final_counts']}",
            flush=True,
        )
    return per_trial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-nodes", type=int, default=1_000_000)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--output", type=str,
                   default="results/rcm_ba_proxy_summary.csv")
    args = p.parse_args()

    device = "cuda"
    N = args.num_nodes
    print(f"BA (m=4) RCM proxy: N={N:,}  tf={TF}  trials={args.trials}")
    print(f"steps_per_launch={STEPS_PER_LAUNCH}  epsilon={EPSILON}")
    print()

    # Build BA graph once
    print("Building BA m=4 ...", flush=True)
    csr_raw, _ei_raw = _build_ba(N, 4, device)

    # Degree stats
    degrees = (csr_raw.row_ptr[1:] - csr_raw.row_ptr[:-1]).float()
    dmax, dmean = float(degrees.max()), float(degrees.mean())
    print(f"  raw degree: max={dmax:.0f}  mean={dmean:.2f}  "
          f"ratio={dmax/dmean:.1f}")

    # RCM reorder
    print("Applying RCM reorder (scipy.sparse.csgraph) ...", flush=True)
    t0 = time.perf_counter()
    csr_rcm, perm = reorder_graph_rcm(csr_raw)
    t_rcm = time.perf_counter() - t0
    print(f"  RCM preprocess: {t_rcm:.2f}s CPU  (amortised over ensemble)")

    model = SEIRModel(beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
                      mean_ir=MEAN_IR, median_ir=MEDIAN_IR)

    gw_raw = _GraphWrapper(csr_raw, N)
    gw_rcm = _GraphWrapper(csr_rcm, N)

    print("\n=== Baseline (raw NetworkX order) ===")
    raw_trials = bench_cell("raw", gw_raw, model, args.trials, device)

    print("\n=== RCM reorder ===")
    rcm_trials = bench_cell("rcm", gw_rcm, model, args.trials, device)

    raw_nups = np.array([t["nups_per_s"] for t in raw_trials])
    rcm_nups = np.array([t["nups_per_s"] for t in rcm_trials])

    raw_mean, raw_std = raw_nups.mean(), raw_nups.std()
    rcm_mean, rcm_std = rcm_nups.mean(), rcm_nups.std()
    ratio = rcm_mean / raw_mean

    # Go/no-go bucket
    if ratio >= 1.20:
        verdict = "RCM-sufficient (skip Rabbit/Gorder)"
    elif ratio <= 1.05:
        verdict = "Needs cache-line-aware reorder (Rabbit/Gorder justified)"
    else:
        verdict = "Grey zone (1.05-1.20x): longer A/B warranted"

    print()
    print("=" * 60)
    print(f"RCM ratio on BA m=4, N={N:,}: {ratio:.3f}x")
    print(f"  raw: {raw_mean:.3e} +/- {raw_std:.1e} NUPS/s")
    print(f"  rcm: {rcm_mean:.3e} +/- {rcm_std:.1e} NUPS/s")
    print(f"  RCM CPU preprocess: {t_rcm:.2f}s "
          f"(amortise over >= {int(t_rcm / max(raw_trials[0]['wall_s'], 1e-6))+1} "
          f"ensemble trials)")
    print(f"Verdict: {verdict}")
    print("=" * 60)

    # Write CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "layout", "trial", "nups_per_s", "wall_s",
            "final_S", "final_E", "final_I", "final_R",
        ])
        for i, out in enumerate(raw_trials):
            w.writerow(["raw", i, out["nups_per_s"], out["wall_s"],
                        *out["final_counts"]])
        for i, out in enumerate(rcm_trials):
            w.writerow(["rcm", i, out["nups_per_s"], out["wall_s"],
                        *out["final_counts"]])
    print(f"\nWrote per-trial CSV: {args.output}")


if __name__ == "__main__":
    main()
