#!/usr/bin/env python
"""
Mixed-precision throughput on the production csr_strategy="auto" path.

After extending MIXED_PRECISION into the merge-path tail kernel, this
bench measures the production-dispatch throughput (auto picks merge
on BA m=4, thread on ER d=8) with and without mixed-precision storage.
The merge pressure scratch buffer stays fp32 per the kernel contract,
so the fp32 atomic accumulation over scale-free hubs is unaffected.

This is the headline measurement that closes the code-freeze loop:
- If BA-merge mixed >= 2.0 G-NUPS we have maximised the pure-Triton
  bandwidth story on the A100 and Step 4 (CUDA-C++ segmented
  reduction) is formally future work.
- If the gain is <5% we document the null and move on.
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

from flashspread import SEIRModel, FixedDegreeGraph
from flashspread.core.graph import GraphCSR
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
    class GW: pass
    gw = GW()
    gw.csr = csr
    gw.edge_index = ei
    gw.num_nodes = N
    gw.num_edges = ei.size(1)
    return gw


def _run(gw, use_mixed, seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SEIRModel(beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
                      mean_ir=MEAN_IR, median_ir=MEDIAN_IR)
    engine = RenewalEngineFusedCUDAGraph(
        gw, model, device=device,
        epsilon=EPSILON, tau_max=TAU_MAX,
        seed=seed,
        csr_strategy="auto",            # production: picks merge on BA
        steps_per_launch=STEPS_PER_LAUNCH,
        use_active_compaction=False,
        use_mixed_precision=use_mixed,
    )
    # Print strategy once for confirmation.
    if seed % 10000 == 101:
        print(f"  (dispatched strategy = {engine.csr_strategy!r})",
              flush=True)

    n_seed = max(10, int(0.01 * gw.num_nodes))
    engine.seed_infection(n_seed, state=model.exposed)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while engine.current_time < TF:
        engine.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    total_nups = gw.num_nodes * engine.total_steps
    final_cts = torch.bincount(
        engine.state.to(torch.int32), minlength=4
    ).tolist()
    return {
        "strategy": engine.csr_strategy,
        "wall_s": wall,
        "nups_per_s": total_nups / wall,
        "total_nups": total_nups,
        "total_steps": engine.total_steps,
        "final_counts": final_cts,
    }


def bench_cell(gw, label, use_mixed, trials, device):
    out = []
    for t in range(trials):
        seed = 40260101 + 100 * t
        r = _run(gw, use_mixed, seed, device)
        out.append(r)
        print(
            f"  [{label} mixed={int(use_mixed)} strat={r['strategy']}] "
            f"trial {t}: {r['nups_per_s']:.3e} NUPS, wall {r['wall_s']:.2f}s, "
            f"counts={r['final_counts']}",
            flush=True,
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-nodes", type=int, default=1_000_000)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--output", type=str,
                   default="results/mixed_precision_merge_summary.csv")
    args = p.parse_args()

    device = "cuda"
    print(f"BA m=4 mixed-precision (auto-dispatch) at N={args.num_nodes:,}  "
          f"tf={TF}  trials={args.trials}  spl={STEPS_PER_LAUNCH}")
    print()

    gw = _build_ba(args.num_nodes, 4, device)

    print("=== BA: baseline (fp32/int32), strategy=auto (-> merge) ===")
    base = bench_cell(gw, "ba", False, args.trials, device)

    print("=== BA: mixed precision, strategy=auto (-> merge) ===")
    mixed = bench_cell(gw, "ba", True, args.trials, device)

    base_nups = np.array([r["nups_per_s"] for r in base])
    mixed_nups = np.array([r["nups_per_s"] for r in mixed])
    base_R = np.array([r["final_counts"][3] for r in base])
    mix_R  = np.array([r["final_counts"][3] for r in mixed])
    rel_err_R = np.abs(mix_R - base_R) / np.maximum(base_R, 1)

    speedup = mixed_nups.mean() / base_nups.mean()
    print()
    print(f"  baseline: {base_nups.mean():.3e} +/- {base_nups.std():.1e}")
    print(f"  mixed   : {mixed_nups.mean():.3e} +/- {mixed_nups.std():.1e}")
    print(f"  speedup : {speedup:.3f}x")
    print(f"  per-seed R error: mean={rel_err_R.mean()*100:.2f}%  "
          f"max={rel_err_R.max()*100:.2f}%")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "strategy", "mixed", "trial", "nups_per_s", "wall_s",
            "final_S", "final_E", "final_I", "final_R",
        ])
        for i, r in enumerate(base):
            w.writerow([r["strategy"], 0, i, r["nups_per_s"], r["wall_s"],
                        *r["final_counts"]])
        for i, r in enumerate(mixed):
            w.writerow([r["strategy"], 1, i, r["nups_per_s"], r["wall_s"],
                        *r["final_counts"]])
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
