#!/usr/bin/env python
"""Legacy mixed-storage ablation; all results must be remeasured.

The production mixed path now retains fp32 age clocks. This script can compare
current fp32 versus mixed storage, but its historical CSV cannot support the
manuscript's old fp16-age working-set, bandwidth, or scale claims. Prefer
``benchmark_acceptance.py`` for publication measurements and provenance.

Baseline:
    state       int32
    age         float32
    infectivity float32
    weights     float32

Mixed precision (use_mixed_precision=True):
    state       int8       (4x state traffic)
    age         float32    (unchanged; prevents small-tau clock freeze)
    infectivity bfloat16   (2x infectivity traffic)
    weights     bfloat16   (2x weight traffic)
    pressure accumulator STILL fp32 inside the kernel (mandatory:
    summing hundreds of bf16 edge contributions on a hub would
    absorb small values via mantissa underflow in the exact regime
    where lambda*tau << 1 matters most).

We measure whole-run throughput on ER d=8 and BA m=4 (both at
N=1e6, TF=50) with 5 trials per cell and a per-seed fidelity check
against the baseline; mixed precision is NOT bit-identical by design
(7-bit bf16 mantissa vs 23-bit fp32), so we use a tolerance check on
final compartment counts.

This historical matrix remains restricted to ``csr_strategy='thread'`` for
comparability; the production kernel supports mixed storage in every strategy.
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


def _build_er(N, device):
    g = FixedDegreeGraph(N, NLINK, device=device)
    class GW:
        pass

    gw = GW()
    gw.csr = g.csr
    gw.edge_index = g.edge_index
    gw.num_nodes = N
    gw.num_edges = g.edge_index.size(1)
    return gw


def _build_ba(N, m, device):
    import networkx as nx
    G = nx.barabasi_albert_graph(N, m, seed=42)
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(ei, N, incoming=True)
    class GW:
        pass

    gw = GW()
    gw.csr = csr
    gw.edge_index = ei
    gw.num_nodes = N
    gw.num_edges = ei.size(1)
    return gw


def _run(gw, use_mixed, seed, device):
    # Seed torch's global PRNG so seed_infection's randperm is the
    # same initial set across baseline and mixed runs.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Fresh model per run: its internal tensors are prepared on
    # construction and shouldn't be shared between engines with
    # different storage dtypes.
    model = SEIRModel(beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
                      mean_ir=MEAN_IR, median_ir=MEDIAN_IR)

    engine = RenewalEngineFusedCUDAGraph(
        gw, model, device=device,
        epsilon=EPSILON, tau_max=TAU_MAX,
        seed=seed,
        csr_strategy="thread",
        steps_per_launch=STEPS_PER_LAUNCH,
        use_active_compaction=False,
        use_mixed_precision=use_mixed,
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
    final_state = engine.state.to(torch.int32).detach().clone()
    final_cts = torch.bincount(final_state, minlength=4).tolist()

    return {
        "wall_s": wall,
        "nups_per_s": total_nups / wall,
        "total_nups": total_nups,
        "total_steps": engine.total_steps,
        "final_counts": final_cts,
    }


def bench_cell(gw, label, use_mixed, trials, device):
    out = []
    for t in range(trials):
        seed = 30260101 + 100 * t
        r = _run(gw, use_mixed, seed, device)
        out.append(r)
        print(
            f"  [{label} mixed={int(use_mixed)}] trial {t}: "
            f"{r['nups_per_s']:.3e} NUPS, wall {r['wall_s']:.2f}s, "
            f"counts={r['final_counts']}",
            flush=True,
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-nodes", type=int, default=1_000_000)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--graphs", type=str, default="er,ba")
    p.add_argument("--output", type=str,
                   default="results/mixed_precision_summary.csv")
    args = p.parse_args()

    device = "cuda"
    print(f"Mixed-precision ablation: N={args.num_nodes:,}  tf={TF}  "
          f"trials={args.trials}  steps_per_launch={STEPS_PER_LAUNCH}")
    print()

    graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]

    rows = []
    for gname in graphs:
        if gname == "er":
            gw = _build_er(args.num_nodes, device)
            label = "er"
        elif gname == "ba":
            gw = _build_ba(args.num_nodes, 4, device)
            label = "ba"
        else:
            print(f"skip: {gname}")
            continue

        print(f"=== {label}: baseline (fp32/int32) ===")
        base = bench_cell(gw, label, False, args.trials, device)

        print(f"=== {label}: mixed storage (int8/fp32/bf16) ===")
        mixed = bench_cell(gw, label, True, args.trials, device)

        base_nups = np.array([r["nups_per_s"] for r in base])
        mixed_nups = np.array([r["nups_per_s"] for r in mixed])
        # Mean fidelity error: |final_R_mixed - final_R_base| /
        # final_R_base, as a proxy for trajectory-level error. We
        # pair trials by seed index so the Bernoulli draws differ
        # only through the storage-dtype rounding.
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
        print()

        rows.append({
            "graph": label,
            "N": args.num_nodes,
            "base_nups_mean": base_nups.mean(),
            "base_nups_std": base_nups.std(),
            "mixed_nups_mean": mixed_nups.mean(),
            "mixed_nups_std": mixed_nups.std(),
            "speedup": speedup,
            "rel_err_R_mean_pct": rel_err_R.mean() * 100,
            "rel_err_R_max_pct": rel_err_R.max() * 100,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
