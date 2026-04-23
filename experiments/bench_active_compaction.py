#!/usr/bin/env python
"""
Active-node compaction ablation (Step 2 of the effective-bandwidth plan).

For each (graph, use_active_compaction) cell we run 5 trials and record
NUPS stratified into 10 temporal buckets, so the NUPS(t) curve can be
overlaid on the epidemic curve in the manuscript's Section 5.6. We also
record final SEIR compartment counts for a fidelity sanity-check against
the established ~6% structural-bias floor (Appendix C).

CUDA Graph note: the compaction path uses the Fixed-Grid, Early-Exit
pattern --- the kernel launch grid stays at cdiv(N, BLOCK_SIZE), but
tail blocks past num_active retire in nanoseconds (no HBM reads). The
active-node list is rebuilt once per CUDA Graph replay window
(steps_per_launch = 50) and the state != R predicate keeps S nodes on
the active list, which is mandatory under the pull-based CSR gather.
"""

from __future__ import annotations

import argparse
import csv
import os
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
NUM_BUCKETS = 10  # NUPS(t) temporal resolution


def _build_er(N, device):
    g = FixedDegreeGraph(N, NLINK, device=device)
    class GW: pass
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
    class GW: pass
    gw = GW()
    gw.csr = csr
    gw.edge_index = ei
    gw.num_nodes = N
    gw.num_edges = ei.size(1)
    return gw


def _run_one(gw, model, device, use_compaction, seed, tf=TF):
    """Single-trial run. Returns (nups_per_bucket, final_counts, total_nups, total_wall)."""
    N = gw.num_nodes

    # Force thread strategy: compaction is only wired into the thread
    # kernel for now. Auto-dispatch on BA would pick merge, so we have
    # to override explicitly to exercise the compaction path on BA.
    engine = RenewalEngineFusedCUDAGraph(
        gw, model, device=device,
        epsilon=EPSILON, tau_max=TAU_MAX,
        seed=seed,
        csr_strategy="thread",
        steps_per_launch=STEPS_PER_LAUNCH,
        use_active_compaction=use_compaction,
    )
    # Seed: 1% of nodes Exposed, minimum 10.
    n_seed = max(10, int(0.01 * N))
    engine.seed_infection(n_seed, state=model.exposed)

    # Temporal bucketing: record wall-clock and simulated steps per
    # bucket so NUPS(t) can be reconstructed.
    bucket_wall = np.zeros(NUM_BUCKETS)
    bucket_steps = np.zeros(NUM_BUCKETS)
    bucket_t_edges = np.linspace(0.0, tf, NUM_BUCKETS + 1)

    torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    while engine.current_time < tf:
        t_before = engine.current_time
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_after = engine.current_time

        wall = t1 - t0
        # Allocate wall/steps proportionally to buckets this step covers
        for b in range(NUM_BUCKETS):
            lo = max(t_before, bucket_t_edges[b])
            hi = min(t_after, bucket_t_edges[b + 1])
            if hi > lo:
                frac = (hi - lo) / max(t_after - t_before, 1e-12)
                bucket_wall[b] += wall * frac
                bucket_steps[b] += STEPS_PER_LAUNCH * frac

    torch.cuda.synchronize()
    total_wall = time.perf_counter() - t_total_start

    final_counts = torch.bincount(engine.state, minlength=4).tolist()
    total_steps = engine.total_steps
    total_nups = N * total_steps

    # NUPS per bucket: N * steps_in_bucket / wall_in_bucket
    nups_per_bucket = np.where(
        bucket_wall > 0, N * bucket_steps / bucket_wall, 0.0
    )

    # Track final num_active for diagnostics
    num_active_final = int(engine._num_active_device.item()) if use_compaction else N

    return {
        "nups_per_bucket": nups_per_bucket,
        "total_nups": total_nups,
        "total_wall": total_wall,
        "total_steps": total_steps,
        "final_counts": final_counts,
        "num_active_final": num_active_final,
    }


def bench_cell(gw, label, use_compaction, trials, device):
    """Aggregate multiple trials, return per-trial list + summary."""
    model = SEIRModel(beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
                      mean_ir=MEAN_IR, median_ir=MEDIAN_IR)

    per_trial = []
    for t in range(trials):
        seed = 12345 + 100 * t
        out = _run_one(gw, model, device, use_compaction, seed)
        per_trial.append(out)
        print(
            f"  [{label} compaction={use_compaction}] trial {t}: "
            f"{out['total_nups']/out['total_wall']:.3e} NUPS, "
            f"counts={out['final_counts']}, "
            f"num_active_final={out['num_active_final']}",
            flush=True,
        )

    return per_trial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-nodes", type=int, default=1_000_000)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--graphs", type=str, default="er,ba")
    p.add_argument("--output-csv", type=str,
                   default="results/active_compaction_summary.csv")
    p.add_argument("--output-buckets", type=str,
                   default="results/active_compaction_buckets.npz")
    args = p.parse_args()

    device = "cuda"
    print(f"N={args.num_nodes:,}  tf={TF}  epsilon={EPSILON}  "
          f"steps_per_launch={STEPS_PER_LAUNCH}  trials={args.trials}")
    print(f"compaction predicate: state != R (keeps S on active list, see")
    print(f"docstring for why E+I alone freezes the epidemic)")
    print()

    graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]

    rows = []                  # summary rows (CSV)
    bucket_data = {}           # {label_compaction: (trials, NUM_BUCKETS)}

    for gname in graphs:
        if gname == "er":
            gw = _build_er(args.num_nodes, device)
            label = "er"
        elif gname == "ba":
            gw = _build_ba(args.num_nodes, 4, device)
            label = "ba"
        else:
            print(f"skipping unknown graph: {gname}")
            continue

        for use_comp in (False, True):
            trials = bench_cell(gw, label, use_comp, args.trials, device)

            # Collect bucket-level NUPS matrix
            key = f"{label}_compaction_{int(use_comp)}"
            bucket_arr = np.stack([t["nups_per_bucket"] for t in trials])
            bucket_data[key] = bucket_arr

            mean_nups = float(np.mean([t["total_nups"] / t["total_wall"]
                                       for t in trials]))
            std_nups = float(np.std([t["total_nups"] / t["total_wall"]
                                     for t in trials]))
            mean_walls = float(np.mean([t["total_wall"] for t in trials]))
            final_cts_mean = np.mean(
                [t["final_counts"] for t in trials], axis=0
            ).tolist()

            rows.append({
                "graph": label,
                "N": args.num_nodes,
                "use_compaction": int(use_comp),
                "mean_nups": mean_nups,
                "std_nups": std_nups,
                "mean_wall_s": mean_walls,
                "final_S": final_cts_mean[0],
                "final_E": final_cts_mean[1],
                "final_I": final_cts_mean[2],
                "final_R": final_cts_mean[3],
                "nups_buckets_csv": ",".join(
                    f"{v:.3e}" for v in bucket_arr.mean(axis=0)
                ),
            })

    # Persist
    out_csv = args.output_csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved summary CSV: {out_csv}")

    np.savez(args.output_buckets, num_buckets=NUM_BUCKETS, tf=TF,
             **bucket_data)
    print(f"Saved bucket NPZ: {args.output_buckets}")


if __name__ == "__main__":
    main()
