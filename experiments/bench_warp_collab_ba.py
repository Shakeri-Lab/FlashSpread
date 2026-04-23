#!/usr/bin/env python
"""
Benchmark: warp-collaborative kernel vs 1-thread-per-node kernel on BA.

Reports Fused CG throughput (NUPS) for both kernels on:
  - Regular d=8 graph (baseline; warp-collab expected near-parity)
  - BA m=4          (target; warp-collab expected to deliver 3-10x)

Also sweeps (NODES_PER_BLOCK) to pick a good block shape on BA.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread import FixedDegreeGraph, SEIRModel
from flashspread.core.graph import GraphCSR
from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph

BETA = 2.0 / 8.0
EPSILON = 0.03
TAU_MAX = 0.1
TF = 50.0
STEPS_PER_LAUNCH = 50


def _make_gw(csr, edge_index, N):
    class GW: pass
    gw = GW()
    gw.csr = csr
    gw.edge_index = edge_index
    gw.num_nodes = N
    gw.num_edges = edge_index.size(1)
    return gw


def _build_er(N, device):
    g = FixedDegreeGraph(N, 8, device=device)
    return _make_gw(g.csr, g.edge_index, N)


def _build_ba(N, m, device):
    import networkx as nx
    G = nx.barabasi_albert_graph(N, m, seed=42)
    edges = []
    for u, v in G.edges():
        edges.append([u, v]); edges.append([v, u])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(ei, N, incoming=True)
    return _make_gw(csr, ei, N)


def bench_mode(gw, model, device, strategy, nodes_per_block=8,
               edges_per_merge_block=4096, trials=5, tf=TF):
    N = gw.num_nodes
    total_nups = 0.0
    total_time = 0.0

    for trial in range(2 + trials):
        engine = RenewalEngineFusedCUDAGraph(
            gw, model, device=device,
            seed=1234 + trial * 100,
            epsilon=EPSILON, tau_max=TAU_MAX,
            steps_per_launch=STEPS_PER_LAUNCH,
            csr_strategy=strategy,
            nodes_per_block=nodes_per_block,
            lanes_per_node=32,
            edges_per_merge_block=edges_per_merge_block,
        )
        engine.state[0] = model.infected
        engine.age[0] = 0.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        steps = 0
        while engine.current_time < tf:
            engine.step()
            steps += STEPS_PER_LAUNCH
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if trial >= 2:
            total_nups += N * steps
            total_time += (t1 - t0)

    return total_nups / total_time if total_time > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=1_000_000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graphs", type=str, default="er,ba")
    parser.add_argument("--sweep", type=str, default="2,4,8",
                        help="Comma-separated NODES_PER_BLOCK values to try")
    parser.add_argument("--output", type=str,
                        default="results/warp_collab_throughput.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    N = args.num_nodes
    requested = [g.strip() for g in args.graphs.split(",") if g.strip()]
    npb_sweep = [int(x) for x in args.sweep.split(",") if x.strip()]

    print("Warp-collaborative kernel throughput comparison")
    print(f"N={N:,}  tf={TF}  epsilon={EPSILON}  steps_per_launch={STEPS_PER_LAUNCH}")
    print(f"Device: {device}  NPB sweep: {npb_sweep}\n")

    model = SEIRModel(
        beta=BETA,
        mean_ei=5.0, median_ei=4.0,
        mean_ir=7.5, median_ir=5.0,
    )

    rows = []
    for gname in requested:
        if gname == "er":
            gw = _build_er(N, device)
            label = "FixedDegree d=8"
        elif gname == "ba":
            gw = _build_ba(N, 4, device)
            label = "BA m=4"
        else:
            print(f"  [skip] unknown graph '{gname}'"); continue

        print(f"== Graph: {label} (N={gw.num_nodes:,}, E={gw.num_edges:,}) ==", flush=True)

        degrees_t = (gw.csr.row_ptr[1:] - gw.csr.row_ptr[:-1]).float()
        dmax, dmean = degrees_t.max().item(), degrees_t.mean().item()
        print(f"  Degree stats: mean={dmean:.2f}  max={dmax:.0f}  ratio={dmax/dmean:.1f}")

        baseline = bench_mode(gw, model, device,
                              strategy="thread", trials=args.trials)
        print(f"  thread         : {baseline:,.0f} NUPS  (baseline)")
        rows.append((label, "thread", 0, 0, baseline))

        for npb in npb_sweep:
            wc = bench_mode(gw, model, device,
                            strategy="warp", nodes_per_block=npb,
                            trials=args.trials)
            ratio = wc / baseline if baseline > 0 else float("nan")
            print(f"  warp NPB={npb:<2d}    : {wc:,.0f} NUPS  "
                  f"({ratio:.2f}x vs thread)")
            rows.append((label, "warp", npb, 0, wc))

        for epb in [1024, 2048, 4096, 8192]:
            try:
                me = bench_mode(gw, model, device,
                                strategy="merge", edges_per_merge_block=epb,
                                trials=args.trials)
            except Exception as e:
                print(f"  merge EPB={epb:<5d} FAILED: {e}")
                continue
            ratio = me / baseline if baseline > 0 else float("nan")
            print(f"  merge EPB={epb:<5d}: {me:,.0f} NUPS  "
                  f"({ratio:.2f}x vs thread)")
            rows.append((label, "merge", 0, epb, me))
        print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("graph,kernel,nodes_per_block,edges_per_merge_block,nups\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]:.1f}\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
