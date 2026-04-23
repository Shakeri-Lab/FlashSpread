#!/usr/bin/env python
"""
Age-dependent vs constant source-node infectivity throughput benchmark.

Validates the paper's claim (JOCS §5.3, §8) that switching
transmission_mode='constant' -> 'age_dependent' incurs zero (or
near-zero) throughput cost in the fused CUDA Graph engine, because
the extra float multiply executes in the shadow of CSR memory
latency.

Runs Fused CG (steps_per_launch=50) at N=10^6 on:
  - degree-8 FixedDegreeGraph (ER proxy)
  - BA(m=4) scale-free graph (avg degree ~8)

For each graph, reports NUPS in both transmission modes and the
relative overhead (age_dependent / constant - 1).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread import FixedDegreeGraph, SEIRModel
from flashspread.core.graph import GraphCSR
from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph

BETA = 2.0 / 8.0
MEAN_EI, MEDIAN_EI = 5.0, 4.0
MEAN_IR, MEDIAN_IR = 7.5, 5.0
EPSILON = 0.03
TAU_MAX = 0.1
TF = 50.0
STEPS_PER_LAUNCH = 50


def _make_graph_wrapper(csr, edge_index, num_nodes):
    class GW:
        pass
    gw = GW()
    gw.csr = csr
    gw.edge_index = edge_index
    gw.num_nodes = num_nodes
    gw.num_edges = edge_index.size(1)
    return gw


def bench_mode(graph_wrapper, model, mode, device, trials=5, tf=TF):
    """Run trials of Fused CG with a given transmission_mode; return NUPS."""
    model.transmission_mode = mode
    N = graph_wrapper.num_nodes
    total_nups = 0
    total_time = 0.0

    for trial in range(2 + trials):  # 2 warmup + trials
        engine = RenewalEngineFusedCUDAGraph(
            graph_wrapper, model, device=device,
            seed=12345 + trial * 100,
            epsilon=EPSILON, tau_max=TAU_MAX,
            steps_per_launch=STEPS_PER_LAUNCH,
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


def bench_graph(name, graph_wrapper, model, device, trials):
    print(f"\n== Graph: {name} (N={graph_wrapper.num_nodes:,}, "
          f"E={graph_wrapper.num_edges:,}) ==", flush=True)

    nups_const = bench_mode(graph_wrapper, model, "constant", device, trials=trials)
    print(f"  constant       : {nups_const:,.0f} NUPS")

    nups_age = bench_mode(graph_wrapper, model, "age_dependent", device, trials=trials)
    print(f"  age_dependent  : {nups_age:,.0f} NUPS")

    overhead = (nups_const / nups_age - 1.0) if nups_age > 0 else float("nan")
    print(f"  relative slowdown (age_dep vs const): {overhead*100:+.2f}%")

    return nups_const, nups_age, overhead


def build_er_proxy(N, d, device):
    graph = FixedDegreeGraph(N, d, device=device)
    return _make_graph_wrapper(graph.csr, graph.edge_index, N)


def build_ba(N, m, device):
    import networkx as nx
    print(f"  Building BA(N={N}, m={m}) via networkx...", flush=True)
    G = nx.barabasi_albert_graph(N, m, seed=42)
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(edge_index, N, incoming=True)
    return _make_graph_wrapper(csr, edge_index, N)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=1_000_000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graphs", type=str, default="er,ba",
                        help="Comma-separated list: er,ba")
    parser.add_argument("--output", type=str,
                        default="results/age_dependent_throughput.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    N = args.num_nodes
    requested = [g.strip() for g in args.graphs.split(",") if g.strip()]

    print("Age-Dependent vs Constant Infectivity Throughput Benchmark")
    print(f"N={N:,}  tf={TF}  epsilon={EPSILON}  steps_per_launch={STEPS_PER_LAUNCH}")
    print(f"Device: {device}")

    model = SEIRModel(
        beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
        mean_ir=MEAN_IR, median_ir=MEDIAN_IR,
    )

    rows = []
    for gname in requested:
        if gname == "er":
            gw = build_er_proxy(N, 8, device)
            label = "ER-proxy (FixedDegree d=8)"
        elif gname == "ba":
            gw = build_ba(N, 4, device)
            label = "BA (m=4)"
        else:
            print(f"  [skip] unknown graph '{gname}'")
            continue

        nups_const, nups_age, overhead = bench_graph(
            label, gw, model, device, trials=args.trials
        )
        rows.append((label, N, nups_const, nups_age, overhead))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("graph,N,nups_constant,nups_age_dependent,rel_slowdown\n")
        for row in rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]:.1f},{row[3]:.1f},{row[4]:.6f}\n"
            )
    print(f"\nSaved: {args.output}")

    print("\n=== SUMMARY ===")
    for label, n, nc, na, ov in rows:
        print(f"  {label:30s}  const={nc:10,.0f}  age_dep={na:10,.0f}  "
              f"slowdown={ov*100:+.2f}%")


if __name__ == "__main__":
    main()
