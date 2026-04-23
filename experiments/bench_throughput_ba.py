#!/usr/bin/env python
"""
Barabási-Albert scale-free network throughput benchmark.

Runs Fused CG at N=10^6 on BA graph (m=4, avg degree ~8) for
apples-to-apples comparison with ER d=8 benchmark.
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from flashspread.core.graph import GraphCSR
from flashspread import SEIRModel
from flashspread.engines.renewal_fused import RenewalEngineFused, RenewalEngineFusedCUDAGraph

BETA = 2.0 / 8
MEAN_EI, MEDIAN_EI = 5.0, 4.0
MEAN_IR, MEDIAN_IR = 7.5, 5.0
EPSILON = 0.03
TAU_MAX = 0.1
TF = 50.0


def bench_engine(engine_class, graph, model, device, trials=5, tf=TF, **kwargs):
    N = graph.num_nodes
    total_nups = 0
    total_time = 0.0

    for trial in range(2 + trials):  # 2 warmup + trials
        class GW:
            pass
        gw = GW()
        gw.csr = graph.csr
        gw.edge_index = graph.edge_index
        gw.num_nodes = graph.num_nodes
        gw.num_edges = graph.num_edges

        engine = engine_class(gw, model, device=device,
                              seed=12345 + trial * 100,
                              epsilon=EPSILON, tau_max=TAU_MAX, **kwargs)
        engine.state[0] = model.infected
        engine.age[0] = 0.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        steps_per_call = getattr(engine, 'steps_per_launch', 1)
        steps = 0
        while engine.current_time < tf:
            engine.step()
            steps += steps_per_call

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if trial >= 2:
            total_nups += N * steps
            total_time += (t1 - t0)

    return total_nups / total_time if total_time > 0 else 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=1000000)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/ba_throughput.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    N = args.num_nodes
    m = args.m

    print(f"BA Scale-Free Throughput Benchmark")
    print(f"N={N:,}, m={m} (avg degree ~{2*m})")
    print(f"Device: {device}")
    print()

    print("Creating BA graph...", flush=True)
    G = nx.barabasi_albert_graph(N, m, seed=42)
    # Convert to directed edge list (both directions for undirected graph)
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(edge_index, N, incoming=True)

    class BAGraph:
        pass
    graph = BAGraph()
    graph.csr = csr
    graph.edge_index = edge_index
    graph.num_nodes = N
    graph.num_edges = edge_index.size(1)
    print(f"  Edges: {graph.num_edges:,}")

    # Report degree statistics
    row_ptr = csr.row_ptr
    degrees = (row_ptr[1:] - row_ptr[:-1]).float()
    print(f"  Degree: mean={degrees.mean():.1f}, max={degrees.max().item()}, "
          f"std={degrees.std():.1f}")

    model = SEIRModel(beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
                      mean_ir=MEAN_IR, median_ir=MEDIAN_IR)

    engines = {
        "Fused eager": (RenewalEngineFused, {}),
        "Fused CG": (RenewalEngineFusedCUDAGraph, {"steps_per_launch": 50}),
    }

    results = {}
    for eng_name, (eng_class, eng_kwargs) in engines.items():
        print(f"\n{eng_name}: ", end="", flush=True)
        try:
            nups = bench_engine(eng_class, graph, model, device,
                                trials=args.trials, **eng_kwargs)
            results[eng_name] = nups
            print(f"{nups:,.0f} NUPS")
        except Exception as e:
            print(f"FAILED: {e}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("engine,graph,N,m,nups\n")
        for eng_name, nups in results.items():
            f.write(f"{eng_name},BA,{N},{m},{nups:.1f}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
