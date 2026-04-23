#!/usr/bin/env python
"""
Multi-topology / multi-size tau-leaping fidelity sweep.

For each (graph type, N) pair we run the fused renewal engine at a few
epsilon values, average over trials, sample the infected-fraction
trajectory on a common time grid, and save to disk.

Epidemic seeding: we infect `max(10, 1% of N)` nodes as Exposed at t=0.
This is deliberately above the single-seed threshold so epidemics
actually take off on every topology and size; we want the fidelity of
the *bulk* of the trajectory, not the early stochastic phase.

Output:
  results/fidelity_multi.npz         — per-(graph,N,eps) mean I(t)/N
  results/fidelity_multi_summary.csv — per-(graph,N,eps) scalar metrics

Downstream plotting in experiments/plot_fidelity_multi.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread import FixedDegreeGraph, SEIRModel
from flashspread.core.graph import GraphCSR
from flashspread.engines.renewal_fused import RenewalEngineFused

# Epidemic parameters matched to the main-text validation benchmark.
MEAN_DEGREE = 8
BETA_OVER_DEG = 2.0              # beta = BETA_OVER_DEG / mean_degree = 0.25
MEAN_EI, MEDIAN_EI = 5.0, 4.0
MEAN_IR, MEDIAN_IR = 7.5, 5.0
TAU_MAX = 0.1
TF = 50.0
SAMPLE_TIMES = np.linspace(0.0, TF, int(TF / 0.5) + 1)  # 0, 0.5, ..., 50 (101 pts)

# Default sweep configuration (overridable from CLI).
DEFAULT_N_LIST = [1_000, 10_000, 100_000]
DEFAULT_EPSILONS = [0.005, 0.03, 0.1]
DEFAULT_GRAPHS = ["er", "ba", "fixed"]
DEFAULT_TRIALS = 20


def _build_gw(csr, edge_index, N):
    class GW: pass
    gw = GW()
    gw.csr = csr
    gw.edge_index = edge_index
    gw.num_nodes = N
    gw.num_edges = edge_index.size(1)
    return gw


def build_er(N, d, device, seed=42):
    import networkx as nx
    G = nx.erdos_renyi_graph(N, d / N, seed=seed, directed=False)
    edges = []
    for u, v in G.edges():
        edges.append([u, v]); edges.append([v, u])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(ei, N, incoming=True)
    return _build_gw(csr, ei, N)


def build_ba(N, m, device, seed=42):
    import networkx as nx
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    edges = []
    for u, v in G.edges():
        edges.append([u, v]); edges.append([v, u])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(ei, N, incoming=True)
    return _build_gw(csr, ei, N)


def build_fixed(N, d, device, seed=42):
    g = FixedDegreeGraph(N, d, device=device)
    return _build_gw(g.csr, g.edge_index, N)


GRAPH_BUILDERS = {
    "er": lambda N, device, seed: build_er(N, MEAN_DEGREE, device, seed),
    "ba": lambda N, device, seed: build_ba(N, MEAN_DEGREE // 2, device, seed),
    "fixed": lambda N, device, seed: build_fixed(N, MEAN_DEGREE, device, seed),
}
GRAPH_LABELS = {
    "er": r"Erd\H{o}s--R\'{e}nyi $d{=}8$",
    "ba": r"Barab\'{a}si--Albert $m{=}4$",
    "fixed": r"Fixed-degree $d{=}8$",
}


def seed_count(N, frac=0.01, floor=10):
    """Number of initially-Exposed nodes. Big enough to avoid fade-out."""
    return max(floor, int(frac * N))


def run_one_trajectory(gw, model, device, seed, epsilon, initial_E):
    engine = RenewalEngineFused(
        gw, model, device=device, seed=seed,
        epsilon=epsilon, tau_max=TAU_MAX,
    )
    # Seed `initial_E` Exposed nodes deterministically (indices [0, initial_E)).
    idx = torch.arange(initial_E, device=device)
    engine.state[idx] = model.exposed
    engine.age[idx] = 0.0
    engine._infectivity_prepass()

    # Walk the simulation and sample I(t)/N on SAMPLE_TIMES grid.
    sample_count = len(SAMPLE_TIMES)
    infected_frac = np.zeros(sample_count, dtype=np.float32)
    # t=0 initial infected fraction (I-state count; seed is E so this starts 0).
    counts = engine.count_by_state().cpu().numpy()
    infected_frac[0] = counts[2] / gw.num_nodes

    idx_next = 1
    while engine.current_time < TF and idx_next < sample_count:
        engine.step()
        while (idx_next < sample_count
               and SAMPLE_TIMES[idx_next] <= engine.current_time):
            counts = engine.count_by_state().cpu().numpy()
            infected_frac[idx_next] = counts[2] / gw.num_nodes
            idx_next += 1
    while idx_next < sample_count:
        counts = engine.count_by_state().cpu().numpy()
        infected_frac[idx_next] = counts[2] / gw.num_nodes
        idx_next += 1
    return infected_frac


def run_sweep(graph_key, N, device, epsilons, trials, graph_seed):
    builder = GRAPH_BUILDERS[graph_key]
    gw = builder(N, device, graph_seed)
    model = SEIRModel(
        beta=BETA_OVER_DEG / MEAN_DEGREE,
        mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
        mean_ir=MEAN_IR, median_ir=MEDIAN_IR,
    )
    initial_E = seed_count(N)
    results_for_eps = {}
    for eps in epsilons:
        ensemble = np.zeros((trials, len(SAMPLE_TIMES)), dtype=np.float32)
        walls = np.zeros(trials, dtype=np.float32)
        for r in range(trials):
            seed = 12345 + r * 7919
            t0 = time.perf_counter()
            infected_frac = run_one_trajectory(
                gw, model, device, seed, eps, initial_E
            )
            t1 = time.perf_counter()
            ensemble[r] = infected_frac
            walls[r] = t1 - t0
        mean_traj = ensemble.mean(axis=0)
        q25 = np.percentile(ensemble, 25, axis=0)
        q75 = np.percentile(ensemble, 75, axis=0)
        results_for_eps[eps] = {
            "mean": mean_traj, "q25": q25, "q75": q75,
            "wall_mean": float(walls.mean()),
            "peak_I": float(mean_traj.max()),
            "peak_t": float(SAMPLE_TIMES[np.argmax(mean_traj)]),
        }
    # Topology stats (for labelling).
    degrees = (gw.csr.row_ptr[1:] - gw.csr.row_ptr[:-1]).float()
    d_mean = float(degrees.mean().item())
    d_max = float(degrees.max().item())
    return {
        "graph": graph_key, "N": N,
        "initial_E": initial_E,
        "d_mean": d_mean, "d_max": d_max,
        "epsilons": epsilons,
        "eps_results": results_for_eps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", default=",".join(DEFAULT_GRAPHS))
    parser.add_argument("--sizes", default="1000,10000,100000")
    parser.add_argument("--epsilons", default="0.005,0.03,0.1")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph-seed", type=int, default=42)
    parser.add_argument("--output-npz",
                        default="results/fidelity_multi.npz")
    parser.add_argument("--output-csv",
                        default="results/fidelity_multi_summary.csv")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    epsilons = [float(x) for x in args.epsilons.split(",") if x.strip()]

    print(f"Multi-topology fidelity sweep (trials={args.trials}, "
          f"device={device})")
    print(f"  graphs:    {graphs}")
    print(f"  sizes:     {sizes}")
    print(f"  epsilons:  {epsilons}")
    print(f"  sample grid: {len(SAMPLE_TIMES)} pts, dt={SAMPLE_TIMES[1]-SAMPLE_TIMES[0]:.2f}")

    all_results = []
    for graph_key in graphs:
        for N in sizes:
            print(f"\n== {graph_key:6s}  N={N:>7,} ==", flush=True)
            t0 = time.perf_counter()
            r = run_sweep(graph_key, N, device, epsilons, args.trials,
                          args.graph_seed)
            print(f"  d_mean={r['d_mean']:.2f}  d_max={r['d_max']:.0f}  "
                  f"initial_E={r['initial_E']}", flush=True)
            for eps in epsilons:
                er = r["eps_results"][eps]
                print(f"  eps={eps:<6}: peak_I={er['peak_I']:.4f} at "
                      f"t={er['peak_t']:.1f}  wall={er['wall_mean']:.3f}s", flush=True)
            print(f"  (sweep took {time.perf_counter()-t0:.1f}s)")
            all_results.append(r)

    # Save NPZ (flatten for easy plotting).
    npz_payload = {
        "sample_times": SAMPLE_TIMES,
        "graphs": np.array(graphs),
        "sizes": np.array(sizes, dtype=np.int64),
        "epsilons": np.array(epsilons, dtype=np.float32),
    }
    for r in all_results:
        for eps in epsilons:
            er = r["eps_results"][eps]
            key = f"{r['graph']}_N{r['N']}_eps{eps:g}"
            npz_payload[f"mean_{key}"] = er["mean"]
            npz_payload[f"q25_{key}"] = er["q25"]
            npz_payload[f"q75_{key}"] = er["q75"]
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **npz_payload)
    print(f"\nWrote: {args.output_npz}")

    # Save summary CSV.
    with open(args.output_csv, "w") as f:
        f.write("graph,N,d_mean,d_max,initial_E,epsilon,peak_I,peak_t,wall_mean\n")
        for r in all_results:
            for eps in epsilons:
                er = r["eps_results"][eps]
                f.write(
                    f"{r['graph']},{r['N']},{r['d_mean']:.4f},{r['d_max']:.0f},"
                    f"{r['initial_E']},{eps},"
                    f"{er['peak_I']:.6f},{er['peak_t']:.3f},{er['wall_mean']:.4f}\n"
                )
    print(f"Wrote: {args.output_csv}")


if __name__ == "__main__":
    main()
