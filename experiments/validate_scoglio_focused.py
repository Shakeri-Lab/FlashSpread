#!/usr/bin/env python
"""
Focused validation: NonMarkov vs Fused engines only.

Uses more initial infections (100 exposed) to ensure a visible epidemic
with the age-dependent transmission model, then compares the two
non-Markovian engines at proper scale.
"""

import sys
import time
from pathlib import Path
import numpy as np

import torch
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread.core.graph import GraphCSR
from flashspread import SEIRModel
from flashspread.engines.renewal import RenewalEngineNonMarkov
from flashspread.engines.renewal_fused import RenewalEngineFused


def create_er_graph(N, p, device, seed=42):
    G = nx.erdos_renyi_graph(N, p, seed=seed, directed=False)
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    return GraphCSR(edge_index, N, incoming=True), edge_index


def run_engine(engine_class, graph_csr, edge_index, model, device, seed,
               initial_exposed=100, target_time=30.0, record_dt=0.05,
               **engine_kwargs):
    class GraphWrap:
        pass
    gw = GraphWrap()
    gw.csr = graph_csr
    gw.edge_index = edge_index
    gw.num_nodes = graph_csr.num_nodes
    gw.num_edges = graph_csr.num_edges

    engine = engine_class(gw, model, device=device, seed=seed,
                          epsilon=0.03, tau_max=1.0, **engine_kwargs)

    # Seed initial exposed (not infected — let incubation play out)
    torch.manual_seed(seed)
    indices = torch.randperm(graph_csr.num_nodes, device=device)[:initial_exposed]
    engine.state[indices] = model.exposed
    engine.age[indices] = 0.0

    N = graph_csr.num_nodes
    times, S, E, I, R = [0.0], [], [], [], []
    counts = engine.count_by_state()
    S.append(counts[0].item() / N)
    E.append(counts[1].item() / N)
    I.append(counts[2].item() / N)
    R.append(counts[3].item() / N)

    next_record = record_dt
    while engine.current_time < target_time:
        engine.step()
        if engine.current_time >= next_record:
            counts = engine.count_by_state()
            times.append(engine.current_time)
            S.append(counts[0].item() / N)
            E.append(counts[1].item() / N)
            I.append(counts[2].item() / N)
            R.append(counts[3].item() / N)
            next_record += record_dt

    return {k: np.array(v) for k, v in
            [("time", times), ("S", S), ("E", E), ("I", I), ("R", R)]}


def interpolate_to_grid(traj, t_grid):
    return {k: np.interp(t_grid, traj["time"], traj[k]) for k in ["S", "E", "I", "R"]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=50000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--initial-exposed", type=int, default=100)
    parser.add_argument("--target-time", type=float, default=30.0)
    parser.add_argument("--output", type=str, default="results/validation_scoglio_focused.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    N = args.num_nodes
    p = 15.0 / N
    beta = 3.0 / 15.0

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Focused Scoglio Validation: NonMarkov vs Fused")
    print(f"N={N}, d=15, β={beta}, initial_exposed={args.initial_exposed}")
    print(f"Device: {device}, Seeds: {args.num_seeds}")
    print()

    print("Creating ER graph...", flush=True)
    graph_csr, edge_index = create_er_graph(N, p, device)
    print(f"  Nodes: {N}, Edges: {graph_csr.num_edges}")

    model = SEIRModel(beta=beta, mean_ei=5.0, median_ei=4.0,
                      mean_ir=3.9, median_ir=1.5)

    t_grid = np.linspace(0, args.target_time, 300)

    engines = {
        "NonMarkov (age-dependent edges)": (RenewalEngineNonMarkov, {}),
        "Fused Triton kernel": (RenewalEngineFused, {}),
    }

    all_results = {}
    for eng_name, (eng_class, eng_kwargs) in engines.items():
        print(f"\n--- {eng_name} ---")
        trajs = {k: [] for k in ["S", "E", "I", "R"]}

        for seed_idx in range(args.num_seeds):
            seed = 12345 + seed_idx * 7919
            t0 = time.time()
            try:
                traj = run_engine(
                    eng_class, graph_csr, edge_index, model, device, seed,
                    initial_exposed=args.initial_exposed,
                    target_time=args.target_time, **eng_kwargs,
                )
                interp = interpolate_to_grid(traj, t_grid)
                for k in ["S", "E", "I", "R"]:
                    trajs[k].append(interp[k])
                dt = time.time() - t0
                peak_I = traj["I"].max()
                final_R = traj["R"][-1]
                print(f"  seed {seed_idx:2d}: {dt:.1f}s  peak_I={peak_I:.4f}  final_R={final_R:.4f}", flush=True)
            except Exception as e:
                print(f"  seed {seed_idx:2d}: FAILED - {e}")

        if trajs["S"]:
            all_results[eng_name] = {
                k: (np.mean(trajs[k], axis=0), np.std(trajs[k], axis=0))
                for k in ["S", "E", "I", "R"]
            }

    # Plot
    print("\nGenerating plot...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compartments = ["S", "E", "I", "R"]
    colors_map = {
        "NonMarkov (age-dependent edges)": "#2ca02c",
        "Fused Triton kernel": "#d62728",
    }
    linestyles_map = {
        "NonMarkov (age-dependent edges)": "-",
        "Fused Triton kernel": "--",
    }
    comp_colors = {"S": "#1f77b4", "E": "#ff7f0e", "I": "#2ca02c", "R": "#d62728"}
    titles = {"S": "Susceptible", "E": "Exposed", "I": "Infectious", "R": "Recovered"}

    engine_names = list(all_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Non-Markovian Engine Validation: SEIR on ER Graph\n"
        f"N={N}, d=15, β={beta}, {args.initial_exposed} initial exposed, "
        f"mean ± 1σ over {args.num_seeds} seeds",
        fontsize=13, fontweight="bold",
    )

    for idx, comp in enumerate(compartments):
        ax = axes[idx // 2][idx % 2]

        for eng_name in engine_names:
            mean, std = all_results[eng_name][comp]
            color = colors_map[eng_name]
            ls = linestyles_map[eng_name]
            ax.plot(t_grid, mean, ls, color=color, linewidth=2, label=eng_name)
            ax.fill_between(t_grid, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_title(titles[comp], fontsize=13, fontweight="bold", color=comp_colors[comp])
        ax.set_ylabel("Fraction of population")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, args.target_time)
        if idx >= 2:
            ax.set_xlabel("Time (days)")
        if idx == 0:
            ax.legend(fontsize=10, loc="best")

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")

    # Summary
    print("\n" + "=" * 65)
    print(f"{'Engine':<40s} {'Peak I':>8s} {'Final R':>8s} {'Attack%':>8s}")
    print("-" * 65)
    for eng_name in engine_names:
        mean_I = all_results[eng_name]["I"][0]
        mean_R = all_results[eng_name]["R"][0]
        print(f"{eng_name:<40s} {mean_I.max():>8.4f} {mean_R[-1]:>8.4f} {mean_R[-1]*100:>7.1f}%")
    print("=" * 65)

    # Quantify agreement between the two engines
    if len(engine_names) == 2:
        print("\nAgreement between engines:")
        for comp in compartments:
            m1 = all_results[engine_names[0]][comp][0]
            m2 = all_results[engine_names[1]][comp][0]
            max_diff = np.abs(m1 - m2).max()
            mean_diff = np.abs(m1 - m2).mean()
            print(f"  {comp}: max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")


if __name__ == "__main__":
    main()
