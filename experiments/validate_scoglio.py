#!/usr/bin/env python
"""
Validation against the Scoglio MATLAB non-Markovian SEIR simulator.

Reproduces the MATLAB main_1_q.m setup:
  - ER random graph: N=50000, p=15/N (avg degree 15)
  - SEIR with beta = 3/15 = 0.2
  - E->I: LogNormal(mean=5, median=4)
  - I->R: LogNormal(mean=3.9, median=1.5)
  - 1 initial infected node (state=Infected, i.e. state 3 in MATLAB)
  - Simulate until t=20

Compares three FlashSpread engines:
  1. RenewalEngine (Markovian edges, original)
  2. RenewalEngineNonMarkov (age-dependent edges, source-node compromise)
  3. RenewalEngineFused (fused Triton kernel)

Produces a 2x2 plot of SEIR compartment trajectories (mean ± std over
multiple seeds) for each engine, saved to results/validation_scoglio.png.
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
from flashspread.engines.renewal import (
    RenewalEngine,
    RenewalEngineNonMarkov,
)
from flashspread.engines.renewal_fused import RenewalEngineFused


def create_er_graph(N: int, p: float, device: str) -> GraphCSR:
    """Create Erdos-Renyi graph matching MATLAB's NetGen_ER_nonM."""
    G = nx.erdos_renyi_graph(N, p, seed=42, directed=False)
    # Convert to directed edge list (both directions)
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    return GraphCSR(edge_index, N, incoming=True), edge_index


def run_engine(engine_class, graph_csr, edge_index, model, device, seed,
               target_time=20.0, record_dt=0.1, **engine_kwargs):
    """Run one simulation, recording compartment counts at regular intervals."""
    # Build graph object that has both .csr and .edge_index
    class GraphWrap:
        pass
    gw = GraphWrap()
    gw.csr = graph_csr
    gw.edge_index = edge_index
    gw.num_nodes = graph_csr.num_nodes
    gw.num_edges = graph_csr.num_edges

    engine = engine_class(gw, model, device=device, seed=seed,
                          epsilon=0.03, tau_max=1.0, **engine_kwargs)

    # MATLAB starts with 1 node in state I (state=3 in MATLAB = state 2 = Infected)
    engine.state[0] = model.infected
    engine.age[0] = 0.0

    N = graph_csr.num_nodes
    times = []
    S, E, I, R = [], [], [], []

    # Record initial state
    counts = engine.count_by_state()
    times.append(0.0)
    S.append(counts[0].item())
    E.append(counts[1].item())
    I.append(counts[2].item())
    R.append(counts[3].item())

    next_record = record_dt

    while engine.current_time < target_time:
        engine.step()

        if engine.current_time >= next_record:
            counts = engine.count_by_state()
            times.append(engine.current_time)
            S.append(counts[0].item())
            E.append(counts[1].item())
            I.append(counts[2].item())
            R.append(counts[3].item())
            next_record += record_dt

    return {
        "time": np.array(times),
        "S": np.array(S) / N,
        "E": np.array(E) / N,
        "I": np.array(I) / N,
        "R": np.array(R) / N,
    }


def interpolate_to_grid(traj, t_grid):
    """Interpolate trajectory to a common time grid."""
    out = {}
    for key in ["S", "E", "I", "R"]:
        out[key] = np.interp(t_grid, traj["time"], traj[key])
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=50000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--target-time", type=float, default=20.0)
    parser.add_argument("--output", type=str, default="results/validation_scoglio.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    N = args.num_nodes
    p = 15.0 / N
    beta = 3.0 / 15.0  # = 0.2, matching MATLAB

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Scoglio Validation: N={N}, p={p:.6f}, beta={beta:.4f}")
    print(f"Device: {device}")
    print(f"Seeds: {args.num_seeds}")
    print()

    # Create graph once (shared across engines and seeds)
    print("Creating ER graph...", flush=True)
    graph_csr, edge_index = create_er_graph(N, p, device)
    print(f"  Nodes: {N}, Edges: {graph_csr.num_edges}")

    model = SEIRModel(
        beta=beta,
        mean_ei=5.0, median_ei=4.0,
        mean_ir=3.9, median_ir=1.5,
    )

    # Common time grid for interpolation
    t_grid = np.linspace(0, args.target_time, 200)

    engines = {
        "RenewalEngine\n(Markovian edges)": (RenewalEngine, {}),
        "RenewalEngineNonMarkov\n(age-dependent edges)": (RenewalEngineNonMarkov, {}),
        "RenewalEngineFused\n(fused Triton kernel)": (RenewalEngineFused, {}),
    }

    all_results = {}
    for eng_name, (eng_class, eng_kwargs) in engines.items():
        print(f"\n--- {eng_name.replace(chr(10), ' ')} ---")
        trajs = {k: [] for k in ["S", "E", "I", "R"]}

        for seed_idx in range(args.num_seeds):
            seed = 12345 + seed_idx * 7919
            t0 = time.time()
            try:
                traj = run_engine(
                    eng_class, graph_csr, edge_index, model, device, seed,
                    target_time=args.target_time, **eng_kwargs,
                )
                interp = interpolate_to_grid(traj, t_grid)
                for k in ["S", "E", "I", "R"]:
                    trajs[k].append(interp[k])
                dt = time.time() - t0
                print(f"  seed {seed_idx:2d}: {dt:.1f}s  "
                      f"peak_I={traj['I'].max():.4f}  "
                      f"final_R={traj['R'][-1]:.4f}", flush=True)
            except Exception as e:
                print(f"  seed {seed_idx:2d}: FAILED - {e}")

        if trajs["S"]:
            all_results[eng_name] = {
                k: (np.mean(trajs[k], axis=0), np.std(trajs[k], axis=0))
                for k in ["S", "E", "I", "R"]
            }

    # Plot
    print("\nGenerating plot...", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        compartments = ["S", "E", "I", "R"]
        colors = {"S": "#1f77b4", "E": "#ff7f0e", "I": "#2ca02c", "R": "#d62728"}
        titles = {
            "S": "Susceptible",
            "E": "Exposed",
            "I": "Infectious",
            "R": "Recovered",
        }

        n_engines = len(all_results)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        fig.suptitle(
            f"Scoglio Validation: SEIR on ER Graph (N={N}, d=15, β={beta})\n"
            f"Mean ± 1σ over {args.num_seeds} seeds",
            fontsize=14, fontweight="bold",
        )

        linestyles = ["-", "--", ":"]
        engine_names = list(all_results.keys())

        for idx, comp in enumerate(compartments):
            ax = axes[idx // 2][idx % 2]
            for eng_idx, eng_name in enumerate(engine_names):
                mean, std = all_results[eng_name][comp]
                ls = linestyles[eng_idx % len(linestyles)]
                label = eng_name.replace("\n", " ")
                ax.plot(t_grid, mean, ls, color=colors[comp], alpha=0.6 + 0.2 * (eng_idx == 0),
                        linewidth=2 - 0.3 * eng_idx, label=label)
                ax.fill_between(t_grid, mean - std, mean + std,
                                color=colors[comp], alpha=0.08)

            ax.set_title(titles[comp], fontsize=13, fontweight="bold")
            ax.set_ylabel("Fraction of population")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, args.target_time)
            if idx >= 2:
                ax.set_xlabel("Time (days)")
            if idx == 0:
                ax.legend(fontsize=8, loc="best")

        plt.tight_layout()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output}")

        # Also print a summary table
        print("\n" + "=" * 70)
        print(f"{'Engine':<35s} {'Peak I':>8s} {'Final R':>8s} {'Attack%':>8s}")
        print("-" * 70)
        for eng_name in engine_names:
            mean_I = all_results[eng_name]["I"][0]
            mean_R = all_results[eng_name]["R"][0]
            peak_I = mean_I.max()
            final_R = mean_R[-1]
            label = eng_name.replace("\n", " ")
            print(f"{label:<35s} {peak_I:>8.4f} {final_R:>8.4f} {final_R*100:>7.1f}%")
        print("=" * 70)

    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    main()
