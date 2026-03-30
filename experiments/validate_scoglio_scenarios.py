#!/usr/bin/env python
"""
Recreate the Scoglio MATLAB validation (log_normal_SEIR.png) with new engines.

Matches MATLAB scoglio.m exactly:
  - N=1000, ER graph with avg degree 8
  - 1 initial infected (node 0, state=Infected)
  - E->I and I->R BOTH use the same LogNormal(mean, median) globals
  - Simulate to t=50, 100 runs averaged
  - 3 scenarios with different (median, mean, beta) combinations

Compares:
  - RenewalEngine (Py): original, already validated against MATLAB
  - RenewalEngineNonMarkov (NM): age-dependent edges, source-node compromise
  - RenewalEngineFused (Fused): fused Triton kernel

Output: results/validation_scoglio_scenarios.png
"""

import sys
import time
from pathlib import Path
import numpy as np

import torch
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread.core.graph import GraphCSR
from flashspread.engines.renewal import RenewalEngine, RenewalEngineNonMarkov
from flashspread.engines.renewal_fused import RenewalEngineFused
from flashspread.models.compartmental import SEIRModel


# ============================================================
# Three scenarios matching the existing log_normal_SEIR.png
# In MATLAB scoglio.m, both E->I and I->R share (mean, median)
# ============================================================
SCENARIOS = [
    {"name": "S1", "median": 1, "mean": 1.1, "beta_per_link": 0.8},
    {"name": "S2", "median": 3, "mean": 3.3, "beta_per_link": 0.4},
    {"name": "S3", "median": 5, "mean": 5.5, "beta_per_link": 0.24},
]


def create_er_graph(N, avg_degree, device, seed=42):
    """ER graph matching MATLAB NetGen_ER_nonM."""
    p = avg_degree / N
    G = nx.erdos_renyi_graph(N, p, seed=seed, directed=False)
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(edge_index, N, incoming=True)
    return csr, edge_index


class SEIRModelSharedHazard(SEIRModel):
    """
    SEIR model where E->I and I->R share the same LogNormal parameters.
    Matches MATLAB scoglio.m which uses global mean/median for both.
    """
    def __init__(self, beta, mean, median):
        super().__init__(
            beta=beta,
            mean_ei=mean, median_ei=median,
            mean_ir=mean, median_ir=median,
        )


def run_one(engine_class, graph_csr, edge_index, model, device, seed,
            target_time=50.0, record_dt=0.5, **engine_kwargs):
    """Run one simulation, return trajectory dict."""
    class GW:
        pass
    gw = GW()
    gw.csr = graph_csr
    gw.edge_index = edge_index
    gw.num_nodes = graph_csr.num_nodes
    gw.num_edges = graph_csr.num_edges

    engine = engine_class(gw, model, device=device, seed=seed,
                          epsilon=0.03, tau_max=1.0, **engine_kwargs)

    # 1 initial infected (matching MATLAB x0(1)=3, state 3 = Infected)
    engine.state[0] = model.infected
    engine.age[0] = 0.0

    N = graph_csr.num_nodes
    times = [0.0]
    counts = engine.count_by_state()
    S = [counts[0].item() / N]
    E = [counts[1].item() / N]
    I = [counts[2].item() / N]
    R = [counts[3].item() / N]

    next_rec = record_dt
    while engine.current_time < target_time:
        engine.step()
        if engine.current_time >= next_rec:
            counts = engine.count_by_state()
            times.append(engine.current_time)
            S.append(counts[0].item() / N)
            E.append(counts[1].item() / N)
            I.append(counts[2].item() / N)
            R.append(counts[3].item() / N)
            next_rec += record_dt

    return {k: np.array(v) for k, v in
            [("time", times), ("S", S), ("E", E), ("I", I), ("R", R)]}


def interpolate(traj, t_grid):
    return {k: np.interp(t_grid, traj["time"], traj[k]) for k in ["S", "E", "I", "R"]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=1000)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--target-time", type=float, default=50.0)
    parser.add_argument("--output", type=str, default="results/validation_scoglio_scenarios.png")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    N = args.num_nodes
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Scoglio Scenarios Validation")
    print(f"N={N}, degree={args.degree}, runs={args.num_runs}, t_max={args.target_time}")
    print(f"Device: {device}")
    print()

    # Create graph once
    print("Creating ER graph...", flush=True)
    graph_csr, edge_index = create_er_graph(N, args.degree, device)
    print(f"  Edges: {graph_csr.num_edges}")

    t_grid = np.linspace(0, args.target_time, 200)

    engines = {
        "Py": (RenewalEngine, {}),
        "NM": (RenewalEngineNonMarkov, {}),
        "Fused": (RenewalEngineFused, {}),
    }

    # results[scenario_name][engine_name][compartment] = (mean, std)
    results = {}

    for sc in SCENARIOS:
        sc_name = sc["name"]
        median_val = sc["median"]
        mean_val = sc["mean"]
        beta = sc["beta_per_link"] / args.degree

        print(f"\n{'='*60}")
        print(f"Scenario {sc_name}: median={median_val}, mean={mean_val}, "
              f"beta={beta:.4f} (beta_link={sc['beta_per_link']})")
        print(f"{'='*60}")

        model = SEIRModelSharedHazard(beta=beta, mean=mean_val, median=median_val)
        results[sc_name] = {}

        for eng_name, (eng_class, eng_kwargs) in engines.items():
            print(f"\n  --- {eng_name} ---")
            trajs = {k: [] for k in ["S", "E", "I", "R"]}
            n_success = 0

            for run_idx in range(args.num_runs):
                seed = 12345 + run_idx * 7919
                try:
                    traj = run_one(eng_class, graph_csr, edge_index, model,
                                   device, seed, target_time=args.target_time,
                                   **eng_kwargs)
                    interp = interpolate(traj, t_grid)
                    for k in ["S", "E", "I", "R"]:
                        trajs[k].append(interp[k])
                    n_success += 1
                except Exception as e:
                    if run_idx == 0:
                        print(f"    Run 0 FAILED: {e}")

                if (run_idx + 1) % 20 == 0:
                    print(f"    {run_idx+1}/{args.num_runs} done", flush=True)

            if trajs["S"]:
                results[sc_name][eng_name] = {
                    k: (np.mean(trajs[k], axis=0), np.std(trajs[k], axis=0))
                    for k in ["S", "E", "I", "R"]
                }
                peak_I = results[sc_name][eng_name]["I"][0].max()
                final_R = results[sc_name][eng_name]["R"][0][-1]
                print(f"    {n_success} runs OK. peak_I={peak_I:.4f}, final_R={final_R:.4f}")

    # ============================================================
    # Plot: match the style of log_normal_SEIR.png
    # All 4 compartments on one plot, 3 scenarios, 3 engines
    # ============================================================
    print("\nGenerating plot...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp_colors = {"S": "#1f77b4", "E": "#ff7f0e", "I": "#2ca02c", "R": "#d62728"}
    eng_styles = {
        "Py": {"ls": "-", "lw": 2.0, "alpha": 0.9},
        "NM": {"ls": "--", "lw": 1.8, "alpha": 0.8},
        "Fused": {"ls": ":", "lw": 2.2, "alpha": 0.8},
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.suptitle(
        f"Scoglio Validation: Python vs NonMarkov vs Fused\n"
        f"N={N}, d={args.degree}, {args.num_runs} runs averaged, t=[0,{args.target_time:.0f}]",
        fontsize=13, fontweight="bold",
    )

    legend_entries = []
    for sc in SCENARIOS:
        sc_name = sc["name"]
        if sc_name not in results:
            continue

        for eng_name in engines:
            if eng_name not in results[sc_name]:
                continue

            style = eng_styles[eng_name]
            for comp in ["S", "E", "I", "R"]:
                mean_vals, std_vals = results[sc_name][eng_name][comp]
                color = comp_colors[comp]
                label = f"{eng_name} {sc_name}" if comp == "S" else None
                line, = ax.plot(t_grid, mean_vals, ls=style["ls"], lw=style["lw"],
                                color=color, alpha=style["alpha"], label=label)
                ax.fill_between(t_grid, mean_vals - std_vals, mean_vals + std_vals,
                                color=color, alpha=0.04)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Fraction of population", fontsize=12)
    ax.set_xlim(0, args.target_time)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Build legend
    from matplotlib.lines import Line2D
    legend_handles = []
    # Engine styles
    for eng_name, style in eng_styles.items():
        legend_handles.append(Line2D([0], [0], color="gray", ls=style["ls"],
                                      lw=style["lw"], label=eng_name))
    # Compartment colors
    for comp, color in comp_colors.items():
        legend_handles.append(Line2D([0], [0], color=color, lw=2, label=comp))
    # Scenario labels
    for sc in SCENARIOS:
        legend_handles.append(Line2D([0], [0], color="white", marker="o",
                                      markerfacecolor="gray", markersize=6,
                                      label=f"{sc['name']}: med={sc['median']}, "
                                            f"mean={sc['mean']}, β_l={sc['beta_per_link']}"))

    ax.legend(handles=legend_handles, fontsize=8, loc="center right",
              ncol=1, framealpha=0.9)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")

    # Summary table
    print("\n" + "=" * 75)
    print(f"{'Scenario':<8s} {'Engine':<8s} {'Peak I':>8s} {'Final R':>8s} {'Attack%':>8s}")
    print("-" * 75)
    for sc in SCENARIOS:
        sc_name = sc["name"]
        if sc_name not in results:
            continue
        for eng_name in engines:
            if eng_name not in results[sc_name]:
                continue
            peak_I = results[sc_name][eng_name]["I"][0].max()
            final_R = results[sc_name][eng_name]["R"][0][-1]
            print(f"{sc_name:<8s} {eng_name:<8s} {peak_I:>8.4f} {final_R:>8.4f} {final_R*100:>7.1f}%")
    print("=" * 75)

    # Agreement table: NM and Fused vs Py
    print("\nMax absolute difference vs Py (RenewalEngine):")
    print(f"{'Scenario':<8s} {'Engine':<8s} {'S':>8s} {'E':>8s} {'I':>8s} {'R':>8s}")
    print("-" * 50)
    for sc in SCENARIOS:
        sc_name = sc["name"]
        if sc_name not in results or "Py" not in results[sc_name]:
            continue
        for eng_name in ["NM", "Fused"]:
            if eng_name not in results[sc_name]:
                continue
            diffs = {}
            for comp in ["S", "E", "I", "R"]:
                py_mean = results[sc_name]["Py"][comp][0]
                eng_mean = results[sc_name][eng_name][comp][0]
                diffs[comp] = np.abs(py_mean - eng_mean).max()
            print(f"{sc_name:<8s} {eng_name:<8s} "
                  f"{diffs['S']:>8.4f} {diffs['E']:>8.4f} "
                  f"{diffs['I']:>8.4f} {diffs['R']:>8.4f}")


if __name__ == "__main__":
    main()
