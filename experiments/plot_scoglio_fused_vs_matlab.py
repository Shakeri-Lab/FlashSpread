#!/usr/bin/env python
"""
Clean validation plot: Fused, Fused CG, and MATLAB only.

Exact Scoglio parameters:
  N=1000, nlink=8, beta=0.25, E->I LN(5,4), I->R LN(7.5,5)
  1 initial infected, 100 runs, t=50
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
from flashspread.engines.renewal_fused import RenewalEngineFused, RenewalEngineFusedCUDAGraph

N = 1000
NLINK = 8
BETA = 2.0 / NLINK
MEAN_EI, MEDIAN_EI = 5.0, 4.0
MEAN_IR, MEDIAN_IR = 7.5, 5.0
TF = 50
NR = 100
EPSILON = 0.03
TAU_MAX = 0.1


def create_er_graph(n, nlink, device, seed=42):
    p = nlink / n
    G = nx.erdos_renyi_graph(n, p, seed=seed, directed=False)
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(edge_index, n, incoming=True)

    class GW:
        pass
    gw = GW()
    gw.csr = csr
    gw.edge_index = edge_index
    gw.num_nodes = n
    gw.num_edges = csr.num_edges
    return gw


def run_one(engine_class, gw, model, device, seed, **kwargs):
    engine = engine_class(gw, model, device=device, seed=seed,
                          epsilon=EPSILON, tau_max=TAU_MAX, **kwargs)
    engine.state[0] = model.infected
    engine.age[0] = 0.0

    time_grid = np.arange(0, TF + 1, dtype=float)
    counts_grid = np.zeros((4, len(time_grid)), dtype=float)
    counts = engine.count_by_state()
    counts_grid[:, 0] = counts[:4].cpu().numpy()
    next_idx = 1

    while engine.current_time < TF and next_idx < len(time_grid):
        engine.step()
        while next_idx < len(time_grid) and engine.current_time >= time_grid[next_idx]:
            c = engine.count_by_state()
            counts_grid[:, next_idx] = c[:4].cpu().numpy()
            next_idx += 1

    if next_idx < len(time_grid):
        c = engine.count_by_state()
        counts_grid[:, next_idx:] = c[:4].cpu().numpy().reshape(4, 1)

    return time_grid, counts_grid


def load_matlab(base_dir):
    try:
        from scipy.io import loadmat
        for name in ["scoglio_matlab_7007987.mat", "scoglio_matlab_N1000_T1000_q10_90.mat"]:
            path = Path(base_dir) / name
            if path.exists():
                d = loadmat(str(path), squeeze_me=True)
                return d["pre_mean"][:, :TF + 1]
    except ImportError:
        pass
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/validation_scoglio_exact.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--orig-dir", type=str,
                        default="/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/"
                                "[old] nonmarkovianGEMF/outputs")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Scoglio Validation: Fused vs Fused CG vs MATLAB")
    print(f"N={N}, d={NLINK}, beta={BETA}, runs={NR}")
    print(f"Device: {device}")
    print()

    gw = create_er_graph(N, NLINK, device)
    print(f"Graph: {gw.num_edges} edges")

    model = SEIRModel(beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
                      mean_ir=MEAN_IR, median_ir=MEDIAN_IR)

    time_grid = np.arange(0, TF + 1, dtype=float)

    engines = {
        "Fused (Triton)": (RenewalEngineFused, {}),
        "Fused CG (CUDA Graph)": (RenewalEngineFusedCUDAGraph, {"steps_per_launch": 50}),
    }

    results = {}
    for eng_name, (eng_class, eng_kwargs) in engines.items():
        print(f"\n--- {eng_name} ---")
        pre_sum = np.zeros((4, len(time_grid)), dtype=np.float64)
        n_ok = 0
        for run in range(NR):
            seed = 12345 + run * 7919
            try:
                tg, counts = run_one(eng_class, gw, model, device, seed, **eng_kwargs)
                pre_sum += counts
                n_ok += 1
            except Exception as e:
                if run == 0:
                    print(f"  Run 0 FAILED: {e}")
            if (run + 1) % 25 == 0:
                print(f"  {run+1}/{NR}", flush=True)
        if n_ok > 0:
            results[eng_name] = pre_sum / n_ok
            peak_I = (results[eng_name][2] / N).max()
            final_R = results[eng_name][3, -1] / N
            print(f"  {n_ok} OK, peak_I={peak_I:.4f}, final_R={final_R:.4f}")

    # Load MATLAB
    print("\nLoading MATLAB data...", flush=True)
    mat_data = load_matlab(args.orig_dir)
    if mat_data is not None:
        results["MATLAB (exact Gillespie)"] = mat_data
        print(f"  Loaded: shape {mat_data.shape}")
    else:
        print("  Not found")

    # Plot
    print("\nGenerating plot...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # S, E, I, R
    comp_names = ["S", "E", "I", "R"]

    styles = {
        "Fused (Triton)":         {"ls": "-",  "lw": 2.5, "alpha": 0.9},
        "Fused CG (CUDA Graph)":  {"ls": "--", "lw": 2.0, "alpha": 0.85},
        "MATLAB (exact Gillespie)": {"ls": ":",  "lw": 2.0, "alpha": 0.8},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for eng_name in ["MATLAB (exact Gillespie)", "Fused (Triton)", "Fused CG (CUDA Graph)"]:
        if eng_name not in results:
            continue
        pre = results[eng_name]
        s = styles[eng_name]
        for i in range(4):
            label = f"{eng_name} {comp_names[i]}"
            ax.plot(time_grid, pre[i] / N, color=comp_colors[i],
                    ls=s["ls"], lw=s["lw"], alpha=s["alpha"], label=label)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Fraction of population", fontsize=12)
    ax.set_title(
        f"SEIR fidelity: fused Triton vs. exact Gillespie (N={N}, {NR} runs)",
        fontsize=11,
    )
    ax.set_xlim(0, TF)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Compact legend: group by engine style, then compartment color
    from matplotlib.lines import Line2D
    handles = []
    for eng_name, s in styles.items():
        if eng_name in results:
            handles.append(Line2D([0], [0], color="gray", ls=s["ls"], lw=s["lw"], label=eng_name))
    for i, (name, color) in enumerate(zip(comp_names, comp_colors)):
        handles.append(Line2D([0], [0], color=color, lw=2.5, label=name))
    ax.legend(handles=handles, fontsize=9, loc="center right", framealpha=0.9)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Engine':<28s} {'peak E':>7s} {'peak I':>7s} {'final R':>8s}")
    print("-" * 60)
    for eng_name, pre in results.items():
        frac = pre / N
        print(f"{eng_name:<28s} {frac[1].max():>7.4f} {frac[2].max():>7.4f} {frac[3,-1]:>8.4f}")
    print("=" * 60)

    # Agreement
    eng_list = [k for k in results if k != "MATLAB (exact Gillespie)"]
    if "MATLAB (exact Gillespie)" in results:
        mat = results["MATLAB (exact Gillespie)"] / N
        print("\nMax abs diff vs MATLAB:")
        for eng_name in eng_list:
            frac = results[eng_name] / N
            for i, c in enumerate(comp_names):
                diff = np.abs(frac[i] - mat[i]).max()
                print(f"  {eng_name} {c}: {diff:.4f}")


if __name__ == "__main__":
    main()
