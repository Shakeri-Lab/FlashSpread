#!/usr/bin/env python
"""
Exact recreation of the Scoglio validation (log_normal_SEIR.png).

Parameters extracted from the original saved .npz files:
  N=1000, nlink=8, med=5.0, mean=7.5, bet=2.0 => beta=0.25
  E->I: LogNormal(mean=5.0, median=4.0)  [hardcoded in original]
  I->R: LogNormal(mean=7.5, median=5.0)  [from --med/--mean]
  1 initial infected (node 0, state=Infected)
  100 runs averaged, simulate to t=50
  epsilon=0.03, max_tau=0.1

Compares 3 engines:
  Py:    RenewalEngine (Markovian edges — original, validated against MATLAB)
  NM:    RenewalEngineNonMarkov (age-dependent edges)
  Fused: RenewalEngineFused (fused Triton kernel)

Also loads original MATLAB and Python .npz/.mat if available for overlay.
"""

import sys
import time
import os
from pathlib import Path
import numpy as np

import torch
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread.core.graph import GraphCSR
from flashspread import SEIRModel
from flashspread.engines.renewal import RenewalEngine, RenewalEngineNonMarkov
from flashspread.engines.renewal_fused import RenewalEngineFused

# ============================================================
# Exact parameters from the original validation
# ============================================================
N = 1000
NLINK = 8
MED_IR = 5.0      # I->R median
MEAN_IR = 7.5     # I->R mean
MEAN_EI = 5.0     # E->I mean (hardcoded in original)
MEDIAN_EI = 4.0   # E->I median (hardcoded in original)
BET = 2.0
BETA = BET / NLINK  # = 0.25
TF = 50
NR = 100
EPSILON = 0.03
TAU_MAX = 0.1
GRAPH_SEED = 42


def create_er_graph(n, nlink, device, seed=GRAPH_SEED):
    p = nlink / n
    G = nx.erdos_renyi_graph(n, p, seed=seed, directed=False)
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(edge_index, n, incoming=True)
    return csr, edge_index


def make_graph_wrap(csr, edge_index):
    class GW:
        pass
    gw = GW()
    gw.csr = csr
    gw.edge_index = edge_index
    gw.num_nodes = csr.num_nodes
    gw.num_edges = csr.num_edges
    return gw


def run_engine(engine_class, gw, model, device, seed,
               target_time=TF, **engine_kwargs):
    engine = engine_class(gw, model, device=device, seed=seed,
                          epsilon=EPSILON, tau_max=TAU_MAX, **engine_kwargs)

    # 1 initial infected: node 0 in state Infected
    engine.state[0] = model.infected
    engine.age[0] = 0.0

    n = gw.num_nodes
    # Record at integer time points [0, 1, ..., TF] to match MATLAB
    time_grid = np.arange(0, target_time + 1, dtype=float)
    counts_grid = np.zeros((4, len(time_grid)), dtype=float)

    # Initial counts
    counts = engine.count_by_state()
    counts_grid[:, 0] = counts[:4].cpu().numpy()
    next_t_idx = 1

    while engine.current_time < target_time and next_t_idx < len(time_grid):
        engine.step()

        while next_t_idx < len(time_grid) and engine.current_time >= time_grid[next_t_idx]:
            c = engine.count_by_state()
            counts_grid[:, next_t_idx] = c[:4].cpu().numpy()
            next_t_idx += 1

    # Fill any remaining
    if next_t_idx < len(time_grid):
        c = engine.count_by_state()
        counts_grid[:, next_t_idx:] = c[:4].cpu().numpy().reshape(4, 1)

    return time_grid, counts_grid


def load_original_data(base_dir):
    """Try to load original MATLAB and Python .npz/.mat results."""
    orig = {}
    base = Path(base_dir)

    # Try Python .npz
    for name in ["scoglio_python_7007986.npz", "scoglio_python_N1000_T1000_q10_90.npz"]:
        path = base / name
        if path.exists():
            d = np.load(str(path), allow_pickle=True)
            orig["Py (original)"] = d["pre_mean"][:, :TF + 1]
            break

    # Try MATLAB .mat
    try:
        from scipy.io import loadmat
        for name in ["scoglio_matlab_7007987.mat", "scoglio_matlab_N1000_T1000_q10_90.mat"]:
            path = base / name
            if path.exists():
                d = loadmat(str(path), squeeze_me=True)
                orig["MATLAB"] = d["pre_mean"][:, :TF + 1]
                break
    except ImportError:
        pass

    # Try GPU .npz
    for name in ["scoglio_gpu_7011364.npz", "scoglio_gpu_N1000_T1000_q10_90.npz"]:
        path = base / name
        if path.exists():
            d = np.load(str(path), allow_pickle=True)
            orig["GPU (original)"] = d["pre_mean"][:, :TF + 1]
            break

    return orig


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

    print(f"Exact Scoglio Recreation")
    print(f"N={N}, nlink={NLINK}, beta={BETA}")
    print(f"E->I: LogNormal(mean={MEAN_EI}, median={MEDIAN_EI})")
    print(f"I->R: LogNormal(mean={MEAN_IR}, median={MED_IR})")
    print(f"Runs: {NR}, T: {TF}, epsilon={EPSILON}, tau_max={TAU_MAX}")
    print(f"Device: {device}")
    print()

    # Create graph
    print("Creating ER graph...", flush=True)
    csr, edge_index = create_er_graph(N, NLINK, device)
    gw = make_graph_wrap(csr, edge_index)
    print(f"  Edges: {csr.num_edges}")

    model = SEIRModel(
        beta=BETA,
        mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
        mean_ir=MEAN_IR, median_ir=MED_IR,
    )

    time_grid = np.arange(0, TF + 1, dtype=float)

    engines = {
        "Py (RenewalEngine)": (RenewalEngine, {}),
        "NM (NonMarkov)": (RenewalEngineNonMarkov, {}),
        "Fused (Triton)": (RenewalEngineFused, {}),
    }

    # Accumulate results: pre_sum[engine][4, T+1]
    results = {}
    for eng_name, (eng_class, eng_kwargs) in engines.items():
        print(f"\n--- {eng_name} ---")
        pre_sum = np.zeros((4, len(time_grid)), dtype=np.float64)
        n_ok = 0

        for run in range(NR):
            seed = 12345 + run * 7919
            try:
                tg, counts = run_engine(eng_class, gw, model, device, seed, **eng_kwargs)
                pre_sum += counts
                n_ok += 1
            except Exception as e:
                if run == 0:
                    print(f"  Run 0 FAILED: {e}")

            if (run + 1) % 25 == 0:
                print(f"  {run+1}/{NR}", flush=True)

        if n_ok > 0:
            results[eng_name] = pre_sum / n_ok
            print(f"  {n_ok}/{NR} succeeded")

    # Load original data for overlay
    print("\nLoading original MATLAB/Python data...", flush=True)
    orig = load_original_data(args.orig_dir)
    for k in orig:
        print(f"  Found: {k} (shape {orig[k].shape})")

    # ============================================================
    # Plot: match log_normal_SEIR.png style exactly
    # Single plot, 4 compartments (S1-S4), multiple engines overlaid
    # ============================================================
    print("\nGenerating plot...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    state_labels = ["S", "E", "I", "R"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Engine line styles
    style_map = {
        "Py (RenewalEngine)": {"ls": "-", "lw": 2.0},
        "NM (NonMarkov)": {"ls": "--", "lw": 1.8},
        "Fused (Triton)": {"ls": ":", "lw": 2.2},
        # Originals
        "Py (original)": {"ls": "-.", "lw": 1.2},
        "MATLAB": {"ls": "--", "lw": 1.2},
        "GPU (original)": {"ls": ":", "lw": 1.2},
    }

    # Plot new engines
    for eng_name, pre_mean in results.items():
        style = style_map.get(eng_name, {"ls": "-", "lw": 1.5})
        for i in range(4):
            label = f"{eng_name} {state_labels[i]}"
            ax.plot(time_grid, pre_mean[i] / N, color=colors[i],
                    ls=style["ls"], lw=style["lw"], label=label)

    # Plot originals (if found)
    for orig_name, pre_mean in orig.items():
        style = style_map.get(orig_name, {"ls": "-.", "lw": 1.0})
        tg_orig = np.arange(pre_mean.shape[1])
        for i in range(4):
            label = f"{orig_name} {state_labels[i]}"
            ax.plot(tg_orig, pre_mean[i] / N, color=colors[i],
                    ls=style["ls"], lw=style["lw"], alpha=0.5, label=label)

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Fraction of population", fontsize=12)
    ax.set_title(
        f"Scoglio Validation: Python vs MATLAB vs NonMarkov vs Fused\n"
        f"N={N}, d={NLINK}, β={BETA}, E→I LN({MEAN_EI},{MEDIAN_EI}), "
        f"I→R LN({MEAN_IR},{MED_IR}), {NR} runs",
        fontsize=11,
    )
    ax.set_xlim(0, TF)
    ax.grid(True, alpha=0.3)

    # Compact legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, ncol=3, loc="center right")

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Engine':<25s}  {'peak E':>7s} {'peak I':>7s} {'final R':>7s} {'attack%':>8s}")
    print("-" * 70)
    for eng_name, pre_mean in {**results, **{k: v for k, v in orig.items()}}.items():
        frac = pre_mean / N
        peak_E = frac[1].max()
        peak_I = frac[2].max()
        final_R = frac[3, -1] if frac.shape[1] > TF else frac[3, -1]
        print(f"{eng_name:<25s}  {peak_E:>7.4f} {peak_I:>7.4f} {final_R:>7.4f} {final_R*100:>7.1f}%")
    print("=" * 70)

    # Agreement: max abs difference between engines
    eng_names = list(results.keys())
    if len(eng_names) >= 2:
        print(f"\nMax abs difference (fraction) vs '{eng_names[0]}':")
        ref = results[eng_names[0]] / N
        for eng_name in eng_names[1:]:
            comp = results[eng_name] / N
            for i, s in enumerate(state_labels):
                diff = np.abs(ref[i] - comp[i]).max()
                print(f"  {eng_name} {s}: {diff:.6f}")


if __name__ == "__main__":
    main()
