#!/usr/bin/env python
"""
Multi-topology/multi-size fidelity plot.

Produces a single 3x3 grid of I(t)/N curves (rows = topology, columns =
network size) overlaying tau-leaping at several epsilon values plus the
fine-epsilon reference. Reads results/fidelity_multi.npz written by
experiments/fidelity_multi_graph.py.

Output: docs/jocs/figures/fig_fidelity_multi.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GRAPH_LABEL = {
    "er":    "Erd\u00f6s\u2013R\u00e9nyi ($d{=}8$)",
    "ba":    "Barab\u00e1si\u2013Albert ($m{=}4$)",
    "fixed": r"Fixed-degree ($d{=}8$)",
}


def _size_label(N):
    if N >= 1_000_000:
        return fr"$N{{=}}10^{{{int(round(np.log10(N)))}}}$"
    if N >= 1000:
        return fr"$N{{=}}{{{int(N/1000)}k}}$" if N % 1000 == 0 else fr"$N{{=}}{N}$"
    return fr"$N{{=}}{N}$"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="results/fidelity_multi.npz")
    parser.add_argument("--exact", default="results/fidelity_multi_exact.npz",
                        help="Optional NPZ of exact Gillespie reference curves.")
    parser.add_argument("--out",
                        default="docs/jocs/figures/fig_fidelity_multi.png")
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    sample_times = d["sample_times"]
    graphs = [str(g) for g in d["graphs"]]
    sizes = [int(x) for x in d["sizes"]]
    epsilons = [float(x) for x in d["epsilons"]]

    exact = None
    exact_times = None
    if Path(args.exact).exists():
        ex = np.load(args.exact, allow_pickle=False)
        exact = ex
        exact_times = ex["sample_times"]
        print(f"Loaded exact Gillespie reference from {args.exact}")
    else:
        print(f"No exact reference at {args.exact}; "
              "plot will use tau-leaping ε=0.005 as reference only")

    # Reference is the smallest epsilon in the sweep.
    ref_eps = min(epsilons)
    coarse_eps = sorted([e for e in epsilons if e != ref_eps])

    # Color cycle for coarse epsilons.
    palette = plt.cm.viridis(np.linspace(0.2, 0.85, len(coarse_eps)))

    n_rows, n_cols = len(graphs), len(sizes)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.4 * n_cols, 2.6 * n_rows),
        sharex=True, sharey="row",
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Proxy handles for a shared figure-level legend so the "exact
    # Gillespie" entry shows up even on panels where exact was not
    # computed (e.g. N = 1e5). We build them once with dummy lines.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    shared_handles = []
    if exact is not None:
        shared_handles.append(Patch(facecolor="red", alpha=0.12,
                                    label="exact Gillespie 25\u201375%"))
        shared_handles.append(Line2D([0], [0], color="red", lw=2.0,
                                     label="exact Gillespie (mean)"))
    shared_handles.append(Line2D([0], [0], color="black", lw=1.5,
                                 label=fr"$\varepsilon{{=}}{ref_eps:g}$ tau-leap"))
    for c, eps in zip(palette, coarse_eps):
        shared_handles.append(Line2D([0], [0], color=c, lw=1.3,
                                     label=fr"$\varepsilon{{=}}{eps:g}$"))

    for i, graph_key in enumerate(graphs):
        for j, N in enumerate(sizes):
            ax = axes[i, j]
            # Exact Gillespie reference, if provided (no per-axis label:
            # the shared figure legend below carries that entry).
            if exact is not None:
                ex_key_mean = f"mean_{graph_key}_N{N}"
                ex_key_q25 = f"q25_{graph_key}_N{N}"
                ex_key_q75 = f"q75_{graph_key}_N{N}"
                if ex_key_mean in exact.files:
                    ax.fill_between(
                        exact_times,
                        exact[ex_key_q25],
                        exact[ex_key_q75],
                        color="red", alpha=0.12, linewidth=0, label=None,
                        zorder=1,
                    )
                    ax.plot(exact_times, exact[ex_key_mean],
                            color="red", lw=2.0, label=None, zorder=3)

            # Tau-leaping ε=0.005 reference (internal limit).
            ref_key = f"mean_{graph_key}_N{N}_eps{ref_eps:g}"
            ref_traj = d[ref_key]
            q25_ref = d[f"q25_{graph_key}_N{N}_eps{ref_eps:g}"]
            q75_ref = d[f"q75_{graph_key}_N{N}_eps{ref_eps:g}"]
            ax.fill_between(sample_times, q25_ref, q75_ref,
                            color="black", alpha=0.08, label=None,
                            zorder=1)
            ax.plot(sample_times, ref_traj, color="black", lw=1.5,
                    zorder=4, label=None)

            for c, eps in zip(palette, coarse_eps):
                k = f"mean_{graph_key}_N{N}_eps{eps:g}"
                if k not in d.files:
                    continue
                ax.plot(sample_times, d[k], color=c, lw=1.3,
                        zorder=5, label=None)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(_size_label(N), fontsize=11)
            if j == 0:
                ax.set_ylabel(GRAPH_LABEL.get(graph_key, graph_key),
                              fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel(r"Time $t$")
            ax.set_xlim(sample_times[0], sample_times[-1])
            ax.set_ylim(bottom=0)

    fig.suptitle(
        r"Monte Carlo ensemble-mean $I(t)/N$ (non-Markovian SEIR): "
        "tau-leaping vs. exact Gillespie",
        fontsize=12, y=0.998,
    )
    # Single shared legend above the grid so the "exact Gillespie"
    # entry appears even on cells where the exact run was omitted
    # (N = 1e5 is tau-leaping only).
    fig.legend(
        handles=shared_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(shared_handles),
        fontsize=8.5, frameon=True, framealpha=0.95,
    )
    fig.text(
        0.5, -0.01,
        r"(20 trajectories per $(G, N, \varepsilon)$ for tau-leaping and per $(G, N)$ for exact; "
        r"initial seed $=\max(10,\;0.01N)$ Exposed nodes; $\beta = 2/\bar d = 0.25$; "
        r"log-normal $E{\to}I$, $I{\to}R$ with the main-text parameters.)",
        ha="center", va="top", fontsize=8.5, color="gray",
    )

    # Leave room at top for the shared legend (0.93) and for the
    # bottom caption fringe (0.02).
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
