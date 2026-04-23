#!/usr/bin/env python
"""
Plot NUPS(t) for the active-node compaction ablation.

Two-panel figure:
    (a) ER d=8, N=1e6 at TF=50, final R ~ 15%
    (b) BA m=4, N=1e6 at TF=50, final R ~ 97%

For each panel, overlay the compaction-on and compaction-off NUPS(t)
curves on one y-axis, and the mean R(t) (as a fraction of N, derived
from the SEIR ODE on the benchmark parameters) on the right y-axis.
The shrinking active set is what drives the tail speedup on BA; the
left panel shows that on ER the attack rate at TF=50 is too low for
compaction to break even in the whole-run mean, even though the last
temporal bucket is already above baseline.

Reads: results/active_compaction_buckets.npz
Writes: docs/jocs/figures/fig_compaction_nups_t.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

BUCKETS = "results/active_compaction_buckets.npz"
SUMMARY = "results/active_compaction_summary.csv"
OUT = "docs/jocs/figures/fig_compaction_nups_t.png"


def main():
    d = np.load(BUCKETS, allow_pickle=True)
    tf = float(d["tf"])
    nb = int(d["num_buckets"])
    edges = np.linspace(0.0, tf, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)

    def fmt_panel(ax, graph, title, ylabel_r):
        base = d[f"{graph}_compaction_0"]
        comp = d[f"{graph}_compaction_1"]
        base_m, base_s = base.mean(axis=0), base.std(axis=0)
        comp_m, comp_s = comp.mean(axis=0), comp.std(axis=0)

        # NUPS(t) curves
        ax.plot(
            centers, base_m / 1e9,
            "-o", color="#555", label="compaction OFF (baseline)", lw=2.0,
        )
        ax.fill_between(
            centers, (base_m - base_s) / 1e9, (base_m + base_s) / 1e9,
            color="#555", alpha=0.15,
        )
        ax.plot(
            centers, comp_m / 1e9,
            "-s", color="#c0392b", label="compaction ON", lw=2.0,
        )
        ax.fill_between(
            centers, (comp_m - comp_s) / 1e9, (comp_m + comp_s) / 1e9,
            color="#c0392b", alpha=0.15,
        )

        ax.set_xlabel("simulated time $t$ (days)")
        ax.set_ylabel("throughput (Giga-NUPS)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25, ls=":")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        # Annotate per-bucket speedup ratios on the top edge
        for c, b, comp_val in zip(centers, base_m, comp_m):
            r = comp_val / b if b > 0 else float("nan")
            if r >= 1.05 or r <= 0.97:
                ax.annotate(
                    f"{r:.2f}$\\times$",
                    xy=(c, max(b, comp_val) / 1e9),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", fontsize=8,
                    # Match the "compaction ON" curve colour so the
                    # speedup annotation is visually linked to the
                    # curve being measured, not to a third hue.
                    color=("#c0392b" if r >= 1.0 else "#999"),
                )

    fmt_panel(
        axes[0], "er",
        "(a) ER $d{=}8$, $N{=}10^6$: tail small (final $R\\approx 15\\%$)",
        "R fraction",
    )
    fmt_panel(
        axes[1], "ba",
        "(b) BA $m{=}4$, $N{=}10^6$: tail dominant (final $R\\approx 97\\%$)",
        "R fraction",
    )

    fig.suptitle(
        "Active-node compaction: NUPS$(t)$ vs epidemic phase",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
