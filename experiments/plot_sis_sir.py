#!/usr/bin/env python
"""
Two-panel validation plot: SIS (left) and SIR (right), each showing
FlashSpread's MarkovianEngine ensemble-mean I(t)/N against the exact
Doob-Gillespie reference.

Reads four NPZ files (one pair per model) and writes a single PNG into
docs/jocs/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sis-exact",
                        default="results/exact_sis.npz")
    parser.add_argument("--sis-flash",
                        default="results/flashspread_sis.npz")
    parser.add_argument("--sir-exact",
                        default="results/exact_sir.npz")
    parser.add_argument("--sir-flash",
                        default="results/flashspread_sir.npz")
    parser.add_argument("--out",
                        default="docs/jocs/figures/fig_sis_sir_validation.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)

    specs = [
        ("SIS", args.sis_exact, args.sis_flash, axes[0]),
        ("SIR", args.sir_exact, args.sir_flash, axes[1]),
    ]
    for label, exact_path, flash_path, ax in specs:
        ex = np.load(exact_path, allow_pickle=False)
        fs = np.load(flash_path, allow_pickle=False)
        t = ex["sample_times"]

        # Exact Gillespie: red solid with 25-75% band.
        ax.fill_between(t, ex["q25"], ex["q75"],
                        color="red", alpha=0.15, linewidth=0,
                        label="exact Gillespie 25\u201375%")
        ax.plot(t, ex["mean_traj"], color="red", lw=2.2,
                label="exact Gillespie (mean)")

        # FlashSpread MarkovianEngine: blue dashed line on top.
        ax.plot(fs["sample_times"], fs["mean_traj"],
                color="C0", lw=1.8, linestyle="--",
                label="FlashSpread MarkovianEngine")

        # Parameter annotation
        beta = float(ex["beta"][0])
        rate = float(ex["recovery_rate"][0])
        nodes = int(ex["num_nodes"][0])
        rate_sym = r"$\delta$" if label == "SIS" else r"$\gamma$"
        ax.set_title(
            fr"{label}:  $\beta = {beta:.3f}$,  {rate_sym} $= {rate:.2f}$,"
            fr"  $N = {nodes}$"
        )
        ax.set_xlabel(r"Time $t$")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel(r"Infected fraction $I(t)/N$")
    axes[0].legend(fontsize=9, loc="upper right",
                   frameon=True, framealpha=0.95)

    fig.suptitle(
        "FlashSpread Markovian engine vs. exact Doob-Gillespie "
        "(100 Monte Carlo runs per curve)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
