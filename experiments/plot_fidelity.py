#!/usr/bin/env python
"""
Produce fidelity appendix plots from results/fidelity_trajectories.npz
and results/fidelity_summary.csv.

Outputs three PNGs into docs/jocs/figures/:
  fig_fidelity_traj.png     — mean (S, E, I, R) trajectories at selected
                              epsilons overlaid on the reference curve.
  fig_fidelity_err_vs_eps.png  — four error metrics vs epsilon (log-log).
  fig_fidelity_pareto.png   — mean wall-clock per trajectory vs L-inf
                              trajectory error; Pareto front highlighted.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Exact-Gillespie reference values for this benchmark, from the main
# paper Table 4 (N=1000, ER d=8, beta=2/d, log-normal E->I/I->R, 100 runs).
# These are the scalar summaries of the true (non-tau-leaped) process and
# define the "structural bias floor" of synchronous updates on networks.
EXACT_PEAK_I = 0.314
EXACT_FINAL_R = 0.863


def load_summary(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (float(v) if k != "reference" else v) for k, v in r.items()})
    rows.sort(key=lambda r: r["epsilon"])
    return rows


def load_trajectories(path):
    return np.load(path, allow_pickle=False)


def _reference_label(npz):
    """Human-readable label for the reference curve."""
    raw = str(npz["reference_label"][0]) if "reference_label" in npz.files else ""
    if "matlab_exact_gillespie" in raw:
        return "exact Gillespie (MATLAB)"
    if "tauleap" in raw:
        # Pull the epsilon from the label (e.g. "tauleap_eps0.005")
        try:
            eps = raw.split("eps")[-1]
            return fr"$\varepsilon={eps}$ tau-leaping limit"
        except Exception:
            return "tau-leaping limit"
    return raw or "reference"


def plot_trajectories(npz, out_path,
                      show_epsilons=(0.005, 0.03, 0.1, 0.2),
                      label_prefix="FlashSpread, "):
    """
    Trajectory overlay for Fig. 6.

    We look up the per-eps mean trajectory by the NPZ key pattern
    ``mean_traj_eps<eps:g>`` rather than iterating the stored
    ``epsilons`` array: the latter was saved as float32 and comparing a
    Python float ``0.005`` against the fp32 round-trip value
    ``0.004999999888...`` silently dropped every epsilon from the plot.
    """
    sample_times = npz["sample_times"]
    reference = npz["reference"]
    has_bands = "reference_q_low" in npz.files and "reference_q_high" in npz.files
    ref_label = _reference_label(npz)

    # 2x2 panel: S, E, I, R
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=False)
    axes = axes.ravel()
    labels = ["Susceptible", "Exposed", "Infected", "Recovered"]
    colors_eps = plt.cm.viridis(np.linspace(0.1, 0.9, len(show_epsilons)))

    # Emphasize the production-default epsilon so the reader has one
    # clear "this is the FlashSpread curve to compare" line.
    default_eps = 0.03

    # Build curve list first so we can collect proxy handles for a
    # shared figure-level legend below.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    shared_handles = []
    if has_bands:
        shared_handles.append(Patch(facecolor="red", alpha=0.10,
                                    label=f"{ref_label} 10\u201390%"))
    shared_handles.append(Line2D([0], [0], color="red", lw=2.2,
                                 label=f"{ref_label} (mean)"))

    for k, state_label in enumerate(labels):
        ax = axes[k]
        if has_bands:
            ax.fill_between(
                sample_times,
                npz["reference_q_low"][:, k],
                npz["reference_q_high"][:, k],
                color="red", alpha=0.10, linewidth=0, zorder=1,
                label=None,
            )
        ax.plot(sample_times, reference[:, k], color="red",
                lw=2.2, zorder=3, label=None)

        for c, eps in zip(colors_eps, show_epsilons):
            key = f"mean_traj_eps{eps:g}"
            if key not in npz.files:
                continue
            traj = npz[key]
            # Production default highlighted with a thicker stroke.
            lw = 2.2 if float(eps) == default_eps else 1.3
            ax.plot(sample_times, traj[:, k], color=c, lw=lw,
                    zorder=4, label=None)

        ax.set_title(state_label)
        ax.set_ylabel("Fraction")
        ax.grid(True, alpha=0.3)
        if k >= 2:
            ax.set_xlabel(r"Time $t$")

    # Populate shared handles for the ε curves now (they exist in the
    # figure regardless of which panel we iterate; Line2D handles are
    # proxies).
    for c, eps in zip(colors_eps, show_epsilons):
        key = f"mean_traj_eps{eps:g}"
        if key not in npz.files:
            continue
        lw = 2.2 if float(eps) == default_eps else 1.3
        suffix = " (production default)" if float(eps) == default_eps else ""
        shared_handles.append(
            Line2D([0], [0], color=c, lw=lw,
                   label=fr"{label_prefix}$\varepsilon{{=}}{eps:g}${suffix}")
        )

    fig.suptitle(
        "Ensemble-mean SEIR compartments: FlashSpread (tau-leaping) "
        "vs. exact non-Markovian Gillespie",
        fontsize=12, y=0.998,
    )
    # Single shared legend above the grid; per-axis legends are cluttered
    # and were silently empty when the float-precision bug dropped the
    # ε curves.
    fig.legend(
        handles=shared_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(3, len(shared_handles)),
        fontsize=9, frameon=True, framealpha=0.95,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def plot_error_vs_eps(summary, out_path):
    eps = np.array([r["epsilon"] for r in summary])
    err_peak = np.array([r["err_peak_I"] for r in summary])
    err_peak_lo = np.array([r["ci_err_peak_I_lo"] for r in summary])
    err_peak_hi = np.array([r["ci_err_peak_I_hi"] for r in summary])
    err_final = np.array([r["err_final_R"] for r in summary])
    err_final_lo = np.array([r["ci_err_final_R_lo"] for r in summary])
    err_final_hi = np.array([r["ci_err_final_R_hi"] for r in summary])
    # Trajectory-level self-consistency metrics (pure discretization).
    linf_self = np.array([r["linf_traj"] for r in summary])
    l2_self = np.array([r["l2_traj"] for r in summary])

    fig, ax = plt.subplots(figsize=(7.3, 4.8))

    # Per-run error against exact, with bootstrap 95% CIs as error bars.
    ax.errorbar(
        eps, err_peak,
        yerr=[err_peak - err_peak_lo, err_peak_hi - err_peak],
        fmt="o-", lw=1.8, ms=6, color="C3", capsize=3,
        label=r"per-run peak-$I$ error (vs exact Gillespie, 95% CI)",
    )
    ax.errorbar(
        eps, err_final,
        yerr=[err_final - err_final_lo, err_final_hi - err_final],
        fmt="s-", lw=1.8, ms=6, color="C1", capsize=3,
        label=r"per-run final-$R$ error (vs exact Gillespie, 95% CI)",
    )

    # Self-consistency (pure discretization component; no CIs needed).
    ax.loglog(
        eps, linf_self, "^--", lw=1.2, ms=5, color="C0", alpha=0.75,
        label=r"trajectory $L_\infty$ (self-consistency vs $\varepsilon{=}$0.005)",
    )
    ax.loglog(
        eps, l2_self, "d--", lw=1.2, ms=5, color="C2", alpha=0.75,
        label=r"trajectory $L_2$ (self-consistency vs $\varepsilon{=}$0.005)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Tau-leaping tolerance $\varepsilon$")
    ax.set_ylabel("Error")
    ax.set_title(
        r"Fidelity vs tolerance (ER $N{=}10^3$, $d{=}8$, 100 MC runs; "
        r"bootstrap 95% CIs)"
    )
    ax.grid(True, which="both", alpha=0.3)

    # Structural-floor annotation: the per-run CI bands overlap heavily,
    # so the "floor" is whatever the tightest lower-CI reaches. Place
    # the text in a corner (upper-left) where no data markers live, so
    # the label never collides with an ε point.
    floor = float(min(err_peak_lo.min(), err_final_lo.min()))
    ax.axhline(floor, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.text(
        eps[0] * 1.15, floor * 0.50,
        "structural bias floor\n(synchronous updates on networks)",
        fontsize=8.5, color="gray", ha="left", va="top",
    )

    ax.legend(fontsize=8.5, frameon=True, framealpha=0.95, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def plot_pareto(summary, out_path):
    eps = np.array([r["epsilon"] for r in summary])
    # Point estimate and CI on the per-run peak-I error (the more
    # conservative of the two scalar errors on this benchmark).
    err_peak = np.array([r["err_peak_I"] for r in summary])
    err_peak_lo = np.array([r["ci_err_peak_I_lo"] for r in summary])
    err_peak_hi = np.array([r["ci_err_peak_I_hi"] for r in summary])
    err_final = np.array([r["err_final_R"] for r in summary])
    err_final_lo = np.array([r["ci_err_final_R_lo"] for r in summary])
    err_final_hi = np.array([r["ci_err_final_R_hi"] for r in summary])
    # "Summary error" for the Pareto axis: max of peak-I and final-R
    # per-run errors against the exact Gillespie reference (so we never
    # hide a worse metric behind a better one). Use the CI envelopes.
    summary_err = np.maximum(err_peak, err_final)
    summary_err_lo = np.maximum(err_peak_lo, err_final_lo)
    summary_err_hi = np.maximum(err_peak_hi, err_final_hi)

    wall = np.array([r["mean_wall_time"] for r in summary])
    wall_lo = np.array([r["ci_wall_lo"] for r in summary])
    wall_hi = np.array([r["ci_wall_hi"] for r in summary])

    fig, ax = plt.subplots(figsize=(7.3, 4.8))

    # Pareto "frontier" guideline (light, ordered by wall-clock).
    order = np.argsort(wall)
    ax.plot(wall[order], summary_err[order], "-", color="lightgray",
            lw=1.0, zorder=1)

    # Cross-hatched error bars on both axes (CI on both wall-clock and error).
    ax.errorbar(
        wall, summary_err,
        xerr=[wall - wall_lo, wall_hi - wall],
        yerr=[summary_err - summary_err_lo, summary_err_hi - summary_err],
        fmt="none", ecolor="black", elinewidth=0.8, capsize=3,
        alpha=0.55, zorder=2,
    )
    sc = ax.scatter(wall, summary_err, c=np.log10(eps), cmap="viridis",
                    s=95, zorder=3, edgecolor="black", linewidth=0.6)

    # Place labels selectively: the fast cluster (eps > 0.03) is tight,
    # so label only the three well-separated slow points and the
    # fastest point; drop labels for 0.03 and 0.05 to avoid overlap
    # (their eps colour on the scatter makes them identifiable).
    labelled = {0.005, 0.01, 0.02, 0.1}
    for i, e in enumerate(eps):
        if float(e) not in labelled:
            continue
        ax.annotate(
            fr"$\varepsilon{{=}}{e:g}$",
            xy=(wall[i], summary_err[i]), xytext=(9, -4),
            textcoords="offset points", fontsize=8.5,
        )

    ax.set_xlabel(
        r"Mean wall-clock per trajectory [seconds, $N = 10^{3}$, A100]"
    )
    ax.set_ylabel(
        r"$\max\,(\, |\Delta I_{\mathrm{peak}}|,"
        r"\; |\Delta R_{\mathrm{final}}|\, )$"
        " vs. exact Gillespie"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Compute\u2013fidelity Pareto (structural-bias dominated)")
    ax.grid(True, which="both", alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, label=r"$\log_{10}\varepsilon$")
    cbar.ax.tick_params(labelsize=8)

    # Vertical band highlighting the "sweet spot" region where the
    # error is within 1.2x of the structural floor.
    floor = float(summary_err.min())
    sweet_mask = summary_err < 1.2 * floor
    if sweet_mask.any():
        # Shade a horizontal band near the floor.
        ax.axhspan(floor * 0.97, 1.2 * floor, alpha=0.12, color="green",
                   label="within 1.2x of structural floor")
        ax.legend(fontsize=8.5, loc="upper right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary",
                        default="results/fidelity_summary.csv")
    parser.add_argument("--trajectories",
                        default="results/fidelity_trajectories.npz")
    parser.add_argument("--outdir",
                        default="docs/jocs/figures")
    args = parser.parse_args()

    summary = load_summary(args.summary)
    npz = load_trajectories(args.trajectories)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_trajectories(npz, outdir / "fig_fidelity_traj.png")
    plot_error_vs_eps(summary, outdir / "fig_fidelity_err_vs_eps.png")
    plot_pareto(summary, outdir / "fig_fidelity_pareto.png")


if __name__ == "__main__":
    main()
