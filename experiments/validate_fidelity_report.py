#!/usr/bin/env python
"""
Fidelity re-validation report + PASS/FAIL gate.

Aggregates the outputs of the fidelity-validation array job and checks that
the (post Phase-0/1) simulator still reproduces the JOCS paper's structural-
bias floor when compared to exact Gillespie:

  * SEIR epsilon-sweep (fidelity_sweep.py vs the exact reference):
      per-run peak-I error ~6% and final-R error ~7%, and epsilon-INDEPENDENT
      (the "structural bias floor"; paper appendix tab:fidelity_budget).
  * Multi-graph (fidelity_multi_graph.py vs exact_gillespie_seir.py):
      |peak(I_hat) - peak(I_exact)| per (graph, N) stays near that floor.
  * SIS/SIR (validate_sis_sir_flashspread.py vs exact_gillespie_sis_sir.py):
      L2 of the mean infected-fraction trajectory stays small.

Reads everything from --val-dir, writes a markdown summary, and exits non-zero
if any active check fails (so the SLURM merge job surfaces regressions).

This does not recompute anything; it only aggregates + asserts.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

# ---------------------------------------------------------------------------
# Tolerance bands. Grounded in the paper's reported floor (appendix
# tab:fidelity_budget: ~6% peak-I, ~7% final-R at eps=0.03, epsilon-independent).
# Bands leave margin around the floor; the point is to catch a REGRESSION
# (error blowing far past the floor), not to assert an exact number.
# ---------------------------------------------------------------------------
BANDS = {
    "eps_err_peak_I_max": 0.10,          # peak-I error at the comparison eps
    "eps_err_final_R_max": 0.12,         # final-R error at the comparison eps
    "eps_independence_spread_max": 0.03,  # max-min of err_peak_I across eps
    "multi_peak_I_err_max": 0.10,        # |peak(FS)-peak(exact)| per (graph,N)
    "sis_sir_l2_max": 0.05,              # L2(mean_traj) FS vs exact
}


def _peak(a: np.ndarray) -> float:
    return float(np.max(a))


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    m = min(len(a), len(b))
    return float(np.sqrt(np.mean((np.asarray(a[:m]) - np.asarray(b[:m])) ** 2)))


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.rows: list[tuple] = []   # (section, name, value, band, status)
        self.n_pass = 0
        self.n_fail = 0
        self.n_skip = 0

    def check(self, section: str, name: str, value, band, ok: bool) -> None:
        status = "PASS" if ok else "FAIL"
        self.rows.append((section, name, value, band, status))
        self.n_pass += ok
        self.n_fail += (not ok)

    def skip(self, section: str, name: str, why: str) -> None:
        self.rows.append((section, name, why, "-", "SKIP"))
        self.n_skip += 1


def load_eps_sweep(path: str):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            "epsilon": float(r["epsilon"]),
            "err_peak_I": float(r["err_peak_I"]),
            "err_final_R": float(r["err_final_R"]),
            "l2_traj": float(r.get("l2_traj", "nan")),
            "reference": r.get("reference", "?"),
        })
    out.sort(key=lambda d: d["epsilon"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", default="results/fidelity_val")
    ap.add_argument("--eps", type=float, default=0.03,
                    help="Comparison epsilon for the multi-graph peak-I check.")
    ap.add_argument("--graphs", default="er,ba,fixed")
    ap.add_argument("--sizes", default="1000,10000")
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    vd = args.val_dir
    out_md = args.out_md or os.path.join(vd, "fidelity_validation_summary.md")
    rep = Report()
    graphs = [g for g in args.graphs.split(",") if g]
    sizes = [int(s) for s in args.sizes.split(",") if s]

    # --- 1. SEIR epsilon-sweep structural-bias floor --------------------------
    sweep_csv = os.path.join(vd, "fidelity_summary.csv")
    if os.path.exists(sweep_csv):
        sweep = load_eps_sweep(sweep_csv)
        ref = sweep[0]["reference"] if sweep else "?"
        rep.lines.append(f"Epsilon-sweep reference: `{ref}`")
        # Pick the row nearest the default epsilon.
        row = min(sweep, key=lambda d: abs(d["epsilon"] - args.eps))
        rep.check("eps-sweep", f"err_peak_I @ eps={row['epsilon']:g}",
                  round(row["err_peak_I"], 4), f"<= {BANDS['eps_err_peak_I_max']}",
                  row["err_peak_I"] <= BANDS["eps_err_peak_I_max"])
        rep.check("eps-sweep", f"err_final_R @ eps={row['epsilon']:g}",
                  round(row["err_final_R"], 4), f"<= {BANDS['eps_err_final_R_max']}",
                  row["err_final_R"] <= BANDS["eps_err_final_R_max"])
        peaks = [d["err_peak_I"] for d in sweep]
        spread = max(peaks) - min(peaks)
        rep.check("eps-sweep", "err_peak_I spread across eps (independence)",
                  round(spread, 4), f"<= {BANDS['eps_independence_spread_max']}",
                  spread <= BANDS["eps_independence_spread_max"])
    else:
        rep.skip("eps-sweep", "fidelity_summary.csv", f"missing: {sweep_csv}")

    # --- 2. Multi-graph peak-I error vs exact Gillespie -----------------------
    fs_npz = os.path.join(vd, "fidelity_multi.npz")
    ex_npz = os.path.join(vd, "fidelity_multi_exact.npz")
    if os.path.exists(fs_npz) and os.path.exists(ex_npz):
        fs = np.load(fs_npz)
        ex = np.load(ex_npz)
        for g in graphs:
            for n in sizes:
                fs_key = f"mean_{g}_N{n}_eps{args.eps:g}"
                ex_key = f"mean_{g}_N{n}"
                if fs_key in fs and ex_key in ex:
                    err = abs(_peak(fs[fs_key]) - _peak(ex[ex_key]))
                    rep.check("multi-graph", f"peak-I err {g} N={n}",
                              round(err, 4), f"<= {BANDS['multi_peak_I_err_max']}",
                              err <= BANDS["multi_peak_I_err_max"])
                else:
                    rep.skip("multi-graph", f"{g} N={n}",
                             f"key missing (fs:{fs_key in fs}, exact:{ex_key in ex})")
    else:
        rep.skip("multi-graph", "npz pair",
                 f"missing fs={os.path.exists(fs_npz)} exact={os.path.exists(ex_npz)}")

    # --- 3. SIS/SIR trajectory match -----------------------------------------
    for model in ("sis", "sir"):
        fs_p = os.path.join(vd, f"flashspread_{model}.npz")
        ex_p = os.path.join(vd, f"exact_{model}.npz")
        if os.path.exists(fs_p) and os.path.exists(ex_p):
            fs = np.load(fs_p)
            ex = np.load(ex_p)
            err = _l2(fs["mean_traj"], ex["mean_traj"])
            rep.check("markovian", f"{model.upper()} mean-traj L2",
                      round(err, 4), f"<= {BANDS['sis_sir_l2_max']}",
                      err <= BANDS["sis_sir_l2_max"])
        else:
            rep.skip("markovian", model.upper(),
                     f"missing fs={os.path.exists(fs_p)} exact={os.path.exists(ex_p)}")

    # --- write markdown -------------------------------------------------------
    overall = "PASS" if rep.n_fail == 0 else "FAIL"
    md = ["# FlashSpread fidelity re-validation",
          "",
          f"**Overall: {overall}**  "
          f"({rep.n_pass} pass, {rep.n_fail} fail, {rep.n_skip} skip)",
          "",
          "Checks the post Phase-0/1 simulator against exact Gillespie, "
          "vs the JOCS structural-bias floor (~6% peak-I, ~7% final-R, "
          "epsilon-independent).",
          ""]
    md += rep.lines + ["", "| Section | Check | Value | Band | Result |",
                       "|---|---|---|---|---|"]
    for section, name, value, band, status in rep.rows:
        md.append(f"| {section} | {name} | {value} | {band} | {status} |")
    md.append("")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(md))

    print("\n".join(md))
    print(f"\nWrote {out_md}")
    return 0 if rep.n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
