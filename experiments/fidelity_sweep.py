#!/usr/bin/env python
"""
Tau-leaping fidelity sweep for the JOCS appendix.

For each epsilon in EPSILONS:
  - run NR stochastic trajectories of the fused renewal engine,
  - record full S/E/I/R trajectories on a common time grid,
  - collect peak-I, peak-time, final-attack-rate, wall-time, steps.

Saves:
  results/fidelity_trajectories.npz  (full ensemble trajectories per eps)
  results/fidelity_summary.csv       (per-eps scalar metrics + L-inf error
                                      vs the MATLAB exact reference if
                                      available, else vs the smallest-
                                      epsilon mean trajectory)

Downstream plots are produced by experiments/plot_fidelity.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread.core.graph import GraphCSR
from flashspread import SEIRModel
from flashspread.engines.renewal_fused import RenewalEngineFused

# Match validate_scoglio_exact.py / validate_epsilon_sweep.py.
N = 1000
NLINK = 8
BETA = 2.0 / NLINK
MEAN_EI, MEDIAN_EI = 5.0, 4.0
MEAN_IR, MEDIAN_IR = 7.5, 5.0
TAU_MAX = 0.1
TF = 50.0
NR = 100
EPSILONS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]
SAMPLE_TIMES = np.linspace(0.0, TF, int(TF / 0.1) + 1)  # 0, 0.1, ..., 50.0
NUM_STATES = 4  # S, E, I, R


def _make_graph(seed=42, device="cuda"):
    p = NLINK / N
    G = nx.erdos_renyi_graph(N, p, seed=seed, directed=False)
    edges = []
    for u, v in G.edges():
        edges.append([u, v]); edges.append([v, u])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(ei, N, incoming=True)
    class GW: pass
    gw = GW()
    gw.csr = csr
    gw.edge_index = ei
    gw.num_nodes = N
    gw.num_edges = csr.num_edges
    return gw


def run_trajectory(gw, model, device, seed, eps):
    """Run one stochastic trajectory, sample at SAMPLE_TIMES."""
    engine = RenewalEngineFused(
        gw, model, device=device, seed=seed,
        epsilon=eps, tau_max=TAU_MAX,
    )
    engine.state[0] = model.infected
    engine.age[0] = 0.0

    t = 0.0
    idx = 0
    traj = np.zeros((len(SAMPLE_TIMES), NUM_STATES), dtype=np.float32)
    # t=0 slot: fill from initial state
    counts = engine.count_by_state().cpu().numpy()
    traj[0] = counts / N
    idx = 1

    t0 = time.perf_counter()
    steps = 0
    while engine.current_time < TF and idx < len(SAMPLE_TIMES):
        engine.step()
        steps += 1
        # Catch up to all sample times we've crossed.
        while idx < len(SAMPLE_TIMES) and SAMPLE_TIMES[idx] <= engine.current_time:
            counts = engine.count_by_state().cpu().numpy()
            traj[idx] = counts / N
            idx += 1
    # Fill any remaining samples with final state.
    while idx < len(SAMPLE_TIMES):
        counts = engine.count_by_state().cpu().numpy()
        traj[idx] = counts / N
        idx += 1
    wall = time.perf_counter() - t0
    return traj, steps, wall


def load_matlab_reference(mat_path):
    """
    Load the exact non-Markovian Gillespie reference produced by the
    companion MATLAB implementation. The file stores `pre_mean` as a
    [num_states, T+1] array (already averaged over NR trials). Optional
    quantile bands are exposed as `pre_q_low` and `pre_q_high` when the
    MATLAB run recorded them.
    """
    try:
        from scipy.io import loadmat
        if not Path(mat_path).exists():
            return None, None, None
        d = loadmat(mat_path, squeeze_me=True)
        if "pre_mean" not in d:
            print(f"  [warn] {mat_path}: no pre_mean key (keys={list(d)[:6]}...)")
            return None, None, None
        pre = d["pre_mean"] / N                   # [4, T_mat+1]
        t_mat = np.arange(pre.shape[1])

        def _resample(arr):
            out = np.zeros((len(SAMPLE_TIMES), NUM_STATES), dtype=np.float32)
            for s in range(NUM_STATES):
                out[:, s] = np.interp(SAMPLE_TIMES, t_mat, arr[s])
            return out

        ref_mean = _resample(pre)
        ref_lo = _resample(d["pre_q_low"] / N) if "pre_q_low" in d else None
        ref_hi = _resample(d["pre_q_high"] / N) if "pre_q_high" in d else None
        return ref_mean, ref_lo, ref_hi
    except Exception as e:
        print(f"  [warn] MATLAB reference unavailable: {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-runs", type=int, default=NR)
    parser.add_argument(
        "--matlab",
        type=str,
        default="/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/"
                "[old] nonmarkovianGEMF/outputs/scoglio_matlab_N1000_T1000_q10_90.mat",
        help="MATLAB exact-Gillespie reference file (must contain pre_mean; "
             "optional pre_q_low / pre_q_high for quantile bands).",
    )
    parser.add_argument("--output-npz", type=str,
                        default="results/fidelity_trajectories.npz")
    parser.add_argument("--output-csv", type=str,
                        default="results/fidelity_summary.csv")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Tau-leaping fidelity sweep: N={N}, d={NLINK}, runs={args.num_runs}")
    print(f"Device: {device}  epsilons: {EPSILONS}")
    print(f"Sample grid: {SAMPLE_TIMES[0]:.1f} .. {SAMPLE_TIMES[-1]:.1f} "
          f"(step {SAMPLE_TIMES[1]-SAMPLE_TIMES[0]:.2f}, {len(SAMPLE_TIMES)} pts)")

    gw = _make_graph(seed=42, device=device)
    model = SEIRModel(
        beta=BETA,
        mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
        mean_ir=MEAN_IR, median_ir=MEDIAN_IR,
    )

    ref_matlab, ref_lo, ref_hi = load_matlab_reference(args.matlab)
    if ref_matlab is not None:
        has_bands = ref_lo is not None and ref_hi is not None
        print(f"  MATLAB reference loaded: shape {ref_matlab.shape}, "
              f"peak_I={ref_matlab[:, 2].max():.4f}, "
              f"final_R={ref_matlab[-1, 3]:.4f}"
              + (", + quantile bands" if has_bands else ""))
    else:
        print("  No MATLAB reference; will use smallest-epsilon mean as reference")

    # Run the sweep.
    results = {}
    for eps in EPSILONS:
        print(f"\n-- eps={eps} --", flush=True)
        ensemble = np.zeros((args.num_runs, len(SAMPLE_TIMES), NUM_STATES), dtype=np.float32)
        wall_times, step_counts = [], []
        for r in range(args.num_runs):
            seed = 12345 + r * 7919
            traj, steps, wall = run_trajectory(gw, model, device, seed, eps)
            ensemble[r] = traj
            wall_times.append(wall)
            step_counts.append(steps)
            if (r + 1) % 20 == 0:
                print(f"  run {r+1}/{args.num_runs}", flush=True)
        mean_traj = ensemble.mean(axis=0)
        results[eps] = {
            "ensemble": ensemble,
            "mean_traj": mean_traj,
            "wall_times": np.array(wall_times, dtype=np.float32),
            "step_counts": np.array(step_counts, dtype=np.int32),
        }

    # Reference for error calculation: prefer MATLAB exact Gillespie,
    # else fall back to the smallest-eps ensemble mean.
    smallest = min(EPSILONS)
    if ref_matlab is not None:
        reference = ref_matlab
        reference_label = "matlab_exact_gillespie"
    else:
        reference = results[smallest]["mean_traj"]
        reference_label = f"tauleap_eps{smallest:g}"

    # Metrics with per-run scalars + bootstrap 95% CIs.
    #
    # Reporting only the ensemble-mean peak / final-R produces spurious
    # non-monotonicity in the error-vs-epsilon curve (the 100-run mean
    # varies by ~0.5% across epsilons purely from sampling noise).
    # Instead we compute each run's peak-I and final-R separately, take
    # the cross-run mean as the point estimate, and a 1000-resample
    # bootstrap to get a 95% confidence interval. L-infinity remains a
    # trajectory-level metric computed from the ensemble mean.
    rng = np.random.default_rng(20260417)

    def _bootstrap_ci(x, n_resamples=1000, lo=2.5, hi=97.5):
        x = np.asarray(x, dtype=np.float64)
        if x.size < 2:
            return float("nan"), float("nan")
        samples = rng.choice(x, size=(n_resamples, x.size), replace=True)
        means = samples.mean(axis=1)
        return float(np.percentile(means, lo)), float(np.percentile(means, hi))

    ref_peak_I = float(reference[:, 2].max())
    ref_peak_time = float(SAMPLE_TIMES[np.argmax(reference[:, 2])])
    ref_final_R = float(reference[-1, 3])

    summary = []
    per_run_store = {}
    for eps in EPSILONS:
        ensemble = results[eps]["ensemble"]  # [NR, T, 4]
        mean_traj = results[eps]["mean_traj"]

        # Per-run scalars.
        per_run_peak_I = ensemble[:, :, 2].max(axis=1)      # [NR]
        per_run_peak_t = SAMPLE_TIMES[ensemble[:, :, 2].argmax(axis=1)]
        per_run_final_R = ensemble[:, -1, 3]               # [NR]

        # Point estimates from the per-run cross-run mean.
        peak_I = float(per_run_peak_I.mean())
        peak_time = float(per_run_peak_t.mean())
        final_R = float(per_run_final_R.mean())

        # Per-run absolute error against the exact reference.
        per_run_err_peak = np.abs(per_run_peak_I - ref_peak_I)
        per_run_err_final = np.abs(per_run_final_R - ref_final_R)

        err_peak_I = float(per_run_err_peak.mean())
        err_peak_time = abs(peak_time - ref_peak_time)
        err_final_R = float(per_run_err_final.mean())

        lo_peak, hi_peak = _bootstrap_ci(per_run_err_peak)
        lo_final, hi_final = _bootstrap_ci(per_run_err_final)
        lo_peakI, hi_peakI = _bootstrap_ci(per_run_peak_I)
        lo_finalR, hi_finalR = _bootstrap_ci(per_run_final_R)

        # Trajectory-level metrics (still from the ensemble mean).
        diff = mean_traj - reference
        linf = float(np.max(np.abs(diff)))
        l2 = float(np.sqrt(np.mean(diff ** 2)))
        mean_wall = float(results[eps]["wall_times"].mean())
        lo_wall, hi_wall = _bootstrap_ci(results[eps]["wall_times"])
        mean_steps = float(results[eps]["step_counts"].mean())

        per_run_store[eps] = {
            "peak_I": per_run_peak_I,
            "peak_time": per_run_peak_t,
            "final_R": per_run_final_R,
            "err_peak_I": per_run_err_peak,
            "err_final_R": per_run_err_final,
        }

        summary.append({
            "epsilon": eps,
            "peak_I": peak_I, "peak_time": peak_time, "final_R": final_R,
            "ci_peak_I_lo": lo_peakI, "ci_peak_I_hi": hi_peakI,
            "ci_final_R_lo": lo_finalR, "ci_final_R_hi": hi_finalR,
            "err_peak_I": err_peak_I,
            "ci_err_peak_I_lo": lo_peak, "ci_err_peak_I_hi": hi_peak,
            "err_peak_time": err_peak_time,
            "err_final_R": err_final_R,
            "ci_err_final_R_lo": lo_final, "ci_err_final_R_hi": hi_final,
            "linf_traj": linf, "l2_traj": l2,
            "mean_wall_time": mean_wall,
            "ci_wall_lo": lo_wall, "ci_wall_hi": hi_wall,
            "mean_steps": mean_steps,
        })

    # Save summary CSV (includes bootstrap 95% CIs).
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w") as f:
        f.write(
            "epsilon,peak_I,ci_peak_I_lo,ci_peak_I_hi,peak_time,"
            "final_R,ci_final_R_lo,ci_final_R_hi,"
            "err_peak_I,ci_err_peak_I_lo,ci_err_peak_I_hi,"
            "err_peak_time,"
            "err_final_R,ci_err_final_R_lo,ci_err_final_R_hi,"
            "linf_traj,l2_traj,"
            "mean_wall_time,ci_wall_lo,ci_wall_hi,"
            "mean_steps,reference\n"
        )
        for r in summary:
            f.write(
                f"{r['epsilon']},"
                f"{r['peak_I']:.6f},{r['ci_peak_I_lo']:.6f},{r['ci_peak_I_hi']:.6f},"
                f"{r['peak_time']:.3f},"
                f"{r['final_R']:.6f},{r['ci_final_R_lo']:.6f},{r['ci_final_R_hi']:.6f},"
                f"{r['err_peak_I']:.6f},{r['ci_err_peak_I_lo']:.6f},{r['ci_err_peak_I_hi']:.6f},"
                f"{r['err_peak_time']:.3f},"
                f"{r['err_final_R']:.6f},{r['ci_err_final_R_lo']:.6f},{r['ci_err_final_R_hi']:.6f},"
                f"{r['linf_traj']:.6f},{r['l2_traj']:.6f},"
                f"{r['mean_wall_time']:.4f},{r['ci_wall_lo']:.4f},{r['ci_wall_hi']:.4f},"
                f"{r['mean_steps']:.0f},{reference_label}\n"
            )

    # Save trajectories NPZ (full ensembles + per-run scalars + MATLAB bands).
    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        sample_times=SAMPLE_TIMES,
        reference=reference,
        reference_label=np.array([reference_label]),
        epsilons=np.array(EPSILONS, dtype=np.float32),
    )
    if ref_lo is not None and ref_hi is not None:
        payload["reference_q_low"] = ref_lo
        payload["reference_q_high"] = ref_hi
    for eps in EPSILONS:
        payload[f"mean_traj_eps{eps:g}"] = results[eps]["mean_traj"]
        payload[f"ensemble_eps{eps:g}"] = results[eps]["ensemble"]
        payload[f"wall_times_eps{eps:g}"] = results[eps]["wall_times"]
        prs = per_run_store[eps]
        payload[f"run_peak_I_eps{eps:g}"] = prs["peak_I"]
        payload[f"run_peak_time_eps{eps:g}"] = prs["peak_time"]
        payload[f"run_final_R_eps{eps:g}"] = prs["final_R"]
        payload[f"run_err_peak_I_eps{eps:g}"] = prs["err_peak_I"]
        payload[f"run_err_final_R_eps{eps:g}"] = prs["err_final_R"]
    np.savez_compressed(args.output_npz, **payload)

    print(f"\nWrote: {args.output_csv}")
    print(f"Wrote: {args.output_npz}")
    print(f"Reference: {reference_label}"
          + (" (with quantile bands)" if ref_lo is not None else ""))
    print("\nSummary (err_peak_I with bootstrap 95% CI):")
    print(f"  {'eps':>6}  {'err_peak_I':>16}  {'err_final_R':>16}  "
          f"{'L_inf':>8}  {'wall(s)':>8}")
    for r in summary:
        ci_p = f"{r['err_peak_I']:.4f} [{r['ci_err_peak_I_lo']:.4f},{r['ci_err_peak_I_hi']:.4f}]"
        ci_r = f"{r['err_final_R']:.4f} [{r['ci_err_final_R_lo']:.4f},{r['ci_err_final_R_hi']:.4f}]"
        print(f"  {r['epsilon']:>6.3f}  {ci_p:>16}  {ci_r:>16}  "
              f"{r['linf_traj']:>8.4f}  {r['mean_wall_time']:>8.3f}")


if __name__ == "__main__":
    main()
