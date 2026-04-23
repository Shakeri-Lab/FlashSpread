#!/usr/bin/env python
"""
Epsilon convergence sweep with quantitative error metrics.

Runs the fused engine at multiple epsilon values and compares against
the MATLAB exact baseline, reporting mean ± 95% CI for peak prevalence
error, final attack rate error, and runtime per epsilon.
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
from flashspread.engines.renewal_fused import RenewalEngineFused

N = 1000
NLINK = 8
BETA = 2.0 / NLINK
MEAN_EI, MEDIAN_EI = 5.0, 4.0
MEAN_IR, MEDIAN_IR = 7.5, 5.0
TAU_MAX = 0.1
TF = 50
NR = 100
EPSILONS = [0.005, 0.01, 0.03, 0.05, 0.1]


def create_graph(n, nlink, device, seed=42):
    p = nlink / n
    G = nx.erdos_renyi_graph(n, p, seed=seed, directed=False)
    edges = []
    for u, v in G.edges():
        edges.append([u, v])
        edges.append([v, u])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(ei, n, incoming=True)
    class GW:
        pass
    gw = GW()
    gw.csr = csr
    gw.edge_index = ei
    gw.num_nodes = n
    gw.num_edges = csr.num_edges
    return gw


def run_one(gw, model, device, seed, epsilon):
    engine = RenewalEngineFused(gw, model, device=device, seed=seed,
                                 epsilon=epsilon, tau_max=TAU_MAX)
    engine.state[0] = model.infected
    engine.age[0] = 0.0

    t0 = time.perf_counter()
    steps = 0
    peak_I = 0
    peak_time = 0.0

    while engine.current_time < TF:
        engine.step()
        steps += 1
        counts = engine.count_by_state()
        i_frac = counts[2].item() / N
        if i_frac > peak_I:
            peak_I = i_frac
            peak_time = engine.current_time

    t1 = time.perf_counter()
    counts = engine.count_by_state()
    final_R = counts[3].item() / N
    return {
        "peak_I": peak_I,
        "peak_time": peak_time,
        "final_R": final_R,
        "wall_time": t1 - t0,
        "steps": steps,
    }


def load_matlab_baseline(orig_dir):
    """Load MATLAB exact results for comparison."""
    try:
        from scipy.io import loadmat
        for name in ["scoglio_matlab_7007987.mat"]:
            path = Path(orig_dir) / name
            if path.exists():
                d = loadmat(str(path), squeeze_me=True)
                pre = d["pre_mean"][:, :TF + 1] / N
                peak_I = pre[2].max()
                peak_time = float(np.argmax(pre[2]))
                final_R = pre[3, -1]
                return {"peak_I": peak_I, "peak_time": peak_time, "final_R": final_R}
    except ImportError:
        pass
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="results/epsilon_sweep.csv")
    parser.add_argument("--orig-dir", type=str,
                        default="/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/"
                                "[old] nonmarkovianGEMF/outputs")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Epsilon Convergence Sweep: N={N}, d={NLINK}, runs={NR}")
    print(f"Device: {device}")
    print()

    gw = create_graph(N, NLINK, device)
    model = SEIRModel(beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
                      mean_ir=MEAN_IR, median_ir=MEDIAN_IR)

    # Load exact baseline
    matlab = load_matlab_baseline(args.orig_dir)
    if matlab:
        print(f"MATLAB baseline: peak_I={matlab['peak_I']:.4f}, "
              f"peak_time={matlab['peak_time']:.1f}, final_R={matlab['final_R']:.4f}")
    else:
        print("MATLAB baseline not found; reporting absolute values only")
    print()

    rows = []
    for eps in EPSILONS:
        print(f"eps={eps}: ", end="", flush=True)
        peak_Is, peak_times, final_Rs, wall_times, step_counts = [], [], [], [], []

        for run in range(NR):
            seed = 12345 + run * 7919
            r = run_one(gw, model, device, seed, eps)
            peak_Is.append(r["peak_I"])
            peak_times.append(r["peak_time"])
            final_Rs.append(r["final_R"])
            wall_times.append(r["wall_time"])
            step_counts.append(r["steps"])

        mean_pI = np.mean(peak_Is)
        ci_pI = 1.96 * np.std(peak_Is) / np.sqrt(NR)
        mean_pT = np.mean(peak_times)
        mean_fR = np.mean(final_Rs)
        ci_fR = 1.96 * np.std(final_Rs) / np.sqrt(NR)
        mean_wt = np.mean(wall_times)
        mean_steps = np.mean(step_counts)

        # Errors vs MATLAB
        err_pI = abs(mean_pI - matlab["peak_I"]) if matlab else 0
        err_fR = abs(mean_fR - matlab["final_R"]) if matlab else 0
        err_pT = abs(mean_pT - matlab["peak_time"]) if matlab else 0

        print(f"peak_I={mean_pI:.4f}±{ci_pI:.4f}  final_R={mean_fR:.4f}±{ci_fR:.4f}  "
              f"steps={mean_steps:.0f}  time={mean_wt:.2f}s")

        rows.append({
            "epsilon": eps,
            "peak_I": mean_pI, "ci_peak_I": ci_pI,
            "peak_time": mean_pT,
            "final_R": mean_fR, "ci_final_R": ci_fR,
            "err_peak_I": err_pI, "err_final_R": err_fR, "err_peak_time": err_pT,
            "mean_steps": mean_steps, "mean_wall_time": mean_wt,
        })

    # Save CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("epsilon,peak_I,ci_peak_I,peak_time,final_R,ci_final_R,"
                "err_peak_I,err_final_R,err_peak_time,mean_steps,mean_wall_time\n")
        for r in rows:
            f.write(f"{r['epsilon']},{r['peak_I']:.6f},{r['ci_peak_I']:.6f},"
                    f"{r['peak_time']:.2f},{r['final_R']:.6f},{r['ci_final_R']:.6f},"
                    f"{r['err_peak_I']:.6f},{r['err_final_R']:.6f},{r['err_peak_time']:.2f},"
                    f"{r['mean_steps']:.0f},{r['mean_wall_time']:.3f}\n")
    print(f"\nSaved: {args.output}")

    # Print LaTeX table
    print("\nLaTeX table:")
    print(r"\begin{tabular}{ccccccc}")
    print(r"\toprule")
    print(r"$\varepsilon$ & Peak $I$ & $\pm$95\%CI & Final $R$ & $\pm$95\%CI & Steps & Time (s) \\")
    print(r"\midrule")
    for r in rows:
        print(f"{r['epsilon']} & {r['peak_I']:.4f} & {r['ci_peak_I']:.4f} & "
              f"{r['final_R']:.4f} & {r['ci_final_R']:.4f} & "
              f"{r['mean_steps']:.0f} & {r['mean_wall_time']:.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
