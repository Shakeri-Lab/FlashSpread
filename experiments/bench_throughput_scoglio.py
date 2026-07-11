#!/usr/bin/env python
"""
Throughput scaling benchmark overlaying new engines with original data.

Reproduces docs/latex/figures/log_normal_SEIR_throughput.png with additional
curves for RenewalEngineNonMarkov and RenewalEngineFused (+CUDAGraph variants).

Uses the exact Scoglio setup:
  nlink=8, beta=0.25, E->I LN(5,4), I->R LN(7.5,5), epsilon=0.03, tau_max=0.1

Network sizes: 100, 1000, 10000, 100000, 1000000
"""

import sys
import time
import os
from pathlib import Path
import numpy as np
import json

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread.core.graph import GraphCSR
from flashspread import SEIRModel
from flashspread.engines.renewal import (
    RenewalEngine,
    RenewalEngineCUDAGraph,
    RenewalEngineNonMarkov,
    RenewalEngineNonMarkovCUDAGraph,
)
from flashspread.engines.renewal_fused import (
    RenewalEngineFused,
    RenewalEngineFusedCUDAGraph,
)

# Scoglio parameters
NLINK = 8
BETA = 2.0 / NLINK  # 0.25
MEAN_EI, MEDIAN_EI = 5.0, 4.0
MEAN_IR, MEDIAN_IR = 7.5, 5.0
EPSILON = 0.03
TAU_MAX = 0.1
TF = 50.0

# Original data from [old] nonmarkovianGEMF benchmarks
ORIGINAL_DATA = {
    "PYTHON": {100: 1026.7, 1000: 1577.0, 10000: 72.7},
    "MATLAB": {100: 1917.1, 1000: 2010.1, 10000: 4.5},
    "GPU (original)": {100: 266.4, 1000: 1966.0, 10000: 14489.5, 100000: 54239.3, 1000000: 42893.4},
}

NETWORK_SIZES = [100, 1000, 10000, 100000, 1000000, 10000000]


def create_graph(N, nlink, device):
    """Fast directed random graph (matching bench_scoglio_gpu.py)."""
    rng = np.random.default_rng(12345)
    k = nlink
    st = np.repeat(np.arange(N, dtype=np.int32), k)
    en = rng.integers(0, N - 1, size=N * k, dtype=np.int32)
    en = en + (en >= st).astype(np.int32)
    edge_index = torch.tensor(np.stack([st, en]), dtype=torch.long, device=device)
    csr = GraphCSR(edge_index, N, incoming=True)

    class GW:
        pass
    gw = GW()
    gw.csr = csr
    gw.edge_index = edge_index
    gw.num_nodes = N
    gw.num_edges = csr.num_edges
    return gw


def bench_engine(engine_class, gw, model, device, trials=20, tf=TF,
                 warmup_trials=3, **engine_kwargs):
    """Benchmark an engine, return events/sec.

    Counts total state transitions (event_mask.sum() per step), matching
    the original GPU benchmark methodology from bench_scoglio_gpu.py.
    """
    N = gw.num_nodes
    total_events = 0
    total_time = 0.0

    # Detect batched engines (CUDAGraph variants)
    is_batched = any(k in str(engine_class) for k in ["CUDAGraph", "Fused"])

    for trial in range(warmup_trials + trials):
        engine = engine_class(gw, model, device=device,
                              seed=12345 + trial * 100,
                              epsilon=EPSILON, tau_max=TAU_MAX,
                              **engine_kwargs)
        engine.state[0] = model.infected
        engine.age[0] = 0.0

        prev_state = engine.state.clone()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        events_trial = 0
        while engine.current_time < tf:
            engine.step()
            # Count transitions: nodes whose state changed this step
            changed = (engine.state != prev_state).sum(dtype=torch.int64).item()
            events_trial += changed
            prev_state.copy_(engine.state)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if trial >= warmup_trials:
            total_events += max(1, events_trial)
            total_time += (t1 - t0)

    if total_time == 0:
        return 0.0
    return total_events / total_time


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/throughput_scoglio_overlay.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    device = args.device
    print(f"Scoglio Throughput Benchmark")
    print(f"beta={BETA}, nlink={NLINK}, epsilon={EPSILON}, tau_max={TAU_MAX}")
    print(f"Trials per size: {args.trials}")
    print()

    model = SEIRModel(beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
                      mean_ir=MEAN_IR, median_ir=MEDIAN_IR)

    # New engines to benchmark
    new_engines = {
        "GPU (RenewalEngine)": (RenewalEngine, {}),
        "GPU CUDAGraph": (RenewalEngineCUDAGraph, {"steps_per_launch": 50}),
        "NonMarkov": (RenewalEngineNonMarkov, {}),
        "NonMarkov CG": (RenewalEngineNonMarkovCUDAGraph, {"steps_per_launch": 50}),
        "Fused": (RenewalEngineFused, {}),
        "Fused CG": (RenewalEngineFusedCUDAGraph, {"steps_per_launch": 50}),
    }

    results = {name: {} for name in new_engines}

    for N in NETWORK_SIZES:
        print(f"\n--- N = {N:,} ---")
        try:
            gw = create_graph(N, NLINK, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"  Graph creation FAILED: {e}")
            torch.cuda.empty_cache()
            continue
        print(f"  Edges: {gw.num_edges:,}")

        # Match original benchmark: always tf=50, scale trials with N
        tf = TF  # always 50.0
        if N <= 10000:
            trials = args.trials
        elif N <= 100000:
            trials = max(5, args.trials)
        else:
            trials = max(3, args.trials // 2)

        for eng_name, (eng_class, eng_kwargs) in new_engines.items():
            try:
                eps = bench_engine(eng_class, gw, model, device,
                                   trials=trials, tf=tf, **eng_kwargs)
                results[eng_name][N] = eps
                print(f"  {eng_name:>20s}: {eps:>10.1f} events/s")
            except torch.cuda.OutOfMemoryError:
                print(f"  {eng_name:>20s}: OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {eng_name:>20s}: FAILED - {e}")
                torch.cuda.empty_cache()

    # Summary CSV first, so a headless node without matplotlib never loses the data.
    csv_path = args.output.replace(".png", ".csv")
    with open(csv_path, "w") as f:
        f.write("engine,N,events_per_sec\n")
        for algo, data in ORIGINAL_DATA.items():
            for n, eps in sorted(data.items()):
                f.write(f"{algo},{n},{eps:.1f}\n")
        for eng_name, data in results.items():
            for n, eps in sorted(data.items()):
                f.write(f"{eng_name},{n},{eps:.1f}\n")
    print(f"Saved: {csv_path}")

    # Plot (optional: matplotlib is not needed for the CSV artefact).
    print("\nGenerating plot...", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as _e:
        print(f"[plot skipped: matplotlib unavailable: {_e}]")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Original data (from the existing plot)
    orig_styles = {
        "PYTHON": {"color": "#1f77b4", "marker": "o", "ls": "-", "lw": 1.5},
        "MATLAB": {"color": "#ff7f0e", "marker": "s", "ls": "-", "lw": 1.5},
        "GPU (original)": {"color": "#2ca02c", "marker": "^", "ls": "-", "lw": 1.5},
    }
    for algo, data in ORIGINAL_DATA.items():
        ns = sorted(data.keys())
        eps = [data[n] for n in ns]
        s = orig_styles[algo]
        ax.plot(ns, eps, label=algo, color=s["color"], marker=s["marker"],
                ls=s["ls"], lw=s["lw"], markersize=7)

    # New engine data
    new_styles = {
        "GPU (RenewalEngine)": {"color": "#2ca02c", "marker": "v", "ls": "--", "lw": 1.5},
        "GPU CUDAGraph": {"color": "#2ca02c", "marker": "D", "ls": ":", "lw": 2.0},
        "NonMarkov": {"color": "#9467bd", "marker": "p", "ls": "-", "lw": 1.5},
        "NonMarkov CG": {"color": "#9467bd", "marker": "H", "ls": "--", "lw": 2.0},
        "Fused": {"color": "#d62728", "marker": "*", "ls": "-", "lw": 1.5},
        "Fused CG": {"color": "#d62728", "marker": "P", "ls": "--", "lw": 2.5},
    }
    for eng_name, data in results.items():
        if not data:
            continue
        ns = sorted(data.keys())
        eps = [data[n] for n in ns]
        s = new_styles.get(eng_name, {"color": "gray", "marker": "x", "ls": "-", "lw": 1})
        ax.plot(ns, eps, label=eng_name, color=s["color"], marker=s["marker"],
                ls=s["ls"], lw=s["lw"], markersize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network size N", fontsize=12)
    ax.set_ylabel("Events per second", fontsize=12)
    ax.set_title("Throughput Scaling (Scoglio Setup)\nOriginal + New Non-Markovian Engines",
                 fontsize=13, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper left")

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
