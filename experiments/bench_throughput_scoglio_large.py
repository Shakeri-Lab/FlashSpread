#!/usr/bin/env python
"""
Extend throughput benchmark to N=10^7 and 10^8, merge with existing CSV,
and regenerate the overlay plot.
"""

import sys
import time
import os
from pathlib import Path
import numpy as np
import csv

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

NLINK = 8
BETA = 2.0 / NLINK
MEAN_EI, MEDIAN_EI = 5.0, 4.0
MEAN_IR, MEDIAN_IR = 7.5, 5.0
EPSILON = 0.03
TAU_MAX = 0.1
TF = 10.0  # shorter for huge graphs


def create_graph(N, nlink, device):
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


def bench_engine(engine_class, gw, model, device, trials=3, tf=TF,
                 warmup_trials=2, **engine_kwargs):
    """Benchmark using cumulative state transitions (matching original metric)."""
    N = gw.num_nodes
    total_events = 0
    total_time = 0.0

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
    parser.add_argument("--existing-csv", type=str,
                        default="results/throughput_scoglio_overlay.csv")
    parser.add_argument("--output", type=str,
                        default="results/throughput_scoglio_overlay.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    device = args.device
    print(f"Large-scale throughput extension: N=10^7, 10^8")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem_gb:.0f} GB")
    print()

    model = SEIRModel(beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
                      mean_ir=MEAN_IR, median_ir=MEDIAN_IR)

    large_sizes = [10_000_000, 100_000_000]

    engines = {
        "GPU (RenewalEngine)": (RenewalEngine, {}),
        "GPU CUDAGraph": (RenewalEngineCUDAGraph, {"steps_per_launch": 50}),
        "NonMarkov": (RenewalEngineNonMarkov, {}),
        "NonMarkov CG": (RenewalEngineNonMarkovCUDAGraph, {"steps_per_launch": 50}),
        "Fused": (RenewalEngineFused, {}),
        "Fused CG": (RenewalEngineFusedCUDAGraph, {"steps_per_launch": 50}),
    }

    new_rows = []
    for N in large_sizes:
        print(f"\n{'='*60}")
        print(f"N = {N:,} (10^{len(str(N))-1})")
        print(f"{'='*60}")

        # Check memory feasibility: E = N * nlink * 2 directions, ~4 bytes each
        est_mem_gb = N * NLINK * 2 * 4 / 1e9  # col_ind
        est_mem_gb += N * NLINK * 2 * 4 / 1e9  # weights
        est_mem_gb += (N + 1) * 4 / 1e9  # row_ptr
        est_mem_gb += N * 4 * 10 / 1e9  # state buffers
        print(f"Estimated GPU memory: {est_mem_gb:.1f} GB")

        if device == "cuda":
            mem_avail = torch.cuda.get_device_properties(0).total_memory / 1e9
            if est_mem_gb > mem_avail * 0.85:
                print(f"  SKIPPING: would need {est_mem_gb:.1f} GB, only {mem_avail:.0f} GB available")
                continue

        try:
            print("Creating graph...", flush=True)
            gw = create_graph(N, NLINK, device)
            print(f"  Edges: {gw.num_edges:,}")
        except Exception as e:
            print(f"  Graph creation FAILED: {e}")
            continue

        for eng_name, (eng_class, eng_kwargs) in engines.items():
            print(f"  {eng_name:>20s}: ", end="", flush=True)
            try:
                eps = bench_engine(eng_class, gw, model, device,
                                   trials=args.trials, tf=TF, **eng_kwargs)
                new_rows.append({"engine": eng_name, "N": N, "events_per_sec": eps})
                print(f"{eps:>10.1f} events/s")
            except torch.cuda.OutOfMemoryError:
                print("OOM")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"FAILED - {e}")
                torch.cuda.empty_cache()

        # Free GPU memory before next size
        del gw
        torch.cuda.empty_cache()

    # Load existing CSV and merge
    print("\nMerging with existing data...", flush=True)
    existing = []
    if os.path.exists(args.existing_csv):
        with open(args.existing_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append({
                    "engine": row["engine"],
                    "N": int(row["N"]),
                    "events_per_sec": float(row["events_per_sec"]),
                })

    # Remove old entries for the large sizes we just benchmarked
    benchmarked_sizes = set(r["N"] for r in new_rows)
    merged = [r for r in existing if r["N"] not in benchmarked_sizes]
    merged.extend(new_rows)

    # Write merged CSV
    csv_path = args.output.replace(".png", ".csv")
    with open(csv_path, "w") as f:
        f.write("engine,N,events_per_sec\n")
        for r in sorted(merged, key=lambda x: (x["engine"], x["N"])):
            f.write(f"{r['engine']},{r['N']},{r['events_per_sec']:.1f}\n")
    print(f"Saved: {csv_path}")

    # Plot
    print("Generating plot...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by engine
    by_engine = {}
    for r in merged:
        by_engine.setdefault(r["engine"], []).append(r)
    for k in by_engine:
        by_engine[k].sort(key=lambda x: x["N"])

    styles = {
        "PYTHON":              {"color": "#1f77b4", "marker": "o",  "ls": "-",  "lw": 1.5, "ms": 7},
        "MATLAB":              {"color": "#ff7f0e", "marker": "s",  "ls": "-",  "lw": 1.5, "ms": 7},
        "GPU (original)":      {"color": "#2ca02c", "marker": "^",  "ls": "-",  "lw": 1.5, "ms": 7},
        "GPU (RenewalEngine)": {"color": "#2ca02c", "marker": "v",  "ls": "--", "lw": 1.5, "ms": 6},
        "GPU CUDAGraph":       {"color": "#2ca02c", "marker": "D",  "ls": ":",  "lw": 2.0, "ms": 6},
        "NonMarkov":           {"color": "#9467bd", "marker": "p",  "ls": "-",  "lw": 1.5, "ms": 7},
        "NonMarkov CG":        {"color": "#9467bd", "marker": "H",  "ls": "--", "lw": 2.0, "ms": 7},
        "Fused":               {"color": "#d62728", "marker": "*",  "ls": "-",  "lw": 1.5, "ms": 9},
        "Fused CG":            {"color": "#d62728", "marker": "P",  "ls": "--", "lw": 2.5, "ms": 9},
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    # Plot order: originals first, then new
    order = ["PYTHON", "MATLAB", "GPU (original)",
             "GPU (RenewalEngine)", "GPU CUDAGraph",
             "NonMarkov", "NonMarkov CG",
             "Fused", "Fused CG"]

    for eng in order:
        if eng not in by_engine:
            continue
        data = by_engine[eng]
        ns = [d["N"] for d in data]
        eps = [d["events_per_sec"] for d in data]
        s = styles.get(eng, {"color": "gray", "marker": "x", "ls": "-", "lw": 1, "ms": 5})
        ax.plot(ns, eps, label=eng, color=s["color"], marker=s["marker"],
                ls=s["ls"], lw=s["lw"], markersize=s["ms"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Network size N", fontsize=12)
    ax.set_ylabel("Events per second", fontsize=12)
    ax.set_title("Throughput Scaling (Scoglio Setup)\n"
                 "Original + New Non-Markovian Engines",
                 fontsize=13, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper left")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
