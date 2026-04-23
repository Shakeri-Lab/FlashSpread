#!/usr/bin/env python
"""
Mixed-precision throughput scaling benchmark (N = 100 ... 1e7).

Produces a full scaling curve for Fused CG + mixed precision directly
comparable to the Fused CG baseline curve already on Figure 6
(fig3-renewal-throughput.tex). Uses the same methodology as
experiments/bench_throughput_scoglio.py: measures realised state
transitions per wall-clock second, the "events/sec" y-axis that
Scoglio-family figures are drawn in, rather than raw NUPS.

Writes results/mixed_precision_scaling.csv which the figure's
pgfplots block picks up.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread.core.graph import GraphCSR
from flashspread import SEIRModel
from flashspread.engines.renewal_fused import RenewalEngineFusedCUDAGraph

NLINK = 8
BETA = 2.0 / NLINK
MEAN_EI, MEDIAN_EI = 5.0, 4.0
MEAN_IR, MEDIAN_IR = 7.5, 5.0
EPSILON = 0.03
TAU_MAX = 0.1
TF = 50.0
STEPS_PER_LAUNCH = 50
# Scan the same N grid Figure 6 uses so the new curve lines up.
# N=1e8 is the zoom-panel endpoint on the reviewer-revised Figure 6:
# it fits on a 40 GB A100 (~9.6 GB baseline, ~6.6 GB mixed) and sits
# well past the L2 cliff at N=1e7.
NETWORK_SIZES = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]


def create_graph(N, nlink, device):
    """Fast directed random graph (matching bench_throughput_scoglio.py)."""
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


def bench_engine(gw, model, device, use_mixed, trials=10, tf=TF,
                 warmup_trials=3):
    """Return realised-events-per-second (matches bench_throughput_scoglio.py)."""
    N = gw.num_nodes
    total_events = 0
    total_time = 0.0

    for trial in range(warmup_trials + trials):
        engine = RenewalEngineFusedCUDAGraph(
            gw, model, device=device,
            seed=12345 + trial * 100,
            epsilon=EPSILON, tau_max=TAU_MAX,
            csr_strategy="thread",            # compatible with mixed
            steps_per_launch=STEPS_PER_LAUNCH,
            use_mixed_precision=use_mixed,
        )
        # Seed: single infected node at index 0, same as the
        # original Fig 6 methodology.
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

    return total_events / total_time if total_time else 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default="results/mixed_precision_scaling.csv")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-n", type=int, default=10_000_000)
    parser.add_argument("--min-n", type=int, default=0,
                        help="Only run sizes >= this value (useful for "
                             "extending an earlier scan with just the "
                             "top-end point, e.g. N=1e8).")
    args = parser.parse_args()

    device = "cuda"
    sizes = [N for N in NETWORK_SIZES
             if args.min_n <= N <= args.max_n]
    print(f"Mixed-precision scaling: trials={args.trials}  sizes={sizes}")
    print()

    model = SEIRModel(beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
                      mean_ir=MEAN_IR, median_ir=MEDIAN_IR)

    rows = []
    for N in sizes:
        print(f"=== N = {N:>10,} ===", flush=True)
        gw = create_graph(N, NLINK, device)
        for use_mixed in (False, True):
            eps_sec = bench_engine(
                gw, model, device, use_mixed,
                trials=args.trials,
            )
            tag = "mixed" if use_mixed else "base"
            print(f"  {tag:>6}: {eps_sec:.3e} events/s", flush=True)
            rows.append({
                "N": N,
                "config": "fused_cg_mixed" if use_mixed else "fused_cg_base",
                "events_per_s": eps_sec,
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "config", "events_per_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
