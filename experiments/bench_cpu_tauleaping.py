#!/usr/bin/env python
"""
CPU tau-leaping benchmark to isolate algorithmic vs hardware speedup.

Runs RenewalEngine on CPU (using reference_influence fallback) at
multiple network sizes, measuring NUPS (Node-Updates Per Second).
"""

import sys
import time
import os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread.core.network import FixedDegreeGraph
from flashspread import SEIRModel
from flashspread.engines.renewal import RenewalEngine

NLINK = 8
BETA = 2.0 / NLINK
MEAN_EI, MEDIAN_EI = 5.0, 4.0
MEAN_IR, MEDIAN_IR = 7.5, 5.0
EPSILON = 0.03
TAU_MAX = 0.1
TF = 50.0


def bench_cpu(N, trials=3, tf=TF):
    """Benchmark RenewalEngine on CPU, return NUPS."""
    ncores = os.cpu_count() or 8
    torch.set_num_threads(min(ncores, 8))

    graph = FixedDegreeGraph(N, NLINK, device="cpu")
    model = SEIRModel(beta=BETA, mean_ei=MEAN_EI, median_ei=MEDIAN_EI,
                      mean_ir=MEAN_IR, median_ir=MEDIAN_IR)

    total_nups_time = 0.0
    total_node_updates = 0

    for trial in range(trials):
        engine = RenewalEngine(graph, model, device="cpu",
                               seed=12345 + trial * 100,
                               epsilon=EPSILON, tau_max=TAU_MAX)
        engine.state[0] = model.infected
        engine.age[0] = 0.0

        t0 = time.perf_counter()
        steps = 0
        while engine.current_time < tf:
            engine.step()
            steps += 1
        t1 = time.perf_counter()

        wall = t1 - t0
        nups = N * steps
        total_nups_time += wall
        total_node_updates += nups

    return total_node_updates / total_nups_time


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="1000,10000,100000,1000000")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--tf", type=float, default=50.0)
    parser.add_argument("--output", type=str, default="results/cpu_tauleaping_nups.csv")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    print(f"CPU Tau-Leaping Benchmark")
    print(f"Threads: {min(os.cpu_count() or 8, 8)}")
    print(f"Trials: {args.trials}, tf: {args.tf}")
    print()

    results = []
    for N in sizes:
        print(f"N={N:>10,}: ", end="", flush=True)
        # For large N, reduce tf to keep runtime manageable
        tf = args.tf if N <= 10000 else min(args.tf, 10.0)
        if N >= 1000000:
            tf = min(tf, 2.0)
        nups = bench_cpu(N, trials=args.trials, tf=tf)
        print(f"{nups:>12,.0f} NUPS")
        results.append({"N": N, "nups": nups})

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("engine,N,nups\n")
        for r in results:
            f.write(f"CPU tau-leaping,{r['N']},{r['nups']:.1f}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
