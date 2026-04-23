#!/usr/bin/env python
"""
Run FlashSpread's MarkovianEngine on SIS and SIR and collect ensemble-mean
trajectories for validation against the exact Doob-Gillespie reference
produced by experiments/exact_gillespie_sis_sir.py.

Both models use edge-driven S -> I at rate beta * |I-neighbours|, and
constant recovery: I -> S at rate delta (SIS) or I -> R at rate gamma
(SIR). No tau-leaping tolerance is exposed: the MarkovianEngine picks
an adaptive tau internally bounded by max_prob and theta.

Output: results/flashspread_{sis,sir}.npz with the ensemble and summary.
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
from flashspread.models.compartmental import SISModel, SIRModel
from flashspread.engines.markovian import MarkovianEngine


def _build_er(N, d, device, seed=42):
    G = nx.erdos_renyi_graph(N, d / N, seed=seed, directed=False)
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
    gw.num_edges = ei.size(1)
    return gw


def run_trajectory(gw, model, device, seed, initial_frac, tf, sample_times):
    engine = MarkovianEngine(gw, model, device=device, seed=seed)
    # Deterministic seeding: infect the first `initial_I` nodes (indices
    # 0..initial_I). Matches the exact Gillespie reference.
    N = gw.num_nodes
    initial_I = max(10, int(initial_frac * N))
    idx = torch.arange(initial_I, device=device)
    engine.state[idx] = model.infected
    engine._recompute_all()

    n_samples = len(sample_times)
    frac = np.zeros(n_samples, dtype=np.float32)
    frac[0] = initial_I / N
    out_idx = 1

    while engine.current_time < tf and out_idx < n_samples:
        engine.step()
        t_now = engine.current_time
        while out_idx < n_samples and sample_times[out_idx] <= t_now:
            counts = torch.bincount(
                engine.state, minlength=model.num_states
            ).cpu().numpy()
            frac[out_idx] = counts[model.infected] / N
            out_idx += 1

    while out_idx < n_samples:
        counts = torch.bincount(
            engine.state, minlength=model.num_states
        ).cpu().numpy()
        frac[out_idx] = counts[model.infected] / N
        out_idx += 1

    return frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sis", "sir"], required=True)
    parser.add_argument("--num-nodes", type=int, default=1000)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tf", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=2.0 / 8.0)
    parser.add_argument(
        "--recovery-rate", type=float, default=0.15,
        help="delta (SIS) or gamma (SIR)",
    )
    parser.add_argument("--initial-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output",
                        default="results/flashspread_sis.npz")
    parser.add_argument("--graph-seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"FlashSpread MarkovianEngine {args.model.upper()} "
          f"(N={args.num_nodes}, d={args.degree}, trials={args.trials}, "
          f"device={device})")
    gw = _build_er(args.num_nodes, args.degree, device,
                   seed=args.graph_seed)
    print(f"  graph: {gw.num_edges} edges")

    # Build the compartment model.
    if args.model == "sis":
        model = SISModel(beta=args.beta, delta=args.recovery_rate)
    else:
        model = SIRModel(beta=args.beta, gamma=args.recovery_rate)
    if hasattr(model, "prepare"):
        model.prepare(device)

    sample_times = np.linspace(0.0, args.tf, int(args.tf / 0.5) + 1)

    trajs = np.zeros((args.trials, len(sample_times)), dtype=np.float32)
    wall_total = 0.0
    for r in range(args.trials):
        t0 = time.perf_counter()
        frac = run_trajectory(
            gw, model, device,
            seed=args.seed + 1000 * r,
            initial_frac=args.initial_frac,
            tf=args.tf, sample_times=sample_times,
        )
        wall_total += (time.perf_counter() - t0)
        trajs[r] = frac
        if (r + 1) % 10 == 0 or r == 0:
            print(f"  run {r+1}/{args.trials}: wall={wall_total/(r+1):.3f}s/trial "
                  f"peak_I={frac.max():.4f}", flush=True)

    mean_traj = trajs.mean(axis=0)
    q25 = np.percentile(trajs, 25, axis=0)
    q75 = np.percentile(trajs, 75, axis=0)

    print(f"\n  mean peak_I = {mean_traj.max():.4f} at "
          f"t = {sample_times[mean_traj.argmax()]:.1f}")
    print(f"  total wall = {wall_total:.1f}s over {args.trials} trials")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        model=np.array([args.model]),
        num_nodes=np.array([args.num_nodes]),
        degree=np.array([args.degree]),
        beta=np.array([args.beta]),
        recovery_rate=np.array([args.recovery_rate]),
        sample_times=sample_times,
        ensemble=trajs,
        mean_traj=mean_traj,
        q25=q25,
        q75=q75,
    )
    print(f"  wrote: {args.output}")


if __name__ == "__main__":
    main()
