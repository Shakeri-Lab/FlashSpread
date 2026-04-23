#!/usr/bin/env python
"""
Exact Markovian Gillespie reference for SIS and SIR on a contact network.

Both dynamics are edge-mediated with exponential waiting times:

  SIS:  S -> I at rate beta * |I-neighbors(i)|
        I -> S at rate delta  (constant)

  SIR:  S -> I at rate beta * |I-neighbors(i)|
        I -> R at rate gamma (constant)

Algorithm (Doob-Gillespie direct method, maintained incrementally):
  * Per S-node: lam_S[i] = beta * |I-neighbors(i)|
  * Per I-node: lam_I[i] = delta (SIS) or gamma (SIR), constant
  * At each step: pick next event time T ~ Exp(Lambda_tot);
    pick reaction type (S->I, I->S/R) by its sub-total; pick a
    specific node proportional to its rate; apply; update neighbours.

No tau-leaping approximation is made: every reaction is resolved exactly.
This is the ground-truth reference for validating FlashSpread's
MarkovianEngine (tau-leaping) on SIS and SIR.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread.core.graph import GraphCSR


def _build_adj_list(edge_index_np, N):
    """Return adj[i] = sorted np.int64 array of i's neighbors."""
    src = edge_index_np[0]
    dst = edge_index_np[1]
    order = np.argsort(src, kind="stable")
    src_s = src[order]
    dst_s = dst[order]
    row_ptr = np.searchsorted(src_s, np.arange(N + 1))
    return [dst_s[row_ptr[i]: row_ptr[i + 1]].astype(np.int64) for i in range(N)]


def run_trajectory_sis(adj, beta, delta, initial_I, tf, sample_times, rng):
    """
    One exact Markovian SIS trajectory; returns infected fraction on grid.
    """
    N = len(adj)
    state = np.zeros(N, dtype=np.int8)          # 0=S, 1=I
    lam_S = np.zeros(N, dtype=np.float64)       # only nonzero on S nodes
    state[initial_I] = 1

    # Initialize lam_S from the initial I-nodes.
    for j in initial_I:
        for nb in adj[j]:
            if state[nb] == 0:
                lam_S[nb] += beta

    I_set = set(int(j) for j in initial_I)

    t = 0.0
    sample_count = len(sample_times)
    frac = np.zeros(sample_count, dtype=np.float32)
    frac[0] = len(I_set) / N
    out_idx = 1

    events = 0
    while t < tf:
        Delta_S = float(lam_S.sum())
        Delta_I = delta * len(I_set)
        Lambda = Delta_S + Delta_I
        if Lambda <= 0.0:
            break  # no more reactions possible

        dt = rng.exponential(1.0 / Lambda)
        t_next = t + dt
        while out_idx < sample_count and sample_times[out_idx] <= t_next:
            frac[out_idx] = len(I_set) / N
            out_idx += 1
        if t_next >= tf:
            break
        t = t_next

        # Reaction category
        u = rng.random() * Lambda
        if u < Delta_S:
            # S -> I: categorical sample over S rates
            cum = np.cumsum(lam_S)
            r = rng.random() * cum[-1]
            node = int(np.searchsorted(cum, r, side="right"))
            if node >= N or state[node] != 0 or lam_S[node] <= 0:
                # Rare fp edge case -- retry exactly.
                s_idx = np.flatnonzero(state == 0)
                if s_idx.size == 0:
                    continue
                node = int(rng.choice(s_idx,
                                      p=lam_S[s_idx] / lam_S[s_idx].sum()))
            state[node] = 1
            lam_S[node] = 0.0
            I_set.add(node)
            for nb in adj[node]:
                if state[nb] == 0:
                    lam_S[nb] += beta
        else:
            # I -> S: uniform over I nodes
            node = int(rng.choice(list(I_set)))
            state[node] = 0
            I_set.discard(node)
            # Rebuild lam_S for this ex-I's S-neighbors.
            for nb in adj[node]:
                if state[nb] == 0:
                    lam_S[nb] -= beta
                    if lam_S[nb] < 0:
                        lam_S[nb] = 0.0
            # Also this node just became S, its lam_S from remaining I-neighbors.
            new_lam = 0.0
            for nb in adj[node]:
                if state[nb] == 1:
                    new_lam += beta
            lam_S[node] = new_lam
        events += 1

    while out_idx < sample_count:
        frac[out_idx] = len(I_set) / N
        out_idx += 1
    return frac, events


def run_trajectory_sir(adj, beta, gamma, initial_I, tf, sample_times, rng):
    """
    One exact Markovian SIR trajectory; returns infected fraction on grid.
    """
    N = len(adj)
    state = np.zeros(N, dtype=np.int8)          # 0=S, 1=I, 2=R
    lam_S = np.zeros(N, dtype=np.float64)
    state[initial_I] = 1
    for j in initial_I:
        for nb in adj[j]:
            if state[nb] == 0:
                lam_S[nb] += beta
    I_set = set(int(j) for j in initial_I)

    t = 0.0
    sample_count = len(sample_times)
    frac = np.zeros(sample_count, dtype=np.float32)
    frac[0] = len(I_set) / N
    out_idx = 1

    events = 0
    while t < tf:
        Delta_S = float(lam_S.sum())
        Delta_I = gamma * len(I_set)
        Lambda = Delta_S + Delta_I
        if Lambda <= 0.0:
            break

        dt = rng.exponential(1.0 / Lambda)
        t_next = t + dt
        while out_idx < sample_count and sample_times[out_idx] <= t_next:
            frac[out_idx] = len(I_set) / N
            out_idx += 1
        if t_next >= tf:
            break
        t = t_next

        u = rng.random() * Lambda
        if u < Delta_S:
            cum = np.cumsum(lam_S)
            r = rng.random() * cum[-1]
            node = int(np.searchsorted(cum, r, side="right"))
            if node >= N or state[node] != 0 or lam_S[node] <= 0:
                s_idx = np.flatnonzero(state == 0)
                if s_idx.size == 0:
                    continue
                node = int(rng.choice(s_idx,
                                      p=lam_S[s_idx] / lam_S[s_idx].sum()))
            state[node] = 1
            lam_S[node] = 0.0
            I_set.add(node)
            for nb in adj[node]:
                if state[nb] == 0:
                    lam_S[nb] += beta
        else:
            # I -> R: uniform over I nodes; R is absorbing, no neighbor refresh
            node = int(rng.choice(list(I_set)))
            state[node] = 2
            I_set.discard(node)
            for nb in adj[node]:
                if state[nb] == 0:
                    lam_S[nb] -= beta
                    if lam_S[nb] < 0:
                        lam_S[nb] = 0.0
        events += 1

    while out_idx < sample_count:
        frac[out_idx] = len(I_set) / N
        out_idx += 1
    return frac, events


def _build_graph(graph_key, N, device, seed=42):
    import networkx as nx
    if graph_key == "er":
        G = nx.erdos_renyi_graph(N, 8 / N, seed=seed, directed=False)
    elif graph_key == "ba":
        G = nx.barabasi_albert_graph(N, 4, seed=seed)
    else:
        raise ValueError(f"Unknown graph {graph_key!r}")
    edges = []
    for u, v in G.edges():
        edges.append([u, v]); edges.append([v, u])
    return np.array(edges, dtype=np.int64).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sis", "sir"], required=True)
    parser.add_argument("--graph", default="er")
    parser.add_argument("--num-nodes", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tf", type=float, default=50.0)
    parser.add_argument("--beta", type=float, default=2.0 / 8.0)
    parser.add_argument(
        "--recovery-rate", type=float, default=0.15,
        help="delta (SIS recovery back to S) or gamma (SIR recovery to R)",
    )
    parser.add_argument("--initial-frac", type=float, default=0.01,
                        help="Fraction of nodes initially infected.")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--output",
                        default="results/exact_sis_sir.npz")
    args = parser.parse_args()

    sample_times = np.linspace(0.0, args.tf, int(args.tf / 0.5) + 1)

    print(f"Exact Markovian {args.model.upper()} Gillespie "
          f"(N={args.num_nodes}, {args.graph}, trials={args.trials})")
    edge_index = _build_graph(args.graph, args.num_nodes, "cpu")
    adj = _build_adj_list(edge_index, args.num_nodes)
    initial_I = np.arange(max(10, int(args.initial_frac * args.num_nodes)),
                          dtype=np.int64)
    print(f"  beta={args.beta}  recovery_rate={args.recovery_rate}  "
          f"initial_I={len(initial_I)}")

    trajs = np.zeros((args.trials, len(sample_times)), dtype=np.float32)
    wall_total = 0.0
    events_total = 0
    for r in range(args.trials):
        rng = np.random.default_rng(args.seed + 1000 * r)
        t0 = time.perf_counter()
        if args.model == "sis":
            frac, events = run_trajectory_sis(
                adj, args.beta, args.recovery_rate,
                initial_I, args.tf, sample_times, rng,
            )
        else:
            frac, events = run_trajectory_sir(
                adj, args.beta, args.recovery_rate,
                initial_I, args.tf, sample_times, rng,
            )
        wall_total += (time.perf_counter() - t0)
        events_total += events
        trajs[r] = frac
        if (r + 1) % 10 == 0 or r == 0:
            print(f"  run {r+1}/{args.trials}: events={events} "
                  f"wall={wall_total/(r+1):.2f}s/trial  peak_I={frac.max():.4f}",
                  flush=True)

    mean_traj = trajs.mean(axis=0)
    q25 = np.percentile(trajs, 25, axis=0)
    q75 = np.percentile(trajs, 75, axis=0)

    print(f"\n  mean peak_I = {mean_traj.max():.4f} at "
          f"t = {sample_times[mean_traj.argmax()]:.1f}")
    print(f"  mean events/trial = {events_total / args.trials:.0f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        model=np.array([args.model]),
        graph=np.array([args.graph]),
        num_nodes=np.array([args.num_nodes]),
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
