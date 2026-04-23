#!/usr/bin/env python
"""
Exact non-Markovian Gillespie simulator for SEIR on a contact network.

The dynamics match the fused renewal engine exactly at the model level:
  * Markovian edge-mediated S -> E with rate lam_S(i) = beta * |I-neighbors(i)|.
  * Non-Markovian nodal E -> I with log-normal(mu_EI, sigma_EI) waiting time.
  * Non-Markovian nodal I -> R with log-normal(mu_IR, sigma_IR) waiting time.

Algorithm (Modified Next Reaction Method / Gibson-Bruck-style):
  1. Pre-sample each active node's firing time when it enters E or I.
  2. Maintain lam_S[i] = beta * |I-neighbors(i)|, updated incrementally
     as neighbors enter/leave I.
  3. At each step, the next candidate event is
        t_S = t + Exp(sum lam_S)              (some S node catches infection)
        t_E = min of pre-sampled E -> I times (an E node progresses)
        t_I = min of pre-sampled I -> R times (an I node recovers)
     Execute the earliest and repeat until tf.

This is the reference simulator used to overlay an "exact" curve against
tau-leaping on Figure 8 of the manuscript. Pure-Python/numpy on CPU; fast
for N up to ~10^4, workable for N = 10^5 with fewer Monte Carlo runs.
"""

from __future__ import annotations

import argparse
import heapq
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread import FixedDegreeGraph
from flashspread.core.graph import GraphCSR


_EPS = 1e-300


def _build_adj_list(edge_index_np, N):
    """Return adj[i] = sorted np.int64 array of neighbors, from a [2, E] edge_index."""
    src = edge_index_np[0]
    dst = edge_index_np[1]
    # For each undirected edge, both directions should already be present
    # (all our builders symmetrize). adj[i] = destinations of edges from i.
    order = np.argsort(src, kind="stable")
    src_s = src[order]
    dst_s = dst[order]
    # Row-start pointer
    row_ptr = np.searchsorted(src_s, np.arange(N + 1))
    adj = [dst_s[row_ptr[i]: row_ptr[i + 1]].astype(np.int64) for i in range(N)]
    return adj


def run_trajectory(adj, beta, mu_ei, sig_ei, mu_ir, sig_ir,
                   initial_E, tf, sample_times, rng):
    """Run one exact trajectory; return infected fraction on `sample_times`."""
    N = len(adj)
    state = np.zeros(N, dtype=np.int8)           # 0=S, 1=E, 2=I, 3=R
    lam_S = np.zeros(N, dtype=np.float64)        # per-S-node rate
    counts = np.zeros(4, dtype=np.int64)
    counts[0] = N

    # Heap entries: (fire_time, node_id, version_token). Lazy deletion
    # via state check at pop time.
    heap_E = []
    heap_I = []

    for i in initial_E:
        state[i] = 1
        counts[0] -= 1
        counts[1] += 1
        heapq.heappush(
            heap_E,
            (0.0 + rng.lognormal(mu_ei, sig_ei), int(i)),
        )

    t = 0.0
    sample_count = len(sample_times)
    infected_frac = np.zeros(sample_count, dtype=np.float32)
    infected_frac[0] = counts[2] / N
    out_idx = 1

    events = 0

    def _peek_valid(heap, expected_state):
        """Return (time, node) of the earliest valid event, discarding stale."""
        while heap and state[heap[0][1]] != expected_state:
            heapq.heappop(heap)
        if heap:
            return heap[0]
        return None

    # Precompute cumulative distribution scratch.
    cum_scratch = np.empty(N, dtype=np.float64)

    while t < tf:
        Delta_S = float(lam_S.sum())
        t_S = t + rng.exponential(1.0 / max(Delta_S, _EPS)) if Delta_S > 0 else float("inf")
        peek_E = _peek_valid(heap_E, 1)
        peek_I = _peek_valid(heap_I, 2)
        t_E = peek_E[0] if peek_E else float("inf")
        t_I = peek_I[0] if peek_I else float("inf")

        t_next = min(t_S, t_E, t_I)
        if t_next >= tf:
            break

        # Emit samples up to t_next.
        while out_idx < sample_count and sample_times[out_idx] <= t_next:
            infected_frac[out_idx] = counts[2] / N
            out_idx += 1

        t = t_next

        if t == t_S:
            # Categorical sample S node weighted by lam_S.
            np.cumsum(lam_S, out=cum_scratch)
            r = rng.random() * cum_scratch[-1]
            node = int(np.searchsorted(cum_scratch, r, side="right"))
            # Guard against fp rounding past the end.
            if node >= N or state[node] != 0 or lam_S[node] <= 0:
                # Retry via vectorised filter; rare.
                s_idx = np.flatnonzero(state == 0)
                node = int(rng.choice(s_idx, p=lam_S[s_idx] / lam_S[s_idx].sum()))
            state[node] = 1
            counts[0] -= 1
            counts[1] += 1
            lam_S[node] = 0.0
            heapq.heappush(heap_E, (t + rng.lognormal(mu_ei, sig_ei), node))
        elif t == t_E:
            _, node = heapq.heappop(heap_E)
            state[node] = 2
            counts[1] -= 1
            counts[2] += 1
            # Neighbor S-nodes gain beta per newly-infectious neighbor.
            for neigh in adj[node]:
                if state[neigh] == 0:
                    lam_S[neigh] += beta
            heapq.heappush(heap_I, (t + rng.lognormal(mu_ir, sig_ir), node))
        else:  # t == t_I
            _, node = heapq.heappop(heap_I)
            state[node] = 3
            counts[2] -= 1
            counts[3] += 1
            for neigh in adj[node]:
                if state[neigh] == 0:
                    lam_S[neigh] -= beta
                    # fp safety: clamp.
                    if lam_S[neigh] < 0.0:
                        lam_S[neigh] = 0.0

        events += 1

    while out_idx < sample_count:
        infected_frac[out_idx] = counts[2] / N
        out_idx += 1
    return infected_frac, events


def _build_graph(graph_key, N, device, seed=42):
    import networkx as nx
    if graph_key == "er":
        G = nx.erdos_renyi_graph(N, 8 / N, seed=seed, directed=False)
    elif graph_key == "ba":
        G = nx.barabasi_albert_graph(N, 4, seed=seed)
    elif graph_key == "fixed":
        g = FixedDegreeGraph(N, 8, device=device)
        ei = g.edge_index.cpu().numpy()
        return ei
    else:
        raise ValueError(f"Unknown graph {graph_key!r}")
    edges = []
    for u, v in G.edges():
        edges.append([u, v]); edges.append([v, u])
    return np.array(edges, dtype=np.int64).T  # [2, E]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", default="er,ba,fixed")
    parser.add_argument("--sizes", default="1000,10000")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--tf", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--output",
                        default="results/fidelity_multi_exact.npz")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    mean_ei, median_ei = 5.0, 4.0
    mean_ir, median_ir = 7.5, 5.0
    # Convert mean/median to log-normal (mu, sigma) by:
    #   median = exp(mu)      =>  mu = ln(median)
    #   mean   = exp(mu + sigma^2/2)
    mu_ei = float(np.log(median_ei))
    sig_ei = float(np.sqrt(2 * (np.log(mean_ei) - mu_ei)))
    mu_ir = float(np.log(median_ir))
    sig_ir = float(np.sqrt(2 * (np.log(mean_ir) - mu_ir)))
    beta = 2.0 / 8.0  # β = 2 / mean_degree = 0.25

    sample_times = np.linspace(0.0, args.tf, int(args.tf / 0.5) + 1)
    graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]

    print(f"Exact Gillespie SEIR (log-normal) -- non-Markovian reference")
    print(f"  graphs: {graphs}  sizes: {sizes}  trials: {args.trials}")
    print(f"  mu_EI={mu_ei:.4f} sig_EI={sig_ei:.4f}  "
          f"mu_IR={mu_ir:.4f} sig_IR={sig_ir:.4f}  beta={beta}")

    payload = {"sample_times": sample_times,
               "graphs": np.array(graphs),
               "sizes": np.array(sizes, dtype=np.int64),
               "trials": args.trials}

    for graph_key in graphs:
        for N in sizes:
            print(f"\n== {graph_key} N={N} ==", flush=True)
            edge_index = _build_graph(graph_key, N, args.device, seed=42)
            adj = _build_adj_list(edge_index, N)
            initial_E = np.arange(max(10, int(0.01 * N)), dtype=np.int64)

            trajs = np.zeros((args.trials, len(sample_times)), dtype=np.float32)
            wall_total = 0.0
            events_total = 0
            for r in range(args.trials):
                rng = np.random.default_rng(args.seed + 1000 * r)
                t0 = time.perf_counter()
                traj, events = run_trajectory(
                    adj, beta, mu_ei, sig_ei, mu_ir, sig_ir,
                    initial_E, args.tf, sample_times, rng,
                )
                t1 = time.perf_counter()
                trajs[r] = traj
                wall_total += (t1 - t0)
                events_total += events
                if (r + 1) % 5 == 0 or r == 0:
                    print(f"  run {r+1}/{args.trials}  "
                          f"events={events}  wall={t1-t0:.2f}s  "
                          f"peak_I={traj.max():.4f}", flush=True)

            mean_traj = trajs.mean(axis=0)
            q25 = np.percentile(trajs, 25, axis=0)
            q75 = np.percentile(trajs, 75, axis=0)
            key = f"{graph_key}_N{N}"
            payload[f"mean_{key}"] = mean_traj
            payload[f"q25_{key}"] = q25
            payload[f"q75_{key}"] = q75
            print(f"  -> peak(mean)={mean_traj.max():.4f} at "
                  f"t={sample_times[mean_traj.argmax()]:.1f}  "
                  f"total wall={wall_total:.1f}s  "
                  f"mean events={events_total/args.trials:.0f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
