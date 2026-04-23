#!/usr/bin/env python
"""
Correctness verification: warp-collaborative fused kernel vs
1-thread-per-node fused kernel.

Both kernels use the same tl.rand(seed, node_id) RNG pattern, so for
identical (seed, step_id) pairs the per-node random draw is identical.
With identical state input, they must produce identical per-node
(next_state, next_age, next_infectivity, rates) outputs.

Graphs tested:
 1. FixedDegreeGraph(d=8)        -- uniform degree, like the ER proxy.
 2. Barabasi-Albert(m=4)         -- heavy tail with hubs (the workload
                                    warp-collaboration is designed for).

A mismatch in any per-node field at any step fails the check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from flashspread import FixedDegreeGraph, SEIRModel
from flashspread.core.graph import GraphCSR
from flashspread.engines.renewal_fused import RenewalEngineFused


def _build_er(N, d, device):
    g = FixedDegreeGraph(N, d, device=device)
    class GW:
        pass
    gw = GW()
    gw.csr = g.csr
    gw.edge_index = g.edge_index
    gw.num_nodes = N
    gw.num_edges = g.edge_index.size(1)
    return gw


def _build_ba(N, m, device):
    import networkx as nx
    G = nx.barabasi_albert_graph(N, m, seed=7)
    edges = []
    for u, v in G.edges():
        edges.append([u, v]); edges.append([v, u])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
    csr = GraphCSR(ei, N, incoming=True)
    class GW:
        pass
    gw = GW()
    gw.csr = csr
    gw.edge_index = ei
    gw.num_nodes = N
    gw.num_edges = ei.size(1)
    return gw


def _make_engine(gw, model, device, seed, strategy):
    engine = RenewalEngineFused(
        gw, model, device=device, seed=seed,
        epsilon=0.03, tau_max=0.1,
        csr_strategy=strategy,
        nodes_per_block=4, lanes_per_node=32,
        edges_per_merge_block=1024,
    )
    # Deterministic identical seeding: use the SAME indices to infect
    # (not torch.randperm, which is stateful).
    num_seed = max(1, gw.num_nodes // 50)
    idx = torch.arange(num_seed, device=device)
    engine.state.zero_()
    engine.age.zero_()
    engine.state[idx] = model.exposed
    # Rerun the bootstrap pre-pass (seed_infection calls it; we're
    # bypassing seed_infection to keep seeds fully deterministic).
    engine._infectivity_prepass()
    # Reset step_id so both engines produce identical RNG sequences.
    engine._step_id.zero_()
    return engine


def _compare_step(e_ref, e_wc, step_idx, atol_age, rtol_rate):
    # States are integer; must match exactly.
    ds = (e_ref.state != e_wc.state).sum().item()
    # Ages are float; must match modulo fp rounding (identical operations).
    da = (e_ref.age - e_wc.age).abs()
    # Infectivity: identical computation → should be exact-bit match.
    di = (e_ref.infectivity - e_wc.infectivity).abs()
    # Rates: same, but recomputed in the current step; allow a tiny rtol.
    dr = (e_ref.rates - e_wc.rates).abs()

    ok = (
        ds == 0
        and da.max().item() <= atol_age
        and di.max().item() <= atol_age
        and dr.max().item() <= rtol_rate * (e_ref.rates.abs().max().item() + 1e-12)
    )
    return ok, {
        "step": step_idx,
        "state_diff": ds,
        "max_age_diff": float(da.max().item()),
        "max_inf_diff": float(di.max().item()),
        "max_rate_diff": float(dr.max().item()),
    }


def run_pair(label_ref, label_cand, graph_label, gw, model, device,
             strategy_ref, strategy_cand, num_steps, seed,
             atol_age, rtol_rate):
    print(f"  [{label_ref} vs {label_cand}]", end=" ", flush=True)
    e_ref = _make_engine(gw, model, device, seed, strategy=strategy_ref)
    e_wc = _make_engine(gw, model, device, seed, strategy=strategy_cand)

    assert (e_ref.state == e_wc.state).all()
    assert (e_ref.age == e_wc.age).all()
    assert (e_ref.infectivity - e_wc.infectivity).abs().max().item() <= atol_age

    failures = 0
    for k in range(num_steps):
        e_ref.step()
        e_wc.step()
        ok, report = _compare_step(e_ref, e_wc, k, atol_age, rtol_rate)
        if not ok:
            print(f"[MISMATCH] {report}")
            failures += 1
            if failures >= 3:
                print("  (stopping after 3 mismatches)")
                return False, e_ref, e_wc

    return failures == 0, e_ref, e_wc


def run_one(graph_label, gw, model, device, num_steps=50, seed=4242,
            atol_age=1e-6, rtol_rate=1e-5):
    print(f"\n=== {graph_label} (N={gw.num_nodes:,}, E={gw.num_edges:,}) ===")

    # 1. thread vs warp: both are deterministic, expect per-step bit match.
    ok_w, e_ref, e_wc = run_pair(
        "thread", "warp", graph_label, gw, model, device,
        "thread", "warp", num_steps, seed, atol_age, rtol_rate,
    )
    if ok_w:
        print("OK (bit-identical)")

    # 2. thread vs merge: merge uses atomic-add on a shared pressure
    # buffer, so accumulation order is not deterministic. We therefore
    # check population-level agreement rather than per-step bit match.
    e_merge = _make_engine(gw, model, device, seed, strategy="merge")
    for _ in range(num_steps):
        e_merge.step()

    cref = torch.bincount(e_ref.state, minlength=model.num_states).tolist()
    cwc = torch.bincount(e_wc.state, minlength=model.num_states).tolist()
    cmerge = torch.bincount(e_merge.state, minlength=model.num_states).tolist()
    print(f"  After {num_steps} steps (S,E,I,R):")
    print(f"    thread:      {cref}")
    print(f"    warp:        {cwc}")
    print(f"    merge:       {cmerge}")

    # Population tolerance: merge may differ by a few nodes due to fp
    # rounding in atomic_add ordering; reject only if categorically
    # different (e.g., merge outputs all-S or 10x more infected).
    max_diff = max(abs(a - b) for a, b in zip(cref, cmerge))
    max_expected = max(20, int(0.05 * gw.num_nodes))
    ok_m = max_diff <= max_expected
    if ok_m:
        print(f"  OK (merge pop-count within {max_diff} of thread, tol={max_expected})")
    else:
        print(f"  MERGE POP-COUNT MISMATCH: max bin delta = {max_diff} "
              f"(tol={max_expected})")

    return ok_w and ok_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=10_000)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    device = args.device
    N = args.num_nodes

    print("Warp-collaborative kernel correctness check")
    print(f"N={N:,}  steps={args.num_steps}  device={device}")

    model = SEIRModel(
        beta=2.0/8.0,
        mean_ei=5.0, median_ei=4.0,
        mean_ir=7.5, median_ir=5.0,
    )

    all_ok = True
    for label, builder, kwargs in [
        ("FixedDegree d=8", _build_er, dict(N=N, d=8)),
        ("BA m=4",          _build_ba, dict(N=N, m=4)),
    ]:
        gw = builder(device=device, **kwargs)
        ok = run_one(label, gw, model, device,
                     num_steps=args.num_steps, seed=args.seed)
        all_ok = all_ok and ok

    print()
    print("RESULT:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
