#!/usr/bin/env python
"""Markovian SIS through the public Simulator and EngineConfig API.

SIS permits reinfection after recovery and can approach an endemic equilibrium.
The same example uses the CPU reference path on CPU-only hosts and the CUDA path
when a supported GPU and the ``gpu`` extra are available.
"""

import sys
import time

import flashspread as fs


def main():
    num_nodes = 2_000
    degree = 8
    beta, delta = 0.5, 1.0
    seed = 0
    initial_infected = 20
    target_time = 10.0
    device = fs.resolve_device()

    print("FlashSpread -- Markovian SIS")
    print("=" * 52)

    # The direct circulant path avoids a NetworkX dependency and builds exact
    # degree-regular incoming CSR on the selected device.
    graph = fs.regular_graph(
        num_nodes,
        degree=degree,
        seed=seed,
        device=device,
        algorithm="circulant",
    )
    model = fs.SISModel(beta=beta, delta=delta)
    config = fs.EngineConfig(
        execution="auto",  # eager by default for Markovian models
        max_prob=0.1,
        theta=0.01,
        tau_min=1e-6,
        tau_max=1.0,
    )
    sim = fs.Simulator(
        graph,
        model,
        device=device,
        seed=seed,
        config=config,
    ).seed_infection(initial_infected)

    print(f"  network : {num_nodes} nodes, degree {degree}")
    print(f"  model   : SIS (beta={beta}, delta={delta})")
    print(f"  engine  : {type(sim.engine).__name__} on {sim.device}")
    print()

    t0 = time.time()
    traj = sim.run(until=target_time, record_every=2.0)
    elapsed = time.time() - t0

    print("      t      S      I")
    for t, (s, i) in zip(traj.times, traj.counts):
        print(f"  {t:5.1f} {s:6d} {i:6d}")
    print()

    print("Results")
    print("=" * 52)
    print(f"  peak infected : {traj.peak_infected}")
    print(f"  wall clock    : {elapsed:.2f}s")
    assert (traj.counts.sum(axis=1) == num_nodes).all(), "population not conserved!"
    print("  population conserved: yes")

    # Endemic equilibrium: mean-field predicts prevalence 1 - 1/R0 for R0 > 1.
    # This is only a homogeneous-mixing reference; a finite network can differ.
    r0 = beta * degree / delta
    final_prevalence = traj.final_prevalence
    print()
    print(f"  R0 (mean-field)      : {r0:.2f}")
    if r0 > 1:
        print(f"  predicted prevalence : {1 - 1 / r0:.3f}  (homogeneous mixing)")
    print(f"  measured prevalence  : {final_prevalence:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
