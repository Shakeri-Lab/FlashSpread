#!/usr/bin/env python
"""Non-Markovian SEIR through the public Simulator and EngineConfig API.

Unlike a Markovian model, where the transition rate is constant and memoryless,
the renewal SEIR model uses *age-dependent* hazards for E->I and I->R. That
captures peaked incubation and recovery timing.

The same example uses the CPU reference path on CPU-only hosts and the fused
CUDA-Graph path when a supported GPU and the ``gpu`` extra are available.
"""

import time

import flashspread as fs


def main():
    num_nodes = 2_000
    degree = 8
    seed = 0
    initial_exposed = 20
    target_time = 20.0
    device = fs.resolve_device()

    # Log-normal dwell times. SEIRModel requires mean > median > 0 (right-skewed).
    model = fs.SEIRModel(
        beta=0.3,
        mean_ei=5.0, median_ei=4.0,    # incubation, E -> I
        mean_ir=3.9, median_ir=1.5,    # infectious period, I -> R
    )

    print("FlashSpread -- non-Markovian SEIR")
    print("=" * 52)
    graph = fs.regular_graph(
        num_nodes,
        degree=degree,
        seed=seed,
        device=device,
        algorithm="circulant",
    )
    config = fs.EngineConfig(
        backend="auto",
        execution="auto",
        traversal="auto",
        transmission="model",
        precision="fp32",
        batch_steps=50,
        epsilon=0.03,
        tau_max=1.0,
    )
    sim = fs.Simulator(
        graph,
        model,
        device=device,
        seed=seed,
        config=config,
    ).seed_infection(initial_exposed)

    print(f"  network : {num_nodes} nodes, degree {degree} ({graph.num_edges} directed edges)")
    print(f"  engine  : {type(sim.engine).__name__} on {sim.device}")
    print(f"  window  : {sim.steps_per_launch} step(s) per call")
    print()

    t0 = time.time()
    traj = sim.run(until=target_time, record_every=2.0)
    elapsed = time.time() - t0

    print("      t      S      E      I      R")
    for t, (s, e, i, r) in zip(traj.times, traj.counts):
        print(f"  {t:5.1f} {s:6d} {e:6d} {i:6d} {r:6d}")
    print()

    print("Results")
    print("=" * 52)
    print(f"  peak infected : {traj.peak_infected} at t={traj.peak_time:.1f}")
    print(f"  attack rate   : {traj.final_attack_rate:.1%}")
    granularity = "window" if sim.steps_per_launch > 1 else "internal step"
    print(
        f"  end time      : {traj.times[-1]:.1f}  "
        f"(first {granularity} at/past {target_time:.1f})"
    )
    print(f"  wall clock    : {elapsed:.2f}s")

    # Population must be conserved at every recorded sample.
    assert (traj.counts.sum(axis=1) == num_nodes).all(), "population not conserved!"
    print(f"  population conserved at all {len(traj)} samples: yes")

    # traj.to_dict() is DataFrame-ready:
    #   import pandas as pd; pd.DataFrame(traj.to_dict())
    print(f"  columns       : {list(traj.to_dict())}")


if __name__ == "__main__":
    main()
