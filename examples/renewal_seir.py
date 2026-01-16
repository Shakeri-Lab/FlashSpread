#!/usr/bin/env python
"""
Example: Non-Markovian SEIR Epidemic Simulation

This example demonstrates the FlashSpread Renewal engine for simulating
SEIR dynamics with log-normal dwell time distributions.

Unlike Markovian models where transition rates are constant, the renewal
SEIR model uses age-dependent hazards for the E->I and I->R transitions,
capturing the realistic peaked timing of incubation and recovery.
"""

import time
import torch
from flashspread import RenewalEngine, SEIRModel, FixedDegreeGraph


def main():
    # Configuration
    num_nodes = 10000
    degree = 15
    beta = 0.3  # Infection rate

    # Log-normal parameters for incubation (E->I)
    mean_ei = 5.0   # Mean incubation period
    median_ei = 4.0  # Median incubation period

    # Log-normal parameters for recovery (I->R)
    mean_ir = 3.9   # Mean infectious period
    median_ir = 1.5  # Median infectious period

    initial_exposed = 100
    target_time = 50.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"FlashSpread Non-Markovian SEIR Example")
    print(f"=" * 50)
    print(f"Network: {num_nodes} nodes, degree {degree}")
    print(f"Model: SEIR (beta={beta})")
    print(f"  E->I: LogNormal(mean={mean_ei}, median={median_ei})")
    print(f"  I->R: LogNormal(mean={mean_ir}, median={median_ir})")
    print(f"Device: {device}")
    print()

    # Create network
    print("Creating network...")
    graph = FixedDegreeGraph(num_nodes, degree, device=device)
    print(f"  Edges: {graph.num_edges}")

    # Create model
    model = SEIRModel(
        beta=beta,
        mean_ei=mean_ei,
        median_ei=median_ei,
        mean_ir=mean_ir,
        median_ir=median_ir,
    )

    # Create engine
    engine = RenewalEngine(graph, model, device=device, epsilon=0.03)

    # Seed initial exposed (not infected, to show incubation)
    engine.seed_infection(initial_exposed, state=model.exposed)
    print(f"  Initial exposed: {initial_exposed}")
    print()

    # Run simulation
    print("Running simulation...")
    start_time = time.time()

    history = {"S": [], "E": [], "I": [], "R": [], "time": []}
    print_interval = 5.0
    next_print = 0.0

    while engine.current_time < target_time:
        engine.step()

        if engine.current_time >= next_print:
            counts = engine.count_by_state()
            history["S"].append(counts[0].item())
            history["E"].append(counts[1].item())
            history["I"].append(counts[2].item())
            history["R"].append(counts[3].item())
            history["time"].append(engine.current_time)

            print(f"  t={engine.current_time:5.1f}: "
                  f"S={counts[0]:5d}, E={counts[1]:5d}, "
                  f"I={counts[2]:5d}, R={counts[3]:5d}")

            next_print += print_interval

    elapsed = time.time() - start_time
    print()

    # Final results
    print(f"Results")
    print(f"=" * 50)
    counts = engine.count_by_state()
    print(f"Final state: S={counts[0]}, E={counts[1]}, I={counts[2]}, R={counts[3]}")
    print(f"Total population: {counts.sum().item()} (should be {num_nodes})")
    print(f"Attack rate: {(num_nodes - counts[0].item()) / num_nodes:.1%}")
    print(f"Simulation steps: {engine.total_steps}")
    print(f"Wall clock time: {elapsed:.2f}s")
    print(f"Steps per second: {engine.total_steps / elapsed:.0f}")

    # Find epidemic peak
    if history["I"]:
        peak_idx = history["I"].index(max(history["I"]))
        print(f"\nEpidemic peak:")
        print(f"  Time: {history['time'][peak_idx]:.1f}")
        print(f"  Infected: {history['I'][peak_idx]}")


if __name__ == "__main__":
    main()
