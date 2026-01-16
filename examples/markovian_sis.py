#!/usr/bin/env python
"""
Example: Markovian SIS Epidemic Simulation

This example demonstrates the FlashSpread Markovian engine for simulating
SIS (Susceptible-Infected-Susceptible) dynamics on a random network.

The SIS model allows reinfection after recovery, leading to endemic
equilibrium where a fraction of the population remains infected.
"""

import time
import torch
from flashspread import MarkovianEngine, SISModel, FixedDegreeGraph


def main():
    # Configuration
    num_nodes = 10000
    degree = 15
    beta = 0.5  # Infection rate
    delta = 1.0  # Recovery rate
    initial_infected = 100
    num_steps = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"FlashSpread Markovian SIS Example")
    print(f"=" * 50)
    print(f"Network: {num_nodes} nodes, degree {degree}")
    print(f"Model: SIS (beta={beta}, delta={delta})")
    print(f"Device: {device}")
    print()

    # Create network
    print("Creating network...")
    graph = FixedDegreeGraph(num_nodes, degree, device=device)
    print(f"  Edges: {graph.num_edges}")

    # Create model
    model = SISModel(beta=beta, delta=delta)

    # Create engine
    engine = MarkovianEngine(graph, model, device=device)

    # Seed initial infection
    engine.seed_infection(initial_infected)
    print(f"  Initial infected: {engine.count_infected()}")
    print()

    # Run simulation
    print("Running simulation...")
    start_time = time.time()

    infected_history = []
    for step in range(num_steps):
        tau, num_events = engine.step()
        infected = engine.count_infected()
        infected_history.append(infected)

        if step % 200 == 0:
            print(f"  Step {step}: time={engine.current_time:.2f}, infected={infected}")

    elapsed = time.time() - start_time
    print()

    # Results
    print(f"Results")
    print(f"=" * 50)
    print(f"Simulation time: {engine.current_time:.2f}")
    print(f"Total events: {engine.total_events}")
    print(f"Final infected: {engine.count_infected()}")
    print(f"Wall clock time: {elapsed:.2f}s")
    print(f"Throughput: {engine.total_events / elapsed:.2e} events/sec")

    # Check for endemic equilibrium
    # Expected endemic prevalence: 1 - delta/(beta * degree) if R0 > 1
    r0 = beta * degree / delta
    if r0 > 1:
        expected_prevalence = 1 - 1/r0
        actual_prevalence = engine.count_infected() / num_nodes
        print(f"\nR0 = {r0:.2f} (epidemic)")
        print(f"Expected endemic prevalence: {expected_prevalence:.3f}")
        print(f"Actual prevalence: {actual_prevalence:.3f}")


if __name__ == "__main__":
    main()
