# FlashSpread Performance Analysis and Optimization Guide

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Roofline Analysis](#roofline-analysis)
4. [Optimization Strategies](#optimization-strategies)
5. [Ablation Study](#ablation-study)
6. [Scalability Guidelines](#scalability-guidelines)
7. [Recommended Configurations](#recommended-configurations)

---

## Overview

FlashSpread is a GPU-accelerated epidemic simulation framework with two engines:

| Engine | Model Type | Complexity | Use Case |
|--------|------------|------------|----------|
| **MarkovianEngine** | SIS, SIR | O(K × D_avg) sparse | Memoryless processes |
| **RenewalEngine** | SEIR (non-Markovian) | O(N) dense | Age-dependent transitions |

Both engines run entirely on GPU using custom Triton kernels for maximum performance.

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Simulation Step                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │ FlashNeighbor│ -> │ Hazard/Rate     │ -> │ Tau Selection │  │
│  │ (Triton)     │    │ Computation     │    │ (Adaptive)    │  │
│  └──────────────┘    └─────────────────┘    └───────────────┘  │
│         │                    │                      │           │
│         v                    v                      v           │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │ CSR Graph    │    │ erfcx for       │    │ Max rate      │  │
│  │ Traversal    │    │ lognormal hazard│    │ computation   │  │
│  └──────────────┘    └─────────────────┘    └───────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Transition Sampling & State Update           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **FlashNeighbor Kernel** (`flashspread/core/flash_neighbor.py`)
   - Triton kernel for sparse neighbor influence computation
   - Computes: I[i] = Σ_{j ∈ N(i)} w_ji × 1{X_j = q}
   - Fuses state lookup, predicate evaluation, and accumulation

2. **Hazard Functions** (`flashspread/models/hazards.py`)
   - `lognormal_hazard_stable()`: Uses erfcx for numerical stability
   - Avoids catastrophic cancellation for large ages

3. **Engines** (`flashspread/engines/`)
   - `MarkovianEngine`: Sparse updates, exploits rate structure
   - `RenewalEngine`: Dense updates, handles age-dependent hazards
   - `RenewalEngineCUDAGraph`: Batched execution via CUDA Graphs

---

## Roofline Analysis

### Hardware: NVIDIA A100

| Specification | Value |
|---------------|-------|
| Peak FP32 (CUDA cores) | 19.5 TFLOPS |
| Memory Bandwidth | 2039 GB/s |
| Ridge Point | 9.6 FLOPs/byte |

**Note:** Tensor cores (156+ TFLOPS) cannot be utilized because:
- FlashNeighbor uses sparse CSR traversal (irregular access)
- Hazard computation uses transcendental ops (log, erfcx)
- Transition sampling is element-wise with branching

### FLOP Estimation

Per simulation step for N=1M nodes, E=15M edges:

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| FlashNeighbor | 45M | 3 FLOPs/edge |
| Hazard (dense) | 110M | 55 FLOPs/node × 2 transitions |
| Tau selection | 4M | max, div, min, compare |
| Transition prob | 5M | mul, neg, exp, neg, add |
| Sampling | 6M | RNG + compare |
| State update | 6M | copy, compare, mask |
| **Total** | **~176M** | |

### Memory Traffic Estimation

| Operation | Bytes | Notes |
|-----------|-------|-------|
| CSR row_ptr | 4M | (N+1) × 4 |
| CSR col_ind | 60M | E × 4 |
| CSR weights | 60M | E × 4 |
| State arrays | 16M | N × 4 × 4 arrays |
| Working buffers | 24M | event_prob, mask, etc. |
| **Total** | **~165M** | |

### Arithmetic Intensity

| Configuration | AI (FLOPs/byte) | Bound |
|---------------|-----------------|-------|
| Baseline (sparse) | 0.66 | Memory |
| Dense hazard | 1.34 | Memory |
| Dense + mult=8 | 8.22 | Memory |
| Dense + mult=16 | **16.08** | **Compute** |

### Benchmark Results (A100)

| Config | AI | GFLOPS | Efficiency | ms/step |
|--------|-----|--------|------------|---------|
| renewal_baseline | 0.66 | 49.5 | 3.7% | 1.49 |
| renewal_batched_100 | 0.66 | 137.5 | **10.3%** | 0.53 |
| renewal_dense | 1.34 | 166.2 | 6.1% | 0.91 |
| renewal_ridge_16 | 16.08 | 664.6 | 3.4% | 2.71 |

**Key insight:** CUDA Graph batching provides the best practical improvement (2.8x) without artificial compute inflation.

---

## Optimization Strategies

### 1. CUDA Graph Batching (Implemented)

**Effect:** 2.8x speedup
**Mechanism:** Captures multiple steps into a single CUDA graph, eliminating kernel launch overhead.

```python
# Without CUDA Graph: 6 kernel launches per step
for step in range(1000):
    pressure = flash_neighbor(state)      # launch 1
    rates = compute_hazard(age, pressure) # launch 2
    tau = select_tau(rates)               # launch 3
    probs = compute_probs(rates, tau)     # launch 4
    events = sample(probs)                # launch 5
    state = update(state, events)         # launch 6

# With CUDA Graph: 1 launch per 50 steps
graph.replay()  # Executes 50 steps
```

### 2. Graph Reordering (RCM)

**Expected:** 10-30% speedup
**Mechanism:** Reverse Cuthill-McKee reordering reduces adjacency matrix bandwidth, improving cache locality during CSR traversal.

```python
from flashspread.core.optimizations import reorder_graph_rcm

reordered_graph, perm = reorder_graph_rcm(graph.csr)
# Bandwidth reduction: typically 50-80%
```

### 3. Kernel Fusion

**Expected:** 20-30% speedup
**Mechanism:** Fuse FlashNeighbor output directly into hazard computation, eliminating intermediate global memory writes.

```python
# Current: Two kernels
pressure = flash_neighbor(state)  # Write 4MB to global memory
hazard = compute_hazard(pressure) # Read 4MB from global memory

# Fused: Single kernel
hazard = fused_flash_hazard(state)  # No intermediate storage
```

### 4. Block Size Tuning

**Expected:** 5-15% speedup
**Mechanism:** Larger block sizes can improve occupancy and reduce synchronization overhead.

```python
# Default: BLOCK_SIZE = 128
# Optimized: BLOCK_SIZE = 256 (better for A100)
```

---

## Ablation Study

### Configurations Tested

| ID | Name | Optimizations |
|----|------|---------------|
| 0 | baseline | None |
| 1 | rcm_only | RCM reordering |
| 2 | fused_only | Fused PyTorch ops |
| 3 | block_256 | Larger block size |
| 4 | cuda_graph_only | CUDA Graph (50 steps) |
| 5 | rcm_fused | RCM + Fused |
| 6 | rcm_cuda_graph | RCM + CUDA Graph |
| 7 | all_optimizations | All above |
| 8 | cuda_graph_100 | CUDA Graph (100 steps) |
| 9 | all_opt_100 | All + 100 steps |

### Running the Ablation Study

```bash
# Submit all 10 configurations
sbatch slurm/run_ablation_study.sbatch

# Monitor progress
squeue -u $USER
tail -f logs/flashspread_ablation-*.out

# Merge results after completion
sbatch slurm/merge_ablation_results.sbatch

# View summary
cat results/ablation/ablation_summary.md
```

### Accuracy Validation

Each configuration is compared against baseline:
- Multiple runs (5) with different seeds
- Final infected count compared
- **Passes** if within 10% of baseline (accounting for stochastic variance)

---

## Scalability Guidelines

### Memory Requirements

| Nodes | Edges (d=15) | GPU Memory | Recommended |
|-------|--------------|------------|-------------|
| 100K | 1.5M | ~200 MB | Any 4GB+ GPU |
| 1M | 15M | ~2 GB | 8GB+ GPU |
| 10M | 150M | ~20 GB | A100 40GB |
| 100M | 1.5B | ~200 GB | Multi-GPU |

### Scaling Formula

```
Memory (GB) ≈ N × 0.00002 + E × 0.000008
            ≈ N × (0.00002 + degree × 0.000008)
```

### Performance Scaling

- **Strong scaling:** Steps/sec roughly constant with graph size (memory-bound)
- **Weak scaling:** Time per step scales linearly with N

---

## Recommended Configurations

### Renewal Engine (Non-Markovian SEIR)

```python
from flashspread import FixedDegreeGraph, SEIRModel
from flashspread.engines.renewal import RenewalEngineCUDAGraph

# Create graph
graph = FixedDegreeGraph(num_nodes=1_000_000, degree=15, device="cuda")

# Create model
model = SEIRModel(
    beta=0.3,      # Infection rate
    mean_ei=5.0,   # Mean incubation period
    median_ei=4.0,
    mean_ir=3.9,   # Mean infectious period
    median_ir=1.5,
)

# RECOMMENDED: Use CUDA Graph engine for best performance
engine = RenewalEngineCUDAGraph(
    graph, model,
    device="cuda",
    epsilon=0.03,         # Accuracy parameter (default is good)
    tau_max=1.0,          # Max time step
    steps_per_launch=50,  # CUDA Graph batch size (50-100 optimal)
)

# Seed infection and run
engine.seed_infection(num_nodes // 100, state=model.exposed)
while engine.current_time < 100.0:
    engine.step()
```

### Markovian Engine (SIS/SIR)

```python
from flashspread import FixedDegreeGraph, SISModel
from flashspread.engines import MarkovianEngine

graph = FixedDegreeGraph(num_nodes=1_000_000, degree=15, device="cuda")
model = SISModel(beta=0.5, delta=1.0)

# RECOMMENDED configuration
engine = MarkovianEngine(
    graph, model,
    device="cuda",
    max_prob=0.1,   # Controls step size (0.1 = accurate, 0.2 = faster)
    theta=0.01,     # Adaptive stepping parameter
    tau_max=1.0,    # Max time step
)

engine.seed_infection(num_nodes // 100)
for _ in range(1000):
    engine.step()
```

### Performance Tips

1. **Always use CUDA Graph** for Renewal engine (2.8x speedup)
2. **Use `steps_per_launch=50-100`** for best throughput
3. **Keep `epsilon=0.03`** for accuracy/speed balance
4. **Pre-allocate graphs** if running multiple simulations
5. **Use `torch.cuda.synchronize()`** only when reading results

---

## Code References

| File | Description |
|------|-------------|
| `flashspread/engines/renewal.py` | RenewalEngine and RenewalEngineCUDAGraph |
| `flashspread/engines/markovian.py` | MarkovianEngine |
| `flashspread/core/flash_neighbor.py` | FlashNeighbor Triton kernel |
| `flashspread/models/hazards.py` | lognormal_hazard_stable |
| `flashspread/core/optimizations.py` | RCM reordering utilities |
| `experiments/benchmark_roofline.py` | Roofline benchmarking |
| `experiments/ablation_study.py` | Optimization ablation study |

---

## References

1. Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An Insightful Visual Performance Model for Multicore Architectures.
2. NVIDIA A100 Tensor Core GPU Architecture (2020).
3. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (2019).
