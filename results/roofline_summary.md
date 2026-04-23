# FlashSpread Roofline Benchmark Summary

## Hardware: NVIDIA A100 (CUDA Cores Only)

| Spec | Value |
|------|-------|
| Peak FP32 (CUDA cores) | 19.5 TFLOPS |
| Memory Bandwidth | 2039 GB/s |
| Ridge Point | 9.6 FLOPs/byte |

**Note:** Tensor cores (156+ TFLOPS) cannot be used - this workload is sparse/irregular.

---

## Configuration Explanations

### Renewal Engine Configs

| Config | What It Tests | Parameters |
|--------|---------------|------------|
| `renewal_baseline` | Default renewal engine behavior | epsilon=0.03, sparse hazard, no batching |
| `renewal_dense` | Dense hazard computation (all N nodes) | sparse_hazard=False - increases FLOPs/step |
| `renewal_batched_10/50/100` | CUDA Graph batching effect | Captures 10/50/100 steps in graph, reduces kernel launch overhead |
| `renewal_small_tau` | Finer time resolution | epsilon=0.01 - smaller steps, same AI |
| `renewal_large_tau` | Coarser time resolution | epsilon=0.1 - larger steps, same AI |
| `renewal_compute_heavy` | Moderate compute intensity | dense + mult=2, tests increased FLOPs |
| `renewal_ridge_8` | Approaching ridge point | dense + mult=8, AI ~8.2 (just below ridge) |
| `renewal_ridge_16` | Compute-bound regime | dense + mult=16, AI ~16 (above ridge) |
| `renewal_compute_bound` | Deep compute-bound | dense + mult=20, AI ~20 |

### Markovian Engine Configs

| Config | What It Tests | Parameters |
|--------|---------------|------------|
| `markov_baseline` | Default Markovian engine | max_prob=0.1, theta=0.01 |
| `markov_aggressive` | Larger time steps | max_prob=0.2, theta=0.05 - faster but less accurate |

---

## Results Summary

| Config | AI (FLOPs/byte) | GFLOPS | ms/step | Bound | Efficiency |
|--------|-----------------|--------|---------|-------|------------|
| renewal_ridge_16 | **16.08** | **664.6** | 2.71 | Compute | 3.4% |
| renewal_compute_bound | 20.00 | 652.8 | 3.43 | Compute | 3.3% |
| renewal_ridge_8 | 8.22 | 597.7 | 1.54 | Memory | 3.6% |
| renewal_compute_heavy | 2.33 | 378.6 | 0.69 | Memory | 8.0% |
| renewal_dense | 1.34 | 166.2 | 0.91 | Memory | 6.1% |
| renewal_batched_100 | 0.66 | 137.5 | 0.53 | Memory | **10.3%** |
| renewal_batched_50 | 0.66 | 137.0 | 0.54 | Memory | 10.2% |
| renewal_batched_10 | 0.66 | 136.1 | 0.54 | Memory | 10.2% |
| renewal_baseline | 0.66 | 49.5 | 1.49 | Memory | 3.7% |
| markov_baseline | 0.47 | 47.6 | 0.79 | Memory | 5.0% |

---

## Key Findings

### 1. Ridge Crossing Achieved
- `renewal_ridge_16` (AI=16.08) and `renewal_compute_bound` (AI=20.0) are **compute-bound**
- Transition happens between mult=8 (AI=8.2) and mult=16 (AI=16.1)

### 2. Low Roofline Efficiency (3-10%)
The gap between achieved and theoretical performance is large because:
- **Sparse irregular access**: FlashNeighbor traverses CSR with unpredictable patterns
- **Special functions**: `erfcx()` uses polynomial approximations, not pure FLOPs
- **Memory latency**: Even compute-bound code waits for data
- **Warp divergence**: State-dependent branching causes thread divergence

### 3. CUDA Graph Batching is Highly Effective
- `renewal_batched_*` achieves **2.8x speedup** over baseline at same AI
- Best efficiency (10.3%) achieved with batching
- Reduces kernel launch overhead significantly

### 4. Practical Performance
For real simulations (not artificial compute inflation):
- **Best throughput**: `renewal_batched_100` at 137.5 GFLOPS, 0.53 ms/step
- **Markovian** is ~3x slower per step but simpler model

---

## Scalability Recommendations

### For Renewal Engine (Non-Markovian SEIR)
```python
# Recommended defaults for best scalability
engine = RenewalEngineCUDAGraph(
    graph, model, device="cuda",
    epsilon=0.03,        # Good accuracy/speed tradeoff
    tau_max=1.0,
    steps_per_launch=50, # CUDA Graph batching - key for performance
)
```

### For Markovian Engine (SIS/SIR)
```python
# Recommended defaults
engine = MarkovianEngine(
    graph, model, device="cuda",
    max_prob=0.1,   # Controls step size
    theta=0.01,     # Convergence threshold
    tau_max=1.0,
)
```

### Graph Size Scaling
| Nodes | Edges (deg=15) | Memory | Recommended |
|-------|----------------|--------|-------------|
| 100K | 1.5M | ~200 MB | Any GPU |
| 1M | 15M | ~2 GB | 8+ GB GPU |
| 10M | 150M | ~20 GB | A100 40GB+ |
| 100M | 1.5B | ~200 GB | Multi-GPU |

---

## Files

- `roofline_plot.png` - Roofline visualization
- `speedup_comparison.png` - Speedup vs baseline
- `timing_breakdown.png` - Time per step comparison
- `roofline_data.json` - Raw benchmark data
