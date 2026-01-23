# FlashSpread Development Notes

## Recent Updates (2026-01-23)

### README Updated with Comprehensive Guidelines

- Performance benchmarking section (roofline analysis, ablation results)
- Scale guidelines: moderate (100K-1M), large (1M-10M), very large (10M+)
- RL training configuration (fast, less accurate: `epsilon=0.1, tau_max=2.0`)
- RL evaluation configuration (accurate: `epsilon=0.01, tau_max=0.5`)
- Benchmarking instructions

### Ablation Study Bug Fix

Fixed `experiments/ablation_study.py` - CUDA Graph configs were running more total simulation steps than baseline (200*50=10000 vs 200). Now all configs run the same total simulation steps for fair comparison.

**To re-run ablation with fix:**
```bash
rm results/ablation/ablation_*.json
sbatch slurm/run_ablation_study.sbatch
sbatch slurm/merge_ablation_results.sbatch  # After completion
```

---

## Ablation Study Results (Partial - Pre-Fix)

| Config | ms/step | Speedup | Notes |
|--------|---------|---------|-------|
| baseline | 1.408 | 1.0x | Reference |
| rcm_only | 1.361 | 1.04x | Cache locality |
| fused_only | 1.374 | 1.03x | Fewer kernel launches |
| block_256 | 1.375 | 1.02x | Better occupancy |
| **cuda_graph_only** | **0.174** | **~8x** | Launch overhead eliminated |

**Key finding:** CUDA Graph provides ~8x speedup per simulation step. Other optimizations provide minor incremental benefits.

---

## Roofline Benchmark (Completed)

**Job ID:** `7376334` - COMPLETED
**Results:** `results/roofline_plot.png`, `results/roofline_summary.md`

| Task | Config | Engine | Parameters |
|------|--------|--------|------------|
| 0 | renewal_baseline | renewal | epsilon=0.03, sparse=true |
| 1 | renewal_dense | renewal | sparse=false |
| 2-4 | renewal_batched_* | cuda_graph | steps_per_launch=10/50/100 |
| 5 | renewal_small_tau | renewal | epsilon=0.01 |
| 6 | renewal_large_tau | renewal | epsilon=0.1 |
| 7 | renewal_compute_heavy | cuda_graph | sparse=false, mult=2 |
| 8-9 | markov_* | markovian | baseline/aggressive |
| **10** | **renewal_ridge_8** | cuda_graph | sparse=false, **mult=8** |
| **11** | **renewal_ridge_16** | cuda_graph | sparse=false, **mult=16** |
| **12** | **renewal_compute_bound** | cuda_graph | sparse=false, **mult=20** |

**Key changes from v1:**
- Fixed FLOP estimation (erfcx ~30 FLOPs, not counted before)
- Added mult=8/16/20 configs to cross ridge point (AI > 9.6)

---

## Roofline Analysis Results (Completed)

**Key Finding:** Ridge crossing achieved at `compute_multiplier=16` (AI=16.08 > 9.6)

**Efficiency:** 3-10% of theoretical roofline (typical for sparse irregular workloads)

**Best Practical Config:** `renewal_batched_100` - 137.5 GFLOPS, 10.3% efficiency

**Output:** `results/roofline_plot.png`, `results/roofline_summary.md`

---

## Scalability Guidance

### Renewal Engine (Non-Markovian SEIR) - Best Practices

```python
from flashspread.engines.renewal import RenewalEngineCUDAGraph

# Optimal for throughput
engine = RenewalEngineCUDAGraph(
    graph, model, device="cuda",
    epsilon=0.03,         # Accuracy parameter (smaller = more steps, more accurate)
    tau_max=1.0,          # Max time step
    steps_per_launch=50,  # CUDA Graph batching - KEY for performance (2.8x speedup)
)
```

**Parameter Tuning:**
| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `epsilon` | Controls step size accuracy | 0.03 (default) good balance |
| `tau_max` | Maximum time step | 1.0 for stability |
| `steps_per_launch` | CUDA Graph batch size | 50-100 for best throughput |

### Markovian Engine (SIS/SIR) - Best Practices

```python
from flashspread.engines.markovian import MarkovianEngine

engine = MarkovianEngine(
    graph, model, device="cuda",
    max_prob=0.1,   # Max transition probability per step
    theta=0.01,     # Target fraction of nodes transitioning
    tau_max=1.0,    # Max time step
)
```

**Parameter Tuning:**
| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `max_prob` | Step size control | 0.1 for accuracy, 0.2 for speed |
| `theta` | Adaptive stepping | 0.01 (default) |
| `tau_max` | Max step | 1.0-2.0 |

### Graph Size Scaling

| Nodes | Edges (d=15) | GPU Memory | Recommended GPU |
|-------|--------------|------------|-----------------|
| 100K | 1.5M | ~200 MB | Any 4GB+ |
| 1M | 15M | ~2 GB | 8GB+ |
| 10M | 150M | ~20 GB | A100 40GB |
| 100M | 1.5B | ~200 GB | Multi-GPU |

---

## Analysis Code Location

```
experiments/
├── benchmark_roofline.py     # Roofline benchmark
├── ablation_study.py         # Optimization ablation study
└── roofline_utils.py         # Plotting utilities

flashspread/
├── engines/
│   ├── __init__.py           # Factory functions: create_renewal_engine(), create_markovian_engine()
│   ├── renewal.py            # RenewalEngine, RenewalEngineCUDAGraph
│   ├── markovian.py          # MarkovianEngine
│   └── renewal_tunable.py    # FLOP/byte estimation
└── core/
    ├── flash_neighbor.py     # FlashNeighbor Triton kernel
    └── optimizations.py      # RCM reordering, OptimizationConfig

docs/
└── PERFORMANCE_ANALYSIS.md   # Comprehensive performance documentation
```

## Factory Functions (New)

```python
from flashspread.engines import create_renewal_engine, create_markovian_engine

# Recommended: Uses CUDA Graph with optimal defaults
engine = create_renewal_engine(graph, model, use_cuda_graph=True)

# For Markovian models
engine = create_markovian_engine(graph, model)
```

---

## Project Structure

```
flashspread/
├── engines/
│   ├── markovian.py          # Sparse O(K) Markovian engine
│   ├── renewal.py            # Dense O(N) non-Markovian engine
│   └── renewal_tunable.py    # Tunable version for roofline analysis
├── models/
│   ├── compartmental.py      # SIS, SIR, SEIR models
│   └── hazards.py            # lognormal_hazard_stable (erfcx)
└── core/
    ├── flash_neighbor.py     # Triton kernel for influence computation
    ├── graph.py              # CSR graph structure
    └── network.py            # Graph generators

experiments/
├── benchmark_roofline.py     # Main benchmark script
└── roofline_utils.py         # Plotting utilities

slurm/
├── run_roofline_array.sbatch # Array job (13 parallel configs)
└── merge_roofline_results.sbatch
```

---

## Archive

### Completed: Initial Release (2026-01-16)
- FlashSpread v1.0.0 with dual-engine architecture
- FlashNeighbor Triton kernel
- Markovian (SIS/SIR) and Renewal (SEIR) engines
- CUDA Graph support for RenewalEngine

### Completed: Device Comparison Fix (2026-01-23)
- Fixed device comparison in FlashNeighbor (`00bfb29`)
