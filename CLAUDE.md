# FlashSpread Development Notes

## Recent Updates (2026-01-23)

### README Updated with Comprehensive Guidelines

- Performance benchmarking section (roofline analysis, ablation results)
- Scale guidelines: moderate (100K-1M), large (1M-10M), very large (10M+)
- RL training configuration (fast, less accurate: `epsilon=0.1, tau_max=2.0`)
- RL evaluation configuration (accurate: `epsilon=0.01, tau_max=0.5`)
- Benchmarking instructions

### Ablation Study Bug Fixes

1. **Step count normalization** - CUDA Graph configs were running more total simulation steps than baseline (200*50=10000 vs 200). Now all configs run the same total simulation steps.

2. **JSON serialization** - Fixed `numpy.bool_` not JSON serializable by converting to Python native types.

---

## Ablation Study Results (Job 7378715 - COMPLETED)

| ID | Config | ms/step | Speedup | Infected | AccErr% | Pass |
|----|--------|---------|---------|----------|---------|------|
| 0 | baseline | 1.382 | 1.00x | 1367 | 0.0 | Y |
| 1 | rcm_only | 1.416 | 0.98x | 1350 | 1.2 | Y |
| 2 | fused_only | 1.353 | 1.02x | 1331 | 2.6 | Y |
| 3 | block_256 | 1.353 | 1.02x | 1383 | 1.2 | Y |
| 4 | **cuda_graph_only** | **0.216** | **6.41x** | **0** | 100.0 | **N** |
| 5 | rcm_fused | 1.418 | 0.97x | 1387 | 1.5 | Y |
| 6 | rcm_cuda_graph | 0.212 | 6.52x | 0 | 100.0 | N |
| 7 | all_optimizations | 0.214 | 6.47x | 0 | 100.0 | N |
| 8 | cuda_graph_100 | 0.215 | 6.44x | 0 | 100.0 | N |
| 9 | all_opt_100 | 0.211 | 6.56x | 0 | 100.0 | N |

### Key Findings

1. **Non-CUDA Graph optimizations**: Marginal speedups (0.97x-1.02x), all pass accuracy
2. **CUDA Graph optimizations**: 6.4x-6.6x speedup BUT **all fail accuracy** (0 infected)

### CRITICAL BUG: CUDA Graph Accuracy Failure

**All CUDA Graph configs show 0 final infected vs ~1367 for baseline** (same 200 simulation steps).

**Likely cause:** CUDA Graph captures RNG state and replays same random numbers, causing deterministic behavior that kills the epidemic.

**Action needed:** Investigate `RenewalEngineCUDAGraph` RNG handling - may need to:
- Update RNG state between graph replays
- Use different random seed strategy
- Or disable CUDA Graph for stochastic simulations

**For now:** Use `RenewalEngine` (non-CUDA Graph) for accurate simulations

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

**WARNING:** CUDA Graph has accuracy issues - use `RenewalEngine` until fixed.

```python
from flashspread.engines.renewal import RenewalEngine

# RECOMMENDED: Use non-CUDA Graph for accurate simulations
engine = RenewalEngine(
    graph, model, device="cuda",
    epsilon=0.03,         # Accuracy parameter (smaller = more steps, more accurate)
    tau_max=1.0,          # Max time step
)
```

**Parameter Tuning:**
| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `epsilon` | Controls step size accuracy | 0.03 (default) good balance |
| `tau_max` | Maximum time step | 1.0 for stability |

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

## Factory Functions

```python
from flashspread.engines import create_renewal_engine, create_markovian_engine

# WARNING: use_cuda_graph=False until CUDA Graph accuracy bug is fixed
engine = create_renewal_engine(graph, model, use_cuda_graph=False)

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
