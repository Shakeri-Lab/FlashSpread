# FlashSpread

**GPU-Accelerated Markovian and Non-Markovian Spreading Processes on Complex Networks**

FlashSpread is the first GPU-accelerated framework for simulating both Markovian and non-Markovian (renewal) stochastic spreading processes on million-node contact networks.

## Key Features

- **Dual-Engine Architecture**: Exploits the computational duality between Markovian (sparse O(K) updates) and non-Markovian (dense O(N) updates) regimes
- **FlashNeighbor Kernel**: IO-aware fused Triton kernel for computing inducer influence via sparse traversal
- **Numerically Stable Hazards**: erfcx-based evaluation for heavy-tailed distributions
- **Adaptive Bernoulli Tau-Leaping**: Tunable fidelity for non-Markovian dynamics
- **1000x Speedup**: Achieves >2x10^7 events/second, orders of magnitude faster than CPU baselines

## Installation

```bash
# Clone the repository
git clone https://github.com/Shakeri-Lab/FlashSpread.git
cd FlashSpread

# Install in development mode
pip install -e .

# Or install directly
pip install .
```

### Requirements
- Python >= 3.10
- PyTorch >= 2.0
- Triton >= 2.1
- NetworkX >= 3.0
- NumPy >= 1.24
- CUDA-capable GPU

## Quick Start

### Markovian SIS Simulation

```python
import torch
from flashspread.engines import MarkovianEngine
from flashspread.models import SISModel
from flashspread.core import FixedDegreeGraph

# Create network (10,000 nodes, degree 15)
graph = FixedDegreeGraph(num_nodes=10000, degree=15, device="cuda")

# Define SIS model
model = SISModel(beta=0.5, delta=1.0)

# Create engine and run
engine = MarkovianEngine(graph, model, device="cuda")
engine.seed_infection(num_infected=100)

# Simulate for 100 time units
for _ in range(1000):
    engine.step()

print(f"Final infected: {engine.count_infected()}")
```

### Non-Markovian SEIR Simulation

```python
import torch
from flashspread.engines import RenewalEngine
from flashspread.models import SEIRModel

# Create network
graph = FixedDegreeGraph(num_nodes=10000, degree=15, device="cuda")

# Define SEIR with log-normal transitions
model = SEIRModel(
    beta=0.3,
    mean_ei=5.0, median_ei=4.0,   # E->I log-normal
    mean_ir=3.9, median_ir=1.5    # I->R log-normal
)

# Create renewal engine
engine = RenewalEngine(graph, model, device="cuda", epsilon=0.03)
engine.seed_infection(num_exposed=100)

# Simulate until time T=50
while engine.current_time < 50.0:
    engine.step()

counts = engine.count_by_state()
print(f"S={counts[0]}, E={counts[1]}, I={counts[2]}, R={counts[3]}")
```

## Architecture

FlashSpread uses a dual-engine architecture:

1. **Markovian Engine**: For memoryless processes (SIS, SIR)
   - Control Mode: Global rate recomputation when control inputs change
   - Inertial Mode: Sparse incremental updates between control changes

2. **Renewal Engine**: For age-dependent processes (SEIR with non-exponential dwell times)
   - Dense synchronous updates with adaptive time stepping
   - Bernoulli tau-leaping for tunable fidelity

Both engines share the **FlashNeighbor** kernel for efficient influence computation.

## Performance and Benchmarking

### Performance Overview

FlashSpread achieves high throughput through GPU acceleration and optimized memory access patterns:

| Engine | Graph Size | Throughput | GPU Memory |
|--------|------------|------------|------------|
| **RenewalEngineCUDAGraph** | 1M nodes, 15M edges | ~5,800 steps/sec | ~2 GB |
| **RenewalEngine** | 1M nodes, 15M edges | ~710 steps/sec | ~2 GB |
| **MarkovianEngine** | 1M nodes, 15M edges | ~1,200 steps/sec | ~2 GB |

**Key optimization: CUDA Graph batching provides ~5.7x speedup** by amortizing kernel launch overhead across multiple simulation steps.

### Roofline Analysis (A100)

The algorithm is **memory-bound** on modern GPUs:

| Metric | Value |
|--------|-------|
| A100 CUDA Core FP32 | 19.5 TFLOPS |
| A100 Memory Bandwidth | 2039 GB/s |
| Ridge Point | 9.6 FLOPs/byte |
| Achieved Arithmetic Intensity | 0.6-1.3 FLOPs/byte |
| Achieved Efficiency | 3-10% (typical for sparse irregular workloads) |

**Note:** Tensor cores cannot be utilized due to sparse irregular memory access patterns and transcendental operations (erfcx for log-normal hazards).

### Ablation Study Results

Individual optimization contributions (100K nodes, degree 15):

| Optimization | Speedup | Notes |
|--------------|---------|-------|
| Baseline | 1.0x | RenewalEngine, no optimizations |
| RCM Reordering | 1.01x | Better cache locality |
| Fused Operations | 0.99x | Minor overhead |
| Block Size 256 | 1.02x | Better occupancy |
| **CUDA Graph (50 steps)** | **5.7x** | Amortized launch overhead |
| RCM + CUDA Graph | 5.8x | Combined benefits |

**Recommendation:** Use `RenewalEngineCUDAGraph` with `steps_per_launch=50` for best accuracy/speed tradeoff. Higher batch sizes (100) may have increased stochastic variance.

---

## Scale Guidelines

### Moderate Graphs (100K - 1M nodes)

Suitable for most research and RL training scenarios.

```python
from flashspread.engines import create_renewal_engine

# Recommended configuration
engine = create_renewal_engine(
    graph, model,
    use_cuda_graph=True,      # ~5.7x speedup
    epsilon=0.03,             # Accuracy/speed balance
    steps_per_launch=50,      # Optimal batch size
)
```

**Memory requirement:** ~2 GB for 1M nodes with degree 15

### Large Graphs (1M - 10M nodes)

For population-scale simulations. Requires careful memory management.

```python
from flashspread import FixedDegreeGraph, SEIRModel
from flashspread.engines import RenewalEngineCUDAGraph

# Large graph setup
graph = FixedDegreeGraph(
    num_nodes=10_000_000,
    degree=15,
    device="cuda"
)

model = SEIRModel(beta=0.3, mean_ei=5.0, median_ei=4.0, mean_ir=3.9, median_ir=1.5)

# Use larger epsilon for scalability
engine = RenewalEngineCUDAGraph(
    graph, model,
    device="cuda",
    epsilon=0.05,            # Slightly less accurate, faster
    tau_max=2.0,             # Allow larger time steps
    steps_per_launch=100,    # Maximize batching benefit
)
```

**Memory requirement:** ~20 GB for 10M nodes (A100 40GB recommended)

### Very Large Graphs (10M+ nodes)

For city/country-scale simulations. Consider:

1. **Multi-GPU distribution** (future feature)
2. **Subgraph sampling** for training
3. **Increased epsilon** for tractable step counts

```python
# Very large scale (use with caution)
engine = RenewalEngineCUDAGraph(
    graph, model,
    epsilon=0.1,             # Trade accuracy for speed
    tau_max=5.0,             # Larger time steps
    steps_per_launch=100,
)
```

**Memory scaling:** `Memory (GB) ≈ N × 0.00002 + E × 0.000008`

---

## Reinforcement Learning Integration

FlashSpread supports both fast approximate simulations for training and accurate simulations for evaluation.

### RL Training Configuration (Fast, Less Accurate)

For gradient-based RL training, prioritize simulation speed over accuracy:

```python
from flashspread.engines import create_renewal_engine

def create_training_env(graph, model):
    """Create fast simulator for RL training."""
    return create_renewal_engine(
        graph, model,
        use_cuda_graph=True,
        epsilon=0.1,             # Larger epsilon = bigger steps, faster
        tau_max=2.0,             # Allow larger leaps
        steps_per_launch=100,    # Maximum batching
    )

# Training loop
env = create_training_env(graph, model)
for episode in range(num_episodes):
    env.reset()
    while not done:
        action = policy(state)
        env.apply_control(action)  # Your control interface
        for _ in range(sim_steps_per_action):
            env.step()
        reward = compute_reward(env)
        # ... RL update
```

**Tradeoffs:**
- `epsilon=0.1`: ~3x faster than `epsilon=0.03`, but larger discretization error
- `tau_max=2.0`: Allows larger time jumps, may miss rapid transitions
- `steps_per_launch=100`: Best for long simulations (>1000 steps)

### RL Evaluation Configuration (Accurate)

For policy evaluation and final testing, prioritize accuracy:

```python
def create_evaluation_env(graph, model):
    """Create accurate simulator for RL evaluation."""
    return create_renewal_engine(
        graph, model,
        use_cuda_graph=True,     # Still use for speed
        epsilon=0.01,            # Small epsilon = accurate
        tau_max=0.5,             # Conservative time steps
        steps_per_launch=50,     # Balanced batching
    )

# Evaluation
env = create_evaluation_env(graph, model)
results = []
for seed in range(num_eval_runs):
    env.reset()
    trajectory = []
    while env.current_time < T_max:
        env.step()
        trajectory.append(env.count_by_state())
    results.append(trajectory)
```

**Accuracy tips:**
- Use `epsilon=0.01` for publication-quality results
- Run multiple seeds (5-10) to quantify stochastic variance
- Validate against analytical solutions for simple models (SIS endemic)

### Configuration Comparison

| Setting | Training | Evaluation |
|---------|----------|------------|
| `epsilon` | 0.05-0.1 | 0.01-0.03 |
| `tau_max` | 1.0-2.0 | 0.5-1.0 |
| `steps_per_launch` | 100 | 50 |
| Relative speed | 1.0x | 0.3-0.5x |
| Use case | Gradient updates | Final metrics |

---

## Benchmarking Your Setup

Run the roofline benchmark to characterize performance on your hardware:

```bash
# Single configuration test
python experiments/benchmark_roofline.py --config renewal_batched_50 --num-nodes 1000000

# Full benchmark suite (SLURM)
sbatch slurm/run_roofline_array.sbatch
sbatch slurm/merge_roofline_results.sbatch  # After completion

# View results
cat results/roofline_summary.md
```

For optimization ablation:

```bash
sbatch slurm/run_ablation_study.sbatch
sbatch slurm/merge_ablation_results.sbatch
cat results/ablation/ablation_summary.md
```

See [docs/PERFORMANCE_ANALYSIS.md](docs/PERFORMANCE_ANALYSIS.md) for detailed performance documentation.

---

## SLURM Execution

For HPC environments with SLURM:

```bash
# Markovian simulation
sbatch slurm/run_markovian.sbatch

# Non-Markovian simulation
sbatch slurm/run_renewal.sbatch

# Performance benchmarking
sbatch slurm/run_roofline_array.sbatch

# Optimization ablation study
sbatch slurm/run_ablation_study.sbatch
```

---

## Reproducing the Paper

Every result in the companion manuscript (`docs/jocs/FlashSpread-JOCS.tex`) is
regenerated from this repository against the commit tagged `jocs-submission`.

**Reference environment.** NVIDIA A100-SXM4-80GB (driver ≥ 535), PyTorch 2.5.1,
Triton 3.1, CUDA 12.1, Python 3.11 (conda env pinned in `environment.yml`). The
SLURM wrappers under [slurm/](slurm/) run each benchmark end-to-end on a
single-GPU queue; random seeds are fixed across scripts so that the JSON and
figure artefacts in [results/](results/) regenerate bitwise-identically.

### Figure and table → script map

| Artefact | Script | Wall-clock |
|----------|--------|-----------|
| Fig. 3 (throughput scaling), Tab. 2 (throughput decomposition) | [experiments/bench_throughput_scoglio.py](experiments/bench_throughput_scoglio.py) | ~5 min |
| Fig. 4 (roofline), Tab. "Roofline" | [experiments/benchmark_roofline.py](experiments/benchmark_roofline.py) | ~15 min |
| Fig. 5 (fidelity vs MATLAB exact) | [experiments/plot_scoglio_fused_vs_matlab.py](experiments/plot_scoglio_fused_vs_matlab.py) | ~10 min |
| Tab. "ε convergence sweep" | [experiments/validate_epsilon_sweep.py](experiments/validate_epsilon_sweep.py) | ~10 min |
| Tab. "CSR dispatch", Tab. "degree dispatch sweep" | [experiments/bench_warp_collab_ba.py](experiments/bench_warp_collab_ba.py) | ~30 min |
| Appendix C.1 (fidelity details), Figs. 6–7 | [experiments/fidelity_sweep.py](experiments/fidelity_sweep.py) + [experiments/plot_fidelity.py](experiments/plot_fidelity.py) | ~20 min |
| Appendix C.2 (multi-topology), Fig. 8 | [experiments/fidelity_multi_graph.py](experiments/fidelity_multi_graph.py) + [experiments/exact_gillespie_seir.py](experiments/exact_gillespie_seir.py) + [experiments/plot_fidelity_multi.py](experiments/plot_fidelity_multi.py) | ~45 min |
| Appendix C.3 (SIS / SIR vs Doob–Gillespie), Fig. 9 | [experiments/exact_gillespie_sis_sir.py](experiments/exact_gillespie_sis_sir.py) + [experiments/validate_sis_sir_flashspread.py](experiments/validate_sis_sir_flashspread.py) + [experiments/plot_sis_sir.py](experiments/plot_sis_sir.py) | ~60 min |
| Appendix B (erfcx approximation bounds) | `pytest tests/smoke_test.py::TestTritonErfcxAccuracy` | <10 s |

### Data sources for Fig. 3 (throughput scaling)

Fig. 3 is drawn from two CSVs that the scripts above produce:

- GPU curves (Fused CG, unfused CG, eager) and the exact-Gillespie CPU
  baselines (MATLAB, Python from Scoglio et al.) are in
  [results/throughput_comparison.csv](results/throughput_comparison.csv) and
  are measured directly in realized transitions per second (events/s).
- CPU tau-leaping (8-core) is measured in NUPS in
  [results/cpu_tauleaping_nups.csv](results/cpu_tauleaping_nups.csv) and
  rendered on the events/s axis of Fig. 3 via the shared events-per-NUPS
  ratio calibrated at *N* = 10⁶ on the Fused CG run (7.24 × 10⁻⁴ events per
  NUPS). The ratio is valid because both CPU and GPU run the same ε = 0.03
  tau-leaping algorithm with matched log-normal SEIR parameters; the ratio
  reflects per-step realized-events-per-node, which is a property of the
  algorithm, not the hardware.
- The Fused CG 8.09 Giga-NUPS headline used in Table 2 and the abstract is
  the 1-thread-per-node kernel row of
  [results/warp_collab_throughput.csv](results/warp_collab_throughput.csv)
  (NUPS measured directly as *N* × steps / wall-clock).
- The unfused-CG NUPS row in Table 2 comes from the `renewal_batched_50`
  entry in
  [results/log_normal_SEIR_throughput.csv](results/log_normal_SEIR_throughput.csv)
  (steps-per-second × *N*).

### CPU tau-leaping implementation

The CPU tau-leaping curve is produced by
[experiments/bench_cpu_tauleaping.py](experiments/bench_cpu_tauleaping.py),
which runs the **same** `RenewalEngine` class as the GPU path
([flashspread/engines/renewal.py](flashspread/engines/renewal.py)) with
`device="cpu"` and `torch.set_num_threads(min(cpu_count, 8))` for intra-op
parallelism. It uses the same configuration as the GPU benchmark —
ε = 0.03, τ_max = 0.1, β = 2/d = 0.25, log-normal E→I (mean 5, median 4)
and I→R (mean 7.5, median 5) — and runs the same adaptive Bernoulli
tau-leaping step (adaptive τ selection, Bernoulli sampling against
1 − exp(−λτ), renewal age reset on transition, log-normal hazard via
erfcx).

Three GPU-specific optimizations are **not** used on CPU because they have
no CPU analogue: the Triton `FlashNeighbor` kernel (CPU falls back to
`reference_influence` using `torch.scatter_add`), the fused Triton step
kernel in [flashspread/engines/renewal_fused.py](flashspread/engines/renewal_fused.py),
and CUDA-Graph batching. The degree-aware CSR dispatch (warp/merge
strategies) is also GPU-only. CPU parallelism comes from PyTorch's
intra-op threading across the 8 cores on the benchmark box. The 217×
"strict hardware" speedup reported in Table 2 is therefore strictly
hardware-plus-implementation: same algorithm, same ε, same SEIR
parameters, the gap is the fused Triton kernel + coalesced CSR + CUDA
Graph on the A100 versus threaded `scatter_add` on an 8-core CPU.

---

## Citation

If you use FlashSpread in your research, please cite:

```bibtex
@article{shakeri2025flashspread,
  title={FlashSpread: A Unified GPU Framework for Markovian and Non-Markovian
         Spreading Processes on Complex Networks},
  author={Shakeri, Heman},
  journal={},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds upon the Generalized Epidemic Modeling Framework (GEMF) and incorporates IO-aware design principles inspired by FlashAttention.
