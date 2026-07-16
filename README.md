# FlashSpread

**GPU-accelerated Markovian and non-Markovian spreading processes on complex networks**

FlashSpread simulates stochastic SIS, SIR, and renewal SEIR dynamics on sparse graphs.
It provides CPU reference execution for correctness and development, Triton/CUDA fast
paths for large workloads, adaptive Bernoulli tau-leaping, and graph-reusing ensembles
with independent trajectories.

## Highlights

- **One scalar API.** `Simulator` selects the Markovian or renewal family, resolves the
  device, owns reproducible random streams, and records a `Trajectory`.
- **Current-rate renewal execution.** Susceptible CSR traversal, log-normal hazard
  evaluation, and compact rate-bound partials are fused; device reductions and a
  finalizer select the current step's `tau`, then a transition kernel samples events.
- **Degree-aware sparse traversal.** Fused renewal execution supports thread, warp, and
  edge-merge CSR strategies with automatic selection for uniform and hub-heavy graphs.
- **Memoryless propagation without stale rates.** Built-in SIS/SIR execution performs
  dense node rate and sampling work while propagating influence through the changed
  outgoing frontier.
- **Independent ensembles.** A shared CSR graph drives node-major `[N, R]` state with
  one adaptive clock and random stream per replica.
- **Scalable graph storage.** `GraphCSR` stores int32 incoming CSR, keeps unit weights
  symbolic, reconstructs COO only on request, and builds its outgoing transpose lazily.
- **Semantics-preserving dispatch.** Only the exact, unmodified built-in `SEIRModel`
  enters the specialized fused scalar path; custom models and subclasses retain
  reference execution.

## Installation

```bash
git clone https://github.com/Shakeri-Lab/FlashSpread.git
cd FlashSpread

# Runtime with CPU reference execution
python -m pip install -e .

# Add Triton GPU kernels and NetworkX-backed graph generators
python -m pip install -e ".[gpu,graph]"

# CPU development environment
python -m pip install -e ".[dev]"

# CUDA development environment, including GPU tests
python -m pip install -e ".[dev,gpu]"
```

Package metadata requires Python 3.10 or newer, PyTorch 2.0 or newer, and NumPy 1.24
or newer. A bare `import flashspread` is lazy: it does not initialize PyTorch, NumPy,
Triton, or the optional graph stack until an exported runtime object is used.
On a CUDA-capable host, install the `gpu` extra or pass `device="cpu"` explicitly.

| Extra | Dependency | Capability |
|---|---|---|
| `gpu` | `triton>=2.1` | Triton GPU kernels |
| `graph` | `networkx>=3.0` | NetworkX-backed graph generators |
| `reorder` | `scipy>=1.10` | reverse Cuthill-McKee preprocessing |
| `all` | all runtime extras | GPU, graph generation, and reordering |
| `dev` | pytest, Ruff, mypy, NetworkX, SciPy | development and testing |

Inspect the active environment with:

```python
import flashspread as fs

fs.check_env()
```

## Quick start

### Renewal SEIR

This example runs through the CPU reference engine on a CPU-only machine and selects the
fused CUDA path when a supported GPU is available. The direct circulant graph constructor
does not require NetworkX.

```python
import flashspread as fs

graph = fs.regular_graph(
    10_000,
    degree=8,
    seed=0,
    algorithm="circulant",
)
model = fs.SEIRModel(
    beta=0.25,
    mean_ei=5.0,
    median_ei=4.0,
    mean_ir=7.5,
    median_ir=5.0,
)
config = fs.EngineConfig(epsilon=0.03, tau_max=1.0)

sim = fs.Simulator(graph, model, seed=0, config=config).seed_infection(100)
trajectory = sim.run(until=50.0, record_every=1.0)

print(trajectory.peak_infected, trajectory.peak_time)
print(trajectory.final_attack_rate)
print(trajectory["I"])
```

`SEIRModel` uses log-normal E-to-I and I-to-R holding times, so each pair must satisfy
`mean > median > 0`. Set `transmission_mode="age_dependent"` to scale infectious-source
shedding by the source node's current I-to-R hazard.

### Markovian SIS and SIR

SIS and SIR also run on CPU or CUDA. Automatic Markovian dispatch is eager to preserve
one-step granularity; CUDA Graph batching is an explicit option.

```python
import flashspread as fs

graph = fs.regular_graph(10_000, degree=8, seed=1, algorithm="circulant")
model = fs.SISModel(beta=0.5, delta=0.2)

sim = fs.Simulator(graph, model, seed=1).seed_infection(100)
trajectory = sim.run(until=30.0)

print(trajectory.peak_infected)
```

To request batched Markovian execution on CUDA:

```python
config = fs.EngineConfig(execution="cuda_graph", batch_steps=50)
sim = fs.Simulator(graph, model, device="cuda", seed=1, config=config)
```

### Independent ensembles

The ensemble factory stores the graph once and advances independent trajectories. State,
age, and rate tensors are node-major `[N, R]`; `tau` and `current_time` have shape `[R]`.
CUDA selects the tiled Triton implementation, while CPU selects the PyTorch reference.

```python
import flashspread as fs
from flashspread.engines import create_ensemble_engine

graph = fs.regular_graph(100_000, degree=8, seed=2, algorithm="circulant")
model = fs.SEIRModel(beta=0.3)
engine = create_ensemble_engine(graph, model, replicas=32, seed=2)

engine.seed_infection(100)
tau, state = engine.step()
counts = engine.count_by_state()

print(tau.shape)                  # [32]
print(state.shape)                # [100_000, 32]
print(counts.shape)               # [32, 4]
```

`Simulator` intentionally remains scalar: a single `Trajectory` cannot represent the
ensemble engine's independently adaptive clocks.

## Configuration and dispatch

`EngineConfig` is the preferred immutable configuration surface. Invalid combinations
fail before engine allocation or CUDA Graph capture.

```python
config = fs.EngineConfig(
    backend="auto",          # auto | reference | fused
    execution="auto",       # auto | eager | cuda_graph
    traversal="auto",       # auto | thread | warp | merge
    transmission="model",   # model | constant | age_dependent
    precision="fp32",       # fp32 | bf16_weights | mixed
    compact=False,
    batch_steps=50,
    epsilon=0.03,
    tau_max=1.0,
)
```

The default policy is:

| Model/device | Automatic execution |
|---|---|
| SIS or SIR, CPU | eager PyTorch reference |
| SIS or SIR, CUDA | eager CUDA/Triton Markovian engine |
| exact built-in SEIR, CPU | eager PyTorch renewal reference |
| exact built-in SEIR, CUDA | fused renewal CUDA Graph |
| custom or modified renewal model | reference backend preserving model hooks |

Legacy `Simulator` engine keywords remain supported, but a call must use either
`config=EngineConfig(...)` or legacy engine keywords, not both. Power users can call
`create_markovian_engine`, `create_renewal_engine`, `create_ensemble_engine`, or
`create_engine` from `flashspread.engines` directly.

Active compaction is restricted to fused CUDA Graph execution with thread traversal.
Mixed storage requires the fused backend and is compatible with the production traversal
strategies. Graph and model parameters are construction-time inputs: rebuild the engine
after changing topology, weights, model parameters, or intervention policy.

## Graphs

`GraphCSR` is the runtime graph contract. By default, rows are target nodes and columns
are source nodes whose state or infectivity contributes to that target. Directed COO input
therefore uses `[source, target]`; undirected input must contain both directions.

```python
import torch
import flashspread as fs

# COO: 0 -> 1 and 1 -> 0
edge_index = torch.tensor([[0, 1], [1, 0]])
graph_from_edges = fs.from_edges(edge_index, num_nodes=2)

# Incoming CSR: row 0 reads source 1; row 1 reads source 0
row_ptr = torch.tensor([0, 1, 2], dtype=torch.int32)
col_ind = torch.tensor([1, 0], dtype=torch.int32)
graph_from_csr = fs.from_csr(row_ptr, col_ind)
```

The public generators are symmetric by default and accept a reproducibility seed:

```python
fs.regular_graph(1_000_000, degree=8, seed=42, algorithm="circulant")
fs.regular_graph(100_000, degree=8, seed=42)       # NetworkX random-regular
fs.barabasi_albert(100_000, m=4, seed=42)
fs.watts_strogatz(100_000, k=8, p=0.1, seed=42)
fs.geometric(100_000, radius=0.01, seed=42)
```

`algorithm="circulant"` builds an exact-simple undirected regular graph directly in
int32 CSR with bounded temporary storage. It is suitable for very large deterministic
workloads, but it is not a uniformly sampled random-regular graph. The other generator
paths require the `graph` extra. Engines accept `GraphCSR` directly or an object whose
`.csr` attribute is a `GraphCSR`.

Graph storage is immutable while an engine is bound. Use `graph.with_weights(...)` and
construct a new engine when edge weights change.

## Execution model

### Markovian family

Each built-in SIS/SIR leap evaluates rates and samples events over all `N` nodes. After
state changes, the engine updates influence by traversing the outgoing edges of the changed
frontier `F`. Its per-leap work is therefore
`O(N + sum(deg_out(v) for v in F))`, not sparse `O(|F|)`.

### Renewal family

Renewal hazards change with compartment age, so rates must be evaluated from the current
state and age on every internal step. The specialized CUDA path is deliberately
multi-phase:

1. Fuse susceptible incoming-CSR traversal, stable `erfcx` log-normal hazards, public
   current-rate writes, and compact maximum-rate partials.
2. Reduce the compact partials and finalize the adaptive `tau` for that same rate field.
3. Sample Bernoulli events, update state and age, and advance the clock.

CUDA Graph execution replays this sequence for a fixed batch of internal steps. Public
rates remain materialized for inspection; the implementation does not claim that the
whole simulation step is one monolithic kernel.

### Ensemble family

Ensembles share incoming CSR metadata but retain independent state, age, adaptive time,
and random streams per replica. The exact constant-transmission built-in SEIR path uses a
packed infectious bitmap, tiled CSR/rate evaluation, compact per-replica reductions, and a
tiled transition phase. Other supported models use the tiled gather with reference model
and transition hooks.

## Numerical semantics

FlashSpread uses adaptive Bernoulli tau-leaping, which is an approximation to the
continuous-time process. `epsilon` controls the renewal per-step hazard scale;
`max_prob` and `theta` control Markovian adaptation; `tau_max` caps one internal leap.
Validate tolerance sensitivity and agreement with an exact reference for the model and
observable of interest. No universal error floor is claimed for the current-rate
implementation.

`run(until=...)` stops after the first completed internal step or CUDA Graph window at or
beyond the requested horizon. `trajectory.times[-1]` is the actual end time. Eager
execution gives one-step granularity; smaller `batch_steps` gives tighter batched
granularity. The fused scalar CUDA Graph engine double-buffers and rounds an odd requested
batch size up to the next even value; inspect `sim.steps_per_launch` for the effective
window.

`reset()` reproduces the base random stream. `reset(episode=k)` derives an independent
episode stream from `base_seed + k`. Engine-owned model parameters are copied at
construction, so mutating the original model does not update a running simulation.

## Performance snapshot

The latest production-path acceptance run used one NVIDIA A100-SXM4-80GB, Python 3.12.2,
PyTorch 2.11.0+cu130 with its CUDA 13.0 build/runtime, and Triton 3.6.0. Scalar workloads used `N=1,000,000`,
approximately 8,000,000 directed CSR entries, and 50 internal steps per timed
`Simulator.step()` call.
The ensemble used the same regular graph with `R=32` and timed one independent-replica
step.

| Workload | Early | Peak | Late | Metric |
|---|---:|---:|---:|---|
| Renewal, constant transmission | 12.421 | 13.665 | 15.576 | G-NUPS |
| Renewal, age-dependent transmission | 11.410 | 12.223 | 14.029 | G-NUPS |
| Renewal, mixed storage | 12.752 | 13.957 | 16.298 | G-NUPS |
| Barabasi-Albert `m=4`, auto/merge | 2.181 | 2.757 | 3.676 | G-NUPS |
| Markovian SIS | 15.548 | 13.931 | 14.234 | G-NUPS |
| Ensemble SEIR, `R=32` | 11.824 | 11.612 | 15.250 | G node-replica updates/s |

For scalar runs, NUPS is `N * internal tau-leaps / target wall time`; it is not the number
of realized transitions, frontier edges, or unique changed nodes. Ensemble throughput is
`N * R / step wall time`.

Early, peak, and late are deterministic synthetic checkpoints restored independently
before each target, not phases observed along one simulated trajectory. Renewal checkpoint
fractions `(S, E, I, R)` are `(0.98, 0.01, 0.01, 0)`,
`(0.45, 0.15, 0.25, 0.15)`, and `(0.05, 0.02, 0.03, 0.90)`; SIS infected fractions are
`0.01`, `0.25`, and `0.03`. Regular cases use a seeded exact-simple circulant, not a
uniform random-regular sample. Timings exclude construction, checkpoint restoration, and
warmup.

These are synchronized wall-time measurements, not a hardware-counter roofline result.
The available Nsight Compute run lacked GPU performance-counter permission, so no
compute-bound or memory-bound classification is asserted.

## Reproducing the acceptance runs

The production harnesses record invocation, environment, graph semantics, checkpoint
definitions, source fingerprints, warmup policy, and timing distributions in JSON.

```bash
# Scalar renewal: regular constant/age/mixed and BA auto-dispatch
python experiments/benchmark_acceptance.py walltime --case regular-constant
python experiments/benchmark_acceptance.py walltime --case regular-age
python experiments/benchmark_acceptance.py walltime --case regular-mixed
python experiments/benchmark_acceptance.py walltime --case ba-auto

# Captured Markovian SIS
python experiments/benchmark_markovian.py walltime

# R=32 shared-graph ensemble
python experiments/benchmark_ensemble.py walltime --replicas 32
```

Use `--dry-run --output -` to validate a command and emit provenance without requiring a
GPU. Each harness also has a `profile` mode for one NVTX-delimited production call and a
`print-ncu-command` mode for generating a profiler command.

The performance models in
[experiments/perf_model.py](https://github.com/Shakeri-Lab/FlashSpread/blob/main/experiments/perf_model.py)
and
[experiments/ensemble_perf_model.py](https://github.com/Shakeri-Lab/FlashSpread/blob/main/experiments/ensemble_perf_model.py)
report logical
algorithmic traffic and storage. They do not substitute for measured HBM counters.

## Testing

```bash
python -m pytest
python -m pytest -m gpu       # requires an installed [gpu] extra and CUDA device

ruff check flashspread tests \
  experiments/benchmark_acceptance.py \
  experiments/benchmark_markovian.py \
  experiments/benchmark_ensemble.py \
  experiments/perf_model.py \
  experiments/ensemble_perf_model.py
```

CUDA/Triton tests skip automatically when a GPU is unavailable. The latest complete local
run reports **347 passed, 45 skipped**; the final selected A100 validation reports
**73 passed**.

## Citation

```bibtex
@article{shakeri2026flashspread,
  title={FlashSpread: IO-Aware GPU Simulation of Non-Markovian Epidemic Dynamics via Kernel Fusion},
  author={Shakeri, Heman and Moradi-Jamei, Behnaz and Vajdi, Aram and Ardjmand, Ehsan},
  journal={arXiv preprint arXiv:2604.22092},
  year={2026}
}
```

## License

FlashSpread is released under the
[MIT License](https://github.com/Shakeri-Lab/FlashSpread/blob/main/LICENSE).

## Acknowledgments

This work builds on the Generalized Epidemic Modeling Framework (GEMF) and incorporates
I/O-aware design principles inspired by FlashAttention.
