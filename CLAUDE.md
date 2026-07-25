# FlashSpread Development Notes

These notes are for contributors and coding agents. Keep user-facing installation,
examples, performance results, and citation details in [README.md](README.md). The current
manuscript source is `docs/jocs/FlashSpread-JOCS.tex` (local and gitignored). When prose,
tests, and implementation disagree, inspect the live implementation and tests before
changing a claim.

## Public API and design direction

`Simulator` + `EngineConfig` is the preferred scalar interface. `Simulator` selects the
model family, resolves the device, owns the seed, and records a `Trajectory`;
`EngineConfig` validates dispatch choices before allocation or CUDA Graph capture.

```python
import flashspread as fs

device = fs.resolve_device()
graph = fs.regular_graph(
    10_000,
    degree=8,
    seed=0,
    device=device,
    algorithm="circulant",
)
model = fs.SEIRModel(beta=0.3)
config = fs.EngineConfig(epsilon=0.03, tau_max=1.0)
trajectory = (
    fs.Simulator(graph, model, device=device, seed=0, config=config)
    .seed_infection(100)
    .run(until=50.0)
)
```

Rules for public API work:

- Prefer `EngineConfig` for new configuration. Legacy `**engine_kwargs` remain for
  compatibility, but callers cannot combine them with `config=`.
- Keep direct engine classes and factory functions available for power users, tests, and
  benchmarks. Do not make them the recommended introductory API.
- `Simulator` is intentionally scalar. Independent shared-graph trajectories use
  `flashspread.engines.create_ensemble_engine`; their per-replica clocks do not fit the
  scalar `Trajectory` contract.
- Keep both examples facade-based and runnable on CPU or CUDA. They should demonstrate
  `EngineConfig`, `Simulator`, `Trajectory`, and CSR-native graph construction.

## Execution contracts

### GraphCSR

`GraphCSR` is the canonical runtime graph:

- CSR rows are incoming by default: a target row stores sources contributing pressure.
- Offsets and node indices are int32, so nodes and stored directed entries must fit the
  package's int32 limits.
- Unit weights remain symbolic until the public `.weights` compatibility property is read.
- The outgoing orientation needed by Markovian propagation is built lazily and cached.
- Graph indices and weights are immutable while an engine is bound. Use
  `graph.with_weights(...)` and construct a new engine for changed weights.
- Engines accept a `GraphCSR` directly or a wrapper whose `.csr` is a `GraphCSR`.

### Markovian SIS/SIR

Markovian execution has CPU reference and CUDA implementations. Each internal leap
evaluates rates and samples events over all `N` nodes, then propagates influence through
the outgoing edges of changed nodes. The work is
`O(N + sum(deg_out(v) for v in changed_frontier))`, not sparse `O(K)`.

Automatic Markovian dispatch is eager on both CPU and CUDA. Exact built-in SIS/SIR can use
explicit CUDA Graph batching, but that is opt-in through `EngineConfig(execution="cuda_graph")`.

### Renewal SEIR

Renewal execution must choose `tau` from the current state/age rate field. The production
fused CUDA path is multi-phase:

1. Traverse incoming CSR for susceptible pressure, evaluate stable log-normal hazards,
   materialize public current rates, and emit compact maximum-rate partials.
2. Reduce the partials and finalize the adaptive `tau` for that same rate field.
3. Sample Bernoulli events, update state and age, and advance the clock.

Do not describe this as one monolithic kernel. CUDA Graph execution replays the phase
sequence for a fixed number of internal steps.

Only the exact, unmodified built-in `SEIRModel` is eligible for the specialized fused
scalar path. Subclasses, shadowed hooks, and custom renewal protocols use reference
execution under automatic dispatch; forcing `backend="fused"` for them must fail rather
than silently replace their semantics.

### Ensembles

Ensemble state, age, and rates are node-major `[N, R]`; `tau` and `current_time` are `[R]`.
Every replica has an independent clock and random stream while sharing one CSR graph. CUDA
auto-selects the tiled implementation; CPU uses the PyTorch reference. The exact built-in
constant-transmission SEIR path additionally uses packed infectious-state bits, compact
per-replica reductions, and a tiled transition phase.

## Dispatch and numerical gotchas

- `resolve_device(None)` chooses CUDA when available and CPU otherwise. CUDA execution
  requires the `gpu` extra; force `device="cpu"` when validating a base install on a
  CUDA-visible host without Triton.
- Both Markovian and renewal scalar families have CPU reference paths.
- The fused scalar CUDA Graph engine double-buffers and rounds odd `batch_steps` up to the
  next even value. Read `sim.steps_per_launch` for the effective window.
- `run(until=T)` stops at the first completed internal step or replay window at or beyond
  `T`; `trajectory.times[-1]` is the actual end time. Recording is sampled at completed
  steps/windows, not interpolated or backfilled.
- `compact=True` requires fused CUDA Graph execution with thread traversal. If traversal is
  `auto`, configuration resolution selects thread. Do not force thread compaction on
  hub-heavy graphs without measuring it against auto/merge.
- `precision="mixed"` requires the fused renewal backend and is valid with warp traversal.
- Model parameters are copied into the engine at construction, and CUDA Graphs bind fixed
  storage. Rebuild the engine after changing model parameters, graph topology, weights, or
  intervention policy; there is no live control API. `RenewalEngineFused` enforces this with
  `_assert_control_unchanged()` beside the existing `GraphCSR._assert_unchanged`, because
  `epsilon`/`tau_max` are runtime kernel arguments: mutating them used to take effect on the
  eager engine (bypassing constructor validation) while the captured engine kept the frozen
  values, so the same attribute meant two different things.
- `reset()` reproduces the base random stream. `reset(episode=k)` derives an independent
  episode stream by *mixing* `(base_seed, k)`, not by adding them. Engines maintain private,
  full-width random streams; do not replace them with process-global RNG calls in hot paths.
- Bernoulli tau-leaping is approximate. Validate tolerance sensitivity and exact-reference
  agreement for the intended model and observable; do not claim a universal fidelity floor.
- Counter-based sampling has a width contract. Triton's `philox` picks its word width from
  the *counter* dtype, and 64-bit random words no longer compare against
  `_bernoulli_from_words`' uint32 threshold — which scales every event probability by
  `2**-32`. Whether a 64-bit counter actually widens the words is *version-dependent*
  (measured: 3.1.0/3.2.0/3.3.1 widen, 3.6.0/3.7.1 do not), so never rely on the observed
  behaviour: lane identities wider than one uint32 go in a second counter word via
  `_sample_bernoulli_counter`. `tl.static_assert` guards the invariant, and
  `tests/test_rng_contract.py` asserts the safety property on any version without a GPU.

### Supported Triton versions, measured

The declared `gpu` floor is `triton>=3.2`, not `>=2.1`. Measured on one A100 by running the
tiled ensemble path and probing `tl.randint4x` under four Triton versions (SLURM jobs
17309977 / 17310442):

| Triton | `randint4x` word width for a **uint64** counter | tiled ensemble kernel |
|---|---|---|
| 3.1.0 | 64 (would break a uint32 threshold) | **does not compile** — `tl.range(loop_unroll_factor=)` is 3.2+ |
| 3.2.0 | 64 | compiles and runs |
| 3.3.1 | 64 | compiles and runs |
| 3.6.0 | 32 | compiles and runs |
| 3.7.1 | 32 | compiles and runs (GitHub CI resolves this) |

Two lessons. First, `>=2.1` advertised a GPU feature that cannot compile. Second, Philox's
word width is *version-dependent*, so never rely on it: keep counters in uint32 words and
let the `tl.static_assert` in `_bernoulli_from_words` enforce it.

### Behaviour changes and what they do (not) invalidate

- **Ensemble trajectories: version-dependent, and the A100 baseline is unaffected.** The
  historical packed 64-bit counter collapsed every transition probability to `p * 2**-32`
  *only on the Triton versions that widen the Philox words*. Measured with the historical
  code in place, 20 steps, 32 replicas, seed 7: Triton 3.2.0 and 3.3.1 accepted **0**
  transitions in every replica, while 3.6.0 accepted 99-170. With the two-word counter all
  three versions accept 99-170 identically. So recorded ensemble trajectories produced on
  **Triton 3.6.0 (including the A100 acceptance runs and every manuscript figure) remain
  valid**; anything produced on 3.2.x or 3.3.x is invalid. Timing measurements were never
  affected either way, because the acceptance harness restores fixed checkpoints and the
  step costs the same whether or not an event is accepted.
- **Episode streams.** `reset(episode=k)` for `k != 0` now yields different streams.
  `reset()`, `reset(episode=0)` and every base-seed run are bitwise unchanged.
- **Large-N initial conditions.** `seed_infection` draws seeds with `sample_distinct_nodes`,
  which keeps the historical `randperm(N)[:k]` for `N <= 2**22` and switches to O(k)
  rejection sampling above it. Initial conditions at or below ~4.19M nodes are therefore
  bitwise unchanged; larger populations select different nodes from the same distribution.

## Graph construction

- `regular_graph(..., algorithm="circulant")` constructs exact-simple undirected regular
  CSR directly with bounded temporary storage and no NetworkX dependency. It is structured,
  not a uniformly sampled random-regular graph.
- The default regular generator and the Barabasi-Albert, Watts-Strogatz, and geometric
  generators use NetworkX and require the `graph` extra.
- Existing CSR should enter through `fs.from_csr(...)`; COO `[source, target]` input should
  enter through `fs.from_edges(...)`. Neither constructor adds reciprocal edges.
- Use the circulant path for large deterministic performance workloads, and disclose its
  graph semantics in every benchmark report.

## Validation and performance evidence

CPU/development checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check flashspread tests examples \
  experiments/benchmark_acceptance.py \
  experiments/benchmark_markovian.py \
  experiments/benchmark_ensemble.py \
  experiments/perf_model.py \
  experiments/ensemble_perf_model.py
```

For GPU tests, install `.[dev,gpu]` and run `python -m pytest -m gpu`, or submit
`slurm/run_gpu_validation.sbatch`, which runs the whole `gpu` selection and writes a JSON
report to shared storage. The latest complete local run collects **425 tests**: the full
invocation reports **370 passed, 55 skipped**, and `-m "not gpu"` (what CI runs) reports
**370 passed, 5 skipped, 50 deselected**. Measured on Python 3.11.14, PyTorch 2.5.1+cu121,
Triton 3.1.0. Quote the environment whenever you quote the counts — the skip total moves
with Triton and CUDA availability, so a bare number goes stale silently.

Kernel tests that execute under `TRITON_INTERPRET=1` are gated by
`tests/_triton_support.py`, which probes the installed interpreter rather than pinning a
version range. Triton 3.1 both rejects valid kernels and, worse, silently fails to deliver
kernel stores to host tensors, which makes interpreter-mode assertions vacuous. Treat those
tests as unavailable, not as coverage, unless the probe reports the interpreter usable.

Production acceptance harnesses. All eight presets are evidence: the hub-heavy traversals
decide whether `auto` dispatch is choosing correctly, and the compaction preset is the only
measurement of `compact=True`.

```bash
python experiments/benchmark_acceptance.py walltime --case regular-constant
python experiments/benchmark_acceptance.py walltime --case regular-age
python experiments/benchmark_acceptance.py walltime --case regular-mixed
python experiments/benchmark_acceptance.py walltime --case regular-late-compact
python experiments/benchmark_acceptance.py walltime --case ba-auto
python experiments/benchmark_acceptance.py walltime --case ba-thread
python experiments/benchmark_acceptance.py walltime --case ba-warp
python experiments/benchmark_acceptance.py walltime --case ba-merge
python experiments/benchmark_markovian.py walltime
python experiments/benchmark_ensemble.py walltime --replicas 32
```

Evidence rules:

- The current A100 snapshot and metric definitions live in the README. Do not duplicate a
  second performance table here.
- Acceptance checkpoints are deterministic synthetic states restored independently, not
  epidemic phases observed along one trajectory.
- Scalar NUPS is `N * internal steps / target wall time`; ensemble throughput is
  `N * replicas / step wall time`. Neither metric is realized transitions or unique nodes.
- The regular acceptance workload is a seeded circulant; say so explicitly.
- Construction, checkpoint restoration, and warmup are outside the timed target.
- `experiments/benchmark_roofline.py` is historical synthetic exploration, not production
  characterization. Nsight Compute failed with `ERR_NVGPUCTRPERM`; do not assert a
  compute-bound or memory-bound roofline classification without hardware counters.
- `results/speed_check/` contains historical measurements from an older pipeline. Use the
  production acceptance harnesses and final local `logs/a100_final_*.json` artifacts for
  current evidence. `logs/` is ignored; never force-add the whole directory.

## Repository layout

```text
flashspread/
├── __init__.py                 # lazy public exports
├── config.py                   # immutable EngineConfig and dispatch policy
├── simulator.py                # scalar public facade
├── trajectory.py               # recorded scalar observables
├── graphs.py                   # public graph constructors
├── engines/
│   ├── __init__.py             # scalar and ensemble factories
│   ├── markovian.py            # CPU/CUDA SIS/SIR execution
│   ├── renewal.py              # renewal reference variants
│   ├── renewal_fused.py        # fused scalar CUDA execution
│   └── ensemble.py             # reference/tiled shared-graph ensembles
├── models/                     # SIS, SIR, SEIR, and hazard functions
└── core/
    ├── graph.py                # canonical GraphCSR
    ├── flash_neighbor.py       # CSR influence kernels
    ├── flash_markovian.py      # Markovian Triton kernels
    ├── flash_renewal_kernel.py # fused renewal kernels/finalizer
    ├── flash_ensemble*.py      # tiled ensemble kernels
    └── host_rng.py             # seed normalization and host streams

examples/                       # Simulator/EngineConfig examples
experiments/                    # acceptance, profiling, and performance models
tests/                          # CPU and GPU correctness/dispatch coverage
slurm/                          # cluster wrappers
docs/jocs/                      # local manuscript source (gitignored)
```

## Cluster and manuscript workflow

- GPU work uses SLURM allocation `-A cdt_computing`, partition `-p gpu`, and typically
  `--gres=gpu:a100:1`. Add the 80 GB constraint only when the workload needs it.
- `/tmp` is node-local. Put SLURM stdout/stderr and artifacts on shared storage when they
  must remain visible after the job.
- Manuscript source: `docs/jocs/FlashSpread-JOCS.tex`. Load the site TeX Live module and
  use `latexmk -pdf`; do not commit generated TeX artifacts unless explicitly requested.
- Preserve unrelated dirty worktree files. Stage explicit paths rather than `git add -A`.
- Author commits as the repository owner; do not add an AI/Claude co-author trailer.
