# Non-Markovian Renewal Engines: Architecture & Design Guide

This document describes the six renewal engine variants in FlashSpread, their algorithmic differences, and when to use each one.

## Background: Why Multiple Engines?

FlashSpread's renewal engines simulate non-Markovian SEIR epidemics where the E→I and I→R transitions depend on *holding time* (age-dependent hazards). All engines share the same mathematical model — adaptive Bernoulli tau-leaping with `erfcx`-based lognormal hazards — but differ in **how** they execute it on the GPU.

The design space has three orthogonal axes:

1. **Edge transmission**: Binary (Markovian) vs. continuous infectivity (non-Markovian)
2. **Kernel fusion**: Separate PyTorch ops vs. single fused Triton kernel
3. **CUDA Graph batching**: Eager (one step at a time) vs. captured graph (50 steps replayed)

This gives 2 × 2 × 2 = 8 combinations, of which we implement the 6 that are useful (the two excluded are fused-without-infectivity, which doesn't make architectural sense since the fused kernel inherently uses the infectivity path).

## Engine Summary

```
                         ┌─────────────────────┐
                         │    Edge Transmission │
                         ├──────────┬──────────┤
                         │  Binary  │  Float   │
                         │ (Markov) │(infect.) │
          ┌──────────────┼──────────┼──────────┤
          │ Separate     │ Renewal  │ NonMarkov│
 Kernel   │ PyTorch ops  │ Engine   │          │
 Fusion   ├──────────────┼──────────┼──────────┤
          │ Fused Triton │    —     │ Fused    │
          │ kernel       │          │          │
          └──────────────┴──────────┴──────────┘

          Each cell has an eager and a CUDAGraph variant (×2).
```

### At a Glance

| Engine | Edge type | Fusion | CUDA Graph | Key file |
|--------|-----------|--------|:----------:|----------|
| `RenewalEngine` | Binary | No | No | `engines/renewal.py` |
| `RenewalEngineCUDAGraph` | Binary | No | Yes | `engines/renewal.py` |
| `RenewalEngineNonMarkov` | Infectivity | No | No | `engines/renewal.py` |
| `RenewalEngineNonMarkovCUDAGraph` | Infectivity | No | Yes | `engines/renewal.py` |
| `RenewalEngineFused` | Infectivity | Yes | No | `engines/renewal_fused.py` |
| `RenewalEngineFusedCUDAGraph` | Infectivity | Yes | Yes | `engines/renewal_fused.py` |

---

## Detailed Architecture

### 1. RenewalEngine (Baseline)

The original engine. Each step executes ~12 separate CUDA kernels via PyTorch eager mode:

```
FlashNeighbor(states)           # Triton: CSR traversal, binary state check
    ↓ writes pressure[N] to VRAM
compute_rates(age, state, pressure)  # PyTorch: erfcx hazard for E/I nodes
    ↓ writes rates[N] to VRAM
rates.max()                     # PyTorch: global reduction
    ↓ scalar tau
1 - exp(-rates * tau)           # PyTorch: 4 elementwise ops
    ↓ writes event_prob[N] to VRAM
_rand_uniform()                 # PyTorch: xorshift RNG
    ↓ writes rand_buffer[N] to VRAM
rand < event_prob               # PyTorch: comparison
    ↓ writes event_mask[N] to VRAM
apply_transitions()             # PyTorch: conditional state update
    ↓ writes next_state[N] to VRAM
age.add_(tau)                   # PyTorch: in-place
age.masked_fill_(changed, 0)    # PyTorch: renewal reset
state.copy_(next_state)         # PyTorch: buffer swap
```

**FlashNeighbor inner loop** (Triton kernel):
```python
# For each node i, traverse its incoming neighbors:
while neighbors_remain:
    j = col_ind[ptr]               # load neighbor index
    state_j = states[j]            # load neighbor state (int32)
    w = weights[ptr]               # load edge weight
    if state_j == INFECTED:        # binary check
        pressure[i] += w
```

**S→E rate**: `beta * pressure[i]` (constant beta, integer neighbor count).

### 2. RenewalEngineCUDAGraph

Identical algorithm to `RenewalEngine`, but the entire step sequence is captured as a CUDA Graph during initialization. On subsequent calls, `graph.replay()` replays all ~12 kernels with zero Python overhead.

**Key constraint**: `sparse_hazard = False` is forced because CUDA Graph capture requires static control flow. The `.any()` guards in sparse mode are data-dependent and would break the captured graph. This means `erfcx` is evaluated for ALL N nodes (including S and R), wasting ~70% of the hazard compute.

**Batching**: `steps_per_launch=50` means each `.step()` call replays 50 simulation steps. This amortizes the ~5μs graph replay overhead over 50 steps.

### 3. RenewalEngineNonMarkov

Adds an **infectivity pre-pass** before FlashNeighbor. Instead of a binary state check, the kernel loads a continuous float `infectivity[j]` per neighbor:

```
compute_infectivity(age, state)      # PyTorch: elementwise
    ↓ writes infectivity[N] to VRAM
FlashNeighborInfectivity(infectivity) # Triton: CSR traversal, float multiply
    ↓ writes pressure[N] to VRAM
compute_rates_nonmarkov(...)          # PyTorch: erfcx for E/I; pressure for S
    ↓ (rest identical to RenewalEngine)
```

**FlashNeighborInfectivity inner loop** (Triton kernel):
```python
while neighbors_remain:
    j = col_ind[ptr]
    inf_j = infectivity[j]        # load float infectivity (not int state)
    w = weights[ptr]
    pressure[i] += inf_j * w      # no branch, just multiply-add
```

**Transmission modes** (controlled by `model.transmission_mode`):

| Mode | `infectivity[j]` for I-nodes | S→E rate | Use case |
|------|------------------------------|----------|----------|
| `"constant"` (default) | `beta` | `beta * count_infected_neighbors` | Validation against original engine |
| `"age_dependent"` | `beta * h_IR(age[j])` | `sum(beta * h_IR(age_j) * w_ji)` | True non-Markovian edge transmission |

In `"constant"` mode, the NonMarkov engine is mathematically equivalent to the original `RenewalEngine` — the infectivity pre-pass sets `infectivity[j] = beta` for I-nodes and `0` otherwise, so `pressure[i] = beta * (weighted infected neighbor count)`, and `rate_S = pressure` directly.

In `"age_dependent"` mode, transmission strength varies with how long the source has been infectious. This captures viral shedding dynamics (e.g., COVID-19 infectiousness peaks ~day 3-5 then decays) without O(E) per-edge age storage — only O(N) per-node ages are needed because all edges from node j share j's infection age.

### 4. RenewalEngineNonMarkovCUDAGraph

CUDA Graph variant of `RenewalEngineNonMarkov`. Same `sparse_hazard = False` constraint applies.

### 5. RenewalEngineFused

Replaces the 12-kernel pipeline with a **single fused Triton kernel** that keeps all intermediate values in GPU registers:

```
_infectivity_prepass()           # PyTorch: elementwise (1 kernel)
_step_counter.add_(N)            # scalar increment (1 kernel)
_flash_renewal_fused_kernel()    # Triton: EVERYTHING below in one launch
    ↓ writes next_state[N], next_age[N], rates[N] to VRAM
state.copy_(next_state)          # buffer swap (1 kernel)
age.copy_(next_age)              # buffer swap (1 kernel)
_compute_tau()                   # rates.max() + divide (1 kernel)
```

**Fused kernel inner structure** (single Triton launch, one thread per node):
```python
# === IN REGISTERS (never touches VRAM) ===

# 1. CSR traversal → pressure
pressure = 0.0
for each neighbor j:
    pressure += infectivity[j] * weight[j]

# 2. Load own state and age
my_state = state[i]
my_age = age[i]

# 3. Compute rate WITH SPARSITY (no control flow divergence)
#    S nodes: rate = pressure
#    E nodes: rate = _erfcx_approx hazard  (only ~30% of nodes)
#    I nodes: rate = _erfcx_approx hazard
#    R nodes: rate = 0                     (skip erfcx entirely)
rate = tl.where(is_S, pressure, 0)
rate = tl.where(is_E, lognormal_hazard_triton(age, mu_ei, sig_ei), rate)
rate = tl.where(is_I, lognormal_hazard_triton(age, mu_ir, sig_ir), rate)

# 4. Bernoulli sampling
p = 1.0 - tl.exp(-rate * tau)
rand = tl.rand(seed, offset)
event = rand < p

# 5. State transition + age update
new_state = apply_seir_transition(my_state, event)
new_age = tl.where(new_state != my_state, 0.0, my_age + tau)

# === SINGLE VRAM WRITE ===
store(next_state[i], next_age[i], rates[i])
```

**Triton-level sparsity**: The `tl.where` chain means S and R nodes never execute the expensive `_erfcx_approx` (~55 FLOPs). The kernel grid is static (N threads), satisfying CUDA Graph requirements, but the ALUs skip dead math for inert nodes. This gives sparsity benefits *without* breaking CUDA Graph capture — something impossible with PyTorch's `.any()` guards.

**erfcx approximation**: Since `torch.special.erfcx` is unavailable in Triton, the fused kernel uses `_erfcx_approx`:
- `|z| ≤ 3.5`: `exp(z²) * (1 - erf(z))` via `tl.math.erf` (safe in fp32)
- `|z| > 3.5`: asymptotic expansion `1/(z√π) * (1 - 1/(2z²) + 3/(4z⁴) - 15/(8z⁶))`
- `z < 0`: `2·exp(z²) - erfcx(-z)`
- Max relative error: ~4×10⁻⁴ (acceptable for stochastic tau-leaping)

**Adaptive tau**: The fused kernel uses the *previous step's* tau (loaded from a device tensor). After the kernel writes `rates[N]`, a lightweight `rates.max()` reduction computes tau for the next step. The first step uses `tau_max`.

### 6. RenewalEngineFusedCUDAGraph

CUDA Graph variant of `RenewalEngineFused`. The entire 5-kernel pipeline is captured. A device-side step counter (`_step_counter`) increments by N each step for unique RNG offsets — this is an in-place tensor operation that survives graph capture, unlike Python integers which get baked in as constants.

---

## Architecture Comparison Tables

### Per-Step Pipeline

| Engine | Kernel launches | Intermediate VRAM buffers | Notes |
|:-------|:---:|:---|:---|
| RenewalEngine | ~12 | pressure, rates, event_prob, event_mask, rand_buffer, next_state | Each buffer is O(N) × 4 bytes |
| RenewalEngineCUDAGraph | ~12 (batched) | Same | Launch overhead amortized over 50 steps |
| RenewalEngineNonMarkov | ~13 | infectivity + all of the above | Extra pre-pass kernel |
| RenewalEngineNonMarkovCUDAGraph | ~13 (batched) | Same as NonMarkov | |
| RenewalEngineFused | ~5 | infectivity, rates, next_state, next_age | pressure/prob/rand/mask all in registers |
| RenewalEngineFusedCUDAGraph | ~5 (batched) | Same as Fused | |

### Feature Matrix

| Feature | Renewal | NonMarkov | Fused |
|:--------|:-------:|:---------:|:-----:|
| Edge transmission | Binary `1{I}` | Float `infectivity[j]` | Float `infectivity[j]` |
| S→E rate computation | `beta × Σ neighbors` | `Σ (infectivity × weight)` | Same, in registers |
| E→I / I→R hazard | `torch.special.erfcx` | `torch.special.erfcx` | `_erfcx_approx` (Triton) |
| Hazard sparsity in CG mode | No (forced dense) | No (forced dense) | Yes (`tl.where` skip) |
| Bernoulli RNG | XorShift → VRAM | XorShift → VRAM | `tl.rand()` in registers |
| State transition | PyTorch → VRAM | PyTorch → VRAM | `tl.where` in registers |
| Global memory writes/step | ~7 × O(N) | ~8 × O(N) | ~3 × O(N) |
| Age-dependent transmission | No | Yes | Yes |
| CUDA Graph variant | Yes | Yes | Yes |

---

## Memory Traffic Analysis

For N nodes and E edges (with degree d, E ≈ N·d):

### RenewalEngine / NonMarkov (per step)

| Operation | Bytes read | Bytes written |
|:----------|:-----------|:-------------|
| FlashNeighbor CSR traversal | (N+1)·4 + E·4 + E·4 + N·4 | N·4 |
| Hazard computation | N·4·3 (age, state, pressure) | N·4 (rates) |
| Transition pipeline | N·4·3 (rates, rand, prob) | N·4·3 (prob, mask, next_state) |
| Age update + swap | N·4·3 | N·4·2 |
| **Total** | **~(8E + 40N) bytes** | **~24N bytes** |

### RenewalEngineFused (per step)

| Operation | Bytes read | Bytes written |
|:----------|:-----------|:-------------|
| Infectivity prepass | N·4·2 (age, state) | N·4 (infectivity) |
| Fused kernel (CSR + everything) | (N+1)·4 + E·4 + E·4 + N·4·3 | N·4·3 (next_state, next_age, rates) |
| Buffer swap + tau | N·4·3 | N·4·2 |
| **Total** | **~(8E + 28N) bytes** | **~20N bytes** |

**Savings**: ~12N bytes/step eliminated (25% of node-related traffic). For N=100K: ~5 MB/step saved.

---

## Performance Characteristics

### Why Fused CG is Fastest

1. **Fewer kernel launches**: 5 vs 12-13 per step. Even inside a CUDA Graph, each kernel node has ~1-2μs dispatch overhead.

2. **No intermediate VRAM round-trips**: `pressure`, `event_prob`, `event_mask`, and `rand_buffer` never leave registers in the fused kernel. The unfused pipeline writes each one to VRAM, waits for the next kernel to read it back.

3. **Triton-level sparsity**: The fused kernel skips `erfcx` for S/R nodes (~70% of N during peak epidemic) via `tl.where` — no warp divergence, no dynamic control flow. The unfused CUDAGraph path computes `erfcx` for ALL nodes because `sparse_hazard=False`.

4. **CUDA Graph batching**: 50 steps replayed with zero Python/driver overhead between them.

### Why Throughput Drops at Very Large N

The CSR traversal dominates at scale. For degree d=8 and N=10⁷:

```
CSR data per step = col_ind[E]·4 + weights[E]·4 + infectivity[N]·4
                  = 80M·4 + 80M·4 + 10M·4
                  = 680 MB
```

At A100 bandwidth of ~2 TB/s, the theoretical minimum traversal time is ~0.34 ms/step. The fused kernel helps by eliminating the *other* passes but cannot reduce the CSR traversal cost. This is fundamental to sparse irregular graph algorithms.

### BF16 Weights

The `bf16_weights=True` option downcasts `weights[E]` from fp32 to bfloat16, halving that array's memory traffic. The Triton kernels promote bf16 loads to fp32 for accumulation, maintaining numerical accuracy. Effect is marginal for uniform weights (all 1.0) but measurable for heterogeneous weight distributions on large graphs.

---

## Architectural Design Decisions

Three design choices are responsible for the majority of the fused kernel's performance advantage. Each resolves a tension that appears fundamental when working within PyTorch's eager execution model.

### Triton-Level Sparsity + CUDA Graphs

PyTorch eager mode forces a brutal choice for CUDA Graph capture:

- **Option A (sparse)**: Use boolean masks (`.any()`) to skip `erfcx` for S/R nodes. Saves ~70% of FLOPs during peak epidemic. But `.any()` forces a CPU synchronization point, breaking CUDA Graph capture.
- **Option B (dense)**: Compute `erfcx` for ALL N nodes to maintain static control flow. Wastes compute on inert nodes but satisfies graph capture.

The fused Triton kernel resolves this by moving the conditional logic into the SMs. The kernel launch grid is always `ceil(N / BLOCK_SIZE)` — completely static, satisfying CUDA Graph capture. But *inside* the kernel:

```python
# Block-level guard: scalar branch (not per-lane predication)
any_e = tl.sum(is_e.to(tl.int32), axis=0)
if any_e > 0:
    hazard_e = tl.where(is_e, _erfcx_approx(...), 0.0)
```

The `tl.sum()` produces a scalar that enables a true hardware branch at the block level. Entire blocks consisting of S or R nodes skip the 55-FLOP `erfcx` computation. This gives **zero-launch-overhead batched execution with dynamic compute sparsity** — the "holy grail" that is impossible with PyTorch's `.any()` guards.

### The Source-Node Compromise

The MATLAB non-Markovian simulator allocates O(E x M) per-edge ages, enabling arbitrary edge dynamics. For GPU parallelism, this is prohibitive at scale.

The source-node compromise recognizes that viral shedding is a biological property of the *host* (the source node), not the *contact* (the edge). All edges from an infected node j share j's infection age. By lifting the age-dependence into an O(N) pre-pass — `infectivity[j] = beta * h(age[j])` — and passing the continuous float into the CSR traversal, the kernel captures empirical shedding curves while keeping memory strictly O(N).

This entirely justifies trading away MATLAB's per-edge age tracking for the standard epidemiological case.

### In-Place PRNG Counter

Managing RNG states inside a CUDA Graph is notoriously painful: Python integers (like `total_steps * N`) get baked in as constants during graph capture, so every replay generates identical random numbers.

The solution is a device-side tensor `_step_counter` incremented via `_step_counter.add_(N)` — an in-place operation that survives graph capture. The fused kernel loads this counter via `tl.load(step_offset_ptr)` to feed unique offsets to `tl.rand()` on every replayed step. This is the textbook approach for Monte Carlo simulations under CUDA Graph constraints.

---

## Numerical Safeguards

### erfcx Overflow Guard (z << 0)

When a node first enters E or I (immediately after a renewal reset), its age is near zero. The standardized variable z = (ln(tau) - mu) / (sigma * sqrt(2)) diverges to -infinity, and the z < 0 branch computes `2 * exp(z^2) - erfcx(-z)`. At z = -9.4, exp(z^2) = exp(88.4) overflows fp32 (max ~3.4e38), producing Inf hazard rates and NaN probabilities.

**Guard**: Before the z < 0 branch, extend the asymptotic form to |z| > 9:
```python
result_pos = tl.where(az > 9.0, RSQRT_PI * inv_z, result_pos)
```
For |z| > 9, `erfcx(z) ~ 1/(z*sqrt(pi))` is numerically safe and gives erfcx -> infinity, making the hazard h(tau) = sqrt(2/pi) / (tau * sigma * erfcx(z)) -> 0. This is biologically correct: the probability of transitioning immediately after entering a state is zero for lognormal dwell times.

### Block-Level Hazard Skip (SIMT Nuance)

Triton's `tl.where(mask, true_op, false_op)` can behave like a ternary select instruction — evaluating **both** operands for the entire block vector before applying the mask. This means S and R nodes might secretly still execute the 55-FLOP `erfcx` math.

**Fix**: Wrap in a block-level `if` using `tl.sum()` to produce a scalar condition:
```python
any_e = tl.sum(is_e.to(tl.int32), axis=0)
if any_e > 0:
    hazard_e = tl.where(is_e, _erfcx_approx(...), 0.0)
```
Because `tl.sum` produces a scalar, this compiles to a uniform branch at the SM level (not per-lane predication). Epidemics cluster spatially, so during peak epidemic ~70% of blocks are pure S or R — the SMs branch over the entire math block.

Overhead: two `tl.sum` reductions per block (~10 cycles). Savings: 55 FLOPs x BLOCK_SIZE per pure S/R block.

---

## Future: Warp-Level Load Balancing

### The Degree Heterogeneity Problem

The memory traffic analysis shows the engine is now bounded by CSR traversal (8E bytes). But on scale-free networks (Barabasi-Albert), the `while neighbors_remain` loop creates severe warp divergence: a "superspreader" hub with 5,000 incoming edges forces the other 31 threads in its warp to sit idle for 4,998 iterations.

### The Solution: Warp-Collaborative Traversal

For future V2.0 targeting 100M+ nodes on scale-free networks, implement warp-level CSR load balancing where an entire warp (32 threads) collaborates to process a single high-degree node's neighbor list, accumulating pressure via parallel reduction (`tl.sum()`). This converts the worst-case serial traversal into a parallel one, bounded by `max_degree / 32` instead of `max_degree`.

This is an established technique in sparse graph computation (Merrill & Garland, SC 2016) but has not been applied to epidemic simulation kernels.

---

## When to Use Each Engine

| Scenario | Recommended Engine |
|:---------|:------------------|
| Quick prototyping, small N (<10K) | `RenewalEngine` |
| Production Markovian-edge simulations | `RenewalEngineCUDAGraph` |
| Validating NonMarkov against baseline | `RenewalEngineNonMarkov` with `transmission_mode="constant"` |
| Age-dependent transmission (viral shedding curves) | `RenewalEngineNonMarkov` with `transmission_mode="age_dependent"` |
| Maximum throughput, Markovian edges | `RenewalEngineFusedCUDAGraph` with `transmission_mode="constant"` |
| Maximum throughput, non-Markovian edges | `RenewalEngineFusedCUDAGraph` with `transmission_mode="age_dependent"` |
| RL training (millions of steps) | `RenewalEngineFusedCUDAGraph` |

### Quick Start

```python
from flashspread import SEIRModel, FixedDegreeGraph
from flashspread.engines import create_renewal_engine

graph = FixedDegreeGraph(100_000, 15, device="cuda")
model = SEIRModel(beta=0.3, mean_ei=5.0, median_ei=4.0, mean_ir=3.9, median_ir=1.5)

# Fastest (fused + CUDA Graph, Markovian-equivalent edges)
engine = create_renewal_engine(graph, model, nonmarkov_edges=True,
                               use_cuda_graph=True, transmission_mode="constant")

# With age-dependent transmission
engine = create_renewal_engine(graph, model, nonmarkov_edges=True,
                               use_cuda_graph=True, transmission_mode="age_dependent")

engine.seed_infection(100, state=model.exposed)
for _ in range(200):
    dt, state = engine.step()
```

---

## File Map

```
flashspread/
├── core/
│   ├── flash_neighbor.py          # FlashNeighbor (binary) + FlashNeighborInfectivity (float)
│   ├── flash_renewal_kernel.py    # Fused Triton kernel + _erfcx_approx
│   └── graph.py                   # GraphCSR + to_bf16_weights()
├── engines/
│   ├── renewal.py                 # RenewalEngine, CUDAGraph, NonMarkov, NonMarkovCUDAGraph
│   └── renewal_fused.py           # RenewalEngineFused, FusedCUDAGraph
├── models/
│   ├── compartmental.py           # SEIRModel: compute_rates, compute_infectivity, compute_rates_nonmarkov
│   └── hazards.py                 # lognormal_hazard_stable (erfcx), erfcx_rational_approx
└── engines/__init__.py            # create_renewal_engine() factory
```

---

## Relationship to the MATLAB Non-Markovian Simulator

The original MATLAB code (`nonmarkovianGEMF/Non-Markovian-Spreading-on-Networks/`) allocates an O(E×M) matrix `L{1}.T` to track **per-edge ages**, enabling arbitrary non-Markovian edge dynamics. FlashSpread deliberately trades this away for GPU parallelism.

The **source-node compromise** (`transmission_mode="age_dependent"`) recovers the key biological feature — age-dependent infectiousness — at O(N) cost rather than O(E). This is exact when the transmission kernel depends only on the source node's infection age, which is the standard epidemiological assumption (e.g., viral shedding profiles).

True per-edge ages would be needed only for:
- Temporal/dynamic networks where contacts form and break independently
- Dose-dependent transmission where cumulative exposure per edge matters
- Edge-specific non-Markovian dynamics (different hazard per edge)
