# Roofline Benchmark Summary

## GPU Specifications

- **Peak FP32 Performance**: 19.5 TFLOPS
- **Memory Bandwidth**: 2039 GB/s
- **Ridge Point**: 9.6 FLOPs/byte

## Best Performers

### Best Renewal Engine: `renewal_compute_bound`
- Achieved: **610.39 GFLOPS**
- Efficiency: 3.1%
- Arithmetic Intensity: 19.33 FLOPs/byte
- Time per step: 3.673 ms

### Best Markovian Engine: `markov_conservative`
- Achieved: **0.05 GFLOPS**
- Efficiency: 0.0%
- Arithmetic Intensity: 0.46 FLOPs/byte
- Time per step: 764.650 ms

### Highest Arithmetic Intensity: `renewal_compute_bound`
- Arithmetic Intensity: **19.33 FLOPs/byte**
- Achieved: 610.39 GFLOPS

## Compute vs Memory Bound Analysis

| Configuration | AI (FLOPs/byte) | Bound | Achieved GFLOPS | Efficiency |
|---------------|-----------------|-------|-----------------|------------|
| markov_baseline | 0.46 | Memory | 0.03 | 0.0% |
| markov_aggressive | 0.46 | Memory | 0.01 | 0.0% |
| markov_conservative | 0.46 | Memory | 0.05 | 0.0% |
| renewal_baseline | 0.65 | Memory | 50.60 | 3.8% |
| renewal_batched_10 | 0.65 | Memory | 120.07 | 9.1% |
| renewal_batched_50 | 0.65 | Memory | 132.37 | 10.0% |
| renewal_batched_100 | 0.65 | Memory | 132.54 | 10.1% |
| renewal_small_tau | 0.65 | Memory | 51.01 | 3.9% |
| renewal_large_tau | 0.65 | Memory | 50.80 | 3.9% |
| renewal_dense | 1.31 | Memory | 163.20 | 6.1% |
| renewal_compute_heavy | 2.26 | Memory | 358.84 | 7.8% |
| renewal_ridge_8 | 7.95 | Memory | 538.59 | 3.3% |
| renewal_ridge_16 | 15.53 | Compute | 596.48 | 3.1% |
| renewal_compute_bound | 19.33 | Compute | 610.39 | 3.1% |

## Recommendations

- **2 configuration(s) achieved compute-bound behavior**
  - `renewal_ridge_16`: AI = 15.53
  - `renewal_compute_bound`: AI = 19.33
