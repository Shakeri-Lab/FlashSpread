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

## SLURM Execution

For HPC environments with SLURM:

```bash
# Markovian simulation
sbatch slurm/run_markovian.sbatch

# Non-Markovian simulation
sbatch slurm/run_renewal.sbatch
```

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
