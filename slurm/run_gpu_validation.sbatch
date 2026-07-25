#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -J fs_gpu_validation
#SBATCH -o logs/gpu_validation-%j.out
#SBATCH -e logs/gpu_validation-%j.err
#
# The whole gpu-marked suite, not just smoke_test.py.
#
# CI can only run the CPU suite, and the Triton interpreter cannot substitute
# for hardware: on Triton 3.1 its stores never reach host tensors, so
# interpreter-mode assertions are vacuous (tests/_triton_support.py probes for
# exactly that and skips instead of pretending). Every kernel-level guarantee
# therefore has to be established here.
#
# In particular this is the job that verifies the ensemble Bernoulli sampler
# actually fires. A widened Philox counter suppressed every transition while
# leaving throughput untouched, so the acceptance harness could not see it --
# tests/test_ensemble_gpu.py asserts per-replica event frequency against the
# sampled probability, which is the property that fails when it regresses.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PY="${FLASHSPREAD_PY:-python}"

echo "=== environment ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem \
    --format=csv,noheader
"${PY}" - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
try:
    import triton
    print("triton", triton.__version__)
except Exception as exc:
    raise SystemExit(f"the gpu extra is required for this job: {exc}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible; check --gres")
print("device", torch.cuda.get_device_name(0))
PYEOF
echo ""

# Report the machine-readable result on shared storage: /tmp is node-local and
# disappears with the allocation.
REPORT="results/gpu_validation-${SLURM_JOB_ID:-local}.json"

echo "=== gpu-marked test suite ==="
# No `|| echo`: a correctness gate must fail the job.
"${PY}" -m pytest -m gpu -v --strict-markers \
    --junit-xml="results/gpu_validation-${SLURM_JOB_ID:-local}.xml"

echo ""
echo "=== ensemble sampler sanity (independent of pytest) ==="
"${PY}" - "${REPORT}" <<'PYEOF'
import json
import sys

import torch

from flashspread.core.graph import GraphCSR
from flashspread.engines import create_ensemble_engine
from flashspread.models import SEIRModel

num_nodes, degree, replicas = 65_536, 8, 32
offsets = [s for step in range(1, degree // 2 + 1) for s in (step, -step)]
node = torch.arange(num_nodes, dtype=torch.int64).repeat_interleave(len(offsets))
shift = torch.tensor(offsets, dtype=torch.int64).repeat(num_nodes)
edge_index = torch.stack(((node + shift) % num_nodes, node)).cuda()

graph = GraphCSR(edge_index, num_nodes, incoming=True)
model = SEIRModel(beta=0.6)
engine = create_ensemble_engine(graph, model, replicas=replicas, seed=12345)
engine.seed_infection(num_nodes // 20, state=model.infected)

initial = engine.state.clone()
accepted, expected = 0, 0.0
for _ in range(40):
    before = engine.state.clone()
    tau, _ = engine.step()
    accepted += int((engine.state != before).sum())
    rates = engine.rates
    expected += float((-torch.expm1(-rates * tau.to(rates.dtype).unsqueeze(0))).sum())

changed = (engine.state != initial).sum(dim=0)
sigma = max(expected ** 0.5, 1.0)
report = {
    "num_nodes": num_nodes,
    "replicas": replicas,
    "storage_profile": engine.storage_profile,
    "accepted_transitions": accepted,
    "expected_transitions": expected,
    "sigma": sigma,
    "z_score": (accepted - expected) / sigma,
    "min_changed_per_replica": int(changed.min()),
    "max_changed_per_replica": int(changed.max()),
    "distinct_replica_states": len({tuple(engine.state[:, r].tolist()) for r in range(replicas)}),
}
with open(sys.argv[1], "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))

failures = []
if report["min_changed_per_replica"] == 0:
    failures.append("a replica accepted zero transitions (suppressed sampler?)")
if abs(report["z_score"]) > 6.0:
    failures.append(f"event rate is {report['z_score']:.1f} sigma from its target")
if report["distinct_replica_states"] != replicas:
    failures.append("replicas are not independent")
if failures:
    raise SystemExit("FAILED: " + "; ".join(failures))
print("ensemble sampler OK")
PYEOF

echo ""
echo "=== wrote ${REPORT} ==="
