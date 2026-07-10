#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p standard
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -J fs_fidval_merge
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Fidelity re-validation -- merge + assert (CPU).
#
# Submit with dependencies on the exact array and the FlashSpread GPU job:
#   sbatch --dependency=afterok:<exact_arrayid>:<gpu_jobid> merge_fidelity_validation.sbatch
#
# Steps: (1) combine the per-cell exact SEIR shards into one npz; (2) best-effort
# figure regeneration; (3) run the PASS/FAIL report -- its non-zero exit on a
# fidelity regression makes this job (and thus the pipeline) fail.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"
mkdir -p logs results/fidelity_val results/fidelity_val/figures
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PY="${FLASHSPREAD_PY:-/home/hs9hd/.conda/envs/hpo_edu/bin/python}"
VD="results/fidelity_val"

# 1. Combine the sharded exact SEIR references into one npz keyed
#    mean_{graph}_N{size} (the schema plot_fidelity_multi.py --exact expects).
echo "=== combining exact SEIR shards ==="
"${PY}" - "${VD}" <<'PYEOF'
import glob, os, sys
import numpy as np
vd = sys.argv[1]
shards = sorted(glob.glob(os.path.join(vd, "exact_seir_*.npz")))
payload, sample_times = {}, None
for s in shards:
    d = np.load(s)
    if sample_times is None and "sample_times" in d:
        sample_times = d["sample_times"]
    for k in d.files:
        if k.startswith(("mean_", "q25_", "q75_")):
            payload[k] = d[k]
if sample_times is not None:
    payload["sample_times"] = sample_times
out = os.path.join(vd, "fidelity_multi_exact.npz")
np.savez_compressed(out, **payload)
print(f"combined {len(shards)} shards -> {out} ({len(payload)} arrays)")
PYEOF

# 2. Best-effort figure regeneration (matplotlib is optional; a failure here
#    must NOT mask the fidelity verdict below).
echo "=== regenerating figures (best-effort) ==="
"${PY}" experiments/plot_fidelity.py \
    --summary "${VD}/fidelity_summary.csv" \
    --trajectories "${VD}/fidelity_trajectories.npz" \
    --outdir "${VD}/figures" || echo "  (plot_fidelity skipped)"
"${PY}" experiments/plot_fidelity_multi.py \
    --npz "${VD}/fidelity_multi.npz" \
    --exact "${VD}/fidelity_multi_exact.npz" \
    --out "${VD}/figures/fig_fidelity_multi.png" || echo "  (plot_fidelity_multi skipped)"
"${PY}" experiments/plot_sis_sir.py \
    --sis-exact "${VD}/exact_sis.npz" --sis-flash "${VD}/flashspread_sis.npz" \
    --sir-exact "${VD}/exact_sir.npz" --sir-flash "${VD}/flashspread_sir.npz" \
    --out "${VD}/figures/fig_sis_sir.png" || echo "  (plot_sis_sir skipped)"

# 3. Aggregate + PASS/FAIL gate (authoritative; sets the job exit code).
echo "=== fidelity PASS/FAIL report ==="
"${PY}" experiments/validate_fidelity_report.py \
    --val-dir "${VD}" \
    --graphs er,ba,fixed \
    --sizes 1000,10000 \
    --eps 0.03 \
    --out-md "${VD}/fidelity_validation_summary.md"

echo "=== fidelity re-validation complete: ${VD}/fidelity_validation_summary.md ==="
