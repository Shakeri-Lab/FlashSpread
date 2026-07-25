#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80gb
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J fs_acceptance
#SBATCH -a 0-9
#SBATCH -o logs/acceptance-%A_%a.out
#SBATCH -e logs/acceptance-%A_%a.err
#
# Re-measure the full production acceptance matrix against one commit.
#
# All ten measurements the project documents as evidence: the eight
# benchmark_acceptance presets (including the four that were previously never
# run as evidence -- every hub-heavy traversal and the compaction path), plus
# the captured Markovian SIS case and the R=32 shared-graph ensemble.
#
# Pinned for comparability with the published snapshot:
#   * -C a100_80gb, because the partition also contains 40 GB A100s and the
#     published table is an A100-SXM4-80GB result;
#   * one array task per case, so a single failure does not invalidate the rest;
#   * Triton >= 3.2 via FLASHSPREAD_TRITON, since the tiled ensemble kernel does
#     not compile below that (see CLAUDE.md's measured version table);
#   * output paths carry the git commit, so a JSON can never be mistaken for a
#     measurement of different code.
#
# Submit with:
#   sbatch --export=ALL,FLASHSPREAD_PY=<python>,FLASHSPREAD_TRITON=<triton_dir> \
#     slurm/run_acceptance_matrix.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results/acceptance

PY="${FLASHSPREAD_PY:-python}"
export PYTHONPATH="$(pwd)${FLASHSPREAD_TRITON:+:$FLASHSPREAD_TRITON}:${PYTHONPATH:-}"

COMMIT="$(git rev-parse --short HEAD)"

# A measurement must be attributable to a commit. Two distinct hazards:
# modifications to tracked files, and untracked Python that could shadow or add
# to the measured code. Untracked outputs and notes are irrelevant, so do not
# reject on those -- an over-broad check just trains people to bypass it.
MODIFIED="$(git diff HEAD --name-only)"
UNTRACKED_CODE="$(git ls-files --others --exclude-standard -- 'flashspread/**/*.py' 'flashspread/*.py' 'experiments/*.py')"
if [ -n "${MODIFIED}" ] || [ -n "${UNTRACKED_CODE}" ]; then
    echo "REFUSING: the worktree does not match ${COMMIT}, so a measurement taken"
    echo "here could not be reproduced from it. Commit or stash first."
    [ -n "${MODIFIED}" ] && { echo "  modified tracked files:"; echo "${MODIFIED}" | sed 's/^/    /'; }
    [ -n "${UNTRACKED_CODE}" ] && { echo "  untracked Python:"; echo "${UNTRACKED_CODE}" | sed 's/^/    /'; }
    exit 1
fi

CASES=(
  "acceptance regular-constant"
  "acceptance regular-age"
  "acceptance regular-mixed"
  "acceptance regular-late-compact"
  "acceptance ba-auto"
  "acceptance ba-thread"
  "acceptance ba-warp"
  "acceptance ba-merge"
  "markovian  sis"
  "ensemble   r32"
)
ENTRY="${CASES[${SLURM_ARRAY_TASK_ID}]}"
HARNESS="$(echo "${ENTRY}" | awk '{print $1}')"
CASE="$(echo "${ENTRY}" | awk '{print $2}')"
OUT="results/acceptance/${CASE}-${COMMIT}.json"

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${HARNESS} / ${CASE} @ ${COMMIT} ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,persistence_mode \
    --format=csv,noheader
"${PY}" - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
import triton
print("triton", triton.__version__)
name = torch.cuda.get_device_name(0)
print("device", name)
if "80GB" not in name:
    raise SystemExit(f"expected an 80 GB A100 for comparability, got {name!r}")
PYEOF

# Clock/DVFS provenance: a 10% drift is otherwise indistinguishable from a 10%
# code change. Sampled before and after the timed work.
sample_clocks () {
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active \
        --format=csv,noheader | sed "s/^/  clocks[$1] /"
}
sample_clocks before

case "${HARNESS}" in
  acceptance)
    "${PY}" experiments/benchmark_acceptance.py walltime --case "${CASE}" --output "${OUT}"
    ;;
  markovian)
    "${PY}" experiments/benchmark_markovian.py walltime --output "${OUT}"
    ;;
  ensemble)
    "${PY}" experiments/benchmark_ensemble.py walltime --replicas 32 --output "${OUT}"
    ;;
esac

sample_clocks after
echo "=== wrote ${OUT} ==="
