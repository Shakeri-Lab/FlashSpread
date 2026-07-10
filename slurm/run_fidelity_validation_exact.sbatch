#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J fs_fidval_exact
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-7
#
# Fidelity re-validation -- EXACT Gillespie references (CPU, parallel array).
#
# Each array task computes one exact-Gillespie reference cell and writes a
# distinct npz under results/fidelity_val/. These are the CPU-heavy part; the
# FlashSpread side runs separately on a GPU (run_fidelity_validation_flashspread.sbatch)
# and the merge job (merge_fidelity_validation.sbatch) aggregates + asserts.
#
# Tasks 0-5: exact non-Markovian SEIR (Gibson-Bruck MNRM) for graph x size.
# Tasks 6-7: exact Markovian SIS / SIR (Doob-Gillespie direct method).

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"
mkdir -p logs results/fidelity_val
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Pin the interpreter (torch 2.5.1+cu121, triton 3.1 -- matches the paper env);
# the exact references are pure NumPy/CPU so no GPU is needed here.
PY="${FLASHSPREAD_PY:-/home/hs9hd/.conda/envs/hpo_edu/bin/python}"
VD="results/fidelity_val"

# Trial counts match the JOCS protocol.
SEIR_TRIALS=20
MARKOV_TRIALS=100
BETA=0.25            # 2 / d, d=8
RECOVERY=0.15        # delta (SIS) / gamma (SIR)

# One (kind|graph|N) spec per array index.
CONFIGS=(
    "seir|er|1000"
    "seir|ba|1000"
    "seir|fixed|1000"
    "seir|er|10000"
    "seir|ba|10000"
    "seir|fixed|10000"
    "sis|er|1000"
    "sir|er|1000"
)

SPEC="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r KIND GRAPH N <<< "${SPEC}"
echo "[task ${SLURM_ARRAY_TASK_ID}] kind=${KIND} graph=${GRAPH} N=${N}"

# Fail fast if the env is wrong.
"${PY}" -c "import numpy, flashspread; print('env OK: numpy', numpy.__version__)"

if [[ "${KIND}" == "seir" ]]; then
    "${PY}" experiments/exact_gillespie_seir.py \
        --graphs "${GRAPH}" \
        --sizes "${N}" \
        --trials "${SEIR_TRIALS}" \
        --tf 50.0 \
        --device cpu \
        --output "${VD}/exact_seir_${GRAPH}_N${N}.npz"
else
    "${PY}" experiments/exact_gillespie_sis_sir.py \
        --model "${KIND}" \
        --graph "${GRAPH}" \
        --num-nodes "${N}" \
        --trials "${MARKOV_TRIALS}" \
        --tf 50.0 \
        --beta "${BETA}" \
        --recovery-rate "${RECOVERY}" \
        --initial-frac 0.01 \
        --output "${VD}/exact_${KIND}.npz"
fi

echo "[task ${SLURM_ARRAY_TASK_ID}] done"
