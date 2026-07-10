#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -J fs_fidval_gpu
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# Fidelity re-validation -- FlashSpread side (GPU).
#
# Runs the tau-leaping simulator on the *corrected* (Phase 0/1) code and writes
# its trajectories/metrics under results/fidelity_val/. Compared against the
# exact-Gillespie references (run_fidelity_validation_exact.sbatch) by the merge
# job (merge_fidelity_validation.sbatch).

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}"
mkdir -p logs results/fidelity_val
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PY="${FLASHSPREAD_PY:-/home/hs9hd/.conda/envs/hpo_edu/bin/python}"
VD="results/fidelity_val"
MATLAB_REF="/sfs/gpfs/tardis/project/shakeri-lab/graph_alg/[old] nonmarkovianGEMF/outputs/scoglio_matlab_N1000_T1000_q10_90.mat"

echo "=== environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PY}" -c "import torch, flashspread; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0))"

# 1. SEIR epsilon-sweep structural-bias floor (ER N=1000, vs the MATLAB exact
#    reference used in the paper). This is the headline fidelity metric.
echo "=== [1/4] SEIR epsilon-sweep (fidelity_sweep) ==="
"${PY}" experiments/fidelity_sweep.py \
    --device cuda \
    --num-runs 100 \
    --matlab "${MATLAB_REF}" \
    --output-csv "${VD}/fidelity_summary.csv" \
    --output-npz "${VD}/fidelity_trajectories.npz"

# 2. Multi-graph fidelity (er/ba/fixed x {1000,10000}); the 'fixed' cells use
#    the corrected symmetric FixedDegreeGraph (C1).
echo "=== [2/4] Multi-graph fidelity (fidelity_multi_graph) ==="
"${PY}" experiments/fidelity_multi_graph.py \
    --graphs er,ba,fixed \
    --sizes 1000,10000 \
    --epsilons 0.005,0.03,0.1 \
    --trials 20 \
    --device cuda \
    --output-npz "${VD}/fidelity_multi.npz" \
    --output-csv "${VD}/fidelity_multi_summary.csv"

# 3-4. SIS / SIR Markovian validation (MarkovianEngine).
echo "=== [3/4] FlashSpread SIS ==="
"${PY}" experiments/validate_sis_sir_flashspread.py \
    --model sis --num-nodes 1000 --degree 8 --trials 100 \
    --beta 0.25 --recovery-rate 0.15 --initial-frac 0.01 \
    --device cuda --output "${VD}/flashspread_sis.npz"

echo "=== [4/4] FlashSpread SIR ==="
"${PY}" experiments/validate_sis_sir_flashspread.py \
    --model sir --num-nodes 1000 --degree 8 --trials 100 \
    --beta 0.25 --recovery-rate 0.15 --initial-frac 0.01 \
    --device cuda --output "${VD}/flashspread_sir.npz"

echo "=== FlashSpread side complete ==="
