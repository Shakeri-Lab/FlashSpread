#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:30:00
#SBATCH -J fs_sissir
#SBATCH -o logs/sissir-%j.out
#SBATCH -e logs/sissir-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "SIS / SIR validation (Markovian Gillespie reference)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=============================================="

# Shared parameters (ER graph, d=8, N=1000, beta=0.25).
# Recovery rate chosen to put R0 > 1 so both models produce a visible
# epidemic on a 1000-node graph:
#   R0_SIS = beta*d/delta   and   R0_SIR = beta*d/gamma .
# With beta=0.25 and d=8, setting recovery=0.15 gives R0 ~ 13 -- easy
# takeoff without stochastic fadeout at 1% seed.
BETA=0.25
REC=0.15
N=1000
TRIALS=100

echo "=== [A] Exact Gillespie SIS (CPU) ==="
python experiments/exact_gillespie_sis_sir.py --model sis \
    --graph er --num-nodes $N --trials $TRIALS \
    --beta $BETA --recovery-rate $REC \
    --output results/exact_sis.npz

echo ""
echo "=== [B] Exact Gillespie SIR (CPU) ==="
python experiments/exact_gillespie_sis_sir.py --model sir \
    --graph er --num-nodes $N --trials $TRIALS \
    --beta $BETA --recovery-rate $REC \
    --output results/exact_sir.npz

echo ""
echo "=== [C] FlashSpread SIS (MarkovianEngine, GPU) ==="
python experiments/validate_sis_sir_flashspread.py --model sis \
    --num-nodes $N --trials $TRIALS \
    --beta $BETA --recovery-rate $REC \
    --output results/flashspread_sis.npz

echo ""
echo "=== [D] FlashSpread SIR (MarkovianEngine, GPU) ==="
python experiments/validate_sis_sir_flashspread.py --model sir \
    --num-nodes $N --trials $TRIALS \
    --beta $BETA --recovery-rate $REC \
    --output results/flashspread_sir.npz

echo ""
echo "=== [E] Figure ==="
python experiments/plot_sis_sir.py \
    --sis-exact results/exact_sis.npz \
    --sis-flash results/flashspread_sis.npz \
    --sir-exact results/exact_sir.npz \
    --sir-flash results/flashspread_sir.npz \
    --out docs/jocs/figures/fig_sis_sir_validation.png

echo "Done."
ls -la docs/jocs/figures/fig_sis_sir_validation.png
