#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -J fs_sissir_fs
#SBATCH -o logs/sissir_fs-%j.out
#SBATCH -e logs/sissir_fs-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "FlashSpread SIS / SIR only (exact refs already cached)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=============================================="

BETA=0.25
REC=0.15
N=1000
TRIALS=100

# Only FlashSpread phases (exact refs already in results/).
echo "=== FlashSpread SIS (MarkovianEngine, GPU) ==="
python experiments/validate_sis_sir_flashspread.py --model sis \
    --num-nodes $N --trials $TRIALS \
    --beta $BETA --recovery-rate $REC \
    --output results/flashspread_sis.npz

echo ""
echo "=== FlashSpread SIR (MarkovianEngine, GPU) ==="
python experiments/validate_sis_sir_flashspread.py --model sir \
    --num-nodes $N --trials $TRIALS \
    --beta $BETA --recovery-rate $REC \
    --output results/flashspread_sir.npz

echo ""
echo "=== Figure ==="
python experiments/plot_sis_sir.py \
    --sis-exact results/exact_sis.npz \
    --sis-flash results/flashspread_sis.npz \
    --sir-exact results/exact_sir.npz \
    --sir-flash results/flashspread_sir.npz \
    --out docs/jocs/figures/fig_sis_sir_validation.png

ls -la docs/jocs/figures/fig_sis_sir_validation.png
