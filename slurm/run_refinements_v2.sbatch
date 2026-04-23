#!/bin/bash
#SBATCH -A cdt_computing
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -J fs_ref2
#SBATCH -o logs/ref2-%j.out
#SBATCH -e logs/ref2-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs results
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "=============================================="
echo "Manuscript refinement pass 2"
echo "Job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "=============================================="

# --- Phase A: Fig 4 validation plot (regenerate with fixed title) ---
echo "=== [A] Fig 4: Scoglio-free validation plot ==="
python experiments/plot_scoglio_fused_vs_matlab.py \
    --output docs/jocs/figures/validation_scoglio_exact.png

# --- Phase B: fidelity_sweep with real MATLAB reference + bootstrap CIs ---
echo ""
echo "=== [B] Fidelity sweep (100 runs x 7 epsilons) with exact MATLAB reference ==="
python experiments/fidelity_sweep.py \
    --num-runs 100 \
    --output-csv results/fidelity_summary.csv \
    --output-npz results/fidelity_trajectories.npz

# --- Phase C: Exact non-Markovian Gillespie for Fig 8 overlay ---
echo ""
echo "=== [C] Exact Gillespie reference for Fig 8 (ER/BA/Fixed x N=1K,10K) ==="
python experiments/exact_gillespie_seir.py \
    --graphs er,ba,fixed \
    --sizes 1000,10000 \
    --trials 20 \
    --output results/fidelity_multi_exact.npz

# --- Phase D: rerun tau-leaping multi-topology sweep (if trajectories are stale) ---
# We keep the existing results/fidelity_multi.npz unless it's missing.
if [ ! -f "results/fidelity_multi.npz" ]; then
    echo ""
    echo "=== [D] Rerun multi-topology tau-leaping (fidelity_multi.npz missing) ==="
    python experiments/fidelity_multi_graph.py \
        --graphs er,ba,fixed \
        --sizes 1000,10000,100000 \
        --epsilons 0.005,0.03,0.1 \
        --trials 20
fi

# --- Phase E: regenerate all appendix figures ---
echo ""
echo "=== [E] Regenerate fidelity figures (6, 7, 9) ==="
python experiments/plot_fidelity.py \
    --summary results/fidelity_summary.csv \
    --trajectories results/fidelity_trajectories.npz \
    --outdir docs/jocs/figures

echo ""
echo "=== [F] Regenerate multi-topology figure (8) with exact overlay ==="
python experiments/plot_fidelity_multi.py \
    --npz results/fidelity_multi.npz \
    --exact results/fidelity_multi_exact.npz \
    --out docs/jocs/figures/fig_fidelity_multi.png

echo ""
echo "Done. Artifacts:"
ls -la docs/jocs/figures/validation_scoglio_exact.png \
       docs/jocs/figures/fig_fidelity_traj.png \
       docs/jocs/figures/fig_fidelity_err_vs_eps.png \
       docs/jocs/figures/fig_fidelity_pareto.png \
       docs/jocs/figures/fig_fidelity_multi.png 2>&1
