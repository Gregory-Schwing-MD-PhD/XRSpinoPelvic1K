#!/bin/bash
#SBATCH --job-name=xrsp_eval
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/xrsp_eval_%j.out
#SBATCH --error=logs/xrsp_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 4 -- held-out evaluation + publication figures.
#
#   sbatch slurm/xrsp_eval_unified.sh
#   CKPT=/data/runs/unified/last.pt sbatch slurm/xrsp_eval_unified.sh
#
# CPU ON PURPOSE. This is a single forward pass over ~1300 images with a small
# UNet -- a couple of minutes either way -- and the GPU partition is where the
# queue is. Waiting hours for a V100 to save ninety seconds of compute is a bad
# trade, and the code already falls back to CPU on its own.
#
# best.pt, not last.pt: last.pt is the resume point written every epoch and is
# simply whatever the final epoch produced, which is not the model you would
# ship. Override with CKPT to evaluate a specific checkpoint.
#
# Outputs, under results/unified/:
#   summary.json         every number, per source, machine-readable
#   per_item_{drr,buu}.csv
#   fig_*.pdf / .png     CED, per-level error, Bland-Altman, agreement scatter,
#                        both 4x4 confusion matrices, PI=SS+PT residual
# =============================================================================
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"

CKPT="${CKPT:-/data/runs/unified/best.pt}"
DRR="${DRR:-/data/xrsp1k}"
BUU="${BUU:-/data/BUU-LSPINE}"
OUT="${OUT:-/data/results/unified}"

# Checked on the HOST, where the path is visible without starting a container, so a
# missing checkpoint costs a second rather than a container start.
_ckpt_host="${DATA_ROOT}${CKPT#/data}"
if [[ ! -f "${_ckpt_host}" ]]; then
    echo "ERROR: no checkpoint at ${_ckpt_host}" >&2
    echo "       stage 3 (xrsp_unified.sh) has to finish first." >&2
    exit 1
fi
# run_config.json carries the exact held-out case lists. evaluate_unified.py refuses to
# re-derive them, so its absence is fatal there too -- say so here, where the fix is
# obvious, rather than after the container has started.
if [[ ! -f "$(dirname "${_ckpt_host}")/run_config.json" ]]; then
    echo "ERROR: run_config.json missing beside the checkpoint." >&2
    echo "       It holds the held-out case lists; re-deriving the split would" >&2
    echo "       quietly turn a test score into a training score." >&2
    exit 1
fi

banner "evaluate  ${CKPT}  ->  ${OUT}"

xrun python scripts/evaluate_unified.py \
    --model "${CKPT}" --drr "${DRR}" --buu "${BUU}" --out "${OUT}"

echo "[eval] figures + summary.json in ${OUT}"
echo "[eval] done $(date)"
