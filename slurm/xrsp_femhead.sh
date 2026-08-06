#!/bin/bash
#SBATCH --job-name=xrsp_femhead
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/xrsp_femhead_%A.out
#SBATCH --error=logs/xrsp_femhead_%A.err
# =============================================================================
# OPTIONAL: femoral-head SEGMENTER on DRRs, as an alternative hip-point route.
#
# The unified model regresses the hip point as a heatmap channel. This trains a
# segmenter instead and takes the mask centroid, which is exact -- for equal
# projected radii the union of the two head discs is symmetric about the
# bicoxofemoral point at any overlap -- and gives a dense gradient plus a mask you
# can eyeball. Run it to compare the two representations on the same data.
#
#   sbatch slurm/xrsp_femhead.sh
# =============================================================================
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"

DATA="${DATA:-/data/xrsp1k}"
RUN_DIR="${RUN_DIR:-/data/runs/femhead}"
EPOCHS="${EPOCHS:-60}"

banner "femoral-head segmenter  (${EPOCHS} epochs)"
xrun --nv python scripts/train_femhead.py \
    --data "${DATA}" --out "${RUN_DIR}" --epochs "${EPOCHS}"

echo "[femhead] done $(date)"
