#!/bin/bash
#SBATCH --job-name=xrsp_buu
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/xrsp_buu_%j.out
#SBATCH --error=logs/xrsp_buu_%j.err
# =============================================================================
# Stage 4b -- BUU Spine evaluation (EXTERNAL validity).
#
#   BUU_IMAGES=/data/buu/images BUU_ANN=/data/buu/corners.json #       sbatch slurm/xrsp_eval_buu.sh
#
# BUU is EVALUATION ONLY and is not redistributed with this repo.
#
# What this measures is agreement with HUMAN READERS, not accuracy against truth.
# BUU corners are modal -- what a reader could see -- and for S1 they carry the very
# ala/body superimposition the model is trained to see through. A disagreement is
# only interpretable next to the DRR reader study, which measures the SIGNED human
# bias against amodal truth. See docs/PIPELINE.md section 4.
# =============================================================================
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"

BUU_IMAGES="${BUU_IMAGES:-/data/buu/images}"
BUU_ANN="${BUU_ANN:-/data/buu/corners.json}"
CKPT="${CKPT:-/data/runs/fold_0/best.ckpt}"
OUT="${OUT:-/data/results/buu}"

banner "evaluate on BUU (real standing laterals)"
xrun --nv python scripts/evaluate_buu.py     --images "${BUU_IMAGES}" --annotations "${BUU_ANN}"     --ckpt "${CKPT}" --out "${OUT}"
