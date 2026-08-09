#!/bin/bash
#SBATCH --job-name=xrsp_train
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:nvidia_h200:1
# msa1 AND msa4 have unusable H200s. msa1 reports "[GPU requires reset]": nvidia-smi enumerates it, cuInit
# returns NO_DEVICE, and torch silently falls back to CPU (25x slower, no error).
# Needs a root-level reset. REMOVE THIS LINE once the node is fixed.
#SBATCH --exclude=msa1,msa4
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/xrsp_train_f%a_%A.out
#SBATCH --error=logs/xrsp_train_f%a_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 3 -- train the landmark model. One array task per fold.
#
#   sbatch slurm/xrsp_train_array.sh              # all 5 folds
#   sbatch --array=0 slurm/xrsp_train_array.sh    # just fold 0
#
# Idempotent and resumable: last.ckpt in the run dir is picked up automatically,
# so a job that hits the wall clock is resubmitted with the same command.
#
# The channel set is derived from the DATA (every vertebral level annotated
# anywhere in the dataset), not hardcoded -- scans differ in how many vertebrae
# they show, and per-view masking handles the variation.
# =============================================================================
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"

FOLD="${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}"
DATA="${DATA:-/data/xrsp1k}"
SPLITS="${SPLITS:-/data/splits.json}"
RUN_DIR="${RUN_DIR:-/data/runs/fold_${FOLD}}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-8}"

banner "train fold ${FOLD}  (${EPOCHS} epochs)"
if [[ -f "${PROJECT_ROOT}/${RUN_DIR#/data/}/last.ckpt" || -f "${RUN_DIR}/last.ckpt" ]]; then
    echo "resuming from last.ckpt"
fi

xrun --nv python scripts/train_landmarks.py \
    --data "${DATA}" --splits "${SPLITS}" --fold "${FOLD}" \
    --out "${RUN_DIR}" --epochs "${EPOCHS}" --batch_size "${BATCH}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"

echo "[fold ${FOLD}] done $(date)"
