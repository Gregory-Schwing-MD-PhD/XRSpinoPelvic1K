#!/bin/bash
#SBATCH --job-name=xrsp_gen
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/xrsp_gen_%A_%a.out
#SBATCH --error=logs/xrsp_gen_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 1 -- DRR generation (SLURM array, CPU only)
#
# Renders N randomly-oblique lateral views per CT and projects the labels, the four
# endplate corners of EVERY vertebra visible in that scan, and the bicoxofemoral
# point through the SAME geometry -- so the 2-D ground truth stays exact at any
# angle. How many levels a case yields varies with its field of view; nothing is
# hardcoded.
#
#   sbatch slurm/xrsp_generate.sh
#   N_VIEWS=12 YAW=15 sbatch slurm/xrsp_generate.sh
#
# Idempotent: build_dataset skips a case whose outputs already exist, so a
# timed-out array task is simply resubmitted with the same command.
# =============================================================================
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"

CT_DIR="${CT_DIR:-/data/ct}"
LABEL_DIR="${LABEL_DIR:-/data/labels}"
OUT_DIR="${OUT_DIR:-/data/xrsp1k}"
N_VIEWS="${N_VIEWS:-8}"
YAW="${YAW:-12}"
PITCH="${PITCH:-8}"
ROLL="${ROLL:-6}"
SPACING="${SPACING:-1.0}"
N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

banner "generate  shard ${SHARD}/${N_SHARDS}  (${N_VIEWS} views/CT, yaw +/-${YAW} deg)"

xrun python -m xrsp.build_dataset \
    --ct_dir "${CT_DIR}" --label_dir "${LABEL_DIR}" --out "${OUT_DIR}" \
    --oblique "${N_VIEWS}" \
    --yaw_deg "${YAW}" --pitch_deg "${PITCH}" --roll_deg "${ROLL}" \
    --pixel_spacing_mm "${SPACING}" \
    --shard_id "${SHARD}" --n_shards "${N_SHARDS}"

echo "[shard ${SHARD}] done $(date)"
