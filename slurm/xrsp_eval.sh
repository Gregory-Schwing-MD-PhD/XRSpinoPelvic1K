#!/bin/bash
#SBATCH --job-name=xrsp_eval
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/xrsp_eval_f%a_%A.out
#SBATCH --error=logs/xrsp_eval_f%a_%A.err
# =============================================================================
# Stage 4a -- held-out DRR evaluation (INTERNAL validity).
#
#   sbatch slurm/xrsp_eval.sh
#
# Ground truth here is amodal and exact, so this measures whether the network
# learned the target. It does NOT measure the domain: same renderer, same physics,
# same distribution. Report it as in-silico and pair it with Stage 4b (BUU) and the
# DRR reader study. See docs/PIPELINE.md section 4.
# =============================================================================
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"

FOLD="${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}"
DATA="${DATA:-/data/xrsp1k}"
SPLITS="${SPLITS:-/data/splits.json}"
RUN_DIR="${RUN_DIR:-/data/runs/fold_${FOLD}}"
OUT="${OUT:-/data/results/fold_${FOLD}}"

banner "evaluate fold ${FOLD} (held-out DRRs)"
xrun --nv python scripts/evaluate.py     --data "${DATA}" --splits "${SPLITS}" --fold "${FOLD}"     --ckpt "${RUN_DIR}/best.ckpt" --out "${OUT}"
