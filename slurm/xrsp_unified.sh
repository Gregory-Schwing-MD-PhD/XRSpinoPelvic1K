#!/bin/bash
#SBATCH --job-name=xrsp_unified
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/xrsp_unified_%A.out
#SBATCH --error=logs/xrsp_unified_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Unified spinopelvic landmark model.
#
#   HIP POINT from DRRs  -- 3-D sphere fit to the femoral head, projected
#   CORNERS   from BUU   -- radiologist annotations, deployment domain
#
# The two streams supervise DISJOINT channels, so the DRR corner convention never
# has to agree with BUU's (the DRR corners are simply unused). That is deliberate:
# an endplate corner is a judgement call, a femoral head is a sphere with a fit
# residual you can check.
#
#   sbatch slurm/xrsp_unified.sh
#
# --requeue plus last.pt written EVERY epoch means preemption costs at most one
# epoch: the resubmitted job picks up where it stopped, same command.
# =============================================================================
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"

DRR="${DRR:-/data/xrsp1k}"
BUU="${BUU:-/data/BUU-LSPINE}"
BUU_SPLITS="${BUU_SPLITS:-/data/buu_splits.json}"
RUN_DIR="${RUN_DIR:-/data/runs/unified}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-8}"

RESUME=""
if [[ -f "${DATA_ROOT}/runs/unified/last.pt" ]]; then
    RESUME="--resume ${RUN_DIR}/last.pt"
    echo "resuming from ${RUN_DIR}/last.pt"
fi

# The split must EXIST before training: deriving it inside the trainer ties the
# assignment to the file list, so a re-extract could move a test case into train.
if [[ ! -f "${DATA_ROOT}/buu_splits.json" ]]; then
    echo "ERROR: ${DATA_ROOT}/buu_splits.json missing -- run 'make buu-splits' first" >&2
    exit 1
fi

banner "unified landmark model  (${EPOCHS} epochs)"
xrun --nv python scripts/train_unified.py \
    --drr "${DRR}" --buu "${BUU}" --out "${RUN_DIR}" \
    --buu_splits "${BUU_SPLITS}" \n    --epochs "${EPOCHS}" --batch "${BATCH}" ${RESUME}

echo "[unified] done $(date)"
