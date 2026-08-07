#!/bin/bash
#SBATCH --job-name=xrsp_unified
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:nvidia_h200:1
# msa1's H200 reports "[GPU requires reset]": nvidia-smi enumerates it, cuInit
# returns NO_DEVICE, and torch silently falls back to CPU (25x slower, no error).
# Needs a root-level reset. REMOVE THIS LINE once the node is fixed.
#SBATCH --exclude=msa1
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

# The DRRs supervise the HIP CHANNEL AND NOTHING ELSE (see xrsp/unified.py). If no
# rendered view carries a bicoxofemoral point, every DRR contributes exactly zero
# gradient and the run silently degrades to BUU-corners-only -- which can produce SS
# and LL, but NOT PI or PT, since both need the hip axis. The only symptom is `hip
# nanpx` in a per-epoch line, and 150 epochs is a long time to spend discovering it.
#
# build_dataset already records has_bicox per view. Nothing read it. Now something does.
_MAN_OK=$(xrun sh -c "awk -F, 'FNR>1 && \$5==1 {n++} END {print n+0}' ${DRR}/manifest*.csv 2>/dev/null" || echo 0)
if [[ "${_MAN_OK}" -eq 0 ]]; then
    echo "" >&2
    echo "ERROR: not one rendered view has a bicoxofemoral point (has_bicox=1 in 0 rows" >&2
    echo "       of ${DRR}/manifest*.csv)." >&2
    echo "" >&2
    echo "  Almost always: the labels are the v2 export, which has no femurs. Femoral" >&2
    echo "  segmentation arrives in v3 ('bone-augmented (femurs + thoracic + S1)')." >&2
    echo "  femoral_head_center then has nothing to fit and every view comes out" >&2
    echo "  has_bicox=0." >&2
    echo "" >&2
    echo "  Point data/labels + data/ct at the v3 export, delete data/xrsp1k, and" >&2
    echo "  re-run stage 1. BUU and the container are unaffected." >&2
    exit 1
fi
echo "[gate] ${_MAN_OK} rendered views carry a bicoxofemoral point"

banner "unified landmark model  (${EPOCHS} epochs)"
# Passed EXPLICITLY rather than left to the default, so the value that produced a run is
# in the job log next to its metrics. P_FLIP=0.5 sbatch ... runs the other arm.
xrun --nv python scripts/train_unified.py \
    --drr "${DRR}" --buu "${BUU}" --out "${RUN_DIR}" \
    --buu_splits "${BUU_SPLITS}" \
    --p_flip "${P_FLIP:-0.0}" --max_rot_deg "${MAX_ROT_DEG:-8.0}" \
    --epochs "${EPOCHS}" --batch "${BATCH}" ${RESUME}

echo "[unified] done $(date)"
