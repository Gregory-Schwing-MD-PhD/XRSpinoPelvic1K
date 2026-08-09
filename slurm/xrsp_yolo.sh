#!/bin/bash
#SBATCH --job-name=xrsp_yolo
#SBATCH -q gpu
# ANY GPU, not specifically an H200. YOLOv11n-Pose at batch 8 / 640 px needs about 4 GB
# -- Bansal et al. trained it on an RTX 4060 LAPTOP GPU with 8 GB (their Table 3), so a
# V100 is already more capable than the paper's hardware and an H200 is pure queue tax.
# All four H200s are currently unusable anyway: msa1 and msa4 report GPUs that enumerate
# but fail cuInit, msa2 and msa3 are drained. The CUDA guard below is what makes this
# safe -- it refuses to run on a card torch cannot see, whichever node we land on.
#SBATCH --gres=gpu:1
#SBATCH --exclude=msa1,msa4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/xrsp_yolo_%j.out
#SBATCH --error=logs/xrsp_yolo_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# YOLO-Pose baseline against Bansal et al. 2026 (PLoS One e0347290).
#
#   sbatch slurm/xrsp_yolo.sh                       # yolo11n-pose, their best balance
#   MODEL=yolov8l-pose.pt sbatch slurm/xrsp_yolo.sh # their highest-accuracy variant
#
# WHY THIS IS A FAIR COMPARISON, AND WHERE IT IS NOT
# -------------------------------------------------
# Same films, same patient-grouped/sex-age-stratified split file, same test set as the
# heatmap model (1398/301/301) -- so the two models are scored on one draw, not two.
# Every hyperparameter is Bansal Table 4 and every Ultralytics augmentation is explicitly
# zeroed, matching their Experimental Standardisation section.
#
# It differs from their paper in exactly ONE way, deliberately: their Mendeley release is
# pre-resized to 640x640 NON-UNIFORMLY, which destroys aspect ratio. Here the original
# films are used and Ultralytics letterboxes them. On this dataset the source aspect
# ratio has a median of 0.799, so squashing turns a true 45 deg endplate into 58 deg, by
# a different amount on every film. Invisible in a pixel metric; fatal for an angle.
#
# ultralytics lives in ./pylibs, installed with --target under constraints pinning
# torch/torchvision/numpy. Unconstrained it resolves to torch 2.13 + numpy 2.4, which
# breaks monai and torchaudio -- i.e. it would disable the heatmap model in the same
# environment meant to compare against it.
# =============================================================================
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"

MODEL="${MODEL:-yolo11n-pose.pt}"
DATA="${DATA:-/data/buu_yolo/buu.yaml}"
OUT="${OUT:-/data/runs/yolo}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
TAG="${TAG:-}"

_data_host="${DATA_ROOT}${DATA#/data}"
if [[ ! -f "${_data_host}" ]]; then
    echo "ERROR: ${_data_host} missing -- run scripts/buu_to_yolo.py first." >&2
    exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/pylibs/ultralytics" ]]; then
    echo "ERROR: ${PROJECT_ROOT}/pylibs/ultralytics missing." >&2
    echo "       Install it with the CONSTRAINED command in the header, not a bare" >&2
    echo "       pip install -- an unconstrained resolve upgrades torch and numpy and" >&2
    echo "       silently breaks monai." >&2
    exit 1
fi

# PYTHONPATH must lead with pylibs so ultralytics is importable; /workspace stays on it
# for xrsp. _common.sh sets PYTHONPATH=/workspace, so this extends rather than replaces.
yrun() {
    singularity exec --nv \
        --env "PYTHONPATH=/workspace/pylibs:/workspace,OMP_NUM_THREADS=${OMP_NUM_THREADS},MPLBACKEND=Agg,YOLO_CONFIG_DIR=/tmp/ultralytics,WANDB_MODE=disabled,MLFLOW_ALLOW_FILE_STORE=true,MLFLOW_TRACKING_URI=" \
        --bind "${PROJECT_ROOT}:/workspace,${DATA_ROOT}:/data,${HOST_CONTAINER_TMP}:/tmp" \
        --pwd /workspace "${CONTAINER}" "$@"
}

banner "YOLO-Pose  ${MODEL}  (Bansal et al. Table 4, ${EPOCHS} epochs)"

# Fail fast if CUDA is not actually visible -- the same silent CPU fallback that cost a
# day on the heatmap model, and Ultralytics will happily train on CPU without comment.
yrun python3 -c "
import torch, sys
if not torch.cuda.is_available():
    sys.exit('SLURM allocated a GPU but torch cannot see it -- refusing to train on CPU. '
             'Check nvidia-smi --query-gpu=mig.mode.current --format=csv for '
             '\"[GPU requires reset]\" and resubmit with --exclude=<node>.')
print('device:', torch.cuda.get_device_name(0))"

yrun python3 scripts/train_yolo_pose.py \
    --data "${DATA}" --model "${MODEL}" --out "${OUT}" --epochs "${EPOCHS}"     --imgsz "${IMGSZ}" ${TAG:+--tag "${TAG}"}

echo "[yolo] done $(date)"
