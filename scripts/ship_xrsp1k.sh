#!/usr/bin/env bash
# =============================================================================
# ship_xrsp1k.sh — generate XRSpinoPelvic1K from CTSpinoPelvic1K @v4, then upload to HF.
#   (1) gen   — CPU array: render DRR + dense mask + landmarks per case   [scripts/gen_xrsp1k.slurm]
#   (2) upload — merge manifests + push the dataset to <repo>@<rev>        [scripts/upload_xrsp1k.py]
#
# Prereqs on the grid: this repo cloned, the ctspinopelvic1k.sif container (the SAME image
# the CTSpinoPelvic1K slurms use — both gen and upload run inside it, so NO pip install),
# the v4 tree present locally (ct/ + labels/), and HF_TOKEN exported.
#
#   HF_TOKEN=hf_xxx CTSP1K_V4=/data/.../hf_export_v4 CTSP1K_ROOT=~/CTSpinoPelvic1K \
#     REPO_ID=anonymous-mlhc/XRSpinoPelvic1K bash scripts/ship_xrsp1k.sh
#
# Re-run after v4 is finalized with RESUME=0 (re-renders every case from the new labels).
# Knobs: N_SHARDS (16), OUT_DIR, REV (main), VIEWS ("lateral ap"), RESUME (1),
#        CTSP1K_ROOT / SIF_PATH (container location).
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

: "${HF_TOKEN:?export HF_TOKEN=hf_xxx (never hard-code it)}"
: "${CTSP1K_V4:?set CTSP1K_V4=/path/to/hf_export_v4 (the v4 tree: ct/ + labels/)}"
REPO_ID="${REPO_ID:-anonymous-mlhc/XRSpinoPelvic1K}"
REV="${REV:-main}"
N_SHARDS="${N_SHARDS:-16}"
OUT_DIR="${OUT_DIR:-${ROOT}/data/xrsp1k}"
RESUME="${RESUME:-1}"

# Same Singularity image as the CTSpinoPelvic1K slurms (containers/ctspinopelvic1k.sif).
CTSP1K_ROOT="${CTSP1K_ROOT:-${HOME}/CTSpinoPelvic1K}"
SIF_PATH="${SIF_PATH:-${CTSP1K_ROOT}/containers/ctspinopelvic1k.sif}"
[[ -f "${SIF_PATH}" ]] || { echo "ERROR: no container at ${SIF_PATH}"; \
  echo "       set SIF_PATH=/path/to/ctspinopelvic1k.sif (or CTSP1K_ROOT=/path/to/CTSpinoPelvic1K)"; exit 1; }
mkdir -p logs

echo "[ship_xrsp1k] container: ${SIF_PATH}"
echo "[ship_xrsp1k] (1) gen array %${N_SHARDS}  v4=${CTSP1K_V4} -> ${OUT_DIR}  resume=${RESUME}"
JG=$(sbatch --parsable \
  --array=0-$((N_SHARDS - 1))%${N_SHARDS} \
  --export=ALL,CTSP1K_V4=${CTSP1K_V4},OUT_DIR=${OUT_DIR},N_SHARDS=${N_SHARDS},RESUME=${RESUME},VIEWS=${VIEWS:-lateral ap},NO_RIBS=${NO_RIBS:-0},SIF_PATH=${SIF_PATH},CTSP1K_ROOT=${CTSP1K_ROOT} \
  scripts/gen_xrsp1k.slurm)
echo "GEN_ARRAY=${JG}"

# Upload also runs inside the container (huggingface_hub lives in the image). Bind the repo
# and the output tree; OUT_DIR is bound explicitly only when it lives outside the repo.
UP_BINDS="${ROOT}:${ROOT}"
case "${OUT_DIR}" in "${ROOT}"/*) : ;; *) UP_BINDS="${UP_BINDS},${OUT_DIR}:${OUT_DIR}" ;; esac
UP_CMD="stdbuf -oL -eL singularity exec --env PYTHONPATH=${ROOT},PYTHONUNBUFFERED=1,HF_TOKEN=${HF_TOKEN} --bind ${UP_BINDS} --pwd ${ROOT} ${SIF_PATH} python3 -u scripts/upload_xrsp1k.py --out_dir ${OUT_DIR} --repo_id ${REPO_ID} --revision ${REV}"

echo "[ship_xrsp1k] (2) upload -> ${REPO_ID}@${REV}  after all ${N_SHARDS} shards of ${JG}"
JU=$(sbatch --parsable -q primary --dependency=afterok:${JG} \
  --job-name=xrsp1k_upload --time=04:00:00 --mem=16G --cpus-per-task=4 \
  --output=logs/xrsp1k_upload_%j.out --error=logs/xrsp1k_upload_%j.err \
  --export=ALL,HF_TOKEN=${HF_TOKEN},OUT_DIR=${OUT_DIR},REPO_ID=${REPO_ID},REV=${REV} \
  --wrap "${UP_CMD}")
echo "UPLOAD_JOB=${JU}"
echo "[ship_xrsp1k] submitted:  gen=${JG}  upload=${JU}"
echo "[ship_xrsp1k]   monitor:  tail -f logs/xrsp1k_gen_${JG}* logs/xrsp1k_upload_${JU}*"
