#!/usr/bin/env bash
#SBATCH --job-name=xrsp1k_gen
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-15%16
#SBATCH --output=logs/xrsp1k_gen_%A_%a.out
#SBATCH --error=logs/xrsp1k_gen_%A_%a.err
# =============================================================================
# gen_xrsp1k.slurm — render XRSpinoPelvic1K from CTSpinoPelvic1K @v4 (CPU array).
#
# Reads the v4 tree (ct/ + labels/, VerSe-native ids 1-57) and writes, per case,
# paired DRR (.npy + .png), dense multi-class mask (.png), and per-level landmarks
# (.json). Pure-Python (numpy/nibabel/Pillow), no GPU.
#
# Runs inside the SAME Singularity image the CTSpinoPelvic1K slurms use
# (containers/ctspinopelvic1k.sif) — those deps live in the image, so there is NO
# `pip install` here; xrsp is made importable by binding this repo onto PYTHONPATH.
#
# REQUIRES: the ctspinopelvic1k.sif container, and the v4 tree present locally.
# Point the container path via CTSP1K_ROOT (your CTSpinoPelvic1K checkout) or SIF_PATH.
#
#   CTSP1K_V4=/path/to/hf_export_v4 CTSP1K_ROOT=~/CTSpinoPelvic1K sbatch scripts/gen_xrsp1k.slurm
#   SIF_PATH=/path/to/ctspinopelvic1k.sif CTSP1K_V4=... sbatch scripts/gen_xrsp1k.slurm
#   N_SHARDS=16 CTSP1K_V4=... OUT_DIR=... RESUME=0 sbatch scripts/gen_xrsp1k.slurm
# Keep the array width (0-15%16) in sync with N_SHARDS.
# =============================================================================
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

CTSP1K_V4="${CTSP1K_V4:?set CTSP1K_V4=/path/to/hf_export_v4 (the v4 tree with ct/ and labels/)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/xrsp1k}"
VIEWS="${VIEWS:-lateral ap}"
GAMMA="${GAMMA:-0.5}"
N_SHARDS="${N_SHARDS:-16}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
RESUME="${RESUME:-1}"           # RESUME=0 -> --no_resume (re-render ALL; use after v4 is final)

# Singularity container — the SAME image as the CTSpinoPelvic1K slurms. Default to the
# containers/ dir of a sibling CTSpinoPelvic1K checkout; override with SIF_PATH directly.
CTSP1K_ROOT="${CTSP1K_ROOT:-${HOME}/CTSpinoPelvic1K}"
SIF_PATH="${SIF_PATH:-${CTSP1K_ROOT}/containers/ctspinopelvic1k.sif}"

mkdir -p logs "${OUT_DIR}"
[[ -f "${SIF_PATH}" ]] || { echo "ERROR: no container at ${SIF_PATH}"; \
  echo "       set SIF_PATH=/path/to/ctspinopelvic1k.sif (or CTSP1K_ROOT=/path/to/CTSpinoPelvic1K)"; exit 1; }
[[ -d "${CTSP1K_V4}/ct" && -d "${CTSP1K_V4}/labels" ]] \
  || { echo "ERROR: ${CTSP1K_V4} must contain ct/ and labels/"; exit 1; }

export SINGULARITY_TMPDIR="/tmp/${USER}_xrsp1k_${SLURM_JOB_ID:-$$}"
mkdir -p "${SINGULARITY_TMPDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

# Identity binds: this repo (so `import xrsp` works via PYTHONPATH), the v4 input tree,
# and the output dir (only if it lives outside the repo, to avoid a redundant bind).
BINDS="${REPO_ROOT}:${REPO_ROOT},${CTSP1K_V4}:${CTSP1K_V4}"
case "${OUT_DIR}" in "${REPO_ROOT}"/*) : ;; *) BINDS="${BINDS},${OUT_DIR}:${OUT_DIR}" ;; esac
ENV_VARS="PYTHONPATH=${REPO_ROOT},PYTHONUNBUFFERED=1"

ARGS=( -m xrsp.build_dataset
       --ct_dir "${CTSP1K_V4}/ct" --label_dir "${CTSP1K_V4}/labels"
       --out "${OUT_DIR}" --views ${VIEWS} --gamma "${GAMMA}"
       --shard_id "${SHARD}" --n_shards "${N_SHARDS}" )
[[ "${RESUME}" == "0" ]] && ARGS+=( --no_resume )
[[ "${NO_RIBS:-0}" == "1" ]] && ARGS+=( --no_ribs )   # ship spine+sacrum+femurs now; ribs later

echo "[gen_xrsp1k] shard ${SHARD}/${N_SHARDS}  sif=${SIF_PATH}"
echo "[gen_xrsp1k]   v4=${CTSP1K_V4} -> ${OUT_DIR}  resume=${RESUME}  $(date)"

# preflight inside the container: fail FAST and loud if the render deps / xrsp aren't
# importable — otherwise build_dataset swallows the per-case ImportError and the shard
# "succeeds" with 0 files written.
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd "${REPO_ROOT}" "${SIF_PATH}" \
    python3 -c "import numpy, nibabel, PIL, xrsp; print('deps + xrsp OK')" || {
      echo "ERROR: numpy/nibabel/PIL or xrsp not importable inside ${SIF_PATH}."; exit 1; }

# shellcheck disable=SC2086
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd "${REPO_ROOT}" "${SIF_PATH}" \
    python3 -u "${ARGS[@]}"

echo "[gen_xrsp1k] shard ${SHARD} done  $(date)"
