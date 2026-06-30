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
# (.json). Pure-Python (numpy/nibabel/Pillow) — no GPU, no container.
#
# REQUIRES: the xrsp package importable (pip install -e . in this repo, or
# PYTHONPATH=<repo>), and the v4 tree present locally.
#
#   CTSP1K_V4=/path/to/hf_export_v4 sbatch scripts/gen_xrsp1k.slurm
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
PY="${PY:-python}"
RESUME="${RESUME:-1}"           # RESUME=0 -> --no_resume (re-render ALL; use after v4 is final)

mkdir -p logs "${OUT_DIR}"
[[ -d "${CTSP1K_V4}/ct" && -d "${CTSP1K_V4}/labels" ]] \
  || { echo "ERROR: ${CTSP1K_V4} must contain ct/ and labels/"; exit 1; }

# preflight: fail FAST and loud if the render deps aren't in THIS python — otherwise
# build_dataset catches the per-case ImportError and the shard "succeeds" with 0 written.
"${PY}" -c "import numpy, nibabel, PIL" 2>/dev/null || {
  echo "ERROR: render deps missing in $(${PY} -c 'import sys; print(sys.executable)')."
  echo "       fix:  ${PY} -m pip install -e .   (installs numpy/nibabel/pillow), then re-submit."
  exit 1; }

ARGS=( -m xrsp.build_dataset
       --ct_dir "${CTSP1K_V4}/ct" --label_dir "${CTSP1K_V4}/labels"
       --out "${OUT_DIR}" --views ${VIEWS} --gamma "${GAMMA}"
       --shard_id "${SHARD}" --n_shards "${N_SHARDS}" )
[[ "${RESUME}" == "0" ]] && ARGS+=( --no_resume )
[[ "${NO_RIBS:-0}" == "1" ]] && ARGS+=( --no_ribs )   # ship spine+sacrum+femurs now; ribs later

echo "[gen_xrsp1k] shard ${SHARD}/${N_SHARDS}  v4=${CTSP1K_V4} -> ${OUT_DIR}  resume=${RESUME}  $(date)"
"${PY}" "${ARGS[@]}"
echo "[gen_xrsp1k] shard ${SHARD} done  $(date)"
