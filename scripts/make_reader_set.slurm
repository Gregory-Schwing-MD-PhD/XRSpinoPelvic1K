#!/usr/bin/env bash
#SBATCH --job-name=xrsp1k_reader_set
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/xrsp1k_reader_set_%j.out
#SBATCH --error=logs/xrsp1k_reader_set_%j.out
# =============================================================================
# make_reader_set.slurm — build the BLINDED N-case reader set from a rendered
# XRSpinoPelvic1K dataset and zip it for the radiology residents.
#
# Runs inside the SAME Singularity image as the other XRSpinoPelvic1K / CTSpinoPelvic1K
# slurms (containers/ctspinopelvic1k.sif), via the SAME env (configs/default.env) — numpy/PIL
# live in the image, xrsp not needed here; the repo is bound only so the script is on disk.
#
# Produces:  <OUT>/R01..RN/lateral_drr.png (+ ap)     <- send this
#            <OUT>/reader_measurements_template.csv   <- reader fills
#            <OUT>/README.txt
#            <OUT>.zip                                 <- ready to scp + send (no KEY inside)
#            <OUT>_KEY.csv   (repo root, NOT in the zip) <- COORDINATOR ONLY, do NOT send
#
#   DATASET=data/xrsp1k CTSP1K_ROOT=~/CTSpinoPelvic1K N=25 sbatch scripts/make_reader_set.slurm
#   SIF_PATH=/path/to/ctspinopelvic1k.sif DATASET=... sbatch scripts/make_reader_set.slurm
# =============================================================================
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
source configs/default.env       # CTSP1K-identical singularity env (conda PATH + scratch) + SIF_PATH

DATASET="${DATASET:-${REPO_ROOT}/data/xrsp1k}"
OUT="${OUT:-${REPO_ROOT}/reader_set}"
N="${N:-25}"
SEED="${SEED:-20260630}"

mkdir -p logs
[[ -f "${SIF_PATH}" ]] || { echo "ERROR: no container at ${SIF_PATH}"; \
  echo "       set SIF_PATH=/path/to/ctspinopelvic1k.sif (or CTSP1K_ROOT=/path/to/CTSpinoPelvic1K)"; exit 1; }
[[ -d "${DATASET}" ]] || { echo "ERROR: no rendered dataset at ${DATASET} — run gen_xrsp1k.slurm first"; exit 1; }
ls "${DATASET}"/*/lateral_drr.png >/dev/null 2>&1 \
  || { echo "ERROR: ${DATASET} has no <case>/lateral_drr.png yet — let gen_xrsp1k.slurm finish (or run more shards)"; exit 1; }

# Identity binds: repo (script) + dataset (only if it lives outside the repo).
BINDS="${REPO_ROOT}:${REPO_ROOT}"
case "${DATASET}" in "${REPO_ROOT}"/*) : ;; *) BINDS="${BINDS},${DATASET}:${DATASET}" ;; esac
ENV_VARS="PYTHONPATH=${REPO_ROOT},PYTHONUNBUFFERED=1"

echo "[make_reader_set] sif=${SIF_PATH}"
echo "[make_reader_set]   dataset=${DATASET}  out=${OUT}  n=${N}  seed=${SEED}  $(date)"

# 1) build the blinded set (writes OUT/ + OUT_KEY.csv next to it)
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd "${REPO_ROOT}" "${SIF_PATH}" \
    python3 -u scripts/make_reader_set.py --dataset "${DATASET}" --out "${OUT}" --n "${N}" --seed "${SEED}"

# 2) zip ONLY the reader_set folder (the KEY sits in the repo root, so it is never included)
stdbuf -oL -eL singularity exec \
    --env "${ENV_VARS}" --bind "${BINDS}" --pwd "${REPO_ROOT}" "${SIF_PATH}" \
    python3 -u -c "import shutil, os; b=os.path.splitext('${OUT}')[0]; \
p=shutil.make_archive(b, 'zip', root_dir=os.path.dirname('${OUT}') or '.', base_dir=os.path.basename('${OUT}')); \
print('wrote', p)"

echo ""
echo "[make_reader_set] DONE  $(date)"
echo "  send   : ${OUT}.zip   (+ protocol link)  to BOTH residents"
echo "  PRIVATE: $(dirname "${OUT}")/$(basename "${OUT}")_KEY.csv   (do NOT send — anon->real case map)"
