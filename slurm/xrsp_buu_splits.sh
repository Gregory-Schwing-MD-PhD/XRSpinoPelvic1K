#!/bin/bash
#SBATCH --job-name=xrsp_buusplit
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/xrsp_buusplit_%j.out
#SBATCH --error=logs/xrsp_buusplit_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 2 -- the BUU train/val/test split, written ONCE to disk.
#
#   sbatch slurm/xrsp_buu_splits.sh
#   BUU=/data/BUU-LSPINE_400 sbatch slurm/xrsp_buu_splits.sh
#
# xrsp_unified.sh requires /data/buu_splits.json and nothing produced it, so the
# unified stage failed on a missing file after waiting for a GPU. This is that
# missing step.
#
# Grouped by patient and stratified by sex x age band -- see make_buu_splits.py
# for why age matters here specifically (PI and LL are maturity-dependent, so a
# test set skewed young or old shifts the very quantity being measured).
#
# IDEMPOTENT BY DESIGN, which matters in a dependency chain. make_buu_splits.py
# refuses to overwrite an existing split without --force, and rightly so: a
# resplit invalidates every number previously reported against the old one. But
# an unconditional run would then FAIL on a rerun and take the whole chain down
# with it. So an existing file is a success here, not an error. Set FORCE=1 to
# deliberately resplit.
# =============================================================================
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"

BUU="${BUU:-/data/BUU-LSPINE}"
OUT="${OUT:-/data/buu_splits.json}"
VIEW="${VIEW:-LA}"
SEED="${SEED:-0}"

banner "buu splits  ${BUU} -> ${OUT}"

# The check runs INSIDE the container, because /data only exists there.
if [[ -z "${FORCE:-}" ]] && xrun test -f "${OUT}"; then
    echo "[skip] ${OUT} already exists — a split is written once and kept."
    echo "       Re-run with FORCE=1 to resplit (this invalidates every number"
    echo "       previously reported against the old split)."
    xrun python -c "
import json; d=json.load(open('${OUT}'))
print('  existing split:', d.get('counts'), 'seed', d.get('seed'),
      'from', d.get('buu_root'))"
    exit 0
fi

xrun python scripts/make_buu_splits.py \
    --buu "${BUU}" --out "${OUT}" --view "${VIEW}" --seed "${SEED}" \
    ${FORCE:+--force}

echo "[buu splits] done $(date)"
