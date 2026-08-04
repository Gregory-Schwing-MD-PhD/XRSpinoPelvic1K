#!/bin/bash
#SBATCH --job-name=xrsp_splits
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/xrsp_splits_%j.out
#SBATCH --error=logs/xrsp_splits_%j.err
# =============================================================================
# Stage 2 -- 5-fold splits. Patient-grouped, LSTV-stratified, leakage-asserted.
#
#   sbatch slurm/xrsp_splits.sh
#
# Runs ONCE and the output is committed. Regenerating a split silently invalidates
# every number previously reported against it, so make_splits.py refuses to
# overwrite an existing file unless FORCE=1.
#
# Grouping is by PATIENT because generation emits many views per CT and one patient
# may contribute several CTs. A view is not a new subject; letting views straddle a
# fold boundary inflates every metric invisibly.
# =============================================================================
source "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm/_common.sh"

MANIFEST="${MANIFEST:-/data/cases.csv}"
OUT="${OUT:-/data/splits.json}"

banner "splits  ${MANIFEST} -> ${OUT}"
xrun python scripts/make_splits.py --manifest "${MANIFEST}" --out "${OUT}" ${FORCE:+--force}
