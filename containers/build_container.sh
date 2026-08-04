#!/usr/bin/env bash
# =============================================================================
# XRSpinoPelvic1K — HPC Singularity/Apptainer pull
# containers/build_container.sh
#
# Run this ON THE GRID (WSU warrior or equivalent).
# Pulls the Docker Hub image and converts it to a Singularity .sif.
#
# Prerequisites on the grid:
#   - apptainer or singularity available (try: module load apptainer)
#   - outbound network from the login node
#   - ~12 GB free disk
#
# Usage:   bash containers/build_container.sh
#
# Overrides:
#   BUILDER=singularity|apptainer   (default: auto-detect)
#   DOCKERHUB_USER=<user>           (default: gregoryschwingmdphd)
#   IMAGE=<name>                    (default: xrspinopelvic)
#   TAG=<tag>                       (default: latest)
#   OUTPUT=<path>                   (default: containers/xrspinopelvic.sif)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERHUB_USER="${DOCKERHUB_USER:-gregoryschwingmdphd}"
IMAGE="${IMAGE:-xrspinopelvic}"
TAG="${TAG:-latest}"
DOCKER_URI="docker://${DOCKERHUB_USER}/${IMAGE}:${TAG}"
OUTPUT="${OUTPUT:-${HERE}/xrspinopelvic.sif}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

if [[ -n "${BUILDER:-}" ]]; then
    :
elif command -v apptainer >/dev/null 2>&1; then BUILDER="apptainer"
elif command -v singularity >/dev/null 2>&1; then BUILDER="singularity"
else
    if module load apptainer 2>/dev/null && command -v apptainer >/dev/null 2>&1; then
        BUILDER="apptainer"
    elif module load singularity 2>/dev/null && command -v singularity >/dev/null 2>&1; then
        BUILDER="singularity"
    else
        die "neither apptainer nor singularity found. Try: module load apptainer"
    fi
fi
log "builder: ${BUILDER}  ($(${BUILDER} --version 2>&1 | head -1))"

# Cache and tmp on scratch: converting a ~10 GB image blows past a small /tmp,
# and a failure here wastes the whole pull.
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-/scratch/${USER}/.singularity_cache}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SINGULARITY_CACHEDIR}}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/scratch/${USER}/.singularity_tmp}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SINGULARITY_TMPDIR}}"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"
log "cache: ${SINGULARITY_CACHEDIR}"

if [[ -f "${OUTPUT}" ]]; then
    log "${OUTPUT} already exists — delete it to re-pull. Nothing to do."
    exit 0
fi

log "pulling ${DOCKER_URI} -> ${OUTPUT}"
"${BUILDER}" pull --force "${OUTPUT}" "${DOCKER_URI}"

log "verifying the image runs"
"${BUILDER}" exec "${OUTPUT}" python3 -c \
    "import torch, monai, pytorch_lightning; print('container OK:', torch.__version__)"

log "done: ${OUTPUT}"
log "next:  sbatch slurm/xrsp_generate.sh"
