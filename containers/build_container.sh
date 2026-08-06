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

# MODULE FIRST, not PATH order. On this grid the system `singularity` (singularity-ce
# 4.0.2) sits on PATH but is broken -- `pull` dies with
#     fork/exec /usr/bin/singularity: no such file or directory
# and `build --fakeroot` with
#     /usr/libexec/singularity/bin/starter-suid not found
# while the apptainer MODULE (1.2.2), which is not on PATH until loaded, pulls correctly.
# Preferring PATH order picks the broken runtime every time.
if [[ -n "${BUILDER:-}" ]]; then
    :
elif module load apptainer 2>/dev/null && command -v apptainer >/dev/null 2>&1; then
    BUILDER="apptainer"
elif command -v apptainer >/dev/null 2>&1; then BUILDER="apptainer"
elif module load singularity 2>/dev/null && command -v singularity >/dev/null 2>&1; then
    BUILDER="singularity"
elif command -v singularity >/dev/null 2>&1; then BUILDER="singularity"
else
    die "neither apptainer nor singularity found. Try: module load apptainer"
fi
log "builder: ${BUILDER}  ($(${BUILDER} --version 2>&1 | head -1))"

# Cache and tmp OFF /tmp: converting a ~10 GB image blows past a small /tmp and the
# failure wastes the whole pull. /scratch is used when it exists, but this grid has no
# /scratch -- an unconditional default there silently created the directory under a path
# nobody watches, or failed outright. So probe, then fall back to $HOME.
_pick_cache_root() {
    for d in "/scratch/${USER}" "/wsu/scratch/${USER}" "${HOME}/.cache/xrsp"; do
        parent="$(dirname "$d")"
        if [[ -d "$parent" ]] && mkdir -p "$d" 2>/dev/null; then echo "$d"; return; fi
    done
    echo "${HOME}/.cache/xrsp"
}
_CACHE_ROOT="${XRSP_CACHE_ROOT:-$(_pick_cache_root)}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${_CACHE_ROOT}/.singularity_cache}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SINGULARITY_CACHEDIR}}"
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-${_CACHE_ROOT}/.singularity_tmp}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SINGULARITY_TMPDIR}}"
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"
# A ~10 GB conversion needs room for the layers AND the assembled .sif
_avail=$(df -Pk "${_CACHE_ROOT}" | awk "NR==2{print \$4}")
if [[ -n "${_avail}" && "${_avail}" -lt 25000000 ]]; then
    log "WARNING: only $((_avail/1024/1024)) GB free at ${_CACHE_ROOT}; a ~10 GB image"
    log "         conversion needs roughly 25 GB. Set XRSP_CACHE_ROOT to somewhere larger."
fi
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
