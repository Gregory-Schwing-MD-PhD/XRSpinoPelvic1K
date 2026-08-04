#!/usr/bin/env bash
# =============================================================================
# XRSpinoPelvic1K — Docker Build & Push
# scripts/build_and_push.sh
#
# Run on your LOCAL WORKSTATION. The grid-side pull is containers/build_container.sh.
#
# Prerequisites:  Docker running, and `docker login`.
#
# Usage:
#   ./scripts/build_and_push.sh
#   DOCKERHUB_USER=myuser TAG=v1.0.0 ./scripts/build_and_push.sh
#   PUSH=0 ./scripts/build_and_push.sh          # build only, no push
# =============================================================================
set -euo pipefail

DOCKERHUB_USER="${DOCKERHUB_USER:-gregoryschwingmdphd}"
IMAGE_NAME="${IMAGE_NAME:-xrspinopelvic}"
TAG="${TAG:-latest}"
PUSH="${PUSH:-1}"
OSTK_REF="${OSTK_REF:-main}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

command -v docker >/dev/null 2>&1 || { echo "docker not found" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker daemon not running" >&2; exit 1; }

log "building ${FULL}  (ostk ref: ${OSTK_REF})"
docker build \
    --build-arg "OSTK_REF=${OSTK_REF}" \
    -f "${HERE}/docker/Dockerfile" \
    -t "${FULL}" \
    "${HERE}"

log "image built. size: $(docker images --format '{{.Size}}' "${FULL}" | head -1)"

if [[ "${PUSH}" == "1" ]]; then
    log "pushing ${FULL}"
    docker push "${FULL}"
    log "pushed. On the grid:  bash containers/build_container.sh"
else
    log "PUSH=0 — not pushing."
fi
