#!/usr/bin/env bash
# =============================================================================
# XRSpinoPelvic1K — Docker build & push.   Run on a machine that HAS Docker.
#
# The grid does not, and cannot: `singularity build --fakeroot` fails there with
# "starter-suid not found" and the account has no /etc/subuid entry. Singularity can
# only PULL a prebuilt image, so the image has to be built somewhere with Docker and
# pushed to a registry first. That is the whole reason this script exists.
#
#   ./scripts/build_and_push.sh
#   DOCKERHUB_USER=myuser TAG=v1.0.0 ./scripts/build_and_push.sh
#   PUSH=0 ./scripts/build_and_push.sh            # build only, no push
#   OSTK_REF=<sha> ./scripts/build_and_push.sh    # pin OpenSpineToolkit
#
# Then, on the grid:   bash containers/build_container.sh
# =============================================================================
set -euo pipefail

# TWO Docker Hub accounts exist and both are real:
#   gregoryschwingmdphd -> this project's images (ctspinopelvic1k, ctspinopelvic1k-ts,
#                          spinesurg-ct*). xrspinopelvic belongs HERE.
#   go2432              -> older work (gomc, namd, deepchem, lstv-*, spine-level-ai-*)
# A push is only permitted into the namespace you are logged in to, so being logged in
# as go2432 while targeting this one fails with "insufficient_scope" -- which reads like
# a repo permission problem rather than the wrong account. The preflight below checks it.
DOCKERHUB_USER="${DOCKERHUB_USER:-gregoryschwingmdphd}"
IMAGE_NAME="${IMAGE_NAME:-xrspinopelvic}"
TAG="${TAG:-latest}"
PUSH="${PUSH:-1}"
OSTK_REF="${OSTK_REF:-main}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FULL="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# ── preflight ────────────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || die "docker not found on PATH."

if ! docker info >/dev/null 2>&1; then
    echo "[ERROR] the Docker daemon is not running." >&2
    echo "" >&2
    echo "  WSL (Docker Engine is installed there):   sudo service docker start" >&2
    echo "  Docker Desktop:                           start the app, then retry" >&2
    echo "" >&2
    echo "  Rootless docker is present but unusable here: dockerd-rootless.sh needs" >&2
    echo "  newuidmap from the 'uidmap' package, and installing that needs root anyway." >&2
    exit 1
fi

[[ -f "${HERE}/docker/Dockerfile" ]] || die "docker/Dockerfile not found under ${HERE}"

# Verify the login BEFORE a ~10 GB build. Docker only rejects an unauthenticated push at
# the very end, so without this the failure arrives after the expensive part.
# Checking only that SOME login exists is not enough -- being logged in as the wrong
# account fails exactly as late and as expensively as not being logged in at all. Docker
# Hub permits pushes only to your own namespace, so the logged-in user must EQUAL
# DOCKERHUB_USER. A mismatch surfaces as "insufficient_scope: authorization failed" after
# the push has already uploaded nothing and the build has already cost 14 GB and ~20
# minutes, and it reads like a permissions problem on the repo rather than a name
# mismatch. Compare the two here, before any of that.
if [[ "${PUSH}" == "1" ]]; then
    _who="$(docker system info 2>/dev/null | sed -n 's/^ *Username: *//p' | head -1)"
    if [[ -z "${_who}" ]]; then
        log "WARNING: no Docker Hub login detected. Push will fail after the build."
        log "         Run 'docker login' now, or re-run with PUSH=0."
        read -r -p "         Continue anyway? [y/N] " _yn
        [[ "${_yn}" =~ ^[Yy]$ ]] || exit 1
    elif [[ "${_who}" != "${DOCKERHUB_USER}" ]]; then
        echo "[ERROR] logged in to Docker Hub as '${_who}', but this build targets" >&2
        echo "        '${DOCKERHUB_USER}/${IMAGE_NAME}'. You can only push to your own" >&2
        echo "        namespace, so the push WILL be denied after the build." >&2
        echo "" >&2
        echo "        Either:  DOCKERHUB_USER=${_who} $0" >&2
        echo "        or:      docker login -u ${DOCKERHUB_USER}" >&2
        exit 1
    else
        log "docker hub  : logged in as ${_who} (matches target namespace)"
    fi
fi

# Disk: the image is ~10 GB and the build needs room for layers on top of that.
_avail_gb=$(df -Pk /var/lib/docker 2>/dev/null | awk 'NR==2{print int($4/1024/1024)}')
[[ -z "${_avail_gb}" ]] && _avail_gb=$(df -Pk / | awk 'NR==2{print int($4/1024/1024)}')
if [[ -n "${_avail_gb}" && "${_avail_gb}" -lt 30 ]]; then
    log "WARNING: only ${_avail_gb} GB free where Docker stores images; ~30 GB is safer."
fi

log "=== XRSpinoPelvic1K image build ==="
log "image     : ${FULL}"
log "ostk ref  : ${OSTK_REF}"
log "push      : ${PUSH}"
log "context   : ${HERE}"

# ── build ────────────────────────────────────────────────────────────────────
# OSTK_REF pins OpenSpineToolkit. It determines every endplate corner and hip point in
# the generated labels, so an unpinned image silently changes the dataset between
# rebuilds. Pin it to a sha for anything you intend to publish.
log "step 1/3 — building (this takes a while; base is pytorch cuda12.4 + MONAI)"
_build() {
    docker build \
        --build-arg "OSTK_REF=${OSTK_REF}" \
        -f "${HERE}/docker/Dockerfile" \
        -t "${FULL}" \
        --progress=plain \
        ${1:-} \
        "${HERE}"
}

# A BuildKit cache left inconsistent by a killed daemon or a killed WSL fails with
#     failed to prepare <id> as <id>: parent snapshot <id> does not exist: not found
# Layers report CACHED while the snapshot chain beneath them is gone. It is not a
# Dockerfile problem and re-running does not fix it -- the cache has to be dropped. So
# retry once automatically instead of making that the user's problem.
# The log is kept at a STABLE path and deleted only on success. It used to go to a
# mktemp that was rm'd on every exit path including failure, which threw away the one
# artefact worth having: a 10 GB build emits thousands of lines, a terminal scrollback
# truncates, and the pip resolver prints the actual conflict once, early, far above the
# final error. Losing it means rebuilding just to read it.
_log_tmp="${HERE}/build.log"
_fail() {
    echo "[ERROR] $*" >&2
    echo "         full build log: ${_log_tmp}" >&2
    echo "         the cause is usually well above the last line; try:" >&2
    echo "           grep -n -i -m20 -E 'ERROR|error:|Cannot install|conflict|Failed building' ${_log_tmp}" >&2
    exit 1
}

if ! _build 2>&1 | tee "${_log_tmp}"; then
    if grep -q "parent snapshot .* does not exist" "${_log_tmp}"; then
        log "BuildKit cache is inconsistent (stale parent snapshot). Pruning, then"
        log "rebuilding with --no-cache. This restarts from the base image."
        docker builder prune -af >/dev/null 2>&1 || true
        _build --no-cache 2>&1 | tee "${_log_tmp}" || _fail "build failed after cache prune."
    else
        _fail "build failed."
    fi
fi
rm -f "${_log_tmp}"

log "step 2/3 — built. size: $(docker images --format '{{.Size}}' "${FULL}" | head -1)"

# ── push, then PROVE it landed ───────────────────────────────────────────────
if [[ "${PUSH}" == "1" ]]; then
    log "step 3/3 — pushing ${FULL}"
    docker push "${FULL}"

    # Confirm from the registry's side. A push can report success while the tag is not
    # actually resolvable, and the next thing to discover that is a grid job that fails
    # at pull time -- this repository returned "object not found" for exactly that reason.
    log "verifying the tag is visible on Docker Hub …"
    if command -v curl >/dev/null 2>&1; then
        _url="https://hub.docker.com/v2/repositories/${DOCKERHUB_USER}/${IMAGE_NAME}/tags/${TAG}"
        if curl -fsS "${_url}" >/dev/null 2>&1; then
            log "  ✓ ${FULL} is pullable"
        else
            die "push reported success but ${FULL} is NOT visible on Docker Hub.
       Check the repository exists and is not private to another account:
         ${_url}"
        fi
    else
        log "  (curl unavailable — skipping registry verification)"
    fi
else
    log "step 3/3 — push skipped (PUSH=0)"
fi

cat <<EOF

  ┌────────────────────────────────────────────────────────────────────────┐
  │  Next, ON THE GRID:                                                    │
  │                                                                        │
  │    cd ~/XRSpinoPelvic1K                                                │
  │    bash containers/build_container.sh                                  │
  │                                                                        │
  │  That pulls and converts to containers/xrspinopelvic.sif.              │
  │  It uses the singularity in ~/mambaforge/envs/nextflow/bin — the one   │
  │  on the default PATH is broken on this grid.                           │
  └────────────────────────────────────────────────────────────────────────┘
EOF
log "done."
