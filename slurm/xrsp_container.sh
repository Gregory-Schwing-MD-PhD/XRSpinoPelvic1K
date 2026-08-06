#!/bin/bash
#SBATCH --job-name=xrsp_sif
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/xrsp_sif_%j.out
#SBATCH --error=logs/xrsp_sif_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 0 -- pull the Docker image and convert it to containers/xrspinopelvic.sif
#
#   mkdir -p logs && sbatch slurm/xrsp_container.sh
#
# (mkdir first: SLURM opens --output BEFORE the script runs, so a missing logs/
# fails the job with no log to explain why. The other stages get logs/ created by
# _common.sh, which is too late for this one -- see below.)
#
# WHY THIS IS NOT LIKE THE OTHER STAGES
# ------------------------------------
# It does NOT source slurm/_common.sh. That file refuses to continue when the .sif
# is absent -- correct for every stage that runs INSIDE the container, and fatal
# for the one whose job is to create it. So the environment setup is repeated here
# rather than shared. If the singularity recipe in _common.sh changes, change it
# here too; the duplication is deliberate but it is still duplication.
#
# THE THING MOST LIKELY TO GO WRONG: NETWORK
# ------------------------------------------
# Pulling needs outbound HTTPS to Docker Hub, and on many clusters compute nodes
# have no route off-site even though the login node does. That failure arrives
# ~30 s in as a DNS or TLS error that reads like Docker Hub being down. So the
# route is probed FIRST, and if it is missing the job stops immediately and tells
# you to run containers/build_container.sh on the login node instead -- which is a
# perfectly good answer, just a slower and less pleasant one.
#
# WHY BOTHER WITH A JOB AT ALL
# ----------------------------
# The conversion unpacks 14.1 GB and assembles a .sif on top of it. On a login
# node that is both antisocial and liable to be killed by a watchdog partway
# through, leaving a corrupt cache. A compute node has the disk, the time limit,
# and node-local scratch that is far faster than NFS for the unpack.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/containers/xrspinopelvic.sif}"
mkdir -p "${PROJECT_ROOT}/logs"

echo "================================================================"
echo " XRSpinoPelvic1K -- container pull + convert"
echo " Job     : ${SLURM_JOB_ID:-local}"
echo " Host    : $(hostname)"
echo " Root    : ${PROJECT_ROOT}"
echo " Output  : ${OUTPUT}"
echo " Started : $(date)"
echo "================================================================"

# ── the same environment _common.sh sets up, for the same reasons ────────────
# The singularity on the default PATH is broken on this grid (pull dies with
# "fork/exec /usr/bin/singularity: no such file or directory"); 3.8.6 lives in the
# conda env. Inherited LD_LIBRARY_PATH/PYTHONPATH/JAVA_HOME/R_LIBS leak host
# libraries into the container and surface as import errors that look like a bad
# image. SINGULARITYENV_HOME silently redirects $HOME inside the container.
export CONDA_PREFIX="${CONDA_PREFIX_XRSP:-${HOME}/mambaforge/envs/nextflow}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
unset SINGULARITYENV_HOME

# ── probe the route to Docker Hub before spending anything on it ─────────────
_have_route() {
    if command -v curl >/dev/null 2>&1; then
        # ANY HTTP status proves the route. /v2/ answers 401 unauthenticated, which
        # is a success for this purpose -- so check that curl completed, not the code.
        curl -s -o /dev/null -m 20 https://registry-1.docker.io/v2/ && return 0
        return 1
    fi
    # No curl: fall back to a raw TCP open, which needs no external binary at all.
    timeout 20 bash -c 'exec 3<>/dev/tcp/registry-1.docker.io/443' 2>/dev/null
}
if _have_route; then
    echo "[net] route to registry-1.docker.io: OK"
else
    echo "" >&2
    echo "[ERROR] no outbound HTTPS from $(hostname) to registry-1.docker.io." >&2
    echo "" >&2
    echo "  Compute nodes on this cluster likely have no off-site route, even though" >&2
    echo "  the login node does. Nothing here can work around that -- the image has to" >&2
    echo "  come over the network." >&2
    echo "" >&2
    echo "  Run it on the login node instead:" >&2
    echo "      cd ${PROJECT_ROOT} && bash containers/build_container.sh" >&2
    echo "" >&2
    echo "  Or, if a proxy is available, export it and resubmit:" >&2
    echo "      export https_proxy=http://<proxy>:<port>" >&2
    exit 1
fi

# ── scratch: unpack on the NODE, land the .sif on shared storage ─────────────
# The unpack is thousands of small writes and is dramatically faster on node-local
# disk than on NFS. The OUTPUT deliberately stays under the repo: anything written
# to node-local scratch disappears when the job ends, which would make a "successful"
# job produce nothing.
NODE_SCRATCH="/tmp/${USER}_xrsp_sif_${SLURM_JOB_ID:-$$}"
mkdir -p "${NODE_SCRATCH}"
trap 'rm -rf "${NODE_SCRATCH}" 2>/dev/null || true' EXIT TERM INT

# Only use node-local scratch if it can actually hold the conversion; otherwise let
# build_container.sh pick (/scratch, then $HOME). A half-filled /tmp fails at 90%.
_node_free=$(df -Pk "${NODE_SCRATCH}" | awk 'NR==2{print $4}')
if [[ -n "${_node_free}" && "${_node_free}" -ge 45000000 ]]; then
    export XRSP_CACHE_ROOT="${NODE_SCRATCH}"
    echo "[scratch] node-local: ${NODE_SCRATCH} ($((_node_free/1024/1024)) GB free)"
else
    echo "[scratch] node-local /tmp has only $(( ${_node_free:-0} /1024/1024)) GB;"
    echo "          letting build_container.sh choose (/scratch, then \$HOME)."
fi

echo "[run] containers/build_container.sh"
OUTPUT="${OUTPUT}" bash "${PROJECT_ROOT}/containers/build_container.sh"

echo "================================================================"
echo " done: $(date)"
ls -lh "${OUTPUT}"
echo " next:  sbatch slurm/xrsp_generate.sh"
echo "================================================================"
