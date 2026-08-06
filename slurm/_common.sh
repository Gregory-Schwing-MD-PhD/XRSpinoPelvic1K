# =============================================================================
# XRSpinoPelvic1K -- shared SLURM helpers.   source slurm/_common.sh
#
# Centralises the singularity invocation so every stage binds identically and a
# grid-path change is a one-file edit rather than a six-file hunt.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONTAINER="${CONTAINER:-${PROJECT_ROOT}/containers/xrspinopelvic.sif}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container missing: ${CONTAINER}" >&2
    echo "       run:  bash containers/build_container.sh" >&2
    exit 1
fi

# Per-job tmp on local scratch. Singularity writing into a shared /tmp across
# concurrent array tasks is a reliable source of mysterious mid-run failures.
export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-/tmp/${USER}_xrsp_${SLURM_JOB_ID:-$$}}"
export APPTAINER_TMPDIR="${SINGULARITY_TMPDIR}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export SINGULARITYENV_PYTHONPATH=/workspace
export SINGULARITYENV_OMP_NUM_THREADS="${OMP_NUM_THREADS}"

# Runtime selection. Order matters and is NOT "whatever is on PATH first":
#
# On this grid the system `singularity` (singularity-ce 4.0.2) is on PATH by default but
# BROKEN -- both `pull` and `build --fakeroot` die with
#     /usr/bin/singularity: no such file or directory
#     /usr/libexec/singularity/bin/starter-suid not found
# because the wrapper points at paths the install does not have. The working runtime is
# the apptainer MODULE, which is not on PATH until loaded and pulls from Docker Hub
# correctly. Preferring PATH order therefore picks the broken one every time.
#
# So: try the module FIRST, fall back to whatever is on PATH.
if module load apptainer 2>/dev/null && command -v apptainer >/dev/null 2>&1; then
    RUNNER=apptainer
elif command -v apptainer >/dev/null 2>&1; then
    RUNNER=apptainer
elif command -v singularity >/dev/null 2>&1; then
    RUNNER=singularity
else
    echo "ERROR: no apptainer/singularity available (try: module load apptainer)" >&2
    exit 1
fi
echo "[container runtime] ${RUNNER} -- $(command -v ${RUNNER})"

# xrun [--nv] <cmd...>   run inside the container with the repo bound at /workspace
xrun() {
    local nv=()
    if [[ "${1:-}" == "--nv" ]]; then nv=(--nv); shift; fi
    "${RUNNER}" exec "${nv[@]}" \
        --bind "${PROJECT_ROOT}:/workspace" \
        --bind "${DATA_ROOT}:/data" \
        --pwd /workspace \
        "${CONTAINER}" "$@"
}

banner() {
    echo "================================================================"
    echo " XRSpinoPelvic1K -- $*"
    echo " Job     : ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ [task ${SLURM_ARRAY_TASK_ID}]}"
    echo " Host    : $(hostname)"
    echo " Root    : ${PROJECT_ROOT}"
    echo " Started : $(date)"
    echo "================================================================"
}
