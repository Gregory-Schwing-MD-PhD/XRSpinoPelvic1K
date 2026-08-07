# =============================================================================
# XRSpinoPelvic1K -- shared SLURM helpers.   source slurm/_common.sh
#
# Centralises the singularity invocation so every stage binds identically and a
# grid-path change is a one-file edit rather than a six-file hunt.
#
# ENVIRONMENT, and why it is this specific incantation
# ----------------------------------------------------
# Copied from the working CTSpinoPelvic1K recipe (slurm/benchmark_totalseg.sh),
# because the obvious setup does not work on this grid:
#
#   * The singularity on the default PATH (/usr/bin, singularity-ce 4.0.2) is BROKEN --
#         pull             -> fork/exec /usr/bin/singularity: no such file or directory
#         build --fakeroot -> /usr/libexec/singularity/bin/starter-suid not found
#     The working one is 3.8.6 inside the conda env, so CONDA_PREFIX/bin goes on PATH
#     FIRST. `which singularity` is echoed below precisely so a job log records which
#     binary actually ran.
#
#   * LD_LIBRARY_PATH, PYTHONPATH, JAVA_HOME and the R_LIBS family are UNSET before the
#     call. Inherited host paths leak into the container and shadow its own libraries,
#     which surfaces as import errors that look like a broken image rather than a broken
#     environment. PYTHONPATH is passed back in explicitly via --env, so /workspace is
#     still importable.
#
#   * SINGULARITYENV_HOME is unset: a stale value silently redirects $HOME inside the
#     container and anything writing to it lands somewhere unexpected.
#
# SCRATCH, split deliberately
# ---------------------------
#   sandbox unpack (~10 GB, read-mostly hot path) -> node-local /tmp   (fast)
#   container /tmp (runtime scratch)              -> project NFS       (survives, large)
#   XDG runtime    (near-empty)                   -> project NFS
# A shared /tmp across concurrent array tasks is a reliable source of mysterious
# mid-run failures, so both are per-job and removed on exit.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONTAINER="${CONTAINER:-${PROJECT_ROOT}/containers/xrspinopelvic.sif}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

# ── the environment singularity actually needs ───────────────────────────────
export CONDA_PREFIX="${CONDA_PREFIX_XRSP:-${HOME}/mambaforge/envs/nextflow}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
unset SINGULARITYENV_HOME
if ! command -v singularity >/dev/null 2>&1; then
    echo "ERROR: no singularity on PATH after prepending ${CONDA_PREFIX}/bin" >&2
    echo "       (the system one is broken on this grid; the conda env has 3.8.6)" >&2
    exit 1
fi
echo "[container] $(command -v singularity)  $(singularity --version 2>&1 | head -1)"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container missing: ${CONTAINER}" >&2
    echo "       run:  bash containers/build_container.sh" >&2
    exit 1
fi

# ── split scratch: sandbox on node /tmp, runtime on NFS ──────────────────────
NODE_SCRATCH="/tmp/${USER}_xrsp_${SLURM_JOB_ID:-$$}"
NFS_SCRATCH="${PROJECT_ROOT}/.scratch/${USER}_${SLURM_JOB_ID:-$$}"
export SINGULARITY_TMPDIR="${NODE_SCRATCH}/singularity_unpack"
HOST_CONTAINER_TMP="${NFS_SCRATCH}/container_tmp"
export XDG_RUNTIME_DIR="${NFS_SCRATCH}/xdg_runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${HOST_CONTAINER_TMP}" "${XDG_RUNTIME_DIR}"
trap 'rm -rf "${NODE_SCRATCH}" "${NFS_SCRATCH}" 2>/dev/null || true' EXIT TERM INT

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"

# ── W&B, if a token is on disk ───────────────────────────────────────────────
# Same convention as spinesurg-ct-nnunet: ~/.wandb/token holds the key.
#
# Passed as SINGULARITYENV_WANDB_API_KEY rather than through xrun's --env list,
# deliberately. --env values land in the singularity command line, so the key would be
# readable in `ps` by every other user on a shared node. The SINGULARITYENV_ prefix is
# consumed from the ENVIRONMENT instead and never appears in argv.
#
# Absent token -> nothing exported -> the trainer runs with logging disabled. Offline
# and local runs therefore need no special casing.
if [[ -z "${WANDB_API_KEY:-}" && -r "${HOME}/.wandb/token" ]]; then
    WANDB_API_KEY="$(< "${HOME}/.wandb/token")"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export SINGULARITYENV_WANDB_API_KEY="${WANDB_API_KEY}"
    echo "[wandb] token found -- run logging enabled"
else
    echo "[wandb] no ~/.wandb/token -- logging disabled (metrics still go to this log)"
fi

# xrun [--nv] <cmd...>   run inside the container with the repo bound at /workspace
xrun() {
    local nv=()
    if [[ "${1:-}" == "--nv" ]]; then nv=(--nv); shift; fi
    local binds="${PROJECT_ROOT}:/workspace"
    binds+=",${DATA_ROOT}:/data"
    binds+=",${HOST_CONTAINER_TMP}:/tmp"
    # wandb writes its cache and offline runs under ~/.wandb; without the bind an
    # interrupted upload has nowhere durable to spool to.
    [[ -d "${HOME}/.wandb" ]] && binds+=",${HOME}/.wandb:${HOME}/.wandb"
    local cenv="PYTHONPATH=/workspace"
    # Without this a SLURM .out is a FILE, so Python block-buffers stdout at ~8 KB and a
    # 12-hour job looks dead for its first hour -- the per-case progress lines exist but
    # sit in a buffer. The Dockerfile sets PYTHONUNBUFFERED, but relying on the image's
    # ENV surviving --env across singularity versions is not worth the ambiguity when
    # "is it running or hung?" is the question this answers.
    cenv+=",PYTHONUNBUFFERED=1"
    cenv+=",OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    cenv+=",NUMEXPR_MAX_THREADS=${OMP_NUM_THREADS}"
    cenv+=",MPLBACKEND=Agg"
    # ${nv[@]+"${nv[@]}"}, NOT "${nv[@]}". Under `set -u` bash 4.2 -- which is what the
    # el7 compute nodes run -- treats an EMPTY array expansion as an unbound variable and
    # aborts. Bash only made "${arr[@]}" safe on an empty array in 4.4, so this worked on
    # a modern workstation and killed all 20 array tasks on the grid, at the first xrun,
    # before a single DRR was rendered. The +alternate form expands to nothing at all when
    # the array is empty and is correct on every version.
    singularity exec ${nv[@]+"${nv[@]}"} \
        --env "${cenv}" \
        --bind "${binds}" \
        --pwd /workspace \
        "${CONTAINER}" "$@"
}

banner() {
    echo "================================================================"
    echo " XRSpinoPelvic1K -- $*"
    echo " Job     : ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ [task ${SLURM_ARRAY_TASK_ID}]}"
    echo " Host    : $(hostname)"
    # nvidia-smi prints its "couldn't communicate with the NVIDIA driver" complaint on
    # STDOUT, not stderr, so 2>/dev/null did not suppress it and every CPU-stage log
    # opened with a four-line driver error that looks like the reason the job failed.
    # It is not -- the CPU stages have no GPU by design. Check the exit status instead.
    local _gpu
    if _gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"; then
        echo " GPU     : $(echo "${_gpu}" | head -1)"
    else
        echo " GPU     : none (CPU stage)"
    fi
    echo " Root    : ${PROJECT_ROOT}"
    echo " Data    : ${DATA_ROOT}  ->  /data"
    echo " SIF     : ${CONTAINER}"
    echo " Started : $(date)"
    echo "================================================================"
}
