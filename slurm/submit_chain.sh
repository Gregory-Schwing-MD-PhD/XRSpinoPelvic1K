#!/usr/bin/env bash
# =============================================================================
# Submit the whole pipeline as ONE dependency chain, then walk away.
#
#   bash slurm/submit_chain.sh
#   BUU_DIR=BUU-LSPINE_400 bash slurm/submit_chain.sh
#   DRY=1 bash slurm/submit_chain.sh          # print the plan, submit nothing
#
#   stage 0  container   pull docker -> containers/xrspinopelvic.sif
#   stage 1  generate    DRRs + landmarks          (array 0-19, CPU)  after 0
#   stage 2  buu splits  buu_splits.json           (CPU)              after 0
#   stage 3  unified     the landmark model        (GPU)              after 1 AND 2
#
# 1 and 2 both depend only on the container, so they run CONCURRENTLY; 3 waits
# for both. afterok throughout -- a failed stage stops everything downstream
# rather than training on whatever happens to be on disk.
#
# WHY THE PREFLIGHT IS NOT OPTIONAL
# ---------------------------------
# The point of a chain is that nobody is watching. Every check below is for a
# condition that costs HOURS when it is discovered late: stage 1 queues, waits,
# starts, and dies on a missing directory; or worse, stage 3 waits out stage 1's
# full runtime and only then finds a file that was never going to exist. All of
# it is knowable now, at submit time, in under a second. So it is checked now.
#
# The BUU pairing check is the subtle one. index_buu matches each .jpg to a .csv
# by stem and SILENTLY SKIPS an image whose annotation is missing -- a
# half-extracted archive therefore reads as a smaller dataset rather than as an
# error, and the only symptom is a film count in a log nobody reads until the
# run is over.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"

DATA_ROOT="${DATA_ROOT:-${HERE}/data}"
BUU_DIR="${BUU_DIR:-BUU-LSPINE}"          # relative to DATA_ROOT
SIF="${HERE}/containers/xrspinopelvic.sif"
DRY="${DRY:-}"

# Every counter below is `|| true`: pipefail would otherwise let a MISSING path
# abort the very block whose job is to report missing paths.
ok()   { echo "  [ok]   $*"; }
bad()  { echo "  [FAIL] $*" >&2; FAILED=1; }
FAILED=0

echo "=== preflight ============================================================"
echo "  project : ${HERE}"
echo "  data    : ${DATA_ROOT}   (bound to /data inside the container)"

mkdir -p logs
ok "logs/ exists (sbatch opens --output before the job runs, so this must precede it)"

command -v sbatch >/dev/null 2>&1 \
    && ok "sbatch found — on the grid" \
    || bad "no sbatch: run this ON THE GRID, not the workstation"

[[ -d "${DATA_ROOT}" ]] && ok "data root present" \
    || bad "no data root at ${DATA_ROOT} (override with DATA_ROOT=...)"

for d in ct labels; do
    n=$(ls "${DATA_ROOT}/${d}" 2>/dev/null | wc -l || true)
    [[ "${n}" -gt 0 ]] && ok "data/${d}: ${n} entries" \
        || bad "data/${d} is missing or empty — stage 1 has nothing to render"
done

# --- the labels must contain FEMURS, or the whole DRR half of the run is inert ------
# The DRRs supervise only the bicoxofemoral point. Without femur/hip ids there is no
# hip point, so 6000+ rendered views contribute zero gradient and the model cannot
# produce PI or PT -- and nothing fails. The single symptom is `hip nanpx` in the
# training log, discovered hours later.
#
# This is exactly the v2-vs-v3 trap: v2 is "model-completed dense spinopelvic labels",
# v3 is "bone-augmented (femurs + thoracic + S1)". Both look complete on disk and both
# pass every other check here.
#
# Needs the container (nibabel), so it is skipped -- loudly -- when the .sif is absent.
if [[ -f "${SIF}" ]]; then
    _CP="${CONDA_PREFIX_XRSP:-${HOME}/mambaforge/envs/nextflow}"
    if PATH="${_CP}/bin:${PATH}" command -v singularity >/dev/null 2>&1; then
        _lab=$(env -u LD_LIBRARY_PATH -u PYTHONPATH -u JAVA_HOME -u SINGULARITYENV_HOME \
               PATH="${_CP}/bin:${PATH}" \
               singularity exec --bind "${HERE}":/workspace --bind "${DATA_ROOT}":/data \
                 --pwd /workspace --env PYTHONPATH=/workspace "${SIF}" \
                 python3 -c "
import glob, numpy as np, nibabel as nib
from xrsp.labels import labels_for
fs = sorted(glob.glob('/data/labels/*.nii*'))
if not fs: print('NOLABELS'); raise SystemExit
lab = np.asanyarray(nib.load(fs[0]).dataobj)
ids = set(int(i) for i in np.unique(lab)) - {0}
L   = labels_for(lab)
miss = [k for k in ('femur_left','femur_right','left_hip','right_hip')
        if L.get(k) not in ids]
print('MISSING:' + ','.join(miss) if miss else 'OK')
" 2>/dev/null | tail -1 || echo "UNCHECKED")
        case "${_lab}" in
            OK)  ok "labels contain femurs + hips — the hip point can be fitted" ;;
            MISSING:*)
                 bad "labels are MISSING ${_lab#MISSING:}
         This is the v2 export. Femurs arrive in v3 ('bone-augmented (femurs +
         thoracic + S1)'). With no femurs there is no bicoxofemoral point, so every
         DRR would train nothing and the model could not produce PI or PT.
         Point data/ct and data/labels at the v3 export and delete data/xrsp1k." ;;
            *)   ok "label femur check inconclusive (${_lab}) — not blocking" ;;
        esac
    else
        ok "no singularity on the login node — skipping the femur check"
    fi
else
    ok "container not built yet — femur check deferred to the unified stage gate"
fi

BUU_ABS="${DATA_ROOT}/${BUU_DIR}"
if [[ -d "${BUU_ABS}/LA" ]]; then
    n_jpg=$(ls "${BUU_ABS}"/LA/*.jpg 2>/dev/null | wc -l || true)
    n_csv=$(ls "${BUU_ABS}"/LA/*.csv 2>/dev/null | wc -l || true)
    if [[ "${n_jpg}" -eq 0 ]]; then
        bad "${BUU_ABS}/LA has no .jpg"
    elif [[ "${n_jpg}" -ne "${n_csv}" ]]; then
        # NOT a warning. index_buu drops the unpaired images without complaint, so
        # this is the only place it will ever be visible.
        bad "BUU pairing mismatch: ${n_jpg} .jpg vs ${n_csv} .csv in ${BUU_ABS}/LA
         index_buu pairs by stem and silently drops unpaired images, so the run
         would quietly train on $(( n_jpg < n_csv ? n_jpg : n_csv )) films. Re-extract the archive."
    else
        ok "BUU: ${n_jpg} films, every one paired with a .csv"
    fi
else
    bad "no ${BUU_ABS}/LA — set BUU_DIR to the extracted BUU directory name"
fi

if [[ -f "${SIF}" ]]; then
    ok "container already built ($(du -h "${SIF}" | cut -f1)) — stage 0 will no-op"
else
    ok "container absent — stage 0 will pull it"
fi

if [[ "${FAILED}" -ne 0 ]]; then
    echo "" >&2
    echo "preflight failed — nothing submitted. Fix the above and re-run." >&2
    exit 1
fi

echo "=== submitting ==========================================================="
# sbatch defaults to --export=ALL, so exporting here is how the override reaches the
# jobs. Without this, BUU_DIR would be validated by the preflight and then ignored by
# stages 2 and 3, which fall back to /data/BUU-LSPINE -- the preflight would pass on
# BUU-LSPINE_400 while the jobs looked somewhere else entirely.
# DATA_ROOT is a HOST path (_common.sh binds it to /data); BUU is a CONTAINER path.
export DATA_ROOT
export BUU="/data/${BUU_DIR}"
echo "  DATA_ROOT=${DATA_ROOT}  ->  /data"
echo "  BUU=${BUU}  (inside the container)"

_sub() {   # _sub <label> <dependency-or-empty> <script> -> echoes the job id
    local desc="$1" dep="$2" script="$3"
    local args=(--parsable)
    [[ -n "${dep}" ]] && args+=(--dependency="${dep}" --kill-on-invalid-dep=yes)
    if [[ -n "${DRY}" ]]; then
        echo "DRY: sbatch ${args[*]} ${script}" >&2
        echo "000000"
        return
    fi
    # --parsable can return "jobid;cluster" on a federated setup; keep the id only.
    sbatch "${args[@]}" "${script}" | cut -d';' -f1
}

J0=$(_sub "container"  ""                  slurm/xrsp_container.sh)
echo "  stage 0  container    ${J0}"
J1=$(_sub "generate"   "afterok:${J0}"     slurm/xrsp_generate.sh)
echo "  stage 1  generate     ${J1}   after ${J0}"
J2=$(_sub "buu splits" "afterok:${J0}"     slurm/xrsp_buu_splits.sh)
echo "  stage 2  buu splits   ${J2}   after ${J0}"
J3=$(_sub "unified"    "afterok:${J1}:${J2}" slurm/xrsp_unified.sh)
echo "  stage 3  unified      ${J3}   after ${J1} and ${J2}"

cat <<EOF

=== submitted ============================================================
  watch:    squeue -u \$USER
  logs:     tail -f logs/xrsp_sif_${J0}.out
  cancel:   scancel ${J0} ${J1} ${J2} ${J3}

  --kill-on-invalid-dep=yes is set, so if a stage fails its dependents are
  CANCELLED rather than left queued forever in DependencyNeverSatisfied.
  Check with:  sacct -j ${J0},${J1},${J2},${J3} --format=JobID,JobName%16,State,Elapsed
EOF
