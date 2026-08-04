# Running on the WSU grid

Docker built locally → pushed to Docker Hub → pulled on the grid as a Singularity
`.sif` → bind-mounted by the SLURM scripts. Same chain as `spinesurg-ct-nnunet`.

**No code is baked into the image.** The repo is bind-mounted at `/workspace`, so a
code change needs no rebuild — only a dependency change does. That keeps the
edit→run loop short and makes the image a stable, citable artefact.

## One-time setup

### 1. Local: build and push the image

```bash
docker login
make image                       # -> gregoryschwingmdphd/xrspinopelvic:latest
# or pin the OpenSpineToolkit commit the labels depend on:
OSTK_REF=1174c02 make image
```

`OSTK_REF` matters. Endplate corners are fitted in 3-D by
`ostk.spine.endplate_corners` (medial band + outlier rejection, which drops the
sacral ala). Pin it for anything you intend to publish — a change there moves every
label in the dataset.

### 2. Grid: pull and convert

```bash
ssh warrior
cd ~/XRSpinoPelvic1K
module load apptainer            # or singularity
make container                   # -> containers/xrspinopelvic.sif  (~10 GB)
```

`containers/build_container.sh` puts the Singularity cache and tmpdir on
`/scratch/$USER`. Do not skip that: converting a ~10 GB image overflows a small
`/tmp` and the pull fails after a long download.

It is idempotent — if the `.sif` exists it exits without re-pulling. Delete the file
to force a refresh.

### 3. Grid: check it runs

```bash
apptainer exec containers/xrspinopelvic.sif \
    python3 -c "import torch, monai, ostk; print(torch.__version__, torch.cuda.is_available())"
```

`cuda.is_available()` is `False` on a login node — that is expected. Only a GPU job
with `--nv` sees a device.

### 4. Grid: smoke-test the training path

```bash
make smoke                       # after Stage 1 has produced some views
```

Runs generation output → dataset → heatmaps → MONAI U-Net → soft-argmax → angles on a
handful of views, in under a minute. `make test` covers the numpy half locally, but the
torch half only runs where torch and MONAI are installed — i.e. in the container.
Finding a shape error four hours into a 24-hour job is an avoidable way to lose a day.

## Pipeline

Every stage is idempotent: finished work is skipped and interrupted work resumes, so
a job killed by the wall clock is resubmitted with the identical command.

| Stage | Command | Resource |
|---|---|---|
| 1. Generate DRRs | `make generate` | CPU array, 20 shards |
| 2. Splits | `make splits` | CPU, minutes, **run once** |
| 3. Train | `make train` | GPU array, one task per fold |
| 4a. Held-out DRRs | `make eval` | GPU array |
| 4b. BUU | `make eval-buu` | GPU, single |
| 4c. Reader set | `make reader-set` | CPU |

```bash
make generate                          # N_VIEWS=12 YAW=15 make generate
make splits                            # once; commit data/splits.json
make train                             # EPOCHS=300 make train
make train-fold FOLD=2                 # a single fold
make eval
BUU_IMAGES=/data/buu/images BUU_ANN=/data/buu/corners.json make eval-buu
make status                            # squeue for this user
```

### Layout expected under `--bind ... :/data`

```
data/
  ct/            <case>_ct.nii.gz          input CT
  labels/        <case>_label.nii.gz       3-D segmentation
  cases.csv      case_id, patient_id, lstv_label, has_l6, femoral_heads_visible
  xrsp1k/        <- Stage 1 output
  splits.json    <- Stage 2 output (COMMIT THIS)
  runs/fold_N/   <- Stage 3 checkpoints
  results/       <- Stage 4 metrics
```

`DATA_ROOT` overrides where `/data` binds from; it defaults to `<repo>/data`. On the
grid, point it at scratch or project space rather than `$HOME`.

## Notes that will save you a wasted job

**Splits run once.** `make_splits.py` refuses to overwrite an existing `splits.json`
without `--force`, because a resplit silently invalidates every number reported
against the old one. Commit the file.

**Grouping is by patient, not case.** Generation emits many views per CT and a
patient may contribute several CTs. `build_folds` asserts non-leakage on its own
output; `tests/test_pipeline.py` proves the guard actually fires.

**The channel set comes from the data.** The trainer takes the union of vertebral
levels annotated anywhere in the dataset — scans differ in how many vertebrae they
show — and masks per view. Do not hardcode a level list unless you mean to exclude
levels.

**Geometry augmentation happens at generation, not in the loader.** Obliquity is
applied when the labels are re-projected through the same plan, so they stay exact.
The loader does appearance only (gamma, contrast, noise, mild blur). Warping images
in the loader would require warping landmarks, and a mismatch there is silent.

**One GPU per fold.** The model is a 2-D U-Net over ~53 channels; it does not need
multi-GPU. Five folds run as five array tasks.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `container missing` | run `make container` on the grid first |
| `FATAL: while extracting ... no space left` | cache/tmp on a small `/tmp` — export `SINGULARITY_CACHEDIR=/scratch/$USER/...` |
| `no generated views found under /data/xrsp1k` | Stage 1 has not run, or `DATA_ROOT` points elsewhere |
| `fold N: train=0 val=0` | `splits.json` case ids do not match the generated case dirs |
| `run_config.json not found beside the checkpoint` | evaluating a checkpoint from a different run dir — it carries the channel order |
| `endplate_corners_2d needs OpenSpineToolkit` | ostk missing from the image; rebuild, or pass `--ostk_path` |
| CUDA OOM | lower `BATCH`, or `--height/--width` |
