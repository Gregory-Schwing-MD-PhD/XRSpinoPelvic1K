# XRSpinoPelvic1K -- common commands.  `make help` for the list.
#
# Local targets run directly; grid targets submit SLURM jobs. Pipeline order is
# container -> generate -> splits -> train -> eval. Every stage is idempotent, so a
# resubmitted job resumes rather than restarting.

SHELL := /bin/bash

DOCKERHUB_USER ?= gregoryschwingmdphd
IMAGE          ?= xrspinopelvic
TAG            ?= latest
DATA           ?= data
N_VIEWS        ?= 8
EPOCHS         ?= 200
FOLD           ?= 0
OSTK_REF       ?= main
BUU            ?= /data/BUU-LSPINE        # container path; host = $(DATA)/BUU-LSPINE
BUU_SPLITS     ?= /data/buu_splits.json
BUU_ZIP        ?= BUU-LSPINE_400.zip

.PHONY: help image container generate splits train train-fold eval eval-buu \
        reader-set test smoke lint clean-logs status

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?##/ \
	  {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# Container  (local build -> Docker Hub -> grid .sif)
# =============================================================================
image:  ## LOCAL: build + push the Docker image
	OSTK_REF=$(OSTK_REF) DOCKERHUB_USER=$(DOCKERHUB_USER) \
	IMAGE_NAME=$(IMAGE) TAG=$(TAG) bash scripts/build_and_push.sh

container:  ## GRID: pull the image and convert to containers/xrspinopelvic.sif
	DOCKERHUB_USER=$(DOCKERHUB_USER) IMAGE=$(IMAGE) TAG=$(TAG) \
	bash containers/build_container.sh

# =============================================================================
# Pipeline
# =============================================================================
generate:  ## Stage 1: render N oblique DRRs per CT + project labels/corners (array)
	N_VIEWS=$(N_VIEWS) sbatch slurm/xrsp_generate.sh

splits:  ## Stage 2: patient-grouped, LSTV-stratified 5-fold splits (run ONCE)
	sbatch slurm/xrsp_splits.sh

train:  ## Stage 3: train all 5 folds (SLURM array)
	EPOCHS=$(EPOCHS) sbatch slurm/xrsp_train_array.sh

train-fold:  ## Stage 3: train ONE fold, e.g. `make train-fold FOLD=2`
	EPOCHS=$(EPOCHS) sbatch --array=$(FOLD) slurm/xrsp_train_array.sh

eval:  ## Stage 4a: held-out DRR evaluation, all folds (in-silico)
	sbatch slurm/xrsp_eval.sh

eval-buu:  ## Stage 4b: BUU real-radiograph evaluation (agreement with readers)
	sbatch slurm/xrsp_eval_buu.sh

reader-set:  ## Stage 4c: blinded DRR reader set for the rater study
	sbatch scripts/make_reader_set.slurm

# =============================================================================
# Development
# =============================================================================
test:  ## Run the test suite (numpy only -- no GPU or container needed)
	python -m pytest tests -q

buu:  ## Stage + verify BUU-LSpine from a local archive (NOT redistributed here)
	python scripts/fetch_buu.py --zip $(BUU_ZIP) --out $(BUU)

buu-check:  ## Verify an existing BUU tree before burning grid time on it
	python scripts/fetch_buu.py --check $(BUU)

buu-splits:  ## Write the BUU train/val/test split ONCE (grouped + stratified)
	python scripts/make_buu_splits.py --buu $(BUU) --out $(BUU_SPLITS)

unified:  ## GRID: train the unified model (hip from DRRs, corners from BUU)
	EPOCHS=$(EPOCHS) BUU=$(BUU) BUU_SPLITS=$(BUU_SPLITS) sbatch slurm/xrsp_unified.sh

femhead:  ## GRID: train the femoral-head segmenter (alternative hip route)
	sbatch slurm/xrsp_femhead.sh

measure:  ## GRID/LOCAL: run the unified model on BUU, QC-gated, report PI
	python scripts/measure_pi_unified.py --buu $(BUU) 	  --model $(DATA)/runs/unified/best.pt --out results/buu_pi.csv

validate-hip:  ## Independent classical circle-fit check of the hip point (no labels)
	python scripts/validate_hip_circlefit.py --buu $(BUU) 	  --model $(DATA)/runs/unified/best.pt --out results/hip_circlefit.csv

nf:  ## GRID: run the whole pipeline under Nextflow (-resume re-runs only failures)
	nextflow run nextflow/main.nf -profile slurm -resume 	  --ct_dir $(DATA)/ct --label_dir $(DATA)/labels --buu $(BUU) --outdir $(DATA)/xrsp1k

smoke-union:  ## LOCAL: end-to-end check of the DRR + BUU union (seconds, CPU)
	python scripts/smoke_union.py --drr $(DATA)/xrsp1k --buu $(BUU)

smoke:  ## Verify the TORCH path inside the container before a long GPU job
	apptainer exec --nv \
	  --bind $(PWD):/workspace --bind $(PWD)/$(DATA):/data \
	  --pwd /workspace containers/xrspinopelvic.sif \
	  python scripts/smoke_test.py --data /data/xrsp1k

lint:  ## Byte-compile every module and script
	python -m compileall -q xrsp scripts && echo "compile OK"

status:  ## Show queued/running jobs for this user
	squeue -u $(USER) -o "%.10i %.12j %.8T %.10M %.6D %R"

clean-logs:  ## Delete SLURM logs
	rm -f logs/*.out logs/*.err
