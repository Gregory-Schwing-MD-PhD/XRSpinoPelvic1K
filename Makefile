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
BUU            ?= ../CTSpinoPelvic1K-1/BUU-LSPINE_400

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
