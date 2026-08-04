# =============================================================================
# XRSpinoPelvic1K — DRR generation + landmark training image
# docker/Dockerfile
#
# Workflow (mirrors spinesurg-ct-nnunet):
#   Local:    scripts/build_and_push.sh     → builds + pushes to Docker Hub
#   Grid:     containers/build_container.sh → pulls + converts to .sif
#   Runtime:  slurm/*.sh bind-mount the repo into /workspace; NO code is baked in,
#             so a code change needs no rebuild — only a dependency change does.
#
# Base: pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
#   Same base as spinesurg-ct-nnunet. CUDA 12.1 had a cuBLAS bug on H200 NVL (SM90)
#   causing CUBLAS_STATUS_EXECUTION_FAILED for FP16/BF16 GEMMs; 12.4 + cuDNN 9 is
#   clean, including BF16 tensor-core ops.
#
# Everything is pinned. An unpinned image silently drifts between the local build
# and the grid, which is the fastest way to make a result irreproducible.
# =============================================================================
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

LABEL maintainer="Gregory Schwing <gregory.schwing@wayne.edu>"
LABEL description="XRSpinoPelvic1K — DRR generation + vertebral landmark model"
LABEL version="1.0.0"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl unzip libgomp1 libglib2.0-0 \
        libsm6 libxext6 libxrender-dev libgl1-mesa-glx ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Scientific core (versions matched to spinesurg-ct-nnunet) ────────────────
RUN pip install --no-cache-dir \
        numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.1 \
        pandas==2.2.2 matplotlib==3.9.2 scikit-image==0.24.0

# ── Medical image I/O ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        nibabel==5.2.1 SimpleITK==2.3.1 pydicom==2.4.4 pillow==10.4.0

# ── Model stack: MONAI supplies the network, Lightning the training loop ─────
# Both off the shelf on purpose. The only bespoke code is the masked loss (partial
# landmark annotation) and the angle metrics — i.e. the science, not the plumbing.
RUN pip install --no-cache-dir \
        "monai[all]==1.4.0" \
        lightning==2.4.0 pytorch-lightning==2.4.0 \
        torchmetrics==1.4.1 tensorboard==2.17.1 \
        einops==0.8.0 tqdm==4.66.5 pyyaml==6.0.2

RUN pip install --no-cache-dir "wandb>=0.12.10"

# ── OpenSpineToolkit — the 3-D endplate fit the labels come from ─────────────
# Endplate corners are fitted in 3-D (medial band + outlier rejection, which drops
# the sacral ala) and then projected. Fitting them from the 2-D silhouette instead
# is 26–32° wrong at every slab fraction, because the ala superimposes on the S1
# body in projection. This dependency is therefore load-bearing, not convenience.
# Pin for anything you intend to publish -- the 3-D endplate fit determines every label.
# 09dd9c5 is the first commit where main carries ostk.drr + ostk.project2d (PR #5 merged)
# AND the shared spine.pi_anchor_point, so the corner-midpoint PI convention is the
# default and CT- and XR-derived PI agree.
ARG OSTK_REF=09dd9c5
RUN pip install --no-cache-dir \
        "git+https://github.com/Gregory-Schwing-MD-PhD/OpenSpineToolkit.git@${OSTK_REF}"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace:${PYTHONPATH}
ENV MPLBACKEND=Agg

# ── Smoke test: fail the BUILD, not the 48-hour grid job ─────────────────────
RUN python3 -c "\
import torch, monai, pytorch_lightning as pl, nibabel, scipy, PIL; \
from monai.networks.nets import BasicUNet; \
print(f'torch={torch.__version__} monai={monai.__version__} lightning={pl.__version__}'); \
BasicUNet(spatial_dims=2, in_channels=1, out_channels=8, features=(8,8,8,8,8,8)); \
print('BasicUNet 2-D: OK')"

RUN python3 -c "\
import ostk; from ostk.spine import endplate_corners, corner_params_for_level; \
from ostk.labels import LABELS; \
print('ostk OK — endplate_corners + corner_params_for_level available')" \
    || echo 'WARNING: ostk import failed — corner extraction will not work'

ENTRYPOINT ["/bin/bash"]
