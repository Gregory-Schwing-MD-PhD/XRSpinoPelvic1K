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

# ── YOLO-Pose baseline (Ultralytics) ─────────────────────────────────────────
# Pinned to the version in Bansal et al. 2026 Table 3, so the baseline is the software
# they actually ran rather than whatever is current. Pure python on top of the torch
# already installed above, so this layer is seconds and leaves everything expensive cached.
RUN pip install --no-cache-dir "ultralytics==8.3.159"

# ── OpenSpineToolkit — the 3-D endplate fit the labels come from ─────────────
# Endplate corners are fitted in 3-D (medial band + outlier rejection, which drops
# the sacral ala) and then projected. Fitting them from the 2-D silhouette instead
# is 26–32° wrong at every slab fraction, because the ala superimposes on the S1
# body in projection. This dependency is therefore load-bearing, not convenience.
# Pin for anything you intend to publish -- the 3-D endplate fit determines every label.
#
# FLOOR: f9e23d3 is where OpenSpineToolkit first became pip-installable. Before it the
# repo had no setup.py and no pyproject.toml, so this step failed at EVERY ref with
# "does not appear to be a Python project" -- including the 09dd9c5 that used to be the
# default here. A pin older than f9e23d3 cannot build, and the error names the ref
# rather than the missing package spec, which reads like a bad sha. It is not.
#
# f9e23d3 also carries the endplate work that 09dd9c5 predates: the body-isolation fit,
# the robust (Tukey) endplate profile, the perpendicular plate band, and the medial-band
# restriction that keeps the sacral ala out of the S1 fit.
ARG OSTK_REF=f9e23d3
RUN pip install --no-cache-dir \
        "git+https://github.com/Gregory-Schwing-MD-PhD/OpenSpineToolkit.git@${OSTK_REF}"

ENV PYTHONUNBUFFERED=1
# Plain assignment, not /workspace:${PYTHONPATH} -- PYTHONPATH is undefined in the
# base image, so the expansion leaves a trailing ':' (an empty path entry, which
# Python reads as the CWD) and BuildKit warns UndefinedVar. The SLURM wrapper passes
# PYTHONPATH explicitly via --env anyway.
ENV PYTHONPATH=/workspace
ENV MPLBACKEND=Agg

# ── Smoke test: fail the BUILD, not the 48-hour grid job ─────────────────────
RUN python3 -c "\
import torch, monai, pytorch_lightning as pl, nibabel, scipy, PIL, ultralytics; \
from monai.networks.nets import BasicUNet; \
from ultralytics import YOLO; \
print(f'torch={torch.__version__} monai={monai.__version__} lightning={pl.__version__} ultralytics={ultralytics.__version__}'); \
BasicUNet(spatial_dims=2, in_channels=1, out_channels=8, features=(8,8,8,8,8,8)); \
print('BasicUNet 2-D: OK')"

# NO `|| echo WARNING` here. This check used to swallow its own failure, which
# contradicts the heading four lines up: a warning scrolls past in a 10 GB build and
# the image ships anyway, so the first thing to notice that ostk is missing is a grid
# job that dies after queueing. ostk is not optional -- it IS the labels -- so a broken
# import must fail the build.
RUN python3 -c "\
import ostk; from ostk.spine import endplate_corners, corner_params_for_level; \
from ostk.labels import LABELS; \
from ostk.vertebral_body import endplate_corners_body, body_mask_by_canal; \
print('ostk OK — endplate_corners + corner_params_for_level + vertebral_body available')"

ENTRYPOINT ["/bin/bash"]
