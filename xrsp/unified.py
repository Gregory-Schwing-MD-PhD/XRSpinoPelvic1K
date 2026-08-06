"""The unified landmark model's data plumbing: one model, two sources, disjoint channels.

The settled design
------------------
  HIP POINT   supervised by DRRs only. Ground truth is a 3-D sphere fitted to the femoral
              head (ostk.metrics.femoral_head_center: seeded from the acetabular
              interface, so the fovea and the head-neck junction cannot drag it) and then
              projected. This is the 3-D form of the accepted radiographic construction --
              Legaye & Duval-Beaupere 1998 define the hip axis as the midpoint of the two
              femoral head centres, and a reader finds each centre with a concentric-circle
              (Mose) template. Under a parallel beam the projection of a sphere is a disc
              centred on the projected centre, so the label is exact.

  CORNERS     supervised by BUU only. Radiologist annotations, in the deployment domain.

The two streams supervise DISJOINT channel sets, which is the whole point. It removes the
corner-convention question entirely: the DRR corners are never used, so they can never
disagree with BUU's. That matters because the endplate corner is genuinely ambiguous --
where the plate ends and the wall begins is a judgement call with a literature behind it --
while a femoral head is a large sphere with a fit residual you can check per case. The
split puts each landmark on the source that can actually pin it down.

What is given up, honestly: with disjoint channels nothing supervises the hip and the
corners TOGETHER on one image, so the model never learns their joint geometry from data.
That is a real loss, but a small one -- a landmark network localises each point mostly from
local image evidence -- and it is the price of not depending on the corner extraction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


class SupervisionMask:
    """Wrap a dataset and keep only `supervise` channels in its loss mask.

    Not a filter on the DATA -- the image and the heatmaps are untouched -- only on the
    per-channel `valid` vector. A channel that is dropped here contributes nothing to the
    loss, exactly as an unannotated channel does, so a stream can be given authority over
    part of the output without ever asserting anything about the rest.
    """

    def __init__(self, base, names: Sequence[str], supervise: Sequence[str]):
        self.base = base
        self.names = list(names)
        keep = set(supervise)
        self.keep_idx = np.array([n in keep for n in self.names], dtype=bool)
        if not self.keep_idx.any():
            raise ValueError("SupervisionMask would mask every channel")

    def __len__(self):
        return len(self.base)

    def channel_names(self):
        return list(self.names)

    def __getitem__(self, i):
        import torch
        img, hm, valid, meta = self.base[i]
        v = valid.clone() if hasattr(valid, "clone") else torch.as_tensor(valid).clone()
        v &= torch.from_numpy(self.keep_idx)
        return img, hm, v, meta


def corner_channels(names: Sequence[str]) -> List[str]:
    from .heatmaps import FEMORAL_KEY
    return [n for n in names if n != FEMORAL_KEY]


def hip_channels(names: Sequence[str]) -> List[str]:
    from .heatmaps import FEMORAL_KEY
    return [n for n in names if n == FEMORAL_KEY]


def build_union(drr_rows, buu_rows, *, levels, out_size=(512, 256), sigma=3.0,
                augment=True, seed=0, drr_weight=1, buu_weight=1,
                drr_supervises="hip", buu_supervises="corners"):
    """DRR + BUU as one dataset with disjoint supervision. Returns (dataset, names).

    `drr_supervises` / `buu_supervises` take "hip", "corners" or "all". The defaults are
    the settled design; "all" exists so the alternative can be measured rather than argued
    about, but note that letting BOTH supervise corners requires the DRR corner convention
    to match BUU's, which is not currently guaranteed.
    """
    from .buu import BUULandmarkDataset
    from .dataset import LandmarkDRRDataset
    from .heatmaps import channel_names

    names = channel_names(list(levels))
    sel = {"hip": hip_channels(names), "corners": corner_channels(names), "all": names}

    drr_ds = buu_ds = None
    if drr_rows:
        drr_ds = LandmarkDRRDataset(drr_rows, out_size=out_size, levels=list(levels),
                                    sigma=sigma, augment=augment, seed=seed)
        drr_ds = SupervisionMask(drr_ds, names, sel[drr_supervises])
    if buu_rows:
        buu_ds = BUULandmarkDataset(buu_rows, levels=list(levels), out_size=out_size,
                                    sigma=sigma, augment=augment, seed=seed)
        buu_ds = SupervisionMask(buu_ds, names, sel[buu_supervises])

    from .buu import UnionDataset
    return UnionDataset(drr_ds, buu_ds, drr_weight=drr_weight,
                        buu_weight=buu_weight), names
