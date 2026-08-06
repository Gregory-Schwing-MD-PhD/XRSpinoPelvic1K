"""Orientation augmentation: flip and rotate the image AND its landmarks together.

Why this is safe here when generic geometry augmentation is not
---------------------------------------------------------------
The DRR loader deliberately does appearance-only augmentation, because warping an image
without warping its labels is a silent corruption. A FLIP and a ROTATION are different:
both are exact, closed-form transforms of a point, so the labels can be moved with the
pixels and the correspondence is provable rather than hoped for. Each function below has
a test that asserts a landmark lands where the transform says it should.

Anatomical channel identity does NOT swap under a flip
------------------------------------------------------
`L1.sup_ant` means the ANTERIOR corner of L1's superior endplate -- an anatomical fact,
not a side of the image. Mirroring the film moves that point to the other side, so the
coordinate mirrors and the channel stays put. There is no anterior/posterior channel swap,
and adding one would be the bug. What the model must then learn is to read anteriority
from the anatomy -- spinous processes point posteriorly, the sacrum and promontory are
unmistakable -- which is exactly what a radiologist does and why this is learnable at all.

What orientation invariance does and does NOT buy you
-----------------------------------------------------
PI and LL are angles between lines, so they are invariant to flip and to in-plane
rotation: the landmarks move, the angles do not. SS and PT are NOT -- both are measured
against TRUE VERTICAL, so rotating the film changes them by the rotation angle. Rotation
augmentation therefore makes the LANDMARK DETECTOR robust to a tilted film; it does not
make SS/PT meaningful on one. If films may be rotated, the horizon has to be recovered
separately or SS/PT reported only for films known to be upright.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def flip_lr(img, points: Dict[str, Optional[Sequence[float]]]):
    """Mirror left-right. Returns (image, points)."""
    im = np.asarray(img)
    W = im.shape[1]
    out = {}
    for k, v in points.items():
        out[k] = None if v is None else [float(W - 1 - v[0]), float(v[1])]
    return im[:, ::-1].copy(), out


def rotate(img, points: Dict[str, Optional[Sequence[float]]], deg: float, *, order: int = 1):
    """Rotate about the image centre. Returns (image, points).

    scipy's `rotate(reshape=False)` turns about the centre of the array, so the matching
    point transform is a rotation about that same centre. The sign is the fiddly part and
    is pinned by a test rather than by reasoning: image row indices grow DOWNWARD, so a
    positive angle in array terms is clockwise on screen.
    """
    from scipy.ndimage import rotate as ndrotate
    im = np.asarray(img, np.float32)
    H, W = im.shape
    out_img = ndrotate(im, deg, reshape=False, order=order, mode="constant", cval=0.0)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    out = {}
    for k, v in points.items():
        if v is None:
            out[k] = None
            continue
        dx, dy = float(v[0]) - cx, float(v[1]) - cy
        # Sign determined by MEASUREMENT against scipy, not by reasoning about it: for a
        # positive `deg`, ndimage.rotate maps a point by (+s*dy, -s*dx), not the textbook
        # (-s*dy, +s*dx). Getting this backwards moves labels the wrong way by 2x the
        # rotation and nothing downstream would flag it, so it is asserted in the tests.
        out[k] = [cx + c * dx + s * dy, cy - s * dx + c * dy]
    return out_img, out


def augment_orientation(img, points, rng, *, p_flip: float = 0.5,
                        max_rot_deg: float = 0.0):
    """Random flip and (optionally) rotation, applied to image and points together."""
    if p_flip and rng.random() < p_flip:
        img, points = flip_lr(img, points)
    if max_rot_deg:
        deg = float(rng.uniform(-max_rot_deg, max_rot_deg))
        if abs(deg) > 1e-3:
            img, points = rotate(img, points, deg)
    return img, points
