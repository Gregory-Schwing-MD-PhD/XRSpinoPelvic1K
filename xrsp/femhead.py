"""Femoral-head segmentation on DRRs, and the bicoxofemoral point derived from it.

Why a segmenter and not a landmark heatmap
------------------------------------------
The bicoxofemoral point is the midpoint of the two femoral head centres, and the centre
of a sphere has NO local image evidence -- there is nothing at that pixel to look at. A
network asked to put a peak there has to infer it from the rim, which is precisely what a
segmenter does explicitly, with a dense gradient over thousands of pixels instead of a
near-delta target. The point then falls out of the mask arithmetically, and a mask can be
eyeballed for QC while a heatmap peak cannot.

Mask -> point is exact, not an approximation
--------------------------------------------
On a lateral the two heads project as overlapping discs. Take the UNION blob and its
centroid: if the two projected radii are equal, the union is symmetric under a 180 deg
rotation about the midpoint of the two centres, so its centroid IS that midpoint -- the
bicoxofemoral point -- at any degree of overlap, including total. The two heads never
need to be separated.

The only bias is the near/far magnification difference making one disc slightly larger.
At a 15-18 cm inter-hip distance and clinical SID that is sub-millimetre, against a PI
sensitivity of roughly 1.3 deg per 3 mm of hip-axis error (a ~130 mm moment arm). It only
matters on grossly rotated films, where the projected centres separate widely -- and
those are flagged rather than measured (see `qc_flags`).

For a parallel-beam DRR there is no magnification difference at all, so on synthetic data
the centroid is the projected midpoint exactly. That is what makes DRRs a clean source of
ground truth here.

Error budget, for calibration
-----------------------------
PI error is dominated by the S1 ENDPLATE, not the hip axis: 3 mm of femoral-centre error
is ~1.3 deg of PI, under inter-observer variability, while 2 deg of S1 endplate tilt is
2 deg of PI, one for one. Effort spent on the endplate is worth more than effort spent
here -- which is the reason this module stays deliberately simple.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

FEM_SIDES = ("femur_left", "femur_right")
HIP_SIDES = ("left_hip", "right_hip")


def head_mask_3d(label, affine, labels: Dict[str, int], *, sup_axis=(0.0, 0.0, 1.0),
                 radius_scale: float = 1.05, ostk_path: Optional[str] = None
                 ) -> Optional[np.ndarray]:
    """Voxel mask of BOTH femoral HEADS (not the whole femur).

    The femur label includes neck, trochanter and shaft, whose centroid sits well below
    and lateral to the head -- using it as the hip axis put PT 6.8 deg and PI 6.2 deg out
    in an earlier check while SS and LL stayed correct, i.e. the geometry was fine and
    only this point was wrong. So the head is cut out explicitly: ostk fits a sphere to
    the acetabular interface and extends it over the head while rejecting the neck, and
    the head is then every femur voxel within `radius_scale` x that radius of the centre.
    """
    import sys
    if ostk_path and ostk_path not in sys.path:
        sys.path.insert(0, ostk_path)
    from ostk.metrics import femoral_head_center

    lab = np.asarray(label)
    aff = np.asarray(affine, float)
    out = np.zeros(lab.shape, bool)
    found = 0
    for fem, hip in zip(FEM_SIDES, HIP_SIDES):
        fid = labels.get(fem)
        if fid is None or not (lab == fid).any():
            continue
        sel = (lab == fid)
        hid = labels.get(hip)
        if hid is not None:
            sel = sel | (lab == hid)
        sl = ndimage.find_objects(sel.astype(np.uint8))[0]
        off = np.array([sl[0].start, sl[1].start, sl[2].start], float)
        a2 = aff.copy()
        a2[:3, 3] = a2[:3, 3] + a2[:3, :3] @ off
        res = femoral_head_center(lab[sl], a2, fem, hip, labels=labels, sup_axis=sup_axis)
        if res is None:
            continue
        c, r, _ = res
        idx = np.array(np.nonzero(lab[sl] == fid)).T
        if not len(idx):
            continue
        w = (np.c_[idx, np.ones(len(idx))] @ a2.T)[:, :3]
        keep = np.linalg.norm(w - np.asarray(c, float), axis=1) <= radius_scale * float(r)
        if keep.sum() < 20:
            continue
        ii = idx[keep] + off.astype(int)
        out[ii[:, 0], ii[:, 1], ii[:, 2]] = True
        found += 1
    return out if found else None


def bicoxofemoral_from_mask(mask2d) -> Optional[List[float]]:
    """Bicoxofemoral point = centroid of the projected head blob. See module docstring
    for why the centroid is the midpoint of the two centres exactly, at any overlap."""
    m = np.asarray(mask2d, bool)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return [float(xs.mean()), float(ys.mean())]


def qc_flags(mask2d, *, image_shape=None, min_frac: float = 2e-4,
             max_frac: float = 8e-2, border_px: int = 2) -> Dict[str, object]:
    """Reject criteria for a predicted head mask. Cheap, and they catch the failures that
    matter when labelling at a scale nobody can inspect by hand.

    `ok` is the conjunction. A rejected case is not measured -- it is not silently
    measured badly, which is the failure mode that poisons a pseudo-labelled set.
    """
    m = np.asarray(mask2d, bool)
    H, W = (image_shape or m.shape)
    area = int(m.sum())
    frac = area / float(H * W)
    lab, n = ndimage.label(m)
    touches = False
    if area:
        ys, xs = np.nonzero(m)
        touches = bool((ys < border_px).any() or (ys >= H - border_px).any()
                       or (xs < border_px).any() or (xs >= W - border_px).any())
    # largest-component share: a clean blob is nearly all of the mask
    share = 1.0
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        share = float(sizes.max() / max(sizes.sum(), 1))
    out = {
        "area_px": area, "area_frac": float(frac), "n_components": int(n),
        "largest_share": share, "touches_border": touches,
        "empty": area == 0,
        "area_out_of_range": bool(frac < min_frac or frac > max_frac),
        "fragmented": bool(n > 1 and share < 0.9),
    }
    out["ok"] = not (out["empty"] or out["area_out_of_range"]
                     or out["fragmented"] or out["touches_border"])
    return out


def tta_spread(points: Sequence[Sequence[float]]) -> Optional[float]:
    """Scatter of the point across test-time augmentations, in pixels.

    An uncertainty estimate that costs only repeated inference and needs no ground
    truth -- which is the whole difficulty with pseudo-labels on a set that has none.
    """
    p = np.asarray([q for q in points if q is not None], float)
    if len(p) < 2:
        return None
    return float(np.sqrt(((p - p.mean(0)) ** 2).sum(axis=1).mean()))


def xray_appearance(img, rng):
    """Aggressive intensity randomisation for the DRR -> real-radiograph gap.

    Deliberately harsher than the DRR loader's augmentation. Closing this gap by making
    DRRs look real is the hard road; making the model indifferent to appearance is the
    cheap one, and femoral heads are high-contrast and geometrically stereotyped, which
    is the easy end of the problem. Nothing here moves a pixel, so the mask stays valid.
    """
    x = np.clip(np.asarray(img, np.float32), 0, 1)
    if rng.random() < 0.9:
        x = x ** float(rng.uniform(0.5, 2.0))                      # gamma
    if rng.random() < 0.8:
        x = np.clip((x - 0.5) * float(rng.uniform(0.6, 1.8)) + 0.5, 0, 1)
    if rng.random() < 0.25:
        x = 1.0 - x                                                 # contrast inversion
    if rng.random() < 0.5:                                          # scatter / veiling glare
        from scipy.ndimage import gaussian_filter
        x = np.clip(0.75 * x + 0.25 * gaussian_filter(x, float(rng.uniform(6, 20))), 0, 1)
    if rng.random() < 0.6:
        x = np.clip(x + rng.normal(0, float(rng.uniform(0.005, 0.05)), x.shape), 0, 1)
    if rng.random() < 0.2:                                          # anti-scatter grid lines
        per = float(rng.uniform(3.0, 9.0))
        amp = float(rng.uniform(0.02, 0.08))
        cols = np.arange(x.shape[1])
        x = np.clip(x * (1.0 + amp * np.sin(2 * np.pi * cols / per))[None, :], 0, 1)
    if rng.random() < 0.3:                                          # random collimation
        H, W = x.shape
        t, b = int(rng.uniform(0, 0.06) * H), int(rng.uniform(0, 0.06) * H)
        l, r = int(rng.uniform(0, 0.06) * W), int(rng.uniform(0, 0.06) * W)
        x[:t], x[H - b:], x[:, :l], x[:, W - r:] = 0, 0, 0, 0
    return x.astype(np.float32)


class FemHeadDataset:
    """DRR + projected femoral-head mask, for a binary segmenter.

    Returns (image [1,H,W], mask [1,H,W], meta). Dense supervision throughout -- there is
    no masking question here, unlike the landmark path, because every DRR that has femora
    in the FOV has a full head mask.
    """

    def __init__(self, rows: Sequence[Dict], *, out_size=(256, 128), augment: bool = True,
                 seed: int = 0):
        self.rows = list(rows)
        self.out_size = tuple(out_size)
        self.augment = bool(augment)
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        from scipy.ndimage import zoom
        r = self.rows[i]
        img = np.load(r["npy"]).astype(np.float32)
        msk = np.load(r["head"]).astype(np.float32)
        H, W = self.out_size
        H0, W0 = img.shape
        if (H, W) != (H0, W0):
            img = zoom(img, (H / H0, W / W0), order=1)
            msk = zoom(msk, (H / H0, W / W0), order=0)
        if self.augment:
            img = xray_appearance(img, self._rng)
        return (torch.from_numpy(img[None].astype(np.float32)),
                torch.from_numpy((msk[None] > 0.5).astype(np.float32)),
                {"case": r.get("case", ""), "view": r.get("view", "")})
