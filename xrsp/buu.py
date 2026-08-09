"""BUU-LSpine 400 as a landmark dataset, in the SAME channel space as the DRRs.

Why this exists
---------------
BUU-LSpine ships radiologist-placed vertebral corners on 400 real lateral radiographs:
per vertebra a superior and an inferior endplate for L1..L5, plus the S1 superior
endplate (`S1a`) and — notably — no `S1b`, since S1 is fused to S2 and has no inferior
plate to mark. That is real ground truth in the real domain, which the DRRs are not.

What it does NOT have is the bicoxofemoral point, so PI and PT cannot be computed from it
alone. The DRRs have that landmark and BUU has the domain, so the two are complementary
rather than alternatives — and the right way to combine them is ONE model over the union
with a per-channel loss mask, not a pseudo-label round trip:

  * a DRR sample supervises every channel — corners AND bicoxofemoral;
  * a BUU sample supervises corners only, and the bicoxofemoral channel is masked out.

`heatmaps.gaussian_heatmaps` already returns exactly that mask (it exists because scans
differ in how many levels they show), so nothing new is needed to express "this landmark
is unlabelled here" as distinct from "this landmark is absent here". Supervising an
unlabelled point with a zero target is the standard way partial annotation silently
poisons a landmark model.

A caveat worth stating plainly: under this scheme the bicoxofemoral channel still
receives NO real-domain gradient — it is learned from DRRs alone and cannot be validated
on BUU, which has no ground truth for it. A two-stage pseudo-label pipeline has the same
hole and additionally compounds the error. The cheap fix is neither: hand-annotate the
femoral head on a subset of BUU. On a true lateral both heads superimpose into a single
blob, so it is one click per case, and it converts the weakest landmark in the chain into
a supervised and independently validatable one. `femoral_csv` below loads such a file
when you have one.

Geometry
--------
BUU images carry ANTERIOR ON THE LEFT; the DRRs put it on the right (see
oblique._grid). A landmark model does not care which convention it is trained in, but it
does care that there is only one, so BUU is mirrored into the DRR convention here.
Determining that was not cosmetic: the sacral endplate slopes DOWN AND FORWARD, so the
promontory — the anterior corner — is the LOWER of the two annotated points, which makes
BUU's FIRST point anterior. Getting it backwards flips the sign of every segmental disc
angle while leaving LL, wedge and aspect (all unsigned) looking perfectly correct.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .heatmaps import FEMORAL_KEY, channel_names, gaussian_heatmaps

# CSV row order, cranial -> caudal. 11 rows: L1sup, L1inf, ... L5inf, S1sup.
BUU_ROWS: List[tuple] = ([(f"L{i}", f) for i in range(1, 6) for f in ("sup", "inf")]
                         + [("S1", "sup")])
BUU_LEVELS: List[str] = [f"L{i}" for i in range(1, 6)] + ["S1"]


def index_buu(root: str, view: str = "LA") -> List[Dict]:
    """All (image, annotation) pairs under `root`/<view>. Pairs are matched by stem, and
    an image without a readable annotation is skipped rather than silently unlabelled."""
    rows = []
    for jpg in sorted(glob.glob(os.path.join(root, view, "*.jpg"))):
        csv = jpg[:-4] + ".csv"
        if os.path.exists(csv):
            rows.append({"case": os.path.basename(jpg)[:-4], "view": view,
                         "img": jpg, "csv": csv})
    return rows


def load_corners(csv_path: str, *, image_width: Optional[int] = None) -> Optional[Dict]:
    """{channel_name: [x, y]} for one BUU case, in the DRR's anterior-right convention.

    Returns None if the file does not carry the full 11-row L1..S1 chain, rather than a
    partial dict that would look like missing landmarks instead of a malformed file.
    """
    a = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    if len(a) < len(BUU_ROWS):
        return None
    out: Dict[str, List[float]] = {}
    for i, (lv, face) in enumerate(BUU_ROWS):
        p_ant, p_post = a[i, 0:2].astype(float), a[i, 2:4].astype(float)
        for key, p in ((f"{lv}.{face}_ant", p_ant), (f"{lv}.{face}_post", p_post)):
            x, y = float(p[0]), float(p[1])
            if image_width is not None:            # mirror into the DRR convention
                x = float(image_width - 1 - x)
            out[key] = [x, y]
    return out


def load_femoral(femoral_csv: Optional[str], case: str,
                 *, image_width: Optional[int] = None) -> Optional[List[float]]:
    """Hand-annotated bicoxofemoral point for `case`, if one exists.

    Expects rows of `case,x,y` in ORIGINAL image pixels. This is the file to produce if
    you annotate the femoral heads yourself; without it the channel is simply masked out
    for every BUU sample and learned from the DRRs alone.
    """
    if not femoral_csv or not os.path.exists(femoral_csv):
        return None
    try:
        with open(femoral_csv, encoding="utf-8") as fh:
            for line in fh:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3 and parts[0] == case:
                    x, y = float(parts[1]), float(parts[2])
                    if image_width is not None:
                        x = float(image_width - 1 - x)
                    return [x, y]
    except Exception:                                          # noqa: BLE001
        return None
    return None


class BUULandmarkDataset:
    """BUU laterals with the same (image, heatmaps, valid, meta) contract as the DRR set.

    Channels are the SHARED set, so a BUU batch and a DRR batch are interchangeable to
    the model. Any channel BUU does not annotate — every thoracic level, and the
    bicoxofemoral point unless `femoral_csv` supplies it — is left absent and therefore
    masked out of the loss, not supervised with zeros.
    """

    def __init__(self, rows: Sequence[Dict], *, levels: Sequence[str],
                 out_size=(512, 256), sigma: float = 3.0, augment: bool = False,
                 seed: int = 0, femoral_csv: Optional[str] = None, mirror: bool = True,
                 p_flip: float = 0.5, max_rot_deg: float = 0.0):
        self.rows = list(rows)
        self.levels = list(levels)
        self.names = channel_names(self.levels)
        self.out_size = tuple(out_size)
        self.sigma = float(sigma)
        self.augment = bool(augment)
        self.femoral_csv = femoral_csv
        self.mirror = bool(mirror)
        self.p_flip = float(p_flip)
        self.max_rot_deg = float(max_rot_deg)
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def channel_names(self):
        return list(self.names)

    def _appearance(self, img):
        r = self._rng
        if r.random() < 0.8:
            img = np.clip(img, 0, 1) ** float(r.uniform(0.7, 1.5))
        if r.random() < 0.8:
            img = np.clip((img - 0.5) * float(r.uniform(0.8, 1.3)) + 0.5, 0, 1)
        if r.random() < 0.5:
            img = np.clip(img + r.normal(0, float(r.uniform(0.005, 0.03)), img.shape), 0, 1)
        return img.astype(np.float32)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        from scipy.ndimage import zoom

        row = self.rows[i]
        im = Image.open(row["img"]).convert("L")
        W0, H0 = im.size
        img = np.asarray(im, dtype=np.float32) / 255.0
        if self.mirror:
            img = img[:, ::-1].copy()
        pts = load_corners(row["csv"], image_width=W0 if self.mirror else None) or {}
        fem = load_femoral(self.femoral_csv, row["case"],
                           image_width=W0 if self.mirror else None)
        if fem is not None:
            pts[FEMORAL_KEY] = fem

        H, W = self.out_size
        # LETTERBOX: one uniform scale for both axes, then pad. NOT independent sy/sx.
        #
        # The previous version scaled x and y separately to fill 512x256 exactly, which
        # distorts every angle in the image. Measured on this dataset: source aspect ratio
        # has a median of 0.799 against a 0.500 target, so the anisotropy is ~0.625 and a
        # true 45 deg endplate is rendered at 58 deg -- by a DIFFERENT amount on every
        # film (anisotropy ranged 0.499 to 1.219 across the test set). Agreement metrics
        # survived it, because prediction and truth were distorted identically, but every
        # absolute SS and LL was wrong and the Roussouly thresholds at 35/45 deg were
        # being applied to angles that had been stretched.
        #
        # Padding is added symmetrically so the anatomy stays centred, and the landmarks
        # are shifted by the same offset. The padded border is 0 (black), which matches
        # the collimated border already present on these films.
        s = min(H / H0, W / W0)
        nh, nw = max(1, int(round(H0 * s))), max(1, int(round(W0 * s)))
        img = zoom(img, (nh / H0, nw / W0), order=1)
        oy, ox = (H - nh) // 2, (W - nw) // 2
        canvas = np.zeros((H, W), dtype=np.float32)
        canvas[oy:oy + nh, ox:ox + nw] = img[:nh, :nw]
        img = canvas
        pts = {k: [v[0] * s + ox, v[1] * s + oy] for k, v in pts.items()}
        sx = sy = s
        if self.augment:
            # ORIENTATION first, then appearance. Both the film's handedness and any
            # rotation move the LANDMARKS, so they have to be applied where the points
            # are still explicit -- after the heatmaps are rendered it is too late.
            from .geom_aug import augment_orientation
            img, pts = augment_orientation(img, pts, self._rng,
                                           p_flip=self.p_flip,
                                           max_rot_deg=self.max_rot_deg)
            img = self._appearance(img)
        hm, valid = gaussian_heatmaps(pts, (H, W), sigma=self.sigma, names=self.names)
        return (torch.from_numpy(img[None].astype(np.float32)),
                torch.from_numpy(hm),
                torch.from_numpy(valid),
                {"case": row["case"], "view": row["view"], "source": "buu",
                 "scale_xy": [float(sx), float(sy)], "pixel_spacing_mm": 0.0})


class UnionDataset:
    """DRR + BUU as one dataset over one channel set.

    Deliberately a plain concatenation with a per-source sampling weight rather than
    anything cleverer: the ONLY coupling the two sources need is the shared channel order
    and the loss mask, both of which are already in place. `drr_weight` repeats the DRR
    rows when the two sets are very different sizes, so a few hundred generated views are
    not drowned by 400 radiographs (or the reverse).
    """

    def __init__(self, drr_ds, buu_ds, *, drr_weight: int = 1, buu_weight: int = 1):
        self.parts = []
        for ds, w in ((drr_ds, int(drr_weight)), (buu_ds, int(buu_weight))):
            if ds is None:
                continue
            for _ in range(max(1, w)):
                self.parts.append(ds)
        self._index = [(pi, i) for pi, ds in enumerate(self.parts) for i in range(len(ds))]

    def __len__(self):
        return len(self._index)

    def __getitem__(self, k):
        pi, i = self._index[k]
        return self.parts[pi][i]

    def channel_names(self):
        return self.parts[0].channel_names() if self.parts else []
