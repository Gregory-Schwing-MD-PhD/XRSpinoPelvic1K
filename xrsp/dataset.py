"""Torch dataset over generated DRR views.

Reads what `xrsp.build_dataset.build_case_oblique` writes:

    <case>/<view>_drr.npy        float DRR in [0, 1]
    <case>/<view>_corners.json   4 corners per present level + bicoxofemoral point

Geometry augmentation deliberately does NOT live here. Rotation/obliquity is applied
at GENERATION time, where the labels are re-projected through the same plan and stay
exact; warping an image in the loader would require warping the landmarks too, and
any mismatch there is silent. The loader does appearance only.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .heatmaps import LEVELS, channel_names, gaussian_heatmaps, points_from_json


def index_views(data_root: str, cases: Optional[Sequence[str]] = None) -> List[Dict]:
    """All (case, view) pairs under `data_root`, optionally restricted to `cases`."""
    keep = set(cases) if cases is not None else None
    rows = []
    for cj in sorted(glob.glob(os.path.join(data_root, "*", "*_corners.json"))):
        case = os.path.basename(os.path.dirname(cj))
        if keep is not None and case not in keep:
            continue
        view = os.path.basename(cj).replace("_corners.json", "")
        npy = os.path.join(os.path.dirname(cj), f"{view}_drr.npy")
        if os.path.exists(npy):
            rows.append({"case": case, "view": view, "npy": npy, "json": cj})
    return rows


def levels_present(rows: Sequence[Dict]) -> List[str]:
    """The UNION of vertebral levels actually annotated anywhere in `rows`, cranial->caudal.

    The channel set is derived from the DATA, never hardcoded. How many vertebrae a scan
    shows varies with the acquisition -- one case here has T12, T13, L1-L5, S1; another
    may reach T8, or include L6 -- and every level that is visible must get its four
    corners. A fixed lumbar-only list would silently drop the thoracic levels the FOV
    does contain, and a fixed long list would allocate channels that are never populated.
    Per-view masking (see gaussian_heatmaps) then handles the per-scan variation.
    """
    seen = set()
    for r in rows:
        try:
            meta = json.loads(open(r["json"]).read())
        except Exception:                                     # noqa: BLE001
            continue
        seen.update((meta.get("endplate_corners") or {}).keys())
    order = {lv: i for i, lv in enumerate(LEVELS)}
    return sorted((lv for lv in seen if lv in order), key=lambda lv: order[lv])


class LandmarkDRRDataset:
    """(image [1,H,W], heatmaps [C,H,W], valid [C], meta) for one generated view."""

    def __init__(self, rows: Sequence[Dict], *, out_size=(512, 256), sigma: float = 3.0,
                 augment: bool = False, levels: Optional[Sequence[str]] = None,
                 seed: int = 0):
        self.rows = list(rows)
        self.out_size = tuple(out_size)
        self.sigma = float(sigma)
        self.augment = bool(augment)
        self.names = channel_names(levels) if levels else channel_names()
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def channel_names(self):
        return list(self.names)

    def _appearance(self, img):
        """Appearance-only augmentation. Nothing here moves a pixel, so the landmark
        coordinates stay valid without being transformed."""
        r = self._rng
        if r.random() < 0.8:                                  # gamma / contrast
            img = np.clip(img, 0, 1) ** float(r.uniform(0.7, 1.5))
        if r.random() < 0.8:
            img = np.clip((img - 0.5) * float(r.uniform(0.8, 1.3)) + 0.5, 0, 1)
        if r.random() < 0.5:                                  # detector noise
            img = np.clip(img + r.normal(0, float(r.uniform(0.005, 0.03)), img.shape), 0, 1)
        if r.random() < 0.3:                                  # mild blur (focal spot)
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, float(r.uniform(0.4, 1.2)))
        return img.astype(np.float32)

    def __getitem__(self, i):
        import torch
        from scipy.ndimage import zoom
        row = self.rows[i]
        img = np.load(row["npy"]).astype(np.float32)
        meta = json.loads(open(row["json"]).read())
        pts = points_from_json(meta.get("endplate_corners"), meta.get("bicoxofemoral_px"),
                               self.names)
        H0, W0 = img.shape
        H, W = self.out_size
        sy, sx = H / H0, W / W0
        if (H, W) != (H0, W0):
            img = zoom(img, (sy, sx), order=1)
            pts = {k: (None if v is None else [v[0] * sx, v[1] * sy]) for k, v in pts.items()}
        if self.augment:
            img = self._appearance(img)
        hm, valid = gaussian_heatmaps(pts, (H, W), sigma=self.sigma, names=self.names)
        return (torch.from_numpy(img[None].astype(np.float32)),
                torch.from_numpy(hm),
                torch.from_numpy(valid),
                {"case": row["case"], "view": row["view"],
                 "scale_xy": [float(sx), float(sy)],
                 "pixel_spacing_mm": float(
                     (meta.get("geometry") or {}).get("pixel_spacing_mm", 1.0))})


def collate(batch):
    """Default collate, keeping `meta` as a list of dicts."""
    import torch
    imgs, hms, valids, metas = zip(*batch)
    return (torch.stack(imgs), torch.stack(hms), torch.stack(valids), list(metas))
