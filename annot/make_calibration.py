"""Build the calibration film: one image with a KNOWN answer.

Everything else in this study measures readers against each other. That leaves the one
question nobody can answer from the ledger: are they both wrong together? Inter-reader
agreement is happily consistent about a centre that is 3 mm medial, and no amount of it
would show that.

So one film has ground truth. It is the DRR of a segmented CT, where the femoral head is a
sphere fitted to the articular surface at its contact with the acetabulum, and A, S and P
are the extremes of that sphere projected -- the same construction the teaching figure
uses, and an objectively correct answer rather than an experienced reader's opinion.

Both femoral heads superimpose exactly on this view, because a DRR is integrated along the
bicoxofemoral axis, so the reader marks ONE head and its centre IS the hip point. That
makes it the cleanest possible calibration target: no pairing, no midpoint, one number.

    python annot/make_calibration.py        # writes calib_film.jpg + calib_truth.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_reference import (CT, WEB, head_surface, landmarks_px,   # noqa: E402
                            plan_for_saved, circumcircle)

OUT_IMG = HERE / "calib_film.jpg"
OUT_TRUTH = HERE / "calib_truth.json"


def main():
    import nibabel as nib
    si = nib.load(str(CT / "seg.nii.gz"))
    seg = np.asanyarray(si.dataobj).astype(np.int16)
    aff = si.affine

    meta = json.loads((WEB / "metrics.json").read_text())
    spacing = float(meta["geometry"]["drr"]["pixel_spacing_mm"])
    src = Image.open(WEB / "image.png").convert("L")
    plan, origin, ant, cranial = plan_for_saved(seg, aff, spacing)
    if list(plan["shape"]) != [src.height, src.width]:
        raise SystemExit(f"framing drift: {plan['shape']} vs "
                         f"{[src.height, src.width]}")

    cL, rL, shL, rms = head_surface(seg, aff, "left")
    lm, ctr, near = landmarks_px(cL, rL, shL, plan, origin, ant, cranial)
    (fx, fy), Rpx = circumcircle(lm["A"], lm["S"], lm["P"])

    W, H = src.width, src.height
    truth = {
        "w": W, "h": H,
        "mm_per_px": spacing,
        # normalised the same way a reader's submission is, so scoring is a subtraction
        "centre": [fx / W, fy / H],
        "radius": Rpx / W,
        "landmarks": {k: [v[0] / W, v[1] / H] for k, v in lm.items()},
        # anterior is image-left on this render: sagittal_drr_from_label ends with
        # img[::-1, ::-1], which is superior up and anterior left
        "facing": "left",
        "source": "DRR of a segmented CT; femoral head = sphere fitted to the articular "
                  "surface at its contact with the acetabulum, projected",
        "sphere_fit_rms_mm": round(rms, 2),
        "nearest_surface_voxel_mm": {k: round(v, 2) for k, v in near.items()},
        "note": "Both femoral heads superimpose on this projection, so ONE marked head is "
                "correct and its centre is the bicoxofemoral point itself.",
    }
    src.convert("RGB").save(OUT_IMG, quality=92, optimize=True)
    OUT_TRUTH.write_text(json.dumps(truth, indent=1))
    print(f"  film   {OUT_IMG.name}  {W}x{H}  "
          f"{OUT_IMG.stat().st_size/1000:.0f} kB")
    print(f"  centre {fx:.1f},{fy:.1f} px   radius {Rpx:.1f} px "
          f"({Rpx*spacing:.1f} mm)")
    for k in ("A", "S", "P"):
        print(f"    {k}  {lm[k][0]:7.1f},{lm[k][1]:7.1f} px   "
              f"nearest real surface voxel {near[k]:.2f} mm")
    print(f"  sphere fit rms {rms:.2f} mm")
    print(f"  wrote {OUT_TRUTH.name}")


if __name__ == "__main__":
    main()
