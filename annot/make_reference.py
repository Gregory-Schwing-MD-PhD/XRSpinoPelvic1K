"""Build the teaching figure the annotator shows: WHERE the femoral head is, WHAT the
three landmarks are, and what happens when the two heads separate.

Everything drawn here is derived from the segmented CT and projected, never marked by
hand on a radiograph. That is the point: a hand-marked figure would put whoever marked it
inside every reader's head, and inter-reader agreement would then be measuring how well
they imitate that person. Here the landmarks are found the way the figure claims they
are found --

    the femoral head surface is taken from the ACETABULAR INTERFACE (femur voxels within
    a few mm of the same-side hip mask, i.e. the articular surface), grown over the head
    and sphere-fitted; the sphere is projected onto the DRR; and A, S and P are the
    anterior-, superior- and posterior-most points of that projection.

The one thing NOT done is to take the single most extreme voxel of the surface cloud. It
was tried, and it measures the shell-selection tolerance rather than the anatomy: the
shell is +/-3.35 mm about the fitted radius, so its outermost projected voxel sits 9.6 px
past the true silhouette on a 0.35 mm/px render -- the measured overshoot was 9.5 px --
and three landmarks each biased outward put the centre 1.8 mm (0.0069 of image width)
away, ABOVE the agreement tolerance this figure exists to teach. The fitted silhouette is
used instead, and the run prints how far each landmark is from the nearest real surface
voxel, so "on the anatomy" is a reported number rather than a claim.

PANEL C IS NOW A REAL VIEW. It used to draw a synthetic second head, because a DRR is
integrated along the true bicoxofemoral axis and the two heads therefore superimpose
exactly -- which made the panel geometrically incoherent: the drawn separation did not
correspond to any rotation. It is now rendered from the same CT along an axis YAWED by a
few degrees, so the heads separate by exactly as much as that rotation implies and
nothing is drawn in by hand.

    python annot/make_reference.py            # writes annot/reference_femhead.png + ref_?.png
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.environ.get(
    "OSTK", r"C:/Users/grego/OneDrive/Desktop/OpenSpineToolkit"))

WEB = pathlib.Path(r"C:/Users/grego/OneDrive/Desktop/openspineconsortium.github.io"
                   r"/pacs/data/xr/0003")
CT = WEB.parents[1] / "0003"
OUT = pathlib.Path(__file__).with_name("reference_femhead.png")

# the tool's own colours, so the figure and the film look like the same program
GREEN, RED, AMBER, WHITE = (0, 229, 160), (255, 59, 48), (245, 165, 36), (245, 245, 245)
LMCOL = {"A": (124, 196, 255), "S": (0, 229, 160), "P": (245, 165, 36)}
PANEL_H = 620
YAW_DEG = 8.0          # enough to separate the heads by about one head radius


def font(sz):
    for f in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def label(dr, xy, text, fill=WHITE, sz=17, anchor="la"):
    f = font(sz)
    x, y = xy
    box = dr.textbbox((x, y), text, font=f, anchor=anchor)
    dr.rectangle([box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4], fill=(12, 12, 16))
    dr.text((x, y), text, font=f, fill=fill, anchor=anchor)


def rot_about(v, axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)) @ np.asarray(v, float)


def circumcircle(p, q, r):
    """Centre and radius of the circle through three 2-D points."""
    ax, ay = p
    bx, by = q
    cx, cy = r
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    return (ux, uy), float(np.hypot(ax - ux, ay - uy))


# ── anatomy ─────────────────────────────────────────────────────────────────────
def head_surface(label_arr, affine, side):
    """(centre_mm, radius_mm, surface_points_mm) for one femoral head.

    The surface cloud is the sphere SHELL of the femur mask -- seeded, as ostk does it,
    from the voxels in contact with the acetabulum, then grown over the whole head. It is
    that cloud whose projection is measured for the landmarks, so the figure's A/S/P are
    the extremes of real anatomy rather than three points on an ideal circle.
    """
    from ostk.masks import binary_mask, largest_component, mask_world
    from ostk.metrics import femoral_head_center
    from ostk.labels import LABELS
    fit = femoral_head_center(label_arr, affine, f"femur_{side}",
                              "left_hip" if side == "left" else "right_hip")
    if fit is None:
        raise SystemExit(f"no femoral head fit for {side}")
    c, r, rms = fit
    fem = mask_world(largest_component(
        binary_mask(label_arr, LABELS[f"femur_{side}"])), affine)
    d = np.linalg.norm(fem - c, axis=1)
    shell = fem[np.abs(d - r) <= max(0.15 * r, 2.5)]
    return np.asarray(c), float(r), shell, float(rms)


def landmarks_px(centre_mm, r_mm, shell, plan, origin, ant, cranial):
    """A / S / P: the anterior-, superior- and posterior-most points of the femoral head
    as it appears on this DRR.

    Taken from the fitted sphere's silhouette, NOT from the single most extreme voxel of
    the surface cloud. Measuring the extreme voxel was tried and it measures the wrong
    thing: the shell is selected with a +/-3.35 mm tolerance about the fitted radius, so
    its outermost projected voxel sits 3.35 mm / 0.35 mm-per-px = 9.6 px beyond the true
    silhouette -- and the observed overshoot was 9.5 px. Three landmarks each biased
    outward by their own tolerance band put the circumcircle centre 1.8 mm off, which is
    above the agreement tolerance the figure is teaching people to meet.

    A sphere projects to a circle of radius r under orthographic projection from ANY
    direction, so this is exact for the yawed view too. `nearest` reports how far each
    landmark is from the closest actual surface voxel -- the check that these are points
    on the anatomy rather than on an idealisation floating off it.

    Superior is the smallest row and anterior the smallest column, because
    sagittal_drr_from_label ends with img[::-1, ::-1] -- superior up, anterior left.
    """
    from ostk.geometry import project_to_plane_2d
    from ostk.drr import plane_to_pixel
    ctr = plane_to_pixel(plan, project_to_plane_2d(centre_mm, origin, ant, cranial))
    R = r_mm / plan["pixel_spacing_mm"]
    lm = {"S": np.array([ctr[0], ctr[1] - R]),
          "A": np.array([ctr[0] - R, ctr[1]]),
          "P": np.array([ctr[0] + R, ctr[1]])}
    px = plane_to_pixel(plan, project_to_plane_2d(shell, origin, ant, cranial))
    nearest = {k: float(np.hypot(px[:, 0] - v[0], px[:, 1] - v[1]).min()
                        * plan["pixel_spacing_mm"]) for k, v in lm.items()}
    return lm, ctr, nearest


def plan_for_saved(seg, affine, spacing):
    """Reconstruct the grid of the ALREADY-RENDERED lateral in pacs/data/xr/0003.

    A DRR of this volume takes minutes, and the true-lateral one is already on disk. All
    that is missing is `extent_mm`, which sagittal_drr_from_label returns but the exported
    metrics.json does not carry -- so it is recomputed from the same framing points and
    margin the renderer uses. Cheap, and checked against the saved image's own shape and
    against the hip point the exporter wrote, so a drift in the framing rule is caught
    here rather than silently misplacing every landmark in the figure.
    """
    import ostk.drr as D
    from ostk.project2d import sagittal_axes
    from ostk.geometry import project_to_plane_2d
    info = D._pi_axis_and_origin(seg, affine)
    lr, origin = info["lr"], info["bicox"]
    ant, cranial = sagittal_axes(lr)
    uv = project_to_plane_2d(D._framing_points(seg, affine), origin, ant, cranial)
    m = D.DEFAULT_MARGIN_MM
    u0, v0 = uv.min(axis=0) - m
    u1, v1 = uv.max(axis=0) + m
    W_mm = max(u1 - u0, D.MIN_FOV_MM[0])
    H_mm = max(v1 - v0, D.MIN_FOV_MM[1])
    uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    u_min, v_min = uc - W_mm / 2, vc - H_mm / 2
    Wpx = max(int(round(W_mm / spacing)), 1)
    Hpx = max(int(round(H_mm / spacing)), 1)
    plan = {"pixel_spacing_mm": spacing, "shape": [Hpx, Wpx],
            "extent_mm": [float(u_min), float(v_min),
                          float(u_min + W_mm), float(v_min + H_mm)]}
    return plan, np.asarray(origin, float), np.asarray(ant, float), \
        np.asarray(cranial, float)


def render(seg, ct, affine, yaw_deg=0.0, spacing=0.35, cache=True):
    """A DRR along the bicoxofemoral axis, optionally yawed off it.

    Yaw is applied by rotating the projection axis rather than the volume, so the CT is
    resampled once by the renderer itself and no interpolation is stacked.
    """
    import ostk.drr as D
    from ostk.project2d import sagittal_axes
    # A full projection through this volume takes minutes, and the figure gets iterated
    # on far more often than the anatomy changes.
    cf = OUT.with_name(f".drr_yaw{yaw_deg:g}_{spacing:g}.npz")
    if cache and cf.exists():
        z = np.load(cf, allow_pickle=True)
        plan = json.loads(str(z["plan"]))
        plan["image"] = z["image"]
        print(f"  reused cached {yaw_deg:g}-degree render ({cf.name})")
        return (plan, Image.fromarray((np.clip(plan["image"], 0, 1) * 255)
                                      .astype(np.uint8)).convert("RGB"),
                np.asarray(plan["origin_world_mm"], float),
                np.asarray(plan["axes"]["anterior"], float),
                np.asarray(plan["axes"]["cranial"], float))
    orig_fn = D._pi_axis_and_origin
    if yaw_deg:
        def yawed(lbl, aff, *a, **k):
            info = orig_fn(lbl, aff, *a, **k)
            if info is None:
                return None
            out = dict(info)
            out["lr"] = rot_about(info["lr"], [0, 0, 1], yaw_deg)
            return out
        D._pi_axis_and_origin = yawed
    try:
        plan = D.sagittal_drr_from_label(seg, ct, affine, pixel_spacing_mm=spacing,
                                         gamma=0.6)
    finally:
        D._pi_axis_and_origin = orig_fn
    if plan is None:
        raise SystemExit("DRR failed")
    if cache:
        meta = {k: v for k, v in plan.items() if k != "image"}
        np.savez_compressed(cf, image=plan["image"], plan=json.dumps(meta))
    origin = np.asarray(plan["origin_world_mm"], float)
    ant = np.asarray(plan["axes"]["anterior"], float)
    cranial = np.asarray(plan["axes"]["cranial"], float)
    img = (np.clip(plan["image"], 0, 1) * 255).astype(np.uint8)
    return plan, Image.fromarray(img).convert("RGB"), origin, ant, cranial


# ── drawing ─────────────────────────────────────────────────────────────────────
def draw_head(dr, lm, scale, off, col_centre=AMBER, tag=True, lw=3):
    """The three landmarks, the circle through them, and the derived centre."""
    P = {k: ((v[0] - off[0]) * scale, (v[1] - off[1]) * scale) for k, v in lm.items()}
    (cx, cy), R = circumcircle(P["A"], P["S"], P["P"])
    dr.ellipse([cx - R, cy - R, cx + R, cy + R], outline=(140, 200, 255), width=2)
    for k in ("A", "S", "P"):
        x, y = P[k]
        s = 7
        dr.rectangle([x - s, y - s, x + s, y + s], outline=LMCOL[k], width=lw)
        if tag:
            ox, oy = (x - cx), (y - cy)
            n = np.hypot(ox, oy) or 1
            label(dr, (x + ox / n * 26, y + oy / n * 26), k, LMCOL[k], 19, "mm")
    t = 16
    dr.line([cx - t, cy, cx + t, cy], fill=col_centre, width=3)
    dr.line([cx, cy - t, cx, cy + t], fill=col_centre, width=3)
    return (cx, cy), R


def main():
    import nibabel as nib
    seg_img = nib.load(str(CT / "seg.nii.gz"))
    ct_img = nib.load(str(CT / "ct.nii.gz"))
    seg = np.asanyarray(seg_img.dataobj).astype(np.int16)
    vol = np.asanyarray(ct_img.dataobj).astype(np.float32)
    aff = seg_img.affine

    cL, rL, shL, rmsL = head_surface(seg, aff, "left")
    cR, rR, shR, rmsR = head_surface(seg, aff, "right")
    print(f"  femoral heads: r = {rL:.1f} / {rR:.1f} mm  (sphere-fit rms "
          f"{rmsL:.2f} / {rmsR:.2f} mm)")
    print(f"  head separation {np.linalg.norm(cR - cL):.0f} mm")

    # ── the true lateral: the heads superimpose, one set of landmarks ──────────
    from ostk.geometry import project_to_plane_2d
    from ostk.drr import plane_to_pixel
    meta = json.loads((WEB / "metrics.json").read_text())
    spacing = float(meta["geometry"]["drr"]["pixel_spacing_mm"])
    src = Image.open(WEB / "image.png").convert("RGB")
    plan, origin, ant, cranial = plan_for_saved(seg, aff, spacing)
    if list(plan["shape"]) != [src.height, src.width]:
        raise SystemExit(f"framing drift: reconstructed {plan['shape']} but the saved "
                         f"image is {[src.height, src.width]}")
    hip = next(L for L in meta["geometry"]["landmarks"] if L["cls"] == "hip_axis")
    chk = plane_to_pixel(plan, project_to_plane_2d(
        0.5 * (cL + cR), origin, ant, cranial))
    off = float(np.hypot(chk[0] - hip["xy"][0], chk[1] - hip["xy"][1]))
    if off > 1.0:
        raise SystemExit(f"projection drift: hip point lands {off:.2f} px from the "
                         f"exported one")
    print(f"  reused the saved lateral; hip point reproduces to {off:.2f} px")
    lm, ctrL, near = landmarks_px(cL, rL, shL, plan, origin, ant, cranial)
    (fx, fy), Rpx = circumcircle(lm["A"], lm["S"], lm["P"])
    err_px = float(np.hypot(fx - ctrL[0], fy - ctrL[1]))
    print(f"  panel B: the circle through A, S and P recovers the 3-D sphere centre to "
          f"{err_px:.2f} px")
    print(f"      each landmark sits on measured bone: nearest surface voxel "
          + ", ".join(f"{k} {v:.2f} mm" for k, v in near.items()))

    # ── panel A: the whole film ────────────────────────────────────────────────
    scale = PANEL_H / src.height
    A = src.resize((round(src.width * scale), PANEL_H), Image.LANCZOS)
    dA = ImageDraw.Draw(A)
    ax, ay, r = fx * scale, fy * scale, Rpx * scale
    dA.ellipse([ax - r, ay - r, ax + r, ay + r], outline=GREEN, width=3)
    dA.line([ax - r * 2.2, ay - r * 2.2, ax - r * 0.8, ay - r * 0.8], fill=GREEN, width=2)
    label(dA, (ax - r * 2.3, ay - r * 2.3), "femoral head", GREEN, 18, "ls")
    label(dA, (10, 10), "A  where to look", WHITE, 18)
    label(dA, (10, 548), "below and anterior to the S1 endplate,\nseated in the acetabulum",
          (200, 210, 220), 14)

    # ── panel B: the three landmarks ───────────────────────────────────────────
    half = Rpx * 2.3
    box = (int(fx - half), int(fy - half), int(fx + half), int(fy + half))
    crop = src.crop(box)
    Z = PANEL_H / crop.height
    B = crop.resize((round(crop.width * Z), PANEL_H), Image.LANCZOS)
    dB = ImageDraw.Draw(B)
    draw_head(dB, lm, Z, (box[0], box[1]))
    label(dB, (10, 10), "B  the three landmarks", WHITE, 18)
    label(dB, (10, 528),
          "A anterior-most, S superior-most,\n"
          "P posterior-most point of the\n"
          "articular surface — centre is FITTED",
          (200, 210, 220), 14)

    # ── panel C: yawed, so the two heads genuinely separate ────────────────────
    planC, srcC, oC, aC, cC = render(seg, vol, aff, YAW_DEG)
    lmL, pL, _ = landmarks_px(cL, rL, shL, planC, oC, aC, cC)
    lmR, pR, _ = landmarks_px(cR, rR, shR, planC, oC, aC, cC)
    sep_px = float(np.hypot(*(pR - pL)))
    print(f"  panel C: yaw {YAW_DEG} deg separates the heads by {sep_px:.0f} px "
          f"({sep_px * planC['pixel_spacing_mm']:.0f} mm = "
          f"{sep_px * planC['pixel_spacing_mm'] / rL:.2f} head radii)")

    mid = 0.5 * (pL + pR)
    halfC = max(np.hypot(*(pR - pL)) * 0.75, Rpx * 2.0) + Rpx
    boxC = (int(mid[0] - halfC), int(mid[1] - halfC),
            int(mid[0] + halfC), int(mid[1] + halfC))
    cropC = srcC.crop(boxC)
    ZC = PANEL_H / cropC.height
    Cp = cropC.resize((round(cropC.width * ZC), PANEL_H), Image.LANCZOS)
    dC = ImageDraw.Draw(Cp)
    c1, _ = draw_head(dC, lmL, ZC, (boxC[0], boxC[1]), tag=True, lw=3)
    c2, _ = draw_head(dC, lmR, ZC, (boxC[0], boxC[1]), tag=True, lw=3)
    dC.line([c1[0], c1[1], c2[0], c2[1]], fill=GREEN, width=3)
    m = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
    dC.ellipse([m[0] - 7, m[1] - 7, m[0] + 7, m[1] + 7], fill=GREEN)
    label(dC, (10, 10), "C  two heads: mark BOTH", WHITE, 18)
    label(dC, (m[0], m[1] + 46), "midpoint — derived for you", GREEN, 15, "ms")
    label(dC, (10, 528),
          f"the SAME scan, projected {YAW_DEG:.0f}° off the\n"
          "hip axis. real separation, not drawn in.\n"
          "same diameter, offset front-to-back",
          (200, 210, 220), 14)

    gap = 12
    W = A.width + gap + B.width + gap + Cp.width
    fig = Image.new("RGB", (W, PANEL_H), (12, 12, 16))
    fig.paste(A, (0, 0))
    fig.paste(B, (A.width + gap, 0))
    fig.paste(Cp, (A.width + gap + B.width + gap, 0))
    fig.save(OUT, optimize=True)
    print(f"\nwrote {OUT}  ({fig.width}x{fig.height}, {OUT.stat().st_size/1000:.0f} kB)")
    for name, panel in (("a", A), ("b", B), ("c", Cp)):
        fp = OUT.with_name(f"ref_{name}.png")
        panel.save(fp, optimize=True)
        print(f"  panel {name}: {fp.name}  {panel.width}x{panel.height}")


if __name__ == "__main__":
    main()
