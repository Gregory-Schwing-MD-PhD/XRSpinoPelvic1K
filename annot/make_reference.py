"""Build the teaching figure the annotator shows: WHERE the femoral head is, and what
"centre of curvature" means when you are looking at it.

Built from a DRR whose hip point ostk derived by fitting a 3-D sphere to the segmented
femoral head and projecting its centre. That matters: the crosshair in this figure is an
objectively correct answer, not one annotator's opinion. Marking up a real radiograph
would put whoever marked it inside every reader's head, and inter-reader agreement would
then be measuring how well they imitate that person.

It is a fixed teaching image and never one of the films being annotated, so it cannot
anchor any particular case.

    python annot/make_reference.py            # writes annot/reference_femhead.png
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont

# ostk supplies the sphere fit that defines the radius drawn here. It is a build-time
# dependency of this figure only -- the Space itself never imports it.
sys.path.insert(0, os.environ.get(
    "OSTK", r"C:/Users/grego/OneDrive/Desktop/OpenSpineToolkit"))

WEB = pathlib.Path(r"C:/Users/grego/OneDrive/Desktop/openspineconsortium.github.io"
                   r"/pacs/data/xr/0003")
OUT = pathlib.Path(__file__).with_name("reference_femhead.png")

GREEN, BLUE, AMBER, WHITE = (0, 229, 160), (108, 180, 255), (245, 165, 36), (245, 245, 245)


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


def head_radius_px(spacing_mm: float) -> float:
    """Femoral head radius in DRR pixels, from ostk's own sphere fit on this case.

    Guessed it first and drew a circle half the size of the anatomy. The radius is not a
    matter of taste -- ostk fits a sphere to the segmented head to compute PI at all, so
    the number is already there: 22.4 mm here, an ordinary adult femoral head.
    """
    import numpy as np
    import nibabel as nib
    from ostk import metrics as M
    # WEB is .../pacs/data/xr/0003; the CT bundle sits at .../pacs/data/0003
    seg = nib.load(str(WEB.parents[1] / "0003" / "seg.nii.gz"))
    rec = M.pelvic_incidence_from_label(
        np.asanyarray(seg.dataobj).astype(np.int16), seg.affine)
    fr = rec.fit_residuals
    r_mm = 0.5 * (fr["femhead_left_radius"] + fr["femhead_right_radius"])
    return r_mm / spacing_mm


def main():
    m = json.loads((WEB / "metrics.json").read_text())
    hip = next(L for L in m["geometry"]["landmarks"] if L["cls"] == "hip_axis")
    hx, hy = hip["xy"]
    spacing = float(m["geometry"]["drr"]["pixel_spacing_mm"])
    R = head_radius_px(spacing)
    src = Image.open(WEB / "image.png").convert("RGB")
    print(f"  head radius {R * spacing:.1f} mm -> {R:.0f} px at {spacing} mm/px")

    # ── panel A: the whole film, head circled ─────────────────────────────────
    scale = 620 / src.height
    A = src.resize((round(src.width * scale), 620), Image.LANCZOS)
    dA = ImageDraw.Draw(A)
    ax, ay, r = hx * scale, hy * scale, R * scale
    dA.ellipse([ax - r, ay - r, ax + r, ay + r], outline=GREEN, width=3)
    dA.line([ax - r * 2.2, ay - r * 2.2, ax - r * 0.8, ay - r * 0.8], fill=GREEN, width=2)
    label(dA, (ax - r * 2.3, ay - r * 2.3), "femoral head", GREEN, 18, "ls")
    label(dA, (10, 10), "A  where to look", WHITE, 18)
    # 548, not 590: two 14 px lines need ~36 px and the panel is 620 tall, so anything
    # below ~560 loses the second line off the bottom edge.
    label(dA, (10, 548), "below and anterior to the S1 endplate,\nseated in the acetabulum",
          (200, 210, 220), 14)

    # ── panel B: zoomed, with the concentric-circle construction ──────────────
    half = round(R * 2.3)                      # head fills ~43% of the panel width
    crop = src.crop((int(hx - half), int(hy - half), int(hx + half), int(hy + half)))
    Z = 620 / crop.height
    B = crop.resize((round(crop.width * Z), 620), Image.LANCZOS)
    dB = ImageDraw.Draw(B)
    cx = cy = 310
    rz = R * Z
    # Mose-style concentric rings. The teaching point is that the CENTRE is common to all
    # of them -- you find it by which circle fits the arc, not by picking one ring.
    for f in (0.70, 0.85, 1.00):
        rr = rz * f
        dB.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=BLUE, width=2)
    # the arc actually being traced: the superolateral subchondral margin
    dB.arc([cx - rz, cy - rz, cx + rz, cy + rz], 185, 355, fill=GREEN, width=5)
    label(dB, (cx, cy - rz - 16), "subchondral cortical arc", GREEN, 16, "ms")
    dB.line([cx - 30, cy, cx + 30, cy], fill=AMBER, width=3)
    dB.line([cx, cy - 30, cx, cy + 30], fill=AMBER, width=3)
    dB.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=AMBER)
    label(dB, (10, 10), "B  what to mark", WHITE, 18)
    label(dB, (cx + 38, cy + 20), "CENTRE", AMBER, 17, "ls")
    label(dB, (10, 554),
          "trace the subchondral cortical arc,\n"
          "then mark its centre of curvature\n"
          "— NOT the brightest shadow", (200, 210, 220), 14)

    # ── panel C: the two-head case, which our own data cannot show ────────────
    # A DRR is integrated ALONG the bicoxofemoral axis, so the two heads superimpose
    # exactly and this case is unreachable from our CT. A real film separates them
    # because the beam is not quite along that axis. The second circle here is therefore
    # SYNTHETIC and labelled as such -- drawn at the same radius, offset front-to-back,
    # which is the geometry that distinguishes two heads from a head plus a lookalike.
    Cp = crop.resize((round(crop.width * Z), 620), Image.LANCZOS)
    dC = ImageDraw.Draw(Cp)
    sep = rz * 0.95                      # a plausible separation on a rotated film
    c1 = (cx - sep / 2, cy)
    c2 = (cx + sep / 2, cy - rz * 0.10)  # rotation offsets mostly front-to-back
    for c in (c1, c2):
        dC.ellipse([c[0] - rz, c[1] - rz, c[0] + rz, c[1] + rz], outline=BLUE, width=3)
        dC.line([c[0] - 22, c[1], c[0] + 22, c[1]], fill=AMBER, width=3)
        dC.line([c[0], c[1] - 22, c[0], c[1] + 22], fill=AMBER, width=3)
    mid = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
    dC.line([c1[0], c1[1], c2[0], c2[1]], fill=GREEN, width=3)
    dC.ellipse([mid[0] - 7, mid[1] - 7, mid[0] + 7, mid[1] + 7], fill=GREEN)
    label(dC, (10, 10), "C  two heads: mark BOTH", WHITE, 18)
    label(dC, (mid[0], mid[1] + 44), "midpoint — derived for you", GREEN, 15, "ms")
    label(dC, (10, 528),
          "on a rotated film the heads separate\n"
          "front-to-back, SAME diameter.\n"
          "schematic: second head drawn in",
          (200, 210, 220), 14)

    # ── join ──────────────────────────────────────────────────────────────────
    gap = 12
    W = A.width + gap + B.width + gap + Cp.width
    fig = Image.new("RGB", (W, 620), (12, 12, 16))
    fig.paste(A, (0, 0))
    fig.paste(B, (A.width + gap, 0))
    fig.paste(Cp, (A.width + gap + B.width + gap, 0))
    fig.save(OUT, optimize=True)
    print(f"wrote {OUT}  ({fig.width}x{fig.height}, {OUT.stat().st_size/1000:.0f} kB)")

    # Also emit the panels separately. The composite is 1648 px wide and unreadable in a
    # ~340 px sidebar; stacked panels at column width are legible, which is where the
    # reader actually needs them.
    for name, panel in (("a", A), ("b", B), ("c", Cp)):
        fp = OUT.with_name(f"ref_{name}.png")
        panel.save(fp, optimize=True)
        print(f"  panel {name}: {fp.name}  {panel.width}x{panel.height}")


if __name__ == "__main__":
    main()
