"""Draw both readers' marks on the same film, so the disagreement is visible.

The pilot numbers say readers disagree by a median 0.0236 of image width and that most of
it is head COUNT rather than precision. That is a claim about pictures, and it should be
checked as one: six films, both readers overlaid, chosen to span the range rather than to
flatter it -- the two they agreed on most, two where they disagreed about how many heads
there are, and the two worst.

Colour carries reader identity and nothing else: slot 1 blue, slot 2 orange, in fixed
order, and every mark is also drawn with the reader's number beside it so identity is
never colour alone. Strokes get a dark halo because a radiograph is black ground AND
bright bone, and a single flat colour is illegible over one or the other.

    python annot/render_pilot.py [--out DIR] [--n 6]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PILOT = "gregoryschwingmdphd/xrsp-femhead-asp-pilot"
IMAGES = "gregoryschwingmdphd/xrsp-femhead-images"
TOL = 0.005

# categorical slots 1 and 2, fixed order, never cycled: reader A is always blue
COL = [(57, 135, 229), (235, 104, 52)]
HALO = (10, 10, 12)
DERIVED = (245, 200, 60)          # the hip point each reader implies
INK = (255, 255, 255)
ROLES = ("A", "S", "P")


def font(sz, bold=True):
    for f in (("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if bold
              else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def halo_line(d, xy, fill, w):
    d.line(xy, fill=HALO, width=w + 3)
    d.line(xy, fill=fill, width=w)


def halo_ellipse(d, box, fill, w):
    d.ellipse(box, outline=HALO, width=w + 3)
    d.ellipse(box, outline=fill, width=w)


def label(d, xy, text, fill, f, anchor="mm"):
    x, y = xy
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx or dy:
                d.text((x + dx, y + dy), text, font=f, fill=HALO, anchor=anchor)
    d.text((x, y), text, font=f, fill=fill, anchor=anchor)


def reads(c):
    s = c.get("slots") or {}
    return [s[k] for k in ("1", "2")
            if (s.get(k) or {}).get("done") and (s.get(k) or {}).get("points")]


def heads(p):
    return [q for q in (p.get("heads") or []) if q]


def pair_gap(A, B, asp):
    d = lambda u, v: math.hypot(u[0] - v[0], (u[1] - v[1]) * asp)   # noqa: E731
    if len(A) == 2 and len(B) == 2:
        return min(max(d(A[0], B[0]), d(A[1], B[1])),
                   max(d(A[0], B[1]), d(A[1], B[0])))
    return min(d(u, v) for u in A for v in B)


def mid(P):
    return [sum(q[0] for q in P) / len(P), sum(q[1] for q in P) / len(P)]


def draw_reader(d, p, W, H, ox, oy, k, sc, R_px):
    """One reader's marks, in their slot colour, with their number on every head."""
    col = COL[k % 2]
    f_small = font(max(11, int(R_px * 0.30)))
    lw = max(2, int(R_px * 0.045))
    rr = max(3, int(R_px * 0.075))
    hs = heads(p)
    rad = p.get("radii") or []
    lms = p.get("landmarks") or []
    ex = p.get("extra") or []
    for i, hc in enumerate(hs):
        cx = (hc[0] * W - ox) * sc
        cy = (hc[1] * H - oy) * sc
        R = ((rad[i] if i < len(rad) and rad[i] else 0.085) * W) * sc
        halo_ellipse(d, [cx - R, cy - R, cx + R, cy + R], col, lw)
        t = R * 0.30
        halo_line(d, [cx - t, cy, cx + t, cy], col, lw)
        halo_line(d, [cx, cy - t, cx, cy + t], col, lw)
        # the reader's number rides on the circle, so identity is never colour alone
        label(d, (cx, cy - R - rr * 3), f"R{k+1}", col, f_small)
        lm = lms[i] if i < len(lms) else {}
        for role in ROLES:
            q = (lm.get(role) or {})
            if q.get("src") == "obs" and q.get("xy"):
                x = (q["xy"][0] * W - ox) * sc
                y = (q["xy"][1] * H - oy) * sc
                d.rectangle([x - rr, y - rr, x + rr, y + rr], outline=HALO, width=lw + 2)
                d.rectangle([x - rr, y - rr, x + rr, y + rr], outline=col, width=lw)
                # role letters are deliberately NOT drawn here. With two readers on one
                # small joint the six letters collide with each other and with the R1/R2
                # tags, and the story this figure tells -- how the two circles differ --
                # is carried by the circles. The squares still mark the landmarks.
        for q in (ex[i] if i < len(ex) else []):
            x = (q[0] * W - ox) * sc
            y = (q[1] * H - oy) * sc
            halo_ellipse(d, [x - rr * 0.7, y - rr * 0.7, x + rr * 0.7, y + rr * 0.7],
                         col, max(1, lw - 1))
    if hs:
        m = mid(hs)
        mx, my = (m[0] * W - ox) * sc, (m[1] * H - oy) * sc
        r2 = max(4, int(R_px * 0.10))
        d.ellipse([mx - r2, my - r2, mx + r2, my + r2], fill=DERIVED, outline=HALO, width=2)
        return (mx, my)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="annot/pilot_review")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download, hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    root = Path(snapshot_download(PILOT, repo_type="dataset",
                                  allow_patterns="cases/*.json", max_workers=16,
                                  token=tok))
    cases = [json.loads(p.read_text()) for p in sorted(root.glob("cases/*.json"))]
    rows = []
    for c in cases:
        R = reads(c)
        if len(R) != 2:
            continue
        P = [r["points"] for r in R]
        A, B = heads(P[0]), heads(P[1])
        if not A or not B:
            continue
        W, H = float(P[0].get("w") or 1), float(P[0].get("h") or 1)
        gap = pair_gap(A, B, H / W)
        hipgap = math.hypot(mid(A)[0] - mid(B)[0], (mid(A)[1] - mid(B)[1]) * H / W)
        rows.append(dict(case=c["case_id"], P=P, who=[r.get("annotator", "?") for r in R],
                         gap=gap, hip=hipgap, na=len(A), nb=len(B), W=W, H=H))
    rows.sort(key=lambda r: r["gap"])
    print(f"  {len(rows)} films with two reads")

    pick, seen = [], set()

    def take(cand, tag):
        for r in cand:
            if r["case"] in seen:
                continue
            seen.add(r["case"])
            r["tag"] = tag
            pick.append(r)
            return

    take(rows, "closest agreement")
    take(rows[1:], "close agreement")
    take([r for r in rows if r["na"] != r["nb"]], "one saw two heads, one saw one")
    take([r for r in rows if r["na"] != r["nb"]], "one saw two heads, one saw one")
    take(rows[::-1], "worst disagreement")
    take(rows[::-1], "second worst")
    pick = pick[:a.n]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panels = []
    for r in pick:
        cid = r["case"]
        fp = None
        for cand in (f"images/{cid}.jpg", f"{cid}.jpg"):
            try:
                fp = hf_hub_download(IMAGES, cand, repo_type="dataset", token=tok)
                break
            except Exception:                                  # noqa: BLE001
                continue
        if not fp:
            print(f"  ! no film for {cid}")
            continue
        im = Image.open(fp).convert("RGB")
        W, H = im.size
        pts = [(q[0] * W, q[1] * H) for p in r["P"] for q in heads(p)]
        rad = max([(p.get("radii") or [0.085])[0] * W for p in r["P"]] or [0.085 * W])
        pad = rad * 2.6
        x0 = max(0, int(min(x for x, _ in pts) - pad))
        y0 = max(0, int(min(y for _, y in pts) - pad))
        x1 = min(W, int(max(x for x, _ in pts) + pad))
        y1 = min(H, int(max(y for _, y in pts) + pad))
        crop = im.crop((x0, y0, x1, y1))
        TARGET = 560
        sc = TARGET / max(crop.width, crop.height)
        crop = crop.resize((max(1, int(crop.width * sc)), max(1, int(crop.height * sc))),
                           Image.LANCZOS)
        d = ImageDraw.Draw(crop)
        hips = []
        for k, p in enumerate(r["P"]):
            hips.append(draw_reader(d, p, W, H, x0, y0, k, sc, rad * sc))
        if all(hips):
            halo_line(d, [hips[0][0], hips[0][1], hips[1][0], hips[1][1]], DERIVED, 2)

        # caption strip under the film
        fh = font(15)
        fs = font(13, bold=False)
        strip = 76
        panel = Image.new("RGB", (crop.width, crop.height + strip), (16, 16, 18))
        panel.paste(crop, (0, 0))
        pd = ImageDraw.Draw(panel)
        y = crop.height + 8
        pd.text((10, y), f"{cid}   {r['tag']}", font=fh, fill=INK)
        pd.text((10, y + 21),
                f"hip points {r['hip']:.4f} W apart  =  {r['hip']/TOL:.1f}x the tolerance",
                font=fs, fill=(200, 200, 205))
        for k in range(2):
            cx = 14 + k * 190
            pd.rectangle([cx - 4, y + 44, cx + 6, y + 54], fill=COL[k])
            pd.text((cx + 14, y + 42),
                    f"R{k+1} {r['who'][k][:14]} · {len(heads(r['P'][k]))} head"
                    f"{'s' if len(heads(r['P'][k])) != 1 else ''}",
                    font=fs, fill=(210, 210, 215))
        panels.append(panel)
        panel.save(out / f"{cid}.png", optimize=True)
        print(f"  {cid:16s} {r['tag']:32s} hip gap {r['hip']:.4f} W "
              f"({r['na']}/{r['nb']} heads)")

    if panels:
        cols = 3 if len(panels) >= 3 else len(panels)
        rowsn = math.ceil(len(panels) / cols)
        cw = max(p.width for p in panels)
        ch = max(p.height for p in panels)
        gap = 10
        head_h = 58
        grid = Image.new("RGB", (cols * cw + (cols + 1) * gap,
                                 head_h + rowsn * ch + (rowsn + 1) * gap), (16, 16, 18))
        gd = ImageDraw.Draw(grid)
        gd.text((gap, 12), "Preston pilot — both readers on the same film",
                font=font(20), fill=INK)
        gd.text((gap, 36),
                "blue = reader 1, orange = reader 2 · squares A/S/P · small rings = extra "
                "rim points · yellow dot = the hip point that read implies",
                font=font(13, bold=False), fill=(190, 190, 196))
        for i, p in enumerate(panels):
            cx = gap + (i % cols) * (cw + gap)
            cy = head_h + gap + (i // cols) * (ch + gap)
            grid.paste(p, (cx, cy))
        grid.save(out / "_grid.png", optimize=True)
        print(f"\n  wrote {out}/_grid.png  ({grid.width}x{grid.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
