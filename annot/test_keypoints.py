"""The keypoint export, against a ledger built to break it.

The thing worth testing here is not that files appear. It is that the VISIBILITY FLAGS
are honest, because they are the only thing standing between "we resampled the circle at
eight clock angles" and "we invented rim where the reader could not see any". Each case
below is a specific way that could go wrong.

    python annot/test_keypoints.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
W, H = 1392, 2428
fails = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        fails.append(what)


def arcread(cx, cy, R, lo_deg, span_deg, n=5):
    """A read that traced `span_deg` of rim starting at `lo_deg`."""
    lo, sp = math.radians(lo_deg), math.radians(span_deg)
    pts = [[(cx + R * math.cos(lo + sp * i / (n - 1))) / W,
            (cy + R * math.sin(lo + sp * i / (n - 1))) / H] for i in range(n)]
    return {"tool": "arc", "heads": [[cx / W, cy / H]], "radii": [R / W], "arcs": [pts],
            "arc_lo": [lo], "arc_span": [sp], "w": W, "h": H}


def case(cid, p1, p2=None, **kw):
    s = {"1": {"done": True, "points": p1, "annotator": "alice"}}
    if p2:
        s["2"] = {"done": True, "points": p2, "annotator": "bob"}
    d = {"case_id": cid, "slots": s}
    d.update(kw)
    return d


T = Path(tempfile.mkdtemp(prefix="kpledger_"))
(T / "cases").mkdir()
RECS = [
    case("full",  arcread(700, 900, 120, 0, 350), arcread(702, 901, 121, 0, 350)),
    case("cap",   arcread(700, 900, 120, 200, 80), arcread(701, 899, 119, 205, 75)),
    # the readers walked DIFFERENT halves. Only the overlap is evidence; a union would
    # quietly turn one reader's word into the other's observation.
    case("halfA", arcread(700, 900, 120, 0, 180), arcread(700, 900, 120, 170, 180)),
    case("legacy", {"tool": "circle", "heads": [[0.5, 0.37]], "radii": [0.086],
                    "w": W, "h": H},
                   {"tool": "circle", "heads": [[0.505, 0.372]], "radii": [0.088],
                    "w": W, "h": H}),
    # two heads, marked in OPPOSITE order by the two readers
    case("two", {"tool": "arc", "heads": [[0.44, 0.37], [0.56, 0.38]],
                 "radii": [0.08, 0.08], "arc_lo": [0.0, 0.0], "arc_span": [6.1, 6.1],
                 "w": W, "h": H},
                {"tool": "arc", "heads": [[0.561, 0.381], [0.441, 0.371]],
                 "radii": [0.08, 0.08], "arc_lo": [0.0, 0.0], "arc_span": [6.1, 6.1],
                 "w": W, "h": H}),
    case("nohead", {"heads": [], "w": W, "h": H}, final={"points": None}),
    case("single", arcread(700, 900, 120, 0, 350)),          # only one read
]
for r in RECS:
    (T / "cases" / f"{r['case_id']}.json").write_text(json.dumps(r))

OUT = Path(tempfile.mkdtemp(prefix="kpout_")) / "kp"
subprocess.run([sys.executable, str(ROOT / "export_keypoints.py"),
                "--ledger", f"local:{T}", "--out", str(OUT)],
               check=True, stdout=subprocess.DEVNULL)

lab = {}
for f in sorted(OUT.rglob("*.txt")):
    lab[f.stem] = [ln.split() for ln in f.read_text().strip().split("\n")]


def vis(row):
    return [int(row[7 + 3 * i]) for i in range(9)]


def kp(row, i):
    return (float(row[5 + 3 * i]), float(row[6 + 3 * i]))


check("nohead" not in lab, "a film settled as no-visible-head is not exported at all")
check("single" not in lab, "a film with one read is held back unless --include-single")
check(set(lab) == {"full", "cap", "halfA", "legacy", "two"},
      f"exactly the usable films are exported ({sorted(lab)})")

check(all(v[0] == 2 for rows in lab.values() for v in map(vis, rows)),
      "the centre keypoint is always present -- it is what the study needs")
check(sum(1 for v in vis(lab["full"][0])[1:] if v) == 8,
      "a rim traced right round exports all 8 clock angles as observed")
check(sum(1 for v in vis(lab["cap"][0])[1:] if v) == 2,
      f"a superior-cap-only read exports 2 of 8 "
      f"({sum(1 for v in vis(lab['cap'][0])[1:] if v)})")
check(sum(1 for v in vis(lab["halfA"][0])[1:] if v) == 1,
      "two readers who traced different halves export only their OVERLAP, not the union")
check(sum(1 for v in vis(lab["legacy"][0])[1:] if v) == 0,
      "a circle-tool read contributes a centre and NO rim evidence by default")

# geometry: the exported rim points must actually sit on the circle the readers fitted
row = lab["full"][0]
cx, cy = kp(row, 0)
R = float(row[3]) / 2 / 1.25                       # bbox width was 2R padded by 1.25
asp = W / H
d = [math.hypot(kp(row, i)[0] - cx, (kp(row, i)[1] - cy) / asp) for i in range(1, 9)]
check(max(abs(x - R) for x in d) < 1e-4,
      f"every rim keypoint lies on the fitted circle, aspect included "
      f"(spread {max(d) - min(d):.2e})")
ang = sorted(round(math.degrees(math.atan2((kp(row, i)[1] - cy) / asp,
                                           kp(row, i)[0] - cx)) % 360)
             for i in range(1, 9))
check(ang == [0, 45, 90, 135, 180, 225, 270, 315],
      f"and at fixed clock angles, so keypoint k means the same place every time ({ang})")

# order-free pairing: reader B marked the two heads the other way round
check(len(lab["two"]) == 2, "two heads export as two instances")
xs = sorted(kp(r, 0)[0] for r in lab["two"])
check(abs(xs[0] - 0.4405) < 1e-3 and abs(xs[1] - 0.5605) < 1e-3,
      f"readers who marked the heads in opposite order are paired correctly ({xs})")
man = json.loads((OUT / "manifest.json").read_text())
mid = next(f["bicoxofemoral"] for f in man["films"] if f["case"] == "two")
check(abs(mid[0] - 0.5005) < 1e-3,
      f"and the bicoxofemoral midpoint survives into the manifest ({mid})")

# a horizontal flip must map image-right rim onto image-left rim, or flip augmentation
# silently trains the model on mismatched keypoints
flip = json.loads([l for l in (OUT / "data.yaml").read_text().splitlines()
                   if l.startswith("flip_idx")][0].split(":", 1)[1])
check(flip[0] == 0 and all(flip[flip[i]] == i for i in range(len(flip))),
      f"flip_idx is a valid involution fixing the centre ({flip})")

print("\nFAILED: " + "; ".join(fails) if fails else "\nALL CHECKS PASSED")
sys.exit(1 if fails else 0)
