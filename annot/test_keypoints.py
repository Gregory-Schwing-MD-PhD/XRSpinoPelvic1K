"""The keypoint export, against a ledger built to break it.

The thing worth testing is not that files appear. It is that the VISIBILITY FLAGS are
honest, because they are the only thing standing between "the readers named these three
extremes" and "we made up a landmark where nobody could see one". Each case below is a
specific way that could go wrong.

    python annot/test_keypoints.py
"""
from __future__ import annotations

import json
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


def arc(cx, cy, R, seen=("A", "S", "P"), facing="left", tilt=0.0):
    """A read that named `seen` and could not trace the rest.

    Facing left puts anterior at image-left, so A sits at cx-R and P at cx+R.
    """
    sgn = -1 if facing == "left" else 1
    P = {"A": [(cx + sgn * R) / W, cy / H],
         "S": [cx / W, (cy - R) / H],
         "P": [(cx - sgn * R) / W, cy / H]}
    lm = {}
    for r in ("A", "S", "P"):
        lm[r] = {"xy": P[r], "src": "obs"} if r in seen else {"src": "derived"}
    return {"tool": "arc", "heads": [[cx / W, cy / H]], "radii": [R / W],
            "landmarks": [lm], "facing": [facing], "ap_tilt_deg": [tilt],
            "extra": [[]], "w": W, "h": H}


def case(cid, p1, p2=None, **kw):
    s = {"1": {"done": True, "points": p1, "annotator": "alice"}}
    if p2:
        s["2"] = {"done": True, "points": p2, "annotator": "bob"}
    d = {"case_id": cid, "slots": s}
    d.update(kw)
    return d


T = Path(tempfile.mkdtemp(prefix="kpledger_"))
(T / "cases").mkdir()

two_a = {"tool": "arc", "heads": [[0.44, 0.37], [0.56, 0.38]], "radii": [0.08, 0.08],
         "landmarks": [{r: {"xy": [0.44, 0.37], "src": "obs"} for r in "ASP"},
                       {r: {"xy": [0.56, 0.38], "src": "obs"} for r in "ASP"}],
         "facing": ["left", "left"], "ap_tilt_deg": [0, 0], "w": W, "h": H}
two_b = {"tool": "arc", "heads": [[0.561, 0.381], [0.441, 0.371]], "radii": [0.08, 0.08],
         "landmarks": [{r: {"xy": [0.561, 0.381], "src": "obs"} for r in "ASP"},
                       {r: {"xy": [0.441, 0.371], "src": "obs"} for r in "ASP"}],
         "facing": ["left", "left"], "ap_tilt_deg": [0, 0], "w": W, "h": H}

RECS = [
    # both readers named all three
    case("full", arc(700, 900, 120), arc(702, 901, 121)),
    # both could only trace the top: A and P are derived, and must not be trained on
    case("captop", arc(700, 900, 120, seen=("S",)), arc(701, 899, 119, seen=("S",))),
    # ONE reader saw the anterior extreme, the other did not
    case("onlyone", arc(700, 900, 120, seen=("A", "S", "P")),
                    arc(701, 899, 119, seen=("S", "P"))),
    # the readers disagree about which way the patient faces: an anterior label on
    # posterior cortex is worse than no label, so this has to be surfaced
    case("faceclash", arc(700, 900, 120), arc(701, 899, 119, facing="right")),
    # a legacy circle-tool read: a centre and nothing else
    case("legacy", {"tool": "circle", "heads": [[0.5, 0.37]], "radii": [0.086],
                    "w": W, "h": H},
                   {"tool": "circle", "heads": [[0.505, 0.372]], "radii": [0.088],
                    "w": W, "h": H}),
    # two heads, marked in OPPOSITE order by the two readers
    case("two", two_a, two_b),
    case("nohead", {"heads": [], "w": W, "h": H}, final={"points": None}),
    case("single", arc(700, 900, 120)),                    # only one read
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

KP = {"centre": 0, "A": 1, "S": 2, "P": 3}


def vis(row, name):
    return int(row[7 + 3 * KP[name]])


def kp(row, name):
    i = KP[name]
    return (float(row[5 + 3 * i]), float(row[6 + 3 * i]))


print("\n[1] what is exported at all")
check("nohead" not in lab, "a film settled as no-visible-head is not exported")
check("single" not in lab, "a film with one read is held back unless --include-single")
check(set(lab) == {"full", "captop", "onlyone", "faceclash", "legacy", "two"},
      f"exactly the usable films are exported ({sorted(lab)})")
check(all(len(r) == 5 + 3 * 4 for rows in lab.values() for r in rows),
      "every line is bbox + 4 keypoints, the fixed positional contract")

print("\n[2] visibility is earned, not assumed")
check(all(vis(r, "centre") == 2 for rows in lab.values() for r in rows),
      "the centre is always present -- derived, but always determined")
check(all(vis(lab["full"][0], n) == 2 for n in "ASP"),
      "three extremes both readers named are all visible")
check(vis(lab["captop"][0], "S") == 2
      and vis(lab["captop"][0], "A") == 0 and vis(lab["captop"][0], "P") == 0,
      "an extreme NEITHER reader could trace is excluded from the loss (v=0)")
check(vis(lab["onlyone"][0], "A") == 1,
      f"an extreme only ONE reader traced is kept at lower confidence "
      f"(v={vis(lab['onlyone'][0], 'A')})")
check(vis(lab["onlyone"][0], "S") == 2, "while the ones both traced stay at v=2")
check(all(vis(lab["legacy"][0], n) == 0 for n in "ASP"),
      "a circle-tool read contributes a centre and no landmarks at all")

print("\n[3] a derived landmark still lands somewhere defensible")
row = lab["captop"][0]
cx, cy = kp(row, "centre")
ax, _ = kp(row, "A")
px, _ = kp(row, "P")
check(ax < cx < px,
      f"the derived anterior point is on the ANTERIOR side for a left-facing film "
      f"(A={ax:.3f} centre={cx:.3f} P={px:.3f})")
check(kp(row, "S")[1] < cy, "and the superior one is above the centre")

print("\n[4] pairing and facing")
check(len(lab["two"]) == 2, "two heads export as two instances")
xs = sorted(kp(r, "centre")[0] for r in lab["two"])
check(abs(xs[0] - 0.4405) < 1e-3 and abs(xs[1] - 0.5605) < 1e-3,
      f"readers who marked the heads in opposite order are paired correctly ({xs})")
man = json.loads((OUT / "manifest.json").read_text())
by = {f["case"]: f for f in man["films"]}
check(abs(by["two"]["bicoxofemoral"][0] - 0.5005) < 1e-3,
      f"the bicoxofemoral midpoint survives into the manifest "
      f"({by['two']['bicoxofemoral']})")
check(by["full"]["facing"] == "left", "facing is carried through the export")
check(by["faceclash"]["facing"] == "",
      "and readers who disagree about facing yield NO facing rather than a coin flip")
check(all(vis(lab["faceclash"][0], n) == 2 for n in "ASP"),
      "their named extremes are still exported -- the positions were agreed, the "
      "direction label was not")

print("\n[5] the dataset declaration")
yaml = (OUT / "data.yaml").read_text()
check("kpt_shape: [4, 3]" in yaml, "kpt_shape matches the four keypoints written")
flip = json.loads([l for l in yaml.splitlines()
                   if l.startswith("flip_idx")][0].split(":", 1)[1])
check(flip == [0, 3, 2, 1],
      f"a horizontal flip exchanges anterior and posterior ({flip})")
check(all(flip[flip[i]] == i for i in range(len(flip))), "and is a valid involution")
check("not a landmark" in yaml and "TANGENCY" in yaml,
      "the yaml says why the centre is derived and the extremes are not")

print("\nFAILED: " + "; ".join(fails) if fails else "\nALL CHECKS PASSED")
sys.exit(1 if fails else 0)
