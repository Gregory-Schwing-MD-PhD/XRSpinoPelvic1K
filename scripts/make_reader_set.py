"""make_reader_set.py — build a BLINDED N-case reader set from a generated XRSpinoPelvic1K
dataset, ready to hand to a radiologist for the manual-measurement study.

For each picked case it copies the lateral synthetic radiograph (and the AP, for orientation
only) into a folder under a SHUFFLED anonymous id (R01..RN) so the reader can't infer the
original order, then writes:
  <out>/R##/lateral_drr.png (+ ap_drr.png)        <- the reader measures lateral_drr.png
  <out>/reader_measurements_template.csv          <- blank, one row per R## (reader fills)
  <out>/README.txt                                <- one-line instructions + protocol link
  <out>_KEY.csv  (NEXT TO <out>, not inside it)   <- anon -> real case_id  [COORDINATOR ONLY]

Cases whose lateral view shows the femoral heads (so PI/PT are measurable) are preferred.

  python scripts/make_reader_set.py --dataset data/xrsp1k --out reader_set --n 25

Then zip and send the <out>/ folder + the protocol link; keep <out>_KEY.csv private.
Protocol: https://github.com/Gregory-Schwing-MD-PhD/XRSpinoPelvic1K/blob/main/docs/reader_study_protocol.md
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

PROTOCOL = ("https://github.com/Gregory-Schwing-MD-PhD/XRSpinoPelvic1K/blob/main/"
            "docs/reader_study_protocol.md")
HIP_IDS = {30, 31, 32, 33}                 # hips + femurs -> femoral heads visible (PI/PT)


def _has_hips(case_dir: Path):
    """True/False if the lateral mask shows a hip/femur; None if it can't be read."""
    m = case_dir / "lateral_mask.png"
    if not m.exists():
        return None
    try:
        import numpy as np
        from PIL import Image
        return bool(set(int(v) for v in np.unique(np.array(Image.open(m)))) & HIP_IDS)
    except Exception:                      # noqa: BLE001 — best-effort preference only
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("data/xrsp1k"),
                    help="generated XRSP1K dir (each <case>/lateral_drr.png)")
    ap.add_argument("--out", type=Path, default=Path("reader_set"))
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260630)
    a = ap.parse_args()

    cases = sorted(d for d in a.dataset.iterdir()
                   if d.is_dir() and (d / "lateral_drr.png").exists())
    if not cases:
        raise SystemExit(f"no cases with lateral_drr.png under {a.dataset} — "
                         f"generate XRSpinoPelvic1K first (scripts/gen_xrsp1k.slurm).")

    rng = random.Random(a.seed)
    hips = [c for c in cases if _has_hips(c)]               # prefer measurable-PI/PT cases
    rest = [c for c in cases if c not in set(hips)]
    rng.shuffle(hips); rng.shuffle(rest)
    picked = (hips + rest)[:a.n]
    rng.shuffle(picked)                                     # randomize presentation order
    if len(picked) < a.n:
        print(f"WARNING: only {len(picked)} cases available (< {a.n}).")

    a.out.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "reader_id", "PI_deg", "SS_deg", "PT_deg", "LL_deg", "seconds", "notes"]
    tmpl, key = [], []
    for i, c in enumerate(picked, 1):
        anon = f"R{i:02d}"
        dst = a.out / anon
        dst.mkdir(exist_ok=True)
        shutil.copy2(c / "lateral_drr.png", dst / "lateral_drr.png")
        if (c / "ap_drr.png").exists():
            shutil.copy2(c / "ap_drr.png", dst / "ap_drr.png")
        tmpl.append({"case_id": anon, "reader_id": "", "PI_deg": "", "SS_deg": "",
                     "PT_deg": "", "LL_deg": "", "seconds": "", "notes": ""})
        key.append({"anon_id": anon, "case_id": c.name, "hips_visible": _has_hips(c)})

    with (a.out / "reader_measurements_template.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(tmpl)
    (a.out / "README.txt").write_text(
        "Spinopelvic reader study — measure SS, PT, PI, LL on lateral_drr.png in EACH R## "
        "folder (ap_drr.png is for orientation only; do not measure it). Record one row per "
        "folder in reader_measurements_template.csv (case_id = the R## folder). PI should "
        "roughly equal SS + PT (self-check).\n\nFull protocol: " + PROTOCOL + "\n")

    keyp = a.out.parent / f"{a.out.name}_KEY.csv"           # coordinator only — do NOT send
    with keyp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["anon_id", "case_id", "hips_visible"])
        w.writeheader(); w.writerows(key)

    nhip = sum(1 for r in key if r["hips_visible"])
    print(f"wrote {len(picked)} blinded cases -> {a.out}/   (zip + send this folder + protocol link)")
    print(f"PRIVATE key (DO NOT send to readers): {keyp}")
    print(f"hip/femur visible (PI & PT measurable): {nhip}/{len(picked)}; "
          f"the rest -> leave PI/PT blank per the protocol.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
