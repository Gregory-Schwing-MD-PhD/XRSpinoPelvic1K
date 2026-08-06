#!/usr/bin/env python3
"""Stage BUU-LSpine 400 from a local archive, and verify it. No re-hosting.

    python scripts/fetch_buu.py --zip BUU-LSPINE_400.zip --out data/BUU-LSPINE_400
    python scripts/fetch_buu.py --check data/BUU-LSPINE_400

BUU-LSpine is a third-party dataset (Burapha University) and is NOT redistributed by this
repo. The archive is obtained from the original distributor under whatever terms they set;
this script only unpacks it into the layout the pipeline expects and then proves the
result is intact, so a run cannot start against a half-extracted tree.

The verification is the useful part. A quietly incomplete dataset does not raise an error,
it just trains on less data and reports a worse number that looks like a modelling problem.
Checked here: both views present, image/annotation pairing, the 11-row L1..S1 chain in
every annotation, and that the images actually open.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPECT_ROWS = 11          # L1..L5 superior+inferior, plus S1 superior


def verify(root: str) -> dict:
    out = {"root": os.path.abspath(root), "views": {}, "problems": []}
    for view in ("LA", "AP"):
        d = os.path.join(root, view)
        if not os.path.isdir(d):
            out["problems"].append(f"missing view directory: {view}")
            continue
        jpgs = sorted(glob.glob(os.path.join(d, "*.jpg")))
        csvs = sorted(glob.glob(os.path.join(d, "*.csv")))
        stems_j = {os.path.basename(p)[:-4] for p in jpgs}
        stems_c = {os.path.basename(p)[:-4] for p in csvs}
        unpaired = sorted(stems_j ^ stems_c)
        bad_rows = []
        if view == "LA":
            for p in csvs:
                try:
                    a = np.loadtxt(p, delimiter=",", ndmin=2)
                except Exception:                              # noqa: BLE001
                    bad_rows.append(os.path.basename(p))
                    continue
                if len(a) < EXPECT_ROWS or a.shape[1] < 4:
                    bad_rows.append(os.path.basename(p))
        out["views"][view] = {"images": len(jpgs), "annotations": len(csvs),
                              "unpaired": len(unpaired),
                              "malformed_annotations": len(bad_rows)}
        if unpaired:
            out["problems"].append(f"{view}: {len(unpaired)} unpaired "
                                   f"(e.g. {unpaired[:3]})")
        if bad_rows:
            out["problems"].append(f"{view}: {len(bad_rows)} annotations without the "
                                   f"{EXPECT_ROWS}-row L1..S1 chain (e.g. {bad_rows[:3]})")
    # a sample of images must actually decode -- a truncated jpg survives an ls
    try:
        from PIL import Image
        la = sorted(glob.glob(os.path.join(root, "LA", "*.jpg")))
        for p in la[:: max(1, len(la) // 20)][:20]:
            Image.open(p).convert("L").load()
    except Exception as exc:                                   # noqa: BLE001
        out["problems"].append(f"image failed to decode: {type(exc).__name__}: {exc}")
    out["ok"] = not out["problems"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", dest="zip_path", default=None, help="local BUU archive")
    ap.add_argument("--out", default="data/BUU-LSPINE_400")
    ap.add_argument("--check", default=None, help="verify an existing directory and exit")
    a = ap.parse_args(argv)

    if a.check:
        rep = verify(a.check)
        print(json.dumps(rep, indent=2))
        return 0 if rep["ok"] else 1

    if not a.zip_path:
        sys.exit("need --zip <archive> (this repo does not redistribute BUU-LSpine) "
                 "or --check <dir>")
    if not os.path.exists(a.zip_path):
        sys.exit(f"archive not found: {a.zip_path}")

    os.makedirs(a.out, exist_ok=True)
    with zipfile.ZipFile(a.zip_path) as z:
        names = z.namelist()
        # tolerate an archive that wraps everything in one top-level folder
        tops = {n.split("/")[0] for n in names if "/" in n}
        strip = tops.pop() + "/" if len(tops) == 1 else None
        for n in names:
            if n.endswith("/"):
                continue
            rel = n[len(strip):] if (strip and n.startswith(strip)) else n
            if not rel:
                continue
            dst = os.path.join(a.out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(n) as src, open(dst, "wb") as fh:
                fh.write(src.read())
    print(f"extracted -> {a.out}")
    rep = verify(a.out)
    print(json.dumps(rep, indent=2))
    if not rep["ok"]:
        print("\nVERIFICATION FAILED -- do not train against this tree", file=sys.stderr)
        return 1
    print("\nverified OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
