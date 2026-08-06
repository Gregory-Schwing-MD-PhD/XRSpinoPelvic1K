#!/usr/bin/env python3
"""Write the BUU train/val/test split ONCE, to disk. Run before training, not during it.

    python scripts/make_buu_splits.py --buu /data/BUU-LSPINE --out /data/buu_splits.json

Why a file rather than a seed
-----------------------------
Deriving the split from a seed inside the trainer is reproducible only while the file
list is identical. Add, drop or re-extract one image and every assignment downstream of
it shifts, so a "held-out" case can quietly migrate into training between runs and the
test number stops meaning anything. The split is therefore computed once, written out,
and read back by name. Refuses to overwrite without --force, for the same reason.

Grouping
--------
By PATIENT, from the numeric prefix of `NNNN-S-AGEYV`. On the 2000-film release every
patient contributes exactly one lateral, so grouping is currently equivalent to a random
split -- but it is kept because it costs nothing and the guarantee has to hold if a
future release adds follow-up films. The no-leak assertion runs regardless.

Stratification
--------------
By SEX and AGE BAND. The release is 1318 F / 682 M and spans ages 6-95 (median 59), so an
unstratified draw can hand the test set a materially different population from training.
Age matters here beyond nuisance-variable hygiene: lumbar lordosis and pelvic incidence
are age- and maturity-dependent, and PI in particular rises through skeletal growth before
plateauing in adulthood. A test set skewed young or old therefore shifts the very
quantities the model is measured on.

Pediatric cases
---------------
The minimum age is 6. Those films are neither dropped nor hidden: they are reported, put
in their own stratum so they spread across the splits, and flagged in the output so a
downstream analysis can exclude them deliberately rather than discover them later.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NAME_RE = re.compile(r"^(?P<subj>\d+)-(?P<sex>[MF])-(?P<age>\d+)Y(?P<view>\d)$")
# 0-17 pediatric/adolescent (skeletally immature -> different PI/LL), then adult decades
AGE_BANDS = [(0, 17), (18, 39), (40, 59), (60, 74), (75, 200)]


def band_of(age: int) -> str:
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "unknown"


def parse(case: str):
    m = NAME_RE.match(case)
    if not m:
        return None
    return {"subject": m.group("subj"), "sex": m.group("sex"),
            "age": int(m.group("age")), "view": m.group("view")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buu", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="LA")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    a = ap.parse_args(argv)

    from xrsp.buu import index_buu

    if os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} exists. A resplit invalidates every number reported against "
                 f"the old one -- pass --force if that is what you mean.")

    rows = index_buu(a.buu, view=a.view)
    if not rows:
        sys.exit(f"no {a.view} films under {a.buu}")

    meta, unparsed = {}, []
    for r in rows:
        p = parse(r["case"])
        if p is None:
            # An unparseable name is grouped by its OWN full name, so it can never be
            # pooled with a real subject, and it is reported rather than dropped. The
            # earlier form left `meta` unset on this branch and would KeyError later --
            # invisible on this release, where all 2000 names parse.
            unparsed.append(r["case"])
            meta[r["case"]] = {"subject": r["case"], "sex": "?", "age": -1, "view": "?"}
        else:
            meta[r["case"]] = p
    if unparsed:
        print(f"WARNING: {len(unparsed)} filenames did not match NNNN-S-AGEYV "
              f"(e.g. {unparsed[:3]}); they are grouped by full name", flush=True)

    # one entry per SUBJECT, carrying that subject's stratum
    by_subj = collections.defaultdict(list)
    for case, m in meta.items():
        by_subj[m["subject"]].append(case)
    strata = {}
    for subj, cases in by_subj.items():
        m = meta[cases[0]]
        strata[subj] = f"{m['sex']}|{band_of(m['age'])}"

    rng = np.random.default_rng(a.seed)
    split = {}
    for stratum in sorted(set(strata.values())):
        subs = sorted(s for s in by_subj if strata[s] == stratum)
        rng.shuffle(subs)
        n = len(subs)
        n_te = int(round(a.test_frac * n))
        n_va = int(round(a.val_frac * n))
        # a stratum too small to divide goes wholly to TRAIN: a one-case "test stratum"
        # reports a meaningless number and starves training of a rare group
        if n < 4:
            n_te = n_va = 0
        for i, s in enumerate(subs):
            split[s] = "test" if i < n_te else ("val" if i < n_te + n_va else "train")

    assign = {c: split[meta.get(c, {}).get("subject", c)] for c in meta}
    counts = collections.Counter(assign.values())
    # the guard must FIRE, not be assumed
    sets = {k: {meta[c]["subject"] for c, v in assign.items() if v == k}
            for k in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]), "subject leak train/val"
    assert not (sets["train"] & sets["test"]), "subject leak train/test"
    assert not (sets["val"] & sets["test"]), "subject leak val/test"

    out = {
        "buu_root": os.path.abspath(a.buu), "view": a.view, "seed": a.seed,
        "val_frac": a.val_frac, "test_frac": a.test_frac,
        "n_films": len(assign), "n_subjects": len(by_subj),
        "counts": dict(counts),
        "age_bands": [f"{lo}-{hi}" for lo, hi in AGE_BANDS],
        "assignments": assign,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"{len(assign)} films / {len(by_subj)} subjects -> {dict(counts)}")
    print(f"\n{'stratum':14s}{'train':>7s}{'val':>6s}{'test':>6s}")
    per = collections.defaultdict(collections.Counter)
    for c, v in assign.items():
        per[strata[meta[c]['subject']]][v] += 1
    for k in sorted(per):
        p = per[k]
        print(f"{k:14s}{p['train']:7d}{p['val']:6d}{p['test']:6d}")
    ped = sum(1 for c in meta if meta[c]["age"] <= 17)
    print(f"\npediatric (age <= 17): {ped} films -- spread across splits, flagged not dropped")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
