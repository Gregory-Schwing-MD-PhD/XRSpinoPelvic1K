"""What the pilot actually measured: per-landmark agreement, and new tool vs old.

The old ledger could only ever answer "how far apart are the two centres". That number
conflates two completely different problems, and they have different fixes:

    the readers disagree about WHERE THE CENTRE IS      -> an imaging limit, or the fit
    the readers disagree about WHICH POINT IS ANTERIOR  -> an instruction problem

Named landmarks separate them, and separating them is the whole reason the tool was
rebuilt. A/S/P agreeing tightly while the centres disagree means the fit or the film is
the problem. A/S/P disagreeing while the centres agree means the readers are finding the
same head by different routes, which is fine and only matters for the keypoint export.

Because the pilot re-reads films the circle tool already read twice, this also reports the
comparison that a fresh sample could not: SAME films, both tools, both pairs of readers.

    python annot/analyze_pilot.py                      # summary to the terminal
    python annot/analyze_pilot.py --csv out.csv        # per-film detail as well
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys

PILOT = "gregoryschwingmdphd/xrsp-femhead-asp-pilot"
OLD = "gregoryschwingmdphd/xrsp-femhead-annot"
ROLES = ("A", "S", "P")


def load(spec: str) -> list:
    if spec.startswith("local:"):
        from pathlib import Path
        return [json.loads(p.read_text())
                for p in sorted((Path(spec[6:]) / "cases").glob("*.json"))]
    from huggingface_hub import snapshot_download
    from pathlib import Path
    root = Path(snapshot_download(spec, repo_type="dataset",
                                  allow_patterns="cases/*.json", max_workers=16,
                                  token=os.environ.get("HF_TOKEN")))
    return [json.loads(p.read_text()) for p in sorted((root / "cases").glob("*.json"))]


def _heads(p):
    if not p:
        return []
    if isinstance(p.get("heads"), list):
        return [q for q in p["heads"] if q]
    return [q for q in (p.get("left"), p.get("right")) if q]


def reads(case):
    s = case.get("slots") or {}
    return [s[k] for k in ("1", "2")
            if (s.get(k) or {}).get("done") and (s.get(k) or {}).get("points")]


def d2(a, b, asp=1.0):
    """Distance in fractions of image WIDTH -- the unit the tolerance is written in."""
    return math.hypot(a[0] - b[0], (a[1] - b[1]) * asp)


def pairing(A, B, asp):
    """Order-free head pairing, the same rule the agreement score uses."""
    if len(A) == 2 and len(B) == 2:
        straight = max(d2(A[0], B[0], asp), d2(A[1], B[1], asp))
        crossed = max(d2(A[0], B[1], asp), d2(A[1], B[0], asp))
        return [(0, 0), (1, 1)] if straight <= crossed else [(0, 1), (1, 0)]
    if not A or not B:
        return []
    i, j = min(((i, j) for i in range(len(A)) for j in range(len(B))),
               key=lambda t: d2(A[t[0]], B[t[1]], asp))
    return [(i, j)]


def summarise(name, vals, tol):
    if not vals:
        print(f"  {name:34s}      no data")
        return
    v = sorted(vals)
    med = statistics.median(v)
    p25 = v[int(0.25 * (len(v) - 1))]
    p75 = v[int(0.75 * (len(v) - 1))]
    within = 100 * sum(x <= tol for x in v) / len(v)
    print(f"  {name:34s} n={len(v):4d}   median {med:.4f}   "
          f"IQR {p25:.4f}-{p75:.4f}   <= tol {within:4.1f}%")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=PILOT)
    ap.add_argument("--old", default=OLD)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--csv")
    a = ap.parse_args()

    cases = load(a.pilot)
    done = [c for c in cases if len(reads(c)) == 2]
    print(f"  pilot ledger   {a.pilot}")
    print(f"  films          {len(cases)}   with two reads: {len(done)}")
    if not done:
        print("\n  Nothing to analyse yet -- no film has both reads in.")
        return 0

    old = {}
    try:
        old = {c["case_id"]: c for c in load(a.old)}
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (old ledger unavailable: {type(exc).__name__})")

    rows = []
    centre, per_role, radius = [], {r: [] for r in ROLES}, []
    obs_counts, facing_bad, swapped, tools = [], 0, 0, {}
    for c in done:
        R = reads(c)
        P = [r["points"] for r in R]
        W = float(P[0].get("w") or 1)
        H = float(P[0].get("h") or 1)
        asp = H / W                       # y is a fraction of HEIGHT; rescale to width
        Hd = [_heads(p) for p in P]
        if not all(Hd):
            continue
        pairs = pairing(Hd[0], Hd[1], asp)
        for t in P:
            tools[t.get("tool", "circle")] = tools.get(t.get("tool", "circle"), 0) + 1
        row = {"case": c["case_id"], "n_heads": min(len(Hd[0]), len(Hd[1]))}
        # worst centre disagreement over the paired heads: same as the ledger's `agree`
        ce = max(d2(Hd[0][i], Hd[1][j], asp) for i, j in pairs)
        centre.append(ce)
        row["centre"] = round(ce, 5)
        # PER LANDMARK, only where BOTH readers actually observed it
        for i, j in pairs:
            lm = [(p.get("landmarks") or [{}] * 9) for p in P]
            for r in ROLES:
                try:
                    x = lm[0][i].get(r) or {}
                    y = lm[1][j].get(r) or {}
                except IndexError:
                    continue
                if x.get("src") == "obs" and y.get("src") == "obs":
                    dv = d2(x["xy"], y["xy"], asp)
                    per_role[r].append(dv)
                    row.setdefault(r, round(dv, 5))
            rr = [(p.get("radii") or [None] * 9) for p in P]
            try:
                if rr[0][i] and rr[1][j]:
                    radius.append(abs(rr[0][i] - rr[1][j]))
            except IndexError:
                pass
        for p in P:
            for L in (p.get("landmarks") or []):
                obs_counts.append(sum(1 for r in ROLES
                                      if (L.get(r) or {}).get("src") == "obs"))
        f = {(p.get("facing") or [""])[0] for p in P if p.get("facing")}
        if len(f) > 1:
            facing_bad += 1
            row["facing_conflict"] = 1
        swapped += any(any(p.get("ap_swapped") or []) for p in P)
        # ...and the same film under the old tool
        o = old.get(c["case_id"])
        if o and o.get("agree") is not None:
            row["old_centre"] = round(o["agree"], 5)
        rows.append(row)

    print(f"\n  tools used: {tools}")
    print(f"\n  DISAGREEMENT between the two readers, in fractions of image width")
    print(f"  (tolerance {a.tol})")
    med_c = summarise("centre (the hip point)", centre, a.tol)
    for r in ROLES:
        summarise(f"landmark {r}", per_role[r], a.tol)
    summarise("radius", radius, a.tol)

    # The comparison a fresh sample could not give: same films, both tools.
    paired = [(r["centre"], r["old_centre"]) for r in rows if "old_centre" in r]
    if paired:
        new = [x for x, _ in paired]
        oldv = [y for _, y in paired]
        print(f"\n  SAME FILMS, BOTH TOOLS   n={len(paired)}")
        print(f"      circle tool   median {statistics.median(oldv):.4f}")
        print(f"      landmark tool median {statistics.median(new):.4f}")
        better = sum(1 for x, y in paired if x < y)
        print(f"      tighter on {better}/{len(paired)} films "
              f"({100*better/len(paired):.0f}%)")
        # a paired sign test: with no real difference this is a coin flip per film
        n, k = len(paired), better
        pv = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
        print(f"      one-sided sign test p = {pv:.2g}"
              + ("   (the landmark tool is tighter)" if pv < 0.05 else
                 "   (not separable at this n)"))

    if obs_counts:
        from collections import Counter
        cnt = Counter(obs_counts)
        print(f"\n  extremes a reader could actually trace, per head (of 3):")
        for k in sorted(cnt):
            print(f"      {k}: {cnt[k]:4d}")
    print(f"\n  films where the readers disagreed about FACING: {facing_bad}")
    print(f"  reads flagged A/P swapped at submit:            {swapped}")

    if med_c is not None:
        print(f"\n  Read this against the gate, not instead of it. The circle tool settled "
              f"71 of 1153\n  films against a 0.005 tolerance -- 6%. A median of "
              f"{med_c:.4f} can be a large improvement\n  and still clear almost nothing, "
              f"which is why the distribution is printed rather\n  than a pass rate. The "
              f"tolerance is an output of this pilot, not an input to it.")

    if a.csv:
        cols = ["case", "n_heads", "centre", "old_centre"] + list(ROLES) + \
               ["facing_conflict"]
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n  wrote {a.csv}  ({len(rows)} films)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
