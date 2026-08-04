"""5-fold splits for XRSpinoPelvic1K — patient-grouped, LSTV-stratified.

THE FAILURE THIS EXISTS TO PREVENT
----------------------------------
Generation emits N randomized views per CT. A view is NOT a new subject. If two
views of one CT land either side of a fold boundary, the model has seen the test
image's anatomy — from a slightly different angle, with the same bone geometry and
the same propagated labels — and every reported number is inflated. It is the
easiest way to make this pipeline look like it works when it does not, and it is
invisible unless you check for it, so `assert_no_leakage` is part of the API and
`build_folds` calls it on its own output before returning.

Grouping is by PATIENT, not by case: one patient can contribute several CTs
(prone/supine, follow-ups), and those share anatomy too.

Stratification mirrors the CTSpinoPelvic1K splitter: LSTV first (L6 /
sacralization cases are rare and are what the paper is about — they must not clump
into one fold), then femoral-head visibility (only those cases can report PI/PT, so
every fold needs a comparable share or the per-fold metrics are not comparable).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

N_FOLDS = 5


def _stratum(rec: dict) -> str:
    """Stratification key: LSTV class x whether PI/PT are measurable."""
    lstv = str(rec.get("lstv_label") or rec.get("stratum") or "normal").lower()
    if rec.get("has_l6"):
        lstv = "l6"
    fem = bool(rec.get("femoral_heads_visible", rec.get("has_femurs", False)))
    return f"{lstv}|{'pi' if fem else 'nopi'}"


def build_folds(records: Iterable[dict], *, n_folds: int = N_FOLDS,
                seed: int = 0) -> Dict[str, List[str]]:
    """Assign every PATIENT to a fold, balancing strata.

    `records`: dicts with at least `case_id` and `patient_id`, optionally
    `lstv_label` / `has_l6` / `femoral_heads_visible`.

    Returns {"fold_0": [case_id, ...], ...}. Deterministic for a given seed.
    """
    recs = list(records)
    if not recs:
        raise ValueError("no records to split")
    for r in recs:
        if not r.get("patient_id"):
            raise ValueError(f"record {r.get('case_id')!r} has no patient_id — "
                             "grouping by patient is not optional")

    # one stratum per PATIENT (the patient's rarest stratum wins, so an LSTV case
    # is never diluted by that patient's other, ordinary scans)
    by_patient: Dict[str, List[dict]] = defaultdict(list)
    for r in recs:
        by_patient[str(r["patient_id"])].append(r)
    freq = Counter(_stratum(r) for r in recs)
    patient_stratum = {
        p: min((_stratum(r) for r in rs), key=lambda s: (freq[s], s))
        for p, rs in by_patient.items()
    }

    # deterministic round-robin WITHIN each stratum, rarest stratum first, so the
    # scarce classes are the ones spread most evenly
    import random
    rng = random.Random(seed)
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    load = [0] * n_folds
    strata = sorted({s for s in patient_stratum.values()},
                    key=lambda s: (sum(1 for v in patient_stratum.values() if v == s), s))
    for s in strata:
        pats = sorted(p for p, v in patient_stratum.items() if v == s)
        rng.shuffle(pats)
        for p in pats:
            k = min(range(n_folds), key=lambda i: (load[i], i))   # least-loaded fold
            for r in by_patient[p]:
                folds[k].append(str(r["case_id"]))
            load[k] += len(by_patient[p])
    out = {f"fold_{i}": sorted(set(v)) for i, v in enumerate(folds)}
    assert_no_leakage(out, recs)
    return out


def assert_no_leakage(folds: Dict[str, Sequence[str]], records: Iterable[dict]) -> None:
    """Raise if any patient — or any case — appears in more than one fold."""
    case_patient = {str(r["case_id"]): str(r["patient_id"]) for r in records}
    seen_case: Dict[str, str] = {}
    patient_folds: Dict[str, set] = defaultdict(set)
    for fold, cases in folds.items():
        for c in cases:
            if c in seen_case:
                raise AssertionError(f"case {c} in both {seen_case[c]} and {fold}")
            seen_case[c] = fold
            p = case_patient.get(c)
            if p is None:
                raise AssertionError(f"case {c} in {fold} has no record")
            patient_folds[p].add(fold)
    bad = {p: sorted(f) for p, f in patient_folds.items() if len(f) > 1}
    if bad:
        raise AssertionError(f"PATIENT LEAKAGE across folds: {bad}")


def view_rows_for_fold(manifest_rows: Iterable[dict], folds: Dict[str, Sequence[str]],
                       fold: int, *, split: str) -> List[dict]:
    """All generated VIEW rows for a split. `split` is 'train' | 'val'.

    Views are expanded here, AFTER the fold assignment, never before — the split is
    decided on cases/patients so no view can cross a boundary.
    """
    val_cases = set(folds[f"fold_{fold}"])
    keep = (lambda c: c in val_cases) if split == "val" else (lambda c: c not in val_cases)
    return [r for r in manifest_rows if keep(str(r.get("case_id")))]


def summarise(folds: Dict[str, Sequence[str]], records: Iterable[dict]) -> str:
    recs = {str(r["case_id"]): r for r in records}
    lines = [f"{'fold':8s}{'cases':>7s}{'patients':>10s}   strata"]
    for f, cases in sorted(folds.items()):
        strata = Counter(_stratum(recs[c]) for c in cases if c in recs)
        pats = {str(recs[c]["patient_id"]) for c in cases if c in recs}
        lines.append(f"{f:8s}{len(cases):7d}{len(pats):10d}   " +
                     ", ".join(f"{k}={v}" for k, v in sorted(strata.items())))
    return "\n".join(lines)


def save(folds: Dict[str, Sequence[str]], path) -> None:
    """Write once, commit, never regenerate silently: a resplit invalidates every
    number previously reported against it."""
    Path(path).write_text(json.dumps(folds, indent=2, sort_keys=True))


def load(path) -> Dict[str, List[str]]:
    return json.loads(Path(path).read_text())
