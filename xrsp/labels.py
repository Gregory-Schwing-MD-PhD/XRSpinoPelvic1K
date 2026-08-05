"""Label-id <-> name map — VerSe-native, kept in SYNC with CTSpinoPelvic1K v4.

Source of truth: CTSpinoPelvic1K scripts/label_scheme.py (label_dict()). Vendored here so
this repo runs standalone. v4 is VerSe-native: the spine keeps its VerSe ids verbatim
(C1-C7=1-7, T1-T12=8-19, L1-L6=20-25, sacrum=26, coccyx=27, T13=28), and every non-VerSe
structure gets a fixed id above the VerSe range (S1=29, hips=30/31, femurs=32/33,
rib_left_1..12=34-45, rib_right_1..12=46-57). The OLD v3 scheme (L1=1, T1=13, no ribs) is
gone — do NOT reintroduce it; masks/landmarks generated under it would be mislabelled.
"""
from __future__ import annotations

_VERSE_NAMES = (["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
                + [f"T{n}" for n in range(1, 13)]      # T1..T12 -> 8..19
                + [f"L{n}" for n in range(1, 7)])      # L1..L6 -> 20..25

LABELS = {}
for _i, _nm in enumerate(_VERSE_NAMES, start=1):       # 1..25
    LABELS[_nm] = _i
LABELS["sacrum"] = 26
LABELS["coccyx"] = 27
LABELS["T13"] = 28
LABELS["S1"] = 29                                       # carved from sacrum top (spinopelvic)
LABELS["left_hip"] = 30
LABELS["right_hip"] = 31
LABELS["femur_left"] = 32
LABELS["femur_right"] = 33
for _n in range(1, 13):
    LABELS[f"rib_left_{_n}"] = 33 + _n                 # 34..45
for _n in range(1, 13):
    LABELS[f"rib_right_{_n}"] = 45 + _n                # 46..57

NAMES = {v: k for k, v in LABELS.items()}

# Vertebral levels we localize on a spinal radiograph (cranio-caudal). Ribs/sacrum/femurs
# are in the dense mask but are not per-level localization targets. T13 is rare; kept in
# anatomical order between T12 and L1 (project_level_points skips levels absent in a case).
SPINE_LEVELS = ([f"T{n}" for n in range(1, 13)] + ["T13"]
                + [f"L{n}" for n in range(1, 7)] + ["S1"])


def lid(name: str) -> int:
    return LABELS[name]


def lname(i: int):
    return NAMES.get(int(i))

# ── Legacy scheme + detection ────────────────────────────────────────────────────
# The v3/legacy scheme (L1=1, S1=7, sacrum=8, hips=9/10, femurs=11/12, T1..T13=13..25,
# ribs from 26) is NOT dead on disk, even though v4 supersedes it: the PACS demo cases
# under openspineconsortium.github.io/pacs/data are still written in it, and it is what
# ostk.labels carries. Reading a v4 volume with the legacy map, or the reverse, does not
# fail -- it silently returns the WRONG STRUCTURE. `L1` resolves to id 1, which is C1 in
# v4, so a generation run produces a whole dataset of confidently mislabelled levels.
# That is exactly what xrsp/build_dataset.py did until this was added: it imported
# ostk.labels (legacy) while this module, the documented source of truth, held v4.
LABELS_LEGACY = {}
for _i, _nm in enumerate(["L1", "L2", "L3", "L4", "L5", "L6", "S1", "sacrum",
                          "left_hip", "right_hip", "femur_left", "femur_right"], start=1):
    LABELS_LEGACY[_nm] = _i
for _n in range(1, 14):
    LABELS_LEGACY[f"T{_n}"] = 12 + _n                      # T1..T13 -> 13..25
for _n in range(1, 9):
    LABELS_LEGACY[f"rib_left_{_n}"] = 25 + _n              # 26..33

LABELS_V4 = dict(LABELS)


def detect_scheme(label) -> str:
    """'v4' or 'legacy', from the SACRUM, which both schemes contain and which is always
    large when it is present at all.

    id 8 is `sacrum` in legacy but `T1` in v4; id 26 is `sacrum` in v4 but `rib_left_1`
    in legacy. On an abdominopelvic CT the sacrum dwarfs both alternatives, so whichever
    id carries the bigger object names the scheme -- measured, on the two forms on disk:

        demo 0003 (legacy):  id 8 = 35797 vox (sacrum),  id 26 =   8542 (rib_left_1)
        HF 0001   (v4):      id 8 = 0                 ,  id 26 = 347776 (sacrum)

    Ties and empties fall back to v4, the current scheme. Pass an ndarray of label ids.
    """
    import numpy as _np
    arr = _np.asarray(label)
    n8 = int((arr == 8).sum())
    n26 = int((arr == 26).sum())
    return "legacy" if n8 > n26 else "v4"


def labels_for(label) -> dict:
    """The name->id map matching `label`'s own scheme. Use this instead of importing a
    fixed map, so a volume in either scheme reads correctly."""
    return LABELS_LEGACY if detect_scheme(label) == "legacy" else LABELS_V4
