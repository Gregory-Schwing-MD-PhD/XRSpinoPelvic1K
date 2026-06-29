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
