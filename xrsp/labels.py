"""Label-id ↔ name map (shared with OpenSpineToolkit / CTSpinoPelvic1K v3 scheme).

Vendored so this repo runs standalone; keep in sync with ostk.labels."""
from __future__ import annotations

LABELS = {
    "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6, "S1": 7,
    "sacrum": 8, "left_hip": 9, "right_hip": 10, "femur_left": 11, "femur_right": 12,
}
for _i in range(1, 14):
    LABELS[f"T{_i}"] = 12 + _i            # T1..T13 -> 13..25

NAMES = {v: k for k, v in LABELS.items()}

# the levels we localize on a spinal radiograph (cranio-caudal)
SPINE_LEVELS = [f"T{n}" for n in range(1, 14)] + \
               ["L1", "L2", "L3", "L4", "L5", "L6", "S1"]


def lid(name: str) -> int:
    return LABELS[name]


def lname(i: int):
    return NAMES.get(int(i))
