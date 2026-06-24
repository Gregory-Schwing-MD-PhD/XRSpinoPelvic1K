"""XRSpinoPelvic1K — open spinal-radiograph DRR dataset + real-time level localization,
generated from segmented CT (CTSpinoPelvic1K / OpenSpineToolkit)."""
from .drr import drr_project, projection_plan, to_uint8
from .project_labels import (footprints_to_mask, project_footprints,
                             project_level_points)
from .localize import (gaussian_heatmaps, level_at_point, points_from_heatmaps)
from .labels import LABELS, NAMES, SPINE_LEVELS

__version__ = "0.1.0"
__all__ = [
    "drr_project", "projection_plan", "to_uint8",
    "project_footprints", "project_level_points", "footprints_to_mask",
    "gaussian_heatmaps", "level_at_point", "points_from_heatmaps",
    "LABELS", "NAMES", "SPINE_LEVELS",
]
