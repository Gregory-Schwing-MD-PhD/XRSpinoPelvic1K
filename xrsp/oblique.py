"""Oblique DRR rendering + geometry-consistent label projection.

WHY THIS EXISTS
---------------
`drr.drr_project` integrates with `mu.sum(axis=plan["proj"])` -- a pure axis-aligned
sum -- so every DRR it can produce is a *perfect* lateral (or AP). Real lateral films
never are: there is always a few degrees of rotation, and that obliquity is exactly
what changes how the sacral ala superimposes on the S1 body. A model trained only on
perfect laterals learns a projection that does not occur clinically, and the S1
endplate -- the thing we are trying to see through the ala -- is where it will fail.

So this module renders along an ARBITRARY direction by resampling the volume along
rays, and projects the labels through the IDENTICAL geometry, so the 2-D ground truth
stays exact no matter how oblique the view is. That turns one training image per CT
into as many as you want, which is the geometric half of the domain randomization
SyntheX relies on. (The appearance half -- polyenergetic spectrum, Compton scatter,
detector noise -- is what DeepDRR does; this module deliberately does not reimplement
it.)

The labels are the point: because they are propagated from 3-D, a vertebral body's
footprint is its TRUE extent, including the part a human cannot see through the
overlying ala or ilium. That is amodal ground truth, and it cannot be drawn on a real
radiograph by anyone.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


def _u(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n else v


def _rot(axis, deg):
    """Rodrigues rotation matrix, `deg` degrees about `axis`."""
    a = _u(axis)
    t = np.deg2rad(float(deg))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def sample_view(rng, *, lr_world=(1, 0, 0), sup_world=(0, 0, 1),
                yaw_deg: float = 12.0, pitch_deg: float = 8.0, roll_deg: float = 6.0
                ) -> Dict:
    """A randomly perturbed lateral viewing direction.

    yaw   -- rotation about the SUPERIOR axis: the patient turned off true lateral.
             This is the one that matters: it changes ala/body superimposition.
    pitch -- rotation about the anterior axis: beam tilted cranio-caudally.
    roll  -- rotation of the detector in-plane (film rotation).

    Defaults are deliberately clinical, not extreme: a real 'lateral' is usually
    within ~10-15 deg. Pass 0 for a perfect lateral (the old behaviour).
    """
    sup = _u(sup_world)
    lr = _u(lr_world)
    ant = _u(np.cross(sup, lr))
    yaw = float(rng.uniform(-yaw_deg, yaw_deg)) if yaw_deg else 0.0
    pitch = float(rng.uniform(-pitch_deg, pitch_deg)) if pitch_deg else 0.0
    roll = float(rng.uniform(-roll_deg, roll_deg)) if roll_deg else 0.0
    R = _rot(sup, yaw) @ _rot(ant, pitch)
    return {"direction": _u(R @ lr), "sup": _u(R @ sup),
            "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll}


def oblique_plan(affine, direction, sup, *, roll_deg: float = 0.0,
                 bounds_world: np.ndarray = None, pixel_spacing_mm: float = 1.0,
                 margin_mm: float = 20.0) -> Dict:
    """Define the detector grid for a projection along `direction`.

    Returns a plan consumed by BOTH `render` and `project_footprints`, so the image
    and the labels can never drift apart -- the failure that makes propagated labels
    worthless.
    """
    d = _u(direction)
    up = _u(sup - np.dot(sup, d) * d)                 # cranial, in-plane
    if not np.any(up):                                # degenerate: beam along sup
        up = _u(np.cross(d, (1, 0, 0)))
    right = _u(np.cross(up, d))                       # in-plane, completes the frame
    if roll_deg:
        R = _rot(d, roll_deg)
        up, right = _u(R @ up), _u(R @ right)
    pts = np.asarray(bounds_world, float)
    u = pts @ right
    v = pts @ up
    depth = pts @ d
    u0, u1 = u.min() - margin_mm, u.max() + margin_mm
    v0, v1 = v.min() - margin_mm, v.max() + margin_mm
    d0, d1 = depth.min() - margin_mm, depth.max() + margin_mm
    W = max(int(round((u1 - u0) / pixel_spacing_mm)), 1)
    H = max(int(round((v1 - v0) / pixel_spacing_mm)), 1)
    return {"direction": d.tolist(), "right": right.tolist(), "up": up.tolist(),
            "u0": float(u0), "v0": float(v0), "d0": float(d0), "d1": float(d1),
            "shape": [H, W], "pixel_spacing_mm": float(pixel_spacing_mm),
            "roll_deg": float(roll_deg)}


def _grid(plan):
    H, W = plan["shape"]
    sp = plan["pixel_spacing_mm"]
    right = np.asarray(plan["right"], float)
    up = np.asarray(plan["up"], float)
    us = plan["u0"] + (np.arange(W) + 0.5) * sp
    vs = plan["v0"] + (np.arange(H) + 0.5) * sp
    U, V = np.meshgrid(us, vs)
    # rows are built so that row 0 is the MOST CRANIAL -> superior appears up.
    #
    # KNOWN ISSUE: columns are NOT flipped, so these renders put anterior on the RIGHT,
    # while xrsp/drr.py, the existing dataset and the PACS demo put it on the LEFT.
    # Do not "fix" it by negating `right` (makes the basis left-handed) or by mirroring
    # the column in the pixel maps (changes which corner measure.py treats as anterior).
    # Both were tried: SS went 32.9 -> 6.9 and the PI identity blew out to 66 deg.
    # The flip has to be applied consistently across _grid, project_footprints, every
    # to_px, AND measure.py's anterior convention, in one deliberate change with the
    # angle checks green at each step.
    base = U[..., None] * right + V[..., None] * up
    return base[::-1, :, :]


def render(volume, affine, plan, *, emphasis: str = "bone", bone_hu: float = 150.0,
           hu_floor: float = -1000.0, hu_ceil: float = 2000.0, gamma: float = 1.0,
           interp_order: int = 1) -> np.ndarray:
    """Path-integrate attenuation along the plan's rays. Parallel beam."""
    d = np.asarray(plan["direction"], float)
    sp = plan["pixel_spacing_mm"]
    base = _grid(plan)
    depths = np.arange(plan["d0"] + 0.5 * sp, plan["d1"], sp)
    inv = np.linalg.inv(np.asarray(affine, float))
    vol = np.asarray(volume, np.float32)
    H, W = plan["shape"]
    acc = np.zeros((H, W), np.float64)
    for t in depths:
        world = base + t * d
        flat = world.reshape(-1, 3)
        ijk = (np.c_[flat, np.ones(len(flat))] @ inv.T)[:, :3]
        hu = ndimage.map_coordinates(vol, ijk.T, order=interp_order,
                                     mode="constant", cval=hu_floor).reshape(H, W)
        if emphasis == "bone":
            acc += np.clip(hu - bone_hu, 0.0, hu_ceil - bone_hu)
        else:
            acc += np.clip(hu, hu_floor, hu_ceil) - hu_floor
    img = acc.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img ** float(gamma) if gamma != 1.0 else img


def project_footprints(label, affine, plan, ids: Optional[Iterable[int]] = None
                       ) -> Dict[int, np.ndarray]:
    """AMODAL 2-D footprint of each label id under the SAME geometry as `render`.

    'Amodal' because it is the union along the ray of the 3-D structure: a vertebral
    body's true silhouette, including where the ala or ilium overlies it on the film.
    A human annotating the radiograph cannot produce this -- which is the whole point.
    """
    lab = np.asarray(label)
    H, W = plan["shape"]
    sp = plan["pixel_spacing_mm"]
    right = np.asarray(plan["right"], float)
    up = np.asarray(plan["up"], float)
    # ONE pass over the volume, then group by label value. The obvious loop -- `lab == i`
    # per id -- rescans all 158M voxels of a torso CT for each of ~25 ids and cost 122 s
    # a case, by far the pipeline's dominant expense (`render` itself is 1.6 s). The
    # projection is identical; only the traversal changed.
    idx = np.array(np.nonzero(lab)).T
    if not len(idx):
        return {}
    vals = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
    keep = set(int(v) for v in (ids if ids is not None else np.unique(vals)))
    world = (np.c_[idx, np.ones(len(idx))] @ np.asarray(affine, float).T)[:, :3]
    c = np.floor((world @ right - plan["u0"]) / sp).astype(int)
    r = (H - 1) - np.floor((world @ up - plan["v0"]) / sp).astype(int)
    ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
    out = {}
    for i in np.unique(vals):
        i = int(i)
        if i == 0 or i not in keep:
            continue
        sel = ok & (vals == i)
        if not sel.any():
            continue
        m = np.zeros((H, W), bool)
        m[r[sel], c[sel]] = True
        out[i] = m
    return out


def _level_points_world(lab, affine, level_id: int, largest_component):
    """World-mm points of one level's largest connected component.

    Crops to the level's bounding box BEFORE the connected-component pass. That pass is
    the pipeline's dominant cost -- it runs once per level per endplate face, ~40 times a
    case, and over a full torso volume it made a single case take longer than the whole
    rest of generation. One vertebra occupies a tiny fraction of the volume, so cropping
    is a pure win: same component, a fraction of the work.
    """
    m = (lab == level_id)
    if not m.any():
        return np.empty((0, 3))
    sl = ndimage.find_objects(m.astype(np.uint8))[0]
    sub = largest_component(m[sl])
    idx = np.array(np.nonzero(sub)).T
    if not len(idx):
        return np.empty((0, 3))
    idx = idx + np.array([sl[0].start, sl[1].start, sl[2].start])
    return (np.c_[idx, np.ones(len(idx))] @ affine.T)[:, :3]


def endplate_corners_2d(label, affine, plan, level_id: int, *, level_name: str = None,
                        which: str = "superior", min_voxels: int = 50,
                        ostk_path: str = None, corner_params: dict = None,
                        anatomic: bool = True, points=None):
    """The two 2-D corners of one vertebra's endplate: FIT IN 3-D, then projected.

    Delegates the fit to ostk (spine.endplate_from_label / corner_params_for_level),
    which restricts to the central medial band -- dropping the sacral ala and the
    transverse processes -- and iteratively rejects outliers. That restriction is the
    whole ballgame and it is only possible in 3-D.

    A first version of this selected the slab by global cranial height instead. On a
    tilted endplate a horizontal slab is a cap ACROSS the endplate, not the endplate:
    it returned both corners at the same height (a flat line) for an S1 endplate
    actually inclined ~32 deg. That is the same failure as fitting the 2-D silhouette,
    which on case 0003 is 26-32 deg off at every slab fraction. Do not reintroduce it.

    Returns (anterior_xy, posterior_xy) in detector pixels, or None.
    """
    import sys
    if ostk_path and ostk_path not in sys.path:
        sys.path.insert(0, ostk_path)
    try:
        from ostk.spine import (endplate_corners, endplate_corners_anatomic,
                                corner_params_for_level)
        from ostk.masks import binary_mask, largest_component, mask_world
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(
            "endplate_corners_2d needs OpenSpineToolkit for the 3-D endplate fit "
            f"(pass ostk_path=...): {type(exc).__name__}: {exc}") from exc
    # Name from the volume's OWN scheme (see xrsp.labels.detect_scheme); ostk.labels is
    # the legacy map and mis-names every level on a v4 volume.
    from .labels import labels_for
    name = level_name or {v: k for k, v in labels_for(label).items()}.get(int(level_id))
    if name is None:
        return None
    # `points` lets the caller extract this level once and reuse it for BOTH faces --
    # the extraction is a full-volume compare plus a component pass, and doing it twice
    # per level was 48 s a case.
    pts = (np.asarray(points, float) if points is not None else
           _level_points_world(np.asarray(label), np.asarray(affine, float),
                               int(level_id), largest_component))
    if len(pts) < min_voxels:
        return None
    d = np.asarray(plan["direction"], float)
    up3 = np.asarray(plan["up"], float)
    _params = dict(corner_params) if corner_params is not None else         dict(corner_params_for_level(name))
    kw = {k: v for k, v in _params.items()
          if k in endplate_corners.__code__.co_varnames}
    # ANATOMIC corners for annotation: same fitted line, anterior end run out to the
    # cortical margin (ostk's `ant_skip` deliberately holds it 6-14 mm inside, which is
    # right for a Cobb line and wrong for a label). The corner slides ALONG the fitted
    # line, so the direction -- hence every angle -- is unchanged; see the function's
    # docstring for the two approaches that failed before this one.
    fn = endplate_corners_anatomic if anatomic else endplate_corners
    res = fn(pts, normal_axis=up3, which=which, lr=d, **kw)
    if res is None:
        return None
    p_lo, p_hi = np.asarray(res[0], float), np.asarray(res[1], float)
    right = np.asarray(plan["right"], float)
    up = up3
    sp = plan["pixel_spacing_mm"]
    H, W = plan["shape"]

    def to_px(P):
        return np.array([(P @ right - plan["u0"]) / sp - 0.5,
                         (H - 1) - ((P @ up - plan["v0"]) / sp - 0.5)])
    a_px, b_px = to_px(p_lo), to_px(p_hi)
    return (a_px, b_px) if a_px[0] <= b_px[0] else (b_px, a_px)


# ── 4-corner vertebral annotation ────────────────────────────────────────────────

def anterior_sign(plan, sup_axis=None) -> float:
    """+1 if patient-anterior runs along increasing pixel-x in `plan`, else -1.

    Derived from the plan's own basis rather than assumed, so it stays right for the
    oblique views too -- yaw rotates `right`, and a hardcoded sign would silently
    mislabel corners on every view except the true lateral.

    Needed because ostk's 3-D corner fit returns (anterior, posterior) in WORLD terms,
    while `_grid` builds columns without a flip (see the KNOWN ISSUE there), so on this
    detector anterior lands at +x. Emitting them in ostk's order put `sup_ant` at x=109
    and `sup_post` at x=143 with the femoral heads -- which are anterior -- at x=178.
    Angles are unaffected either way (measure.py takes the undirected line, verified:
    SS/LL/PI/PT identical before and after), but the names have to be right to compare
    against BUU, whose reader corners carry true anterior/posterior identity.
    """
    from .oblique import _u                                   # local: keeps import order
    sup = np.asarray(sup_axis if sup_axis is not None else (0.0, 0.0, 1.0), float)
    d = _u(np.asarray(plan["direction"], float))
    right = np.asarray(plan["right"], float)
    # anterior = sup x lr, the same right-handed convention ostk.spine.anterior_axis uses
    ant = _u(np.cross(sup, d))
    sgn = float(np.dot(ant, right))
    return 1.0 if sgn >= 0 else -1.0


CORNER_KEYS = ("sup_ant", "sup_post", "inf_ant", "inf_post")

# Corner parameters for ANNOTATION, which is not the same job as plane fitting.
#
# ostk's corner_params_for_level is tuned to fit a robust endplate PLANE: it drops the
# posterior 42% and skips the anterior 8% so an osteophyte lip or the canal down-slope
# cannot tilt the fit. Excellent for an angle; wrong for a training target, because the
# surviving "corners" then sit well inside the body -- measured on case 0003 they spanned
# only 36-53% of the vertebral body's A-P width, and a model trained on them would place
# corners inside the vertebra while BUU's human annotators mark the true cortical margins.
#
# So annotation keeps the MEDIAL BAND (lat_frac -- that is what drops the sacral ala, and
# it is the whole reason the endplate is recoverable at all) and drops the trimming.
# Span goes to 69-92%.
# CORRECTION: an earlier version set drop_post=0.0, ant_skip=0.0 here to make the corner
# span cover more of the vertebral body (36-53% -> 69-92%). That was WRONG and is kept
# here as a warning. drop_post is not cosmetic trimming: it exists to stop the posterior
# down-slope toward the spinal canal from tilting the endplate fit. Removing it wrecked
# the ORIENTATION, which is what every angle depends on -- SS went 32.89 -> 6.87 and LL
# 45.06 -> 30.72 while the corners still passed every "on-bone" check. Span is not
# correctness.
#
# So annotation uses ostk's own per-level parameters. The open question this leaves is
# real but separate: these corners are ENDPLATE-LINE endpoints, not the true cortical
# margins a human annotator marks, so a model trained on them will not be directly
# comparable to BUU's corner annotations without reconciling the two definitions. That
# needs a deliberate decision (extend along the fitted line to the cortical edge? train
# on the line and derive corners? annotate a reconciliation subset?), not a parameter
# tweak -- see docs/PIPELINE.md.
ANNOTATION_CORNER_PARAMS = None


def extend_to_cortex(p_a, p_b, footprint, **kw):
    """DISABLED -- returns the fitted corners unchanged. Kept as a record.

    Three attempts to push the endplate ends out to the cortex all failed, and the
    reason they failed is that the premise was wrong: ostk's corners are ALREADY at the
    vertebral body's cortical margins. Measured on case 0003, superior endplate span vs
    vertebral BODY A-P depth --

        L1  30.5 mm / 32.8 mm      L2  33.9 mm / 35.9 mm      L3  35.3 mm / 34.4 mm

    (an adult lumbar body is ~30-38 mm A-P). They only looked short because the span was
    first compared against the WHOLE vertebra, 86-92 mm, which includes the spinous
    process and posterior elements.

    The three failures, so they are not repeated:
      1. walk outward from the midpoint until leaving the footprint -- the endplate lies
         ON the footprint boundary, so it exits at once (spans 0-16 px, L3 exactly 0);
      2. take the outright extremes of a band around the line -- swallows the posterior
         elements (spans 51-73 mm);
      3. grow outward but stop at a gap -- there IS no gap: in projection the pedicles
         and lamina superimpose over the spinal canal, so the bone is contiguous from
         body to spinous process along the line (spans 53-74 mm).

    Any future attempt has to separate body from posterior elements in 3-D BEFORE
    projecting, because that separation does not survive the projection.
    """
    return p_a, p_b
def vertebra_corners_2d(label, affine, plan, level_id: int, *, level_name: str = None,
                        min_voxels: int = 50, ostk_path: str = None,
                        corner_params: dict = None, footprint=None):
    """All FOUR corners of one vertebra -- both endplates -- in detector pixels.

        sup_ant  ---- sup_post      superior endplate
           |             |
        inf_ant  ---- inf_post      inferior endplate

    Four corners per level is the standard vertebral annotation on spine radiographs
    (and what corner-annotated real datasets such as BUU provide), so a model trained
    on these is directly comparable to real ground truth without a conversion step.
    It also gives every SEGMENTAL angle for free: the disc space at L4/5 is the angle
    between L4's inferior endplate and L5's superior endplate.

    Each endplate is fitted in 3-D and then projected -- never fitted from the 2-D
    silhouette, which is 26-32 deg wrong because the ala superimposes on the body.

    Returns {"sup_ant": [x, y], ...} or None.
    """
    out = {}
    try:
        from ostk.masks import largest_component
        _pts = _level_points_world(np.asarray(label), np.asarray(affine, float),
                                   int(level_id), largest_component)
    except Exception:                                      # noqa: BLE001
        _pts = None
    # S1 gets NO inferior endplate: it is fused to S2, so there is no disc space and no
    # inferior endplate to mark. Emitting one drew a line across the middle of the
    # sacrum, which is exactly as wrong as it looked.
    _sides = (("superior", ("sup_ant", "sup_post")),)
    if (level_name or "") != "S1":
        _sides = _sides + (("inferior", ("inf_ant", "inf_post")),)
    for which, keys in _sides:
        try:
            c = endplate_corners_2d(label, affine, plan, level_id, level_name=level_name,
                                    which=which, min_voxels=min_voxels, points=_pts,
                                    ostk_path=ostk_path, corner_params=corner_params)
        except Exception:                                       # noqa: BLE001
            c = None
        if c is None:
            continue
        a, b = c
        if footprint is not None:
            a, b = extend_to_cortex(a, b, footprint)     # out to the cortical border
        # Name the ends by where they actually land on THIS detector, not by the order
        # ostk returns them in -- see anterior_sign.
        if (b[0] - a[0]) * anterior_sign(plan) > 0:
            a, b = b, a
        out[keys[0]] = [float(x) for x in a]
        out[keys[1]] = [float(x) for x in b]
    return out or None


def pi_anchor_2d(corners: dict, *, mode: str = "corner"):
    """The S1 point PI/PT are measured from, in detector pixels.

    mode="corner"   (DEFAULT, radiographic convention) -- bisect the superior endplate
        between its anterior and posterior corners. This is the operational method
        Legaye/Duval-Beaupere defined PI with on lateral radiographs, so it is the
        convention the published PI norms carry. It is also the only one derivable
        from landmarks a model can see: both corners are visible cortical points.

    mode="overmask"  -- ostk's alternative: the centre of the endplate portion actually
        backed by vertebral body, projected onto the same rim line. Anatomically
        well-argued (the endplate IS the body's superior surface) and more robust to a
        degenerate or osteophytic corner. It sits ~21% of the rim anterior of the
        bisector on case 0003 (7.4 mm along the endplate, 0.0 mm perpendicular), which
        moves PI/PT by ~2-3 deg and leaves SS/LL untouched. NOT derivable from the two
        corners -- it needs the 3-D body mask -- so on a radiograph it would have to be
        its own predicted channel, with no visible landmark under it. Kept because the
        argument for it is real and it may render better; see docs/PIPELINE.md.
    """
    if mode == "corner":
        a, p = corners.get("sup_ant"), corners.get("sup_post")
        if a is None or p is None:
            return None
        return [0.5 * (a[0] + p[0]), 0.5 * (a[1] + p[1])]
    if mode == "overmask":
        return corners.get("sup_overmask")            # emitted by the builder when asked
    raise ValueError(f"unknown pi anchor mode {mode!r}")
