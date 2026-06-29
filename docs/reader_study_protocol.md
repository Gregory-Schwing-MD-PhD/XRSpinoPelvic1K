# XRSpinoPelvic1K reader study — how to measure spinopelvic angles on a lateral radiograph

**Purpose.** We are validating an automatic 2D spinopelvic measurement against a 3D ground
truth (the same angles computed from the source CT). Your hand measurements on the **lateral
synthetic radiographs** are the *realistic manual standard*: we compare your numbers to both
the automatic 2D measurement and the 3D ground truth. Please measure exactly as you would on a
routine lateral spine film.

> You are measuring on a **lateral** (side) view. Treat it like any standing lateral lumbar/
> spinopelvic film. Do **not** look up or infer the "computer" values — measure independently.

---

## What you will measure (4 angles)

All four use standard definitions; the identity **PI = SS + PT** should hold to a few degrees
and is a built-in self-check. Radiopaedia references are linked for each (confirm the exact URL
by searching the article title on radiopaedia.org).

### 1. Sacral slope (SS)
- **Definition:** angle between the **superior endplate of S1** and the **horizontal**.
- **How:** draw a line along the S1 superior endplate; draw a horizontal reference; measure the
  acute angle between them.
- Ref: Radiopaedia — *"Sacral slope."*

### 2. Pelvic tilt (PT)
- **Definition:** angle between the **vertical** and the line from the **center of the femoral
  heads** to the **midpoint of the S1 superior endplate**.
- **How:** mark the femoral-head center (on a true lateral the two heads nearly overlap — use
  the center of the overlapping circle; if they are offset, use the midpoint of the two
  centers). Mark the midpoint of the S1 endplate. Draw the connecting line and a vertical
  reference; measure the angle.
- Ref: Radiopaedia — *"Pelvic tilt (spinal)."*

### 3. Pelvic incidence (PI)
- **Definition:** angle between (a) the **line perpendicular to the S1 superior endplate** at
  its midpoint and (b) the **line from the S1 endplate midpoint to the femoral-head center**.
- **How:** at the S1 midpoint, draw the perpendicular to the S1 endplate; draw the line to the
  femoral-head center; measure the angle between them. **Sanity check: PI should ≈ SS + PT.**
- Ref: Radiopaedia — *"Pelvic incidence."*

### 4. Lumbar lordosis (LL)
- **Definition (the one our engine uses):** **Cobb angle** between the **superior endplate of
  L1** and the **superior endplate of S1**.
- **How:** draw a line along the L1 superior endplate and along the S1 superior endplate;
  measure the Cobb angle (the angle between them, via their perpendiculars if your tool does
  Cobb directly).
- Refs: Radiopaedia — *"Lumbar lordosis"* and *"Cobb angle."*

Overview reference: Radiopaedia — *"Spinopelvic alignment / sagittal balance."*

---

## Tool

The films are PNGs. Any viewer with a line/angle (goniometer) tool works. Recommended free
option: **Fiji/ImageJ** (`Straight line` + `Measure` for angles, or the `Angle` tool). RadiAnt,
Horos/OsiriX, or even the angle tool in PowerPoint/Preview are fine. Use whatever you already
measure films with.

Each XRSpinoPelvic1K case folder contains:
- `lateral_drr.png` — **measure on this** (the lateral synthetic radiograph).
- `ap_drr.png` — frontal view, for orientation only (do not measure on it).
- (`*_mask.png`, `*_levels.json`, `*_drr.npy` are for the automatic pipeline — **ignore** them
  while reading; do not open them, to stay blind.)

---

## Procedure (blinded)

1. You will receive a worklist of case IDs in **randomized order** (a coordinator assigns them;
   do not read them in numeric order, and do not open the mask/landmark files).
2. For each case, open `lateral_drr.png`, measure **SS, PT, PI, LL** in degrees.
3. Record each value to one decimal place in the data sheet (below). Note anything that made a
   case hard (e.g., femoral heads unclear, S1 endplate ambiguous, transitional anatomy).
4. If a structure needed for an angle is not visible (e.g., hips cropped → no PI/PT), leave it
   blank and note "not visible."
5. Optional but useful: record measurement time per case (supports an efficiency comparison).

Two readers measuring the same cases lets us report **inter-reader variability** and an
intraclass correlation — please coordinate so there is overlap.

---

## Data entry

Fill in `reader_measurements_template.csv` (one row per case per reader):

| column | meaning |
|---|---|
| `case_id` | case folder name (e.g. `0003`) |
| `reader_id` | your initials/code |
| `PI_deg` | pelvic incidence (°), blank if hips not visible |
| `SS_deg` | sacral slope (°) |
| `PT_deg` | pelvic tilt (°), blank if hips not visible |
| `LL_deg` | lumbar lordosis L1–S1 Cobb (°) |
| `seconds` | time to measure this case (optional) |
| `notes` | difficulty / transitional anatomy / anything odd |

Return the completed CSV. We compare your `*_deg` values against the automatic 2D measurement
and the 3D ground truth (mean absolute error, intraclass correlation, Bland–Altman limits of
agreement), and report inter-reader variability where two readers overlap.

---

## Notes / caveats for readers

- These are **digitally reconstructed radiographs** from CT — they look like a clean lateral
  spine film by design. Measure them as you would a real lateral.
- The patient is **supine/prone** (as in CT and as in intraoperative fluoroscopy), not standing
  — absolute SS/PT/LL therefore differ from a standing film, but that does **not** affect this
  study, which compares your 2D reading to the 3D value on the *same* image.
- Measure independently; the point is to capture genuine human reading variability.
