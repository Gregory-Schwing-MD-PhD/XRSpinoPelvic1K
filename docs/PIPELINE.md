# Pipeline — CT masks → DRR → amodal endplate model → real radiographs

End-to-end plan for learning the **lumbosacral endplate corners** on lateral
radiographs, trained on labels propagated from 3-D CT.

## The claim being tested

On a lateral film the sacral ala superimposes on the S1 body, so the S1 endplate
cannot be recovered from what is visible. Measured on case 0003: fitting the
endplate from the projected **silhouette** is **26–32° wrong** at every slab
fraction from 0.30 down to 0.04 — it does not improve with tuning, because the
medial–lateral coordinate that separates ala from body does not survive projection.

Select the endplate in **3-D** and project the result and the same case comes out
at **0.6°** (S1 endplate 32.9° vs ostk SS 32.31°; L1→S1 45.1° vs LL 45.70°).

So: the target is recoverable **iff the ground truth comes from 3-D**. That is what
the model is being asked to learn, and why it can beat a human on a real film —
published manual inter-rater ICC for the sacral endplate is as low as 0.41, and the
cause named in the literature is exactly this ala/ilium shadow overlap.

## Stage 1 — Generation

`xrsp.oblique` renders along an arbitrary direction and projects labels through the
identical plan, so ground truth stays exact at any angle.

- **N views per CT** via `sample_view(rng, yaw_deg=12, pitch_deg=8, roll_deg=6)`.
  Yaw is the one that matters: it changes ala/body superimposition. Defaults are
  clinical (a real "lateral" is within ~10–15°), not extreme.
- **Appearance randomization is DeepDRR's job**, not ours: polyenergetic spectrum,
  Compton scatter, detector noise. `xrsp.oblique.render` is parallel-beam and
  deliberately does not reimplement it. Plug DeepDRR in behind the same plan.
- Per view, emit: DRR, amodal per-class masks, **endplate corners** (L1…L5, S1;
  `endplate_corners_2d`), femoral-head centre, and the view parameters.

**Every view of one CT carries the same `case_id` and the same `patient_id`.**
Generation must never be the thing that decides the split.

## Stage 2 — Splitting (`xrsp.splits`)

Three constraints, in priority order:

1. **Patient-grouped.** All views of all CTs of one patient land in one fold. A
   randomized view is not a new subject; leaking views across folds inflates every
   number and is the single easiest way to make this work invalidly.
2. **LSTV-stratified.** Mirrors the CTSpinoPelvic1K splitter. L6/sacralization
   cases are rare and are the cases the paper is about — they must not clump.
3. **Level-coverage stratified.** Cases where the femoral heads are in the FOV are
   the only ones where PI/PT are measurable; balance them so every fold can report
   the same parameters.

5-fold, generated **once**, stored as JSON, and version-controlled. Never regenerate
silently — a resplit invalidates every previously reported number.

## Stage 3 — Training

Target: **landmark heatmaps**, not a mask.

A mask cannot express the endplate on a projection (Stage 0's whole point), and a
1–2 px endplate ribbon is unstable to segment and would still need a line fit —
reintroducing the fitting step that fails. Corners are unambiguous, and the angle
you care about is far more sensitive to line *orientation* than to a couple of
pixels of position, which favours two well-localised endpoints.

- Channels: 2 corners × {L1…L5, S1} + femoral head = 13.
- Keep the **amodal segmentation as an auxiliary head**. It is useful for the
  dataset and the paper, and it is a strong auxiliary task — but do not compute
  angles from it.
- Augment: intensity/gamma/contrast, random crop and FOV, mild elastic. Geometry
  augmentation belongs in *generation* (Stage 1) where labels follow exactly, not
  in the loader where they would have to be warped.

## Stage 4 — Evaluation: two test sets, two different questions

This is the part that is easy to get wrong. They are **not** two attempts at the
same measurement.

### 4a. Held-out CTSP1K DRRs — *internal validity*

Ground truth is amodal and exact. Measures whether the network learned the target.

- Corner error in **mm**, SS / LL / PI error in **degrees**, per LSTV stratum.
- This is an upper bound, not a claim about real radiographs. Same renderer, same
  physics, same distribution: a DRR hold-out validates the label propagation and
  the fit, not the domain. **Reported as in-silico.** On its own it is the weakest
  possible position for a claim about occluded anatomy.

### 4b. BUU Spine (real standing laterals) — *external validity*

Real domain, real posture (so SS/PT are meaningful, not just posture-invariant PI),
and human-annotated corners. **Evaluation only — do not re-host** (see ROADMAP §3).

The catch that is also the opportunity: **BUU's corners are what a human could see.**
For S1 that ground truth carries the very ala confusion the model is meant to beat.
So BUU **cannot** score amodal accuracy. Against BUU you report *agreement with human
readers*, in the statistic radiologists use (corner distance; ICC on SS/PI).

### 4c. The bridge — reader study on DRRs (`scripts/make_reader_set.py`)

This is what makes 4b interpretable, and it needs no CT/X-ray pairs.

Hand raters blinded DRRs — where amodal truth **is** known — and have them draw the
S1 endplate. That yields:

- the **human noise floor** in our own data (compare to the published ICC 0.41);
- the **signed human bias**: how far, and in which direction, readers place the S1
  endplate when the ala overlies it.

Then the logic closes:

> If the model disagrees with BUU's human annotations **by the same signed bias**
> measured in 4c, that disagreement is evidence the model is right and the readers
> are systematically wrong — not evidence of model error.

Without 4c, a disagreement on BUU is uninterpretable and a reviewer will read it as
failure. With it, disagreement in the predicted direction is the result.

## Stage 5 — Real pairs (the honest ceiling)

None of the above is a substitute for paired CT + radiograph of the same subject.
Two routes that do not need a prospective protocol:

- **DeepFluoro** — 6 cadaver pelvic CTs, 366 real X-rays, calibrated intrinsics and
  ground-truth pose per image, plus 2-D pelvis/femur segmentations. Known pose means
  the propagated labels can be rendered at the exact geometry of a real radiograph:
  genuine pixel correspondence. Six specimens is a domain-gap probe, not a cohort —
  but it is C-arm imagery, so it also feeds the fluoroscopy work.
- **Retrospective chart pull.** Any patient with a lumbar CT and standing films
  weeks apart qualifies. A few dozen moves this from "in-silico" to "validated", and
  it is the kind of ask OpenSpineConsortium is built to distribute across sites.

## What would make this fail

Named so they get checked, not discovered at review:

- **View leakage** across folds. Guarded in Stage 2; assert it in tests.
- **Perfect-lateral-only training.** Ablate it: randomized vs fixed view, on the
  reader set. If randomization does not help, the domain claim is weak.
- **Supine CT vs standing radiograph.** PI is posture-invariant; SS and PT are not.
  Any SS/PT claim transferred from supine DRRs to standing films must say so.
- **Amodal targets that no one can verify.** Mitigated by 4c, which is the only
  place a human and the amodal truth can be compared at all.
