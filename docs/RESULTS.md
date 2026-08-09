# Results

Held-out evaluation on **BUU-LSPINE**, 301 test films, patient-grouped and stratified by
sex × age band. Every model in this document was trained on the same 1398/301/301 split
from the same `buu_splits.json`, and every number was produced by the same metric
implementation (`xrsp/evalmetrics.py`) — so differences are differences in the model, not
in the scoring code.

---

## 1. Landmark localisation on real radiographs

### 1.1 The scale problem, stated before the comparison

Bansal et al. (PLoS One 2026, e0347290) report keypoint accuracy at Euclidean-distance
thresholds of 5/10/15 px on images **resized to 640 × 640**. Our films are used at their
original resolution (median diagonal **3167 px**); theirs live on a 905 px diagonal. **A
pixel in that paper is not a pixel here** — their 5 px is 0.55% of the image diagonal,
ours is 0.16%. Quoting the raw columns side by side would overstate their result by a
factor of ~3.5.

Everything below is therefore compared as a **fraction of image diagonal**, which is
scale-free. Their thresholds are converted into that unit and applied to our errors.

### 1.2 Corner localisation vs the published baseline

| | ED ≤ 0.55% diag<br>(their "5 px") | ED ≤ 1.10% diag<br>(their "10 px") | ED ≤ 1.66% diag<br>(their "15 px") |
|---|---|---|---|
| **YOLOv8l-Pose @1024 (ours)** | **92.4 %** | **99.3 %** | **99.8 %** |
| YOLOv8l-Pose @640 (ours) | 91.4 % | 98.8 % | 99.7 % |
| YOLOv8n-Pose (Bansal, test) | 75.6 % | 98.5 % | 100.0 % |
| YOLOv11n-Pose (Bansal, test) | 76.8 % | 97.4 % | 100.0 % |
| Detectron2-R50 (Bansal, test) | 75.8 % | 98.3 % | 99.8 % |

At the strict threshold we localise **92.4 %** of corners against their **75.6–76.8 %**.
The published models catch up by the loose threshold and marginally exceed us there
(100 % vs 99.8 %) — our distribution is much tighter in the core and retains a thin tail.

Median error is **0.203 % of the image diagonal** (≈ 6.4 px on a typical film).

Two caveats that favour *their* numbers and are not corrected for:

* their keypoint metrics are computed only inside bounding boxes already matched at
  IoU ≥ 0.5, so vertebrae their detector missed never reach the keypoint score. Our
  `all_gt` protocol counts every annotated landmark.
* their dataset is 698 images (208 Honduran + 508 BUU sagittal) against our 2000; part of
  any gap is data volume, not architecture.

### 1.3 Architecture and resolution sweep

All at Bansal Table 4 hyperparameters with every Ultralytics augmentation explicitly
zeroed. `med px` and the ED columns are in **original film pixels** and so are comparable
*within* this table only.

| run | med px | ED5 % | ED10 % | ED15 % | missed | corner-identity F1 | SS ICC | SS MAE ° | SS ≤5° | LL MAE ° |
|---|---|---|---|---|---|---|---|---|---|---|
| **v8l_640** | 6.89 | 33.2 | 70.5 | 87.6 | 62 | **1.000** | 0.952 | **2.01** | **94 %** | **3.93** |
| **v8l_1024** | **6.38** | **36.3** | **74.1** | **89.2** | 160 | 0.999 | **0.953** | 2.05 | 94 % | 4.29 |
| v11n_1024 | 7.64 | 28.2 | 66.1 | 85.7 | 100 | 1.000 | 0.940 | 2.24 | 93 % | 3.75 |
| v11n_640 | 8.30 | 24.8 | 60.8 | 82.2 | 130 | 0.999 | 0.943 | 2.30 | 92 % | 4.31 |
| v11n_1280 | 9.09 | 20.8 | 55.7 | 79.4 | 218 | 0.999 | 0.923 | 2.75 | 89 % | 4.88 |
| combined (BUU+DRR) | 9.74 | 18.2 | 51.8 | 76.1 | 82 | 1.000 | 0.928 | 2.64 | 88 % | 4.35 |

**Capacity beats recency.** YOLOv8l outperforms every v11 variant — the same ordering
Bansal reports, where YOLOv8l-Pose was their most accurate model despite being the older
and, in their words, not the "architecturally recent" choice.

**Resolution helps, then hurts.** 640 → 1024 improves median error (6.89 → 6.38 px) and
strict accuracy (33.2 → 36.3 %). 1280 is *worse than 640* on every column. Past ~1024 the
vertebra occupies more pixels than the receptive field usefully covers, and the
fixed-epoch budget buys fewer effective passes.

---

## 2. Spinopelvic parameters

Computed from predicted corners via `ostk.metrics2d`, on real film, in undistorted
geometry (YOLO letterboxes, so angles are true).

| | ICC(2,1) | MAE | ≤ 5° | bias | 95 % LoA |
|---|---|---|---|---|---|
| **SS** (v8l_640) | **0.952** | **2.01°** | **94 %** | −0.36° | −6.3 to +5.6° |
| **LL** (v11n_1024) | 0.940 | 3.75° | — | — | — |
| SS (heatmap model) | 0.887 | 2.95° | 85 % | +0.9° | −9.8 to +11.6° |
| LL (heatmap model) | 0.865 | 4.85° | 68 % | −0.0° | −17.8 to +17.8° |

SS at **ICC 0.952, MAE 2.01°** is the headline clinical number. For context, published
manual **inter-rater** ICC for the sacral endplate runs as low as 0.41 — the limiting
factor on this measurement in practice is the human reader, not the model.

**LL is consistently the weaker parameter** and the per-level errors say why: it is the
Cobb angle between the two *least reliable* endplates. Per-level median radial error is
1.15–1.20 px at L3/L4 but **1.96 px at S1** and 1.35 px at L1 — and LL is measured between
S1 and L1.

A Theil–Sen fit across all lumbar endplates was tested as a more robust estimator and
**rejected**: MAE 6.68° vs 4.85°. A straight line cannot represent a lordosis with a
mid-lumbar apex, and the linearity error exceeds the outlier resistance it buys.

---

## 3. Corner identity

For each predicted corner, which of its vertebra's four ground-truth corners is nearest.

```
            s_ant s_post i_ant i_post          macro-F1 0.986   kappa 0.981  (heatmap)
  sup_ant  [1480     9     16     0 ]          macro-F1 0.999-1.000          (YOLO)
  sup_post [   1  1484      2    18 ]
  inf_ant  [  20     0   1483     2 ]
  inf_post [   0    17      1   1487 ]
```

Anterior↔posterior confusion is **essentially zero** (9, 1, 2, 0 …); every off-diagonal
entry is a superior↔inferior slip on the same wall. This is a direct empirical
justification for disabling horizontal-flip augmentation: the model has learned
handedness cleanly, which is exactly what a mirror augmentation destroys. Bansal's own
ablation (their Table 11) independently measured horizontal flip as the single most
damaging augmentation on this task.

---

## 4. Negative results

These cost real compute and each one closes a direction.

**A pelvis class trained only on DRRs does not transfer to real film.** It fires on
**40/40 DRR test images at 0.94 confidence and 0/40 real films**, with exactly one
detection at 0.010 when the threshold is dropped 250×. Vertebrae detect normally on both
(359 vs 243 instances), so this is not a loading fault, a class-index fault, or a
threshold artefact — it is specifically the class with no real-domain supervision, and at
inference it does not exist.

Femoral heads ARE within the exposed field. They lie inside the image bounds on 99.8 % of
films (median 3.78 S1-endplate lengths of margin below S1), and inspection of the expected
region at full resolution shows cortical arcs consistent with femoral head and acetabulum
on a substantial fraction. They are low-contrast and superimposed with bowel gas — hard,
not absent.

> **Retraction.** An earlier draft of this section claimed the heads were *outside the
> exposed anatomy* and that PI was therefore unobtainable from BUU. That was inferred from
> downsampled thumbnails plus the zero-detection result — and the second is circular, since
> a model that cannot cross the synthetic-to-real gap finds nothing whether or not a head
> is visible. Full-resolution inspection contradicts it.

What remains open is a question only a reader can settle: on what fraction of films can a
trained eye place the centre with acceptable confidence? The annotation pilot answers it
directly — the skip rate *is* that measurement — and PI on BUU stands or falls on that
number rather than on any automatic method's opinion.

**The classical circle fit is not a usable pseudolabeller.** `xrsp/hipfit.py` "converged"
on 301/301 films — the warning sign rather than the result, since it always returns some
circle. Against the anatomically expected femoral-head location it sits a median **0.54
S1-lengths** away, with only 45 % within 0.5. Whether that reflects a poor fit or a poor
"expected location" prior cannot be separated without human annotation, so it is reported
as unreliable rather than as measured error.

**Mixing synthetic DRRs into training degraded real-film accuracy.** The combined run is
worst or near-worst on every corner column (ED5 18.2 % vs 33.2 %; SS MAE 2.64° vs 2.01°).
Synthetic data did not act as augmentation here.

**Synthetic corner supervision does not transfer, in either direction.** Corner channels
trained on BUU give 85 px median error on DRRs; the pelvis class trained on DRRs gives
nothing on BUU.

---

## 4b. Whole-spine standing films: a scale cliff, and the fix

The detector is trained on coned lateral lumbar films, where L1–S1 fills the frame. On a
standing C2–S1 radiograph the same six vertebrae occupy roughly a third of it. Nothing
else changes — same anatomy, same projection, same weights — so the question is purely
one of **scale at the network input**, and it can be measured on films that already have
ground truth.

Each of 40 test films was pasted into a taller canvas so the lumbar spine occupies a
fraction **f** of the height (`scripts/standing_scale_sweep.py`). f = 1.00 is the coned
film as acquired; **f ≈ 0.36** is where L1–S1 sits on a C2–S1 standing view. Every error
is mapped back to original film pixels, so the columns are comparable across f.

| f | 1.00 | 0.80 | 0.65 | 0.50 | 0.42 | **0.36** | 0.30 | 0.25 | 0.20 |
|---|---|---|---|---|---|---|---|---|---|
| **single pass** — vertebrae found /6 | **6.00** | 5.72 | 2.90 | 0.05 | 0.03 | **0.00** | 0.05 | 0.03 | 0.12 |
| **tiled** — vertebrae found /6 | 5.40 | 5.42 | 5.58 | 5.45 | 5.65 | **5.45** | 5.42 | 5.53 | 5.45 |
| tiled — median corner error, % diag | 0.284 | 0.284 | 0.323 | 0.332 | 0.330 | 0.305 | 0.292 | 0.324 | 0.302 |
| tiled — SS MAE ° | 3.66 | 4.25 | 3.96 | 4.74 | 3.76 | **4.79** | 5.95 | 4.98 | 5.54 |

**A single pass does not degrade on a standing film — it stops working.** Detections go
from 6.0/6 to **zero** between f = 0.65 and f = 0.42. This is not a gradual accuracy loss
that could be tolerated; below about two-thirds framing the model returns nothing at all,
and the handful of "detections" at f ≤ 0.30 are spurious (median error 14–46 % of the
diagonal — the width of the image).

**Tiling is flat.** Overlapping square windows the width of the film, stepped down it at
50 % overlap with one global NMS, hold 5.4–5.7 of 6 at **every** framing tested down to
f = 0.20, with median corner error steady at 0.28–0.33 % of the diagonal. No retraining,
no new labels, no standing-film dataset — the model was never scale-invariant, the
*protocol* was.

The cost is at the other end: at f = 1.00 tiling finds 5.40/6 against the single pass's
6.00, because a tile edge can cut a vertebra the whole-film pass sees intact, and SS MAE
rises 2.79° → 3.66°. So neither is right everywhere, and the deployed policy is to take
the cheap pass and fall back to tiles only when it comes up short.

Two caveats that make this an **upper bound** on standing-film performance:

* the padding is uniform grey. A real standing film's margins contain thoracic vertebrae
  — distractors that look exactly like the objects being counted, and that the level
  chain then has to name correctly.
* the lumbar spine is at native detail, only re-framed. A standing film is usually
  acquired at lower magnification, so the lumbar segment carries fewer real pixels too.

Verified end to end in the browser (`pacs/tools/test_page.py`): at f = 0.36 a single pass
reports *no detections*, and auto-fallback tiles the film into 8 windows and recovers 7
vertebrae with SS within **1.9°** of the same film's coned reading.

---

## 4c. The landmarks do not extrapolate to unlabelled vertebrae

If the detector had learned the generic concept *vertebral body*, whole-spine landmarks
would come free: the class is single ("vertebra"), and most BUU laterals include T12,
T11 and often more, entirely unannotated. Every test film is therefore already a
held-out trial of the question — and it fails.

Across 40 test films (`scripts/extrapolate_above_l1.py`), counting detections that
overlap no annotated vertebra (IoU < 0.2) and sit cranial to L1:

| | detections / film | extra, above L1 | films with any |
|---|---|---|---|
| single pass | 6.10 | **0.10** | **10 %** |
| tiled | 4.72 | 0.07 | 8 % |

Ground truth is six objects per film. The model returns **6.10** — it finds the six it
is scored on and, on nine films in ten, nothing else, even where four or five thoracic
vertebrae are clearly visible in the same exposure.

Where an extra detection does occur it is **T12 and only T12** — the level immediately
above L1, at up to 0.967 confidence — with T11, T10 and T9 visible and ignored in the
same film. Extrapolation reaches about one level and stops.

**The cause is in the labels, not the architecture.** BUU annotates L1–S1 only, so on
all 1398 training films every thoracic vertebra was a positive example of *background*.
The model was not merely left uninformed about them; it was trained to suppress them.
That the one adjacent level survives is consistent with box-boundary slack rather than
with any generic notion of "vertebra".

Consequences for whole-spine landmarking:

* **A standing-film protocol built on tiling still only yields L1–S1.** §4b shows tiling
  restores detection at standing framing; it does not create thoracic landmarks, because
  there is nothing to restore.
* **The no-human-labelling route that remains is DRR pseudo-labels.** The 3-D
  segmentations already label thoracic vertebrae, and `endplate_corners()` (written for
  the demo's reference pane) projects a corner for any level with a label — whole-spine
  corners at zero annotation cost. The blocker is the synthetic-to-real gap, which is
  measured and unsolved here: mixing DRRs into training made real-film accuracy **worse**
  on every column, and a class supervised only on DRRs scored 40/40 on DRRs and 0/40 on
  film.
* **Simply cropping training films to the annotated block would remove the suppression
  signal, not supply a positive one.** It is worth testing as an ablation, but it cannot
  by itself teach a level the model has never been shown as an object.

---

## 5. Methodological findings that changed the numbers

**Anisotropic resize was distorting every angle.** Both loaders filled 512 × 256 by
scaling x and y independently. Source aspect ratio has a median of 0.799 against a 0.500
target, so a **true 45° endplate rendered at 58.7°** — and by a different amount on every
film (anisotropy 0.499–1.219). Agreement metrics survived, since prediction and truth were
distorted identically, but every absolute SS/LL was wrong and the Roussouly thresholds at
35°/45° were being applied to stretched angles. Now letterboxed; verified 10/30/45/60/80°
render as 10.0/30.0/45.0/60.0/80.0.

**Checkpoint selection on validation loss was keeping the untrained network.** MSE over
sparse Gaussian heatmaps is minimised by predicting nothing. Validation loss was lowest at
epoch 0 (38.9 px error) and rising by epoch 29 (3.1 px), so `best.pt` was the untrained
model — and the evaluation stage reads `best.pt`. Selection is now on landmark error.

**Confidence is a weak failure predictor.** AUC **0.662** for flagging a >15 px error;
rejecting the least-confident 10 % of landmarks cuts gross failures only from 2.23 % to
1.51 %. Of 6622 landmarks, **0** were abstentions — every failure is confident and wrong.
Selective prediction is not a viable safety mechanism for the heatmap model.

**Image contrast does not explain the failures.** Correlation of log max-error with
intensity SD is **+0.072**, with p99–p1 range **−0.004**; the single highest-contrast film
is among the worst performers. A CLAHE/normalisation training run was planned on this
hypothesis and cancelled.

---

## 6. Reproduction

```
scripts/buu_to_yolo.py          BUU -> Ultralytics keypoint format (aspect preserved)
scripts/train_yolo_pose.py      Bansal Table 4 hyperparameters, augmentation zeroed
scripts/evaluate_yolo.py        both scoring protocols, same metrics as the heatmap model
scripts/tabulate_experiments.py one table across all runs
slurm/xrsp_yolo.sh              MODEL=... IMGSZ=... TAG=... sbatch
```

Conversion is verified by round-trip: decoded labels return to the source annotations
within **0.0014 px**.
