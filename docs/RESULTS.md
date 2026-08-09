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

**Pelvic incidence is not obtainable from BUU.** BUU is a lumbar series collimated on
L1–S1. Femoral heads are inside the image bounds on 99.8 % of films (median 3.78
S1-endplate lengths of margin below S1) but are **outside the exposed anatomy** — there is
no cortical arc to find. A pelvis class trained on DRRs fires on **40/40 DRR test images at
0.94 confidence and 0/40 real films**, with exactly one detection at 0.010 when the
threshold is dropped 250×. Vertebrae detect normally on both, so this is not a loading or
class-index fault. PI needs a dataset whose films include the hips.

**The classical circle fit is not a usable pseudolabeller.** `xrsp/hipfit.py` "converged"
on 301/301 films — which is the warning sign, not the result. Against the anatomically
expected femoral-head location it sits a median **0.54 S1-lengths** away, with only 45 %
within 0.5. It is fitting soft-tissue texture.

**Mixing synthetic DRRs into training degraded real-film accuracy.** The combined run is
worst or near-worst on every corner column (ED5 18.2 % vs 33.2 %; SS MAE 2.64° vs 2.01°).
Synthetic data did not act as augmentation here.

**Synthetic corner supervision does not transfer, in either direction.** Corner channels
trained on BUU give 85 px median error on DRRs; the pelvis class trained on DRRs gives
nothing on BUU.

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
