# Roadmap — XRSpinoPelvic1K

## Status
**Done (working + tested):**
- DRR engine: parallel-beam lateral/AP, plus **arbitrary-angle oblique** rendering
  (`xrsp/oblique.py`) with labels re-projected through the identical geometry
- 4 endplate corners for **every vertebra visible in a scan** + the bicoxofemoral
  point, all fitted in 3-D then projected -- the level set follows the data, nothing
  is hardcoded
- Patient-grouped, LSTV-stratified 5-fold splits with an asserted leakage guard
- Landmark model: MONAI U-Net + Lightning, masked loss for partial annotation,
  sub-pixel soft-argmax decode
- PI / SS / PT / LL / segmental angles **from landmarks alone** -- reproduces the 3-D
  pipeline to 0.6 deg on SS and LL
- Full grid chain: Docker -> Docker Hub -> Singularity .sif -> SLURM (see docs/HPC.md)
- 18 tests over the failure modes that would otherwise be invisible in the metrics

**Next:** run generation at scale; train the 5 folds; the DRR reader study (the piece
that makes the BUU comparison interpretable); DeepDRR as an appearance backend.

**Key measurement so far:** fitting the S1 endplate from the projected silhouette is
26-32 deg wrong at every slab fraction, because the ala superimposes on the body.
Fitted in 3-D and projected, the same case is accurate to 0.6 deg. That gap is the
thesis, and it is why the labels have to come from CT.

## 1. Train the level localizer
- Targets: `xrsp.localize.gaussian_heatmaps` from `project_level_points`.
- Model: `xrsp.localize.build_model(n_levels)` (U-Net heatmap regressor; `pip install xrsp[train]`).
- Loss: per-channel MSE on heatmaps; decode with `points_from_heatmaps`.
- Split: patient-grouped, stratified by level coverage / LSTV (mirror CTSpinoPelvic1K splits).
- Trainer lives in `scripts/train_localizer.py` (to add) so the core package stays dep-light.

## 2. Realism (close the synthetic→real gap)
- **Domain randomization:** DONE for geometry -- `xrsp.oblique.sample_view` jitters yaw
  (the axis that changes ala/body superimposition), pitch and roll, and
  `build_dataset --oblique N` emits N views per CT. Appearance jitter (gamma, contrast,
  noise, blur) is in the loader. Scatter/spectrum realism is DeepDRR's job -- plug it in
  behind the same plan rather than reimplementing it.
- **Perspective/cone-beam:** swap the parallel-beam integral for a DeepDRR-style renderer
  behind the same `projection_plan` interface (fluoroscopy geometry).

## 3. Domain adaptation to real radiographs
- Fine-tune / adapt on real lateral films (e.g. BUU-LSPINE — evaluation only, do not re-host).
- Report level-ID accuracy + landmark error on real data.

## 4. Real-time fluoroscopy
- Frame → heatmap → `level_at_point(C-arm centre)` overlay; target interactive latency.
- "Wrong-level guard": flag the predicted level vs the intended one.

## Design contract (so models stay swappable)
- Dataset unit: `(drr.png [1×H×W], mask.png, levels.json{point,bbox per level})`.
- Localizer I/O: image → per-level heatmaps → points → `level_at_point` readout.
- Any backend (segmentation, keypoint regression, detection) that honors this contract drops in.
