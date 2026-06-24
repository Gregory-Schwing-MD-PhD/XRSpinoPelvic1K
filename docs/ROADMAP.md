# Roadmap — XRSpinoPelvic1K

## Status
**Done (working + tested):** DRR engine (parallel-beam, lateral + AP), same-geometry label
& landmark projection, dataset builder CLI, level-localization targets (heatmaps) + readout,
U-Net heatmap-regressor scaffold.

**Next:** train + evaluate the localizer; realism + domain adaptation; real-time fluoro loop.

## 1. Train the level localizer
- Targets: `xrsp.localize.gaussian_heatmaps` from `project_level_points`.
- Model: `xrsp.localize.build_model(n_levels)` (U-Net heatmap regressor; `pip install xrsp[train]`).
- Loss: per-channel MSE on heatmaps; decode with `points_from_heatmaps`.
- Split: patient-grouped, stratified by level coverage / LSTV (mirror CTSpinoPelvic1K splits).
- Trainer lives in `scripts/train_localizer.py` (to add) so the core package stays dep-light.

## 2. Realism (close the synthetic→real gap)
- **Domain randomization:** jitter projection angle (±°), intensity/gamma, add scatter/noise,
  random crops/FOV — generate many DRRs per CT.
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
