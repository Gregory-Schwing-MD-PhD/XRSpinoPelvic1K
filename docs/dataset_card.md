# XRSpinoPelvic1K — dataset card

## Summary
Synthetic spinal radiographs (DRRs) rendered from segmented CT, each paired with a 2-D
vertebra mask and per-level landmarks (point + bbox) projected from 3-D ground truth.
Views: lateral and AP. Labels: T1–T13, L1–L6, S1, sacrum, hips, femurs (where in FOV).

## Provenance & how it's made
- **Source:** [CTSpinoPelvic1K](https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K)
  CT + 3-D segmentations (itself built from CTSpine1K, CTPelvic1K, TCIA COLONOG).
- **Rendering:** `xrsp.drr.drr_project` (parallel-beam attenuation integral) +
  `xrsp.project_labels` (same-geometry projection of the masks). Deterministic.

## Intended use
Training/evaluation of radiograph models for **vertebral-level localization**, spinopelvic
landmark detection, and DRR→real-radiograph domain adaptation. **Research/education only —
not a medical device, not for clinical or intra-operative use.**

## Licensing (read before redistributing)
- **Code:** Apache-2.0.
- **Rendered DRRs / masks:** a derivative of the source CT, so they inherit its terms.
  CTSpinoPelvic1K is released **CC-BY-NC-4.0** (non-commercial), constituents CC-BY-3.0.
  → Redistribute the rendered images **only** under those terms, with attribution.
- Do **not** mix in any dataset whose license forbids redistribution (e.g. controlled-access
  radiograph sets used only for *evaluation* must not be re-hosted here).

## Known limitations
- DRRs are **parallel-beam** and idealized — a synthetic-to-real **domain gap** remains
  (no scatter/detector response); domain randomization + real-data fine-tuning are on the
  roadmap.
- Projected 2-D masks are **silhouettes** of 3-D structures, so overlapping anatomy shares
  pixels (inherent to projection); the per-level **landmarks** avoid this ambiguity.
- Pelvic/femoral landmarks exist only where the source CT FOV includes them.

## Citation
Cite XRSpinoPelvic1K, CTSpinoPelvic1K, and OpenSpineToolkit, plus the upstream CT sources
(CTSpine1K, CTPelvic1K, TCIA COLONOG).
