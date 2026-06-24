# data/

Put your CT + segmentation pairs here (or point `--in` anywhere):

```
data/raw/
  0003_ct.nii.gz      0003_label.nii.gz
  0007_ct.nii.gz      0007_label.nii.gz
  ...
```

Then build the DRR dataset:

```bash
python -m xrsp.build_dataset --in data/raw --out data/xrsp1k --views lateral ap
```

Outputs land in `data/xrsp1k/<case>/` (gitignored — these are large and licensed via the
source CT; see `docs/dataset_card.md`). A small rendered example is committed under
`examples/`.

> Source CT volumes are **not** committed (they carry their own licenses). Get
> CTSpinoPelvic1K from its repo / dataset card.
