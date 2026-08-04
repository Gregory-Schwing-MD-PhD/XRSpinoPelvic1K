"""Landmark heatmap model — MONAI U-Net + a Lightning module.

Both are off-the-shelf on purpose: MONAI supplies the network, Lightning the loop,
checkpointing, AMP and DDP. The only bespoke parts are the masked loss (partial
annotation) and the angle metrics, which is where the science is.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .heatmaps import channel_names, masked_mse, soft_argmax


def build_unet(n_channels: int, *, in_channels: int = 1, features=(32, 64, 128, 256, 512)):
    """MONAI 2-D U-Net regressing `n_channels` heatmaps.

    No final activation: heatmaps are regressed against Gaussians with an MSE loss,
    the standard formulation. A sigmoid would squash the peak and blunt sub-pixel
    localisation, which is what the angle accuracy depends on.
    """
    from monai.networks.nets import BasicUNet
    return BasicUNet(spatial_dims=2, in_channels=in_channels, out_channels=n_channels,
                     features=tuple(features) + (32,), act=("LeakyReLU", {"inplace": True}),
                     norm=("instance", {"affine": True}), dropout=0.0)


class LandmarkNet:
    """Lightning module. Imported lazily so the core package stays dependency-light."""

    def __new__(cls, *a, **kw):
        import pytorch_lightning as pl                        # noqa: F401
        return _make_module()(*a, **kw)


def _make_module():
    import pytorch_lightning as pl
    import torch

    class _LandmarkNet(pl.LightningModule):
        def __init__(self, n_channels: int, *, lr: float = 1e-3, weight_decay: float = 1e-5,
                     features=(32, 64, 128, 256, 512), names: Optional[Sequence[str]] = None,
                     max_epochs: int = 200):
            super().__init__()
            self.save_hyperparameters()
            self.net = build_unet(n_channels, features=features)
            self.names = list(names) if names else channel_names()

        def forward(self, x):
            return self.net(x)

        def _step(self, batch, stage):
            img, hm, valid, meta = batch
            pred = self(img)
            loss = masked_mse(pred, hm, valid)
            self.log(f"{stage}/loss", loss, prog_bar=(stage == "val"),
                     batch_size=img.shape[0], sync_dist=True)
            if stage == "val":
                err = landmark_error_px(pred, hm, valid)
                if np.isfinite(err):
                    self.log("val/landmark_px", float(err), prog_bar=True,
                             batch_size=img.shape[0], sync_dist=True)
            return loss

        def training_step(self, b, i):
            return self._step(b, "train")

        def validation_step(self, b, i):
            return self._step(b, "val")

        def configure_optimizers(self):
            opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.hparams.max_epochs)
            return {"optimizer": opt, "lr_scheduler": sch}

    return _LandmarkNet


def landmark_error_px(pred, target, valid) -> float:
    """Mean Euclidean error, in pixels, over valid channels."""
    import torch
    with torch.no_grad():
        p, _ = soft_argmax(pred)
        t, _ = soft_argmax(target)
        d = torch.linalg.norm(p - t, dim=-1)                  # [B, C]
        m = valid & torch.isfinite(d)
        if m.sum() == 0:
            return float("nan")
        return float(d[m].mean())
