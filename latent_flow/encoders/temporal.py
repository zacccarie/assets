"""Temporal encoder — sliding-window mean over per-frame ViT features.

A pragmatic stand-in for VideoMAE / S3D: a temporal smoothing on top of a
strong per-frame encoder. Reveals temporal structure that pure per-frame
encoders ignore, without requiring a video-trained backbone.
"""

from __future__ import annotations

import numpy as np

from latent_flow.core.types import EncoderCard, LatentTrajectory, VideoTensor
from latent_flow.encoders import ENCODERS
from latent_flow.encoders._torch import run_backbone
from latent_flow.encoders.base import Encoder

_TIMM_MODEL = "vit_small_patch14_dinov2.lvd142m"
_DIM = 384


@ENCODERS.register("temporal")
class TemporalEncoder(Encoder):
    family = "temporal"
    name = "temporal"

    def __init__(self, batch_size: int = 16, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.batch_size = batch_size
        self.window = window

    def encode(self, video: VideoTensor) -> LatentTrajectory:
        z = run_backbone(
            video.frames, _TIMM_MODEL, batch_size=self.batch_size, image_size=224,
        )
        pad = self.window // 2
        zp = np.pad(z, ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.stack(
            [zp[i : i + len(z)] for i in range(self.window)], axis=0
        ).mean(axis=0)
        return self._wrap(smoothed.astype(np.float32), video)

    def latent_dim(self) -> int:
        return _DIM

    def describe(self) -> EncoderCard:
        return EncoderCard(
            name=self.name,
            family=self.family,
            latent_dim=_DIM,
            principle=(
                "A temporal smoothing operator on top of a per-frame ViT. "
                "Each frame's feature is averaged over a centered window, "
                "introducing a controlled temporal receptive field — a "
                "low-cost alternative to training a true video backbone."
            ),
            references=["VideoMAE (Tong et al., 2022)"],
        )
