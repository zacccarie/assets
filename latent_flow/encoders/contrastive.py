"""Contrastive encoder — DINOv2 features, L2-normalized.

DINOv2 is trained by self-distillation with multi-crop augmentation —
operationally a contrastive objective. L2-normalizing puts the features on
the unit sphere, the canonical contrastive embedding space.
"""

from __future__ import annotations

import numpy as np

from latent_flow.core.types import EncoderCard, LatentTrajectory, VideoTensor
from latent_flow.encoders import ENCODERS
from latent_flow.encoders._torch import run_backbone
from latent_flow.encoders.base import Encoder

_TIMM_MODEL = "vit_small_patch14_dinov2.lvd142m"
_DIM = 384


@ENCODERS.register("contrastive")
class ContrastiveEncoder(Encoder):
    family = "contrastive"
    name = "contrastive"

    def __init__(self, batch_size: int = 16) -> None:
        self.batch_size = batch_size

    def encode(self, video: VideoTensor) -> LatentTrajectory:
        z = run_backbone(
            video.frames, _TIMM_MODEL, batch_size=self.batch_size, image_size=224,
        )
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
        return self._wrap(z, video)

    def latent_dim(self) -> int:
        return _DIM

    def describe(self) -> EncoderCard:
        return EncoderCard(
            name=self.name,
            family=self.family,
            latent_dim=_DIM,
            principle=(
                "Self-supervised features learned by invariance to "
                "augmentations (DINOv2), then L2-normalized onto the unit "
                "sphere. The contrastive geometry: angular distance reflects "
                "semantic similarity rather than raw pixel similarity."
            ),
            references=["DINOv2 (Oquab et al., 2023)", "SimCLR (Chen et al., 2020)"],
        )
