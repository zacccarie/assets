"""ViT encoder — patch tokens, global attention, self-supervised features.

Backbone: DINOv2 ViT-S/14. Per-frame CLS-style pooled feature; the latent
trajectory is the sequence of those features.
"""

from __future__ import annotations

from latent_flow.core.types import EncoderCard, LatentTrajectory, VideoTensor
from latent_flow.encoders import ENCODERS
from latent_flow.encoders._torch import run_backbone
from latent_flow.encoders.base import Encoder

_TIMM_MODEL = "vit_small_patch14_dinov2.lvd142m"
_DIM = 384
_IMAGE_SIZE = 224  # multiple of patch size 14


@ENCODERS.register("vit")
class ViTEncoder(Encoder):
    family = "vit"
    name = "vit"

    def __init__(self, batch_size: int = 16) -> None:
        self.batch_size = batch_size

    def encode(self, video: VideoTensor) -> LatentTrajectory:
        z = run_backbone(
            video.frames, _TIMM_MODEL,
            batch_size=self.batch_size, image_size=_IMAGE_SIZE,
        )
        return self._wrap(z, video)

    def latent_dim(self) -> int:
        return _DIM

    def describe(self) -> EncoderCard:
        return EncoderCard(
            name=self.name,
            family=self.family,
            latent_dim=_DIM,
            principle=(
                "A Vision Transformer splits each frame into patches, embeds "
                "them as tokens, and mixes them with global self-attention. "
                "Pretrained self-supervised (DINOv2), its features are "
                "semantically structured without labels — a strong, "
                "general-purpose observation map for the latent trajectory."
            ),
            references=[
                "ViT (Dosovitskiy et al., 2021)",
                "DINOv2 (Oquab et al., 2023)",
            ],
        )
