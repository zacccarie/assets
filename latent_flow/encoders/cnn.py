"""CNN encoder — spatial baseline, hierarchical convolutional features.

Backbone: ConvNeXt-Tiny. Encodes each frame independently into a pooled
feature vector; the latent trajectory is the sequence of those vectors.
"""

from __future__ import annotations

from latent_flow.core.types import EncoderCard, LatentTrajectory, VideoTensor
from latent_flow.encoders import ENCODERS
from latent_flow.encoders._torch import run_backbone
from latent_flow.encoders.base import Encoder

_TIMM_MODEL = "convnext_tiny"
_DIM = 768


@ENCODERS.register("cnn")
class CNNEncoder(Encoder):
    family = "cnn"
    name = "cnn"

    def __init__(self, batch_size: int = 16) -> None:
        self.batch_size = batch_size

    def encode(self, video: VideoTensor) -> LatentTrajectory:
        z = run_backbone(video.frames, _TIMM_MODEL, batch_size=self.batch_size)
        return self._wrap(z, video)

    def latent_dim(self) -> int:
        return _DIM

    def describe(self) -> EncoderCard:
        return EncoderCard(
            name=self.name,
            family=self.family,
            latent_dim=_DIM,
            principle=(
                "A convolutional network builds translation-equivariant "
                "features through a hierarchy of local filters. Each frame is "
                "pooled into a single vector; the video becomes the sequence "
                "of per-frame vectors. Strong spatial prior, no temporal "
                "modelling — the baseline against which richer encoders are "
                "judged."
            ),
            references=["ConvNeXt (Liu et al., 2022)"],
        )
