"""The Encoder contract — frozen in the MVP.

Every encoder maps a VideoTensor to a LatentTrajectory and describes itself
with an EncoderCard. Nothing else. This homogeneity is what makes encoders
comparable, which is the real purpose of the platform.
"""

from __future__ import annotations

import abc

import numpy as np

from latent_flow.core.types import EncoderCard, LatentTrajectory, VideoTensor


class Encoder(abc.ABC):
    #: one of: cnn vit ae vae contrastive temporal world_model rssm jepa
    family: str = "unknown"
    name: str = "base"

    @abc.abstractmethod
    def encode(self, video: VideoTensor) -> LatentTrajectory:
        """Map frames to a latent trajectory Z in R^{T x d}."""

    @abc.abstractmethod
    def latent_dim(self) -> int:
        ...

    @abc.abstractmethod
    def describe(self) -> EncoderCard:
        """Static description: principle, family, references."""

    def _wrap(self, z: np.ndarray, video: VideoTensor) -> LatentTrajectory:
        """Helper: build a LatentTrajectory and attach the compression ratio."""
        h, w = video.resolution
        raw_dim = h * w * 3
        return LatentTrajectory(
            z=z,
            timestamps=video.timestamps,
            encoder_name=self.name,
            video_fingerprint=video.fingerprint,
            meta={
                "family": self.family,
                "latent_dim": int(z.shape[1]),
                "compression_ratio": float(raw_dim / max(z.shape[1], 1)),
            },
        )
