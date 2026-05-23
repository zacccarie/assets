"""Autoencoder family — PCA baseline.

A linear autoencoder trained to minimum reconstruction error converges to
PCA. We use PCA fitted on the video's own frames as the deterministic,
no-training lower bound for the AE family. Any properly trained nonlinear
AE should beat it on rate-distortion; otherwise the nonlinearity is wasted
on this content.
"""

from __future__ import annotations

import numpy as np

from latent_flow.core.types import EncoderCard, LatentTrajectory, VideoTensor
from latent_flow.encoders import ENCODERS
from latent_flow.encoders.base import Encoder


@ENCODERS.register("ae")
class PCABaselineEncoder(Encoder):
    family = "ae"
    name = "ae"

    def __init__(self, latent_dim: int = 64) -> None:
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        self._dim = latent_dim

    def encode(self, video: VideoTensor) -> LatentTrajectory:
        x = video.frames.reshape(len(video.frames), -1).astype(np.float32) / 255.0
        x -= x.mean(axis=0, keepdims=True)
        # economy SVD; we only need the first `latent_dim` singular vectors.
        k = min(self._dim, min(x.shape) - 1)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        z = (u[:, :k] * s[:k]).astype(np.float32)
        if z.shape[1] < self._dim:
            z = np.pad(z, ((0, 0), (0, self._dim - z.shape[1])))
        return self._wrap(z, video)

    def latent_dim(self) -> int:
        return self._dim

    def describe(self) -> EncoderCard:
        return EncoderCard(
            name=self.name,
            family=self.family,
            latent_dim=self._dim,
            principle=(
                "PCA on the video's own frames — the optimal linear AE under "
                "MSE. No training. Deterministic. Serves as the rate-distortion "
                "lower bound for the AE family; nonlinear AEs (Phase 3) are "
                "judged against it."
            ),
            references=["Baldi & Hornik (1989); Hinton & Salakhutdinov (2006)"],
        )
