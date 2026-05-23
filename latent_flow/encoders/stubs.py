"""Encoder stubs — the contract is frozen, the implementation is deferred.

VAE, RSSM, world-model and JEPA-like encoders need either per-video
training or external pretrained checkpoints with non-trivial integration.
They are registered here with full `describe()` cards so the comparison
matrix knows about them; `encode()` raises a clear, actionable error.
"""

from __future__ import annotations

from latent_flow.core.types import EncoderCard, LatentTrajectory, VideoTensor
from latent_flow.encoders import ENCODERS
from latent_flow.encoders.base import Encoder


class _Deferred(Encoder):
    family = "stub"
    name = "stub"
    _dim = 0
    _principle = ""
    _refs: list[str] = []
    _phase = "Phase 3"

    def encode(self, video: VideoTensor) -> LatentTrajectory:
        raise NotImplementedError(
            f"encoder '{self.name}' is a {self._phase} feature. "
            f"The contract is frozen here; the implementation will be added "
            f"once training infrastructure / checkpoints are wired up."
        )

    def latent_dim(self) -> int:
        return self._dim

    def describe(self) -> EncoderCard:
        return EncoderCard(
            name=self.name, family=self.family, latent_dim=self._dim,
            principle=self._principle, references=self._refs,
            notes={"status": "stub", "phase": self._phase},
        )


@ENCODERS.register("vae")
class VAE(_Deferred):
    family = "vae"
    name = "vae"
    _dim = 64
    _principle = (
        "Variational autoencoder: probabilistic latent regularized toward an "
        "isotropic prior (β-VAE controls disentanglement vs reconstruction). "
        "Allows sampling and quantifies aleatoric uncertainty in z."
    )
    _refs = ["Kingma & Welling (2014)", "β-VAE (Higgins et al., 2017)"]


@ENCODERS.register("rssm")
class RSSM(_Deferred):
    family = "rssm"
    name = "rssm"
    _dim = 200
    _principle = (
        "Recurrent State-Space Model: a stochastic latent z_t and a "
        "deterministic hidden h_t evolve jointly via a GRU. Trained to "
        "predict frames k steps ahead in latent space. The dynamics live in "
        "z; the observation map is just a decoder."
    )
    _refs = ["PlaNet (Hafner et al., 2019)"]


@ENCODERS.register("world_model")
class WorldModel(_Deferred):
    family = "world_model"
    name = "world_model"
    _dim = 1024
    _principle = (
        "Full Dreamer-style world model: encoder + RSSM + decoder + reward. "
        "Latent is structured for *planning*, not just reconstruction; rolls "
        "out the future in z without ever re-rendering pixels."
    )
    _refs = ["DreamerV3 (Hafner et al., 2023)"]


@ENCODERS.register("jepa")
class JEPA(_Deferred):
    family = "jepa"
    name = "jepa"
    _dim = 384
    _principle = (
        "Joint-Embedding Predictive Architecture: a predictor matches latents "
        "across views/time WITHOUT reconstructing pixels. Indifferent to "
        "high-frequency unpredictable noise — exactly what a VAE would waste "
        "capacity reconstructing."
    )
    _refs = ["I-JEPA (Assran et al., 2023)", "V-JEPA (Bardes et al., 2024)"]
