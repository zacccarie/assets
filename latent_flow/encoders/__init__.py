"""Encoder zoo. MVP ships CNN + ViT; the registry contract is frozen here.

Phase 2 adds AE, VAE, contrastive, temporal, world-model, RSSM and JEPA
encoders — each is just another `@ENCODERS.register(...)` factory.
"""

from latent_flow.core.registry import Registry
from latent_flow.encoders.base import Encoder

ENCODERS: Registry[Encoder] = Registry("encoder")

# Import side-effect: register the built-in encoders.
from latent_flow.encoders import cnn, vit  # noqa: E402,F401


def build(name: str, **kwargs) -> Encoder:
    return ENCODERS.create(name, **kwargs)


def available() -> list[str]:
    return ENCODERS.names()


__all__ = ["Encoder", "ENCODERS", "build", "available"]
