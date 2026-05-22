"""Latent Flow — analyze video as a trajectory in a latent state space.

Phase 1 (MVP) scaffold. The three load-bearing abstractions are frozen here:

    VideoTensor       — normalized output of ingestion
    LatentTrajectory  — Z in R^{T x d}, the common currency of all analysis
    Encoder           — x -> z, the plugin contract every encoder implements

Everything downstream (dynamics, geometry, multiscale, causal) consumes a
LatentTrajectory and returns a trajectory or a descriptor.
"""

from latent_flow.core.types import EncoderCard, LatentTrajectory, VideoTensor

__version__ = "0.1.0"

__all__ = ["VideoTensor", "LatentTrajectory", "EncoderCard", "__version__"]
