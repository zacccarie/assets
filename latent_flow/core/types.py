"""Core data structures shared across every module.

These three types are the frozen contract of the MVP. Adding an encoder or an
analyzer must never require changing them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VideoTensor:
    """Canonical output of the ingestion pipeline.

    frames: uint8 array, shape (T, H, W, 3), RGB — natural decode order.
    timestamps: float seconds for each sampled frame, shape (T,).
    fps: effective frame rate after (adaptive) sampling.
    source: original path / URL / device descriptor.
    sampling_manifest: how the original stream was reduced to these T frames.
    transcode_manifest: any ffmpeg transform applied before decoding.
    """

    frames: np.ndarray
    timestamps: np.ndarray
    fps: float
    source: str
    sampling_manifest: dict[str, Any] = field(default_factory=dict)
    transcode_manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError(
                f"frames must be (T, H, W, 3), got {self.frames.shape}"
            )
        if len(self.timestamps) != len(self.frames):
            raise ValueError("timestamps length must match frame count")

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self.frames.shape[1]), int(self.frames.shape[2])

    @property
    def fingerprint(self) -> str:
        """Content hash — stable key for the cache. Cheap: hashes a stride."""
        h = hashlib.blake2b(digest_size=16)
        h.update(np.ascontiguousarray(self.frames[::8]).tobytes())
        h.update(repr((self.fps, self.frames.shape)).encode())
        return h.hexdigest()


@dataclass
class LatentTrajectory:
    """Z in R^{T x d}: a video seen through one encoder.

    The common currency of the platform. Dynamics, geometry, multiscale and
    causal modules all consume this and nothing else.
    """

    z: np.ndarray
    timestamps: np.ndarray
    encoder_name: str
    video_fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.z = np.asarray(self.z, dtype=np.float32)
        if self.z.ndim != 2:
            raise ValueError(f"z must be (T, d), got {self.z.shape}")
        if len(self.timestamps) != len(self.z):
            raise ValueError("timestamps length must match T")

    @property
    def num_steps(self) -> int:
        return int(self.z.shape[0])

    @property
    def dim(self) -> int:
        return int(self.z.shape[1])


@dataclass
class EncoderCard:
    """Human- and machine-readable description of an encoder.

    Populated incrementally: `principle` is static; the analysis fields are
    filled by the comparison module once the encoder has run on a video.
    """

    name: str
    family: str
    principle: str
    references: list[str] = field(default_factory=list)
    latent_dim: int | None = None
    compression_ratio: float | None = None
    stability_score: float | None = None
    notes: dict[str, Any] = field(default_factory=dict)
