"""High-level orchestration: video -> latent -> dynamics + geometry.

This is the MVP's "tube that works": the end-to-end path the design document
calls the exit criterion for Phase 1. A proper lazy JobGraph replaces this
straight-line function in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from latent_flow.core.store import TrajectoryStore
from latent_flow.core.types import LatentTrajectory, VideoTensor
from latent_flow.dynamics import embedding, recurrence
from latent_flow.encoders import build
from latent_flow.geometry import project
from latent_flow.ingest import CostEstimator, ingest_video


@dataclass
class EncoderResult:
    encoder_name: str
    trajectory: LatentTrajectory
    projection: np.ndarray
    phase_space: dict[str, Any]
    recurrence_rate: float
    determinism: float
    cached: bool = False


@dataclass
class AnalysisResult:
    video: VideoTensor
    encoders: list[EncoderResult] = field(default_factory=list)


def analyze(
    source: str,
    encoders: list[str],
    target_frames: int = 256,
    max_side: int = 512,
    projection: str = "umap",
    store: TrajectoryStore | None = None,
) -> AnalysisResult:
    """Run the full MVP pipeline for one video across several encoders."""
    store = store or TrajectoryStore()
    video = ingest_video(source, target_frames=target_frames, max_side=max_side)
    estimator = CostEstimator()
    result = AnalysisResult(video=video)

    for name in encoders:
        cost = estimator.estimate(video.num_frames, video.resolution, name)
        if cost.verdict == "REFUSE":
            raise RuntimeError(f"encoder '{name}': {cost.render()}")

        cached = store.get(video.fingerprint, name)
        if cached is not None:
            traj = cached
        else:
            traj = build(name).encode(video)
            store.put(traj)

        recon = embedding.reconstruct(traj.z)
        rmat, _ = recurrence.recurrence_plot(traj.z)

        result.encoders.append(
            EncoderResult(
                encoder_name=name,
                trajectory=traj,
                projection=project(traj.z, method=projection, n_components=3),
                phase_space={"tau": recon["tau"], "dimension": recon["dimension"]},
                recurrence_rate=recurrence.recurrence_rate(rmat),
                determinism=recurrence.determinism(rmat),
                cached=cached is not None,
            )
        )
    return result
