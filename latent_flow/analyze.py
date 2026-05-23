"""High-level orchestration: video -> latent -> dynamics + geometry (+ full).

Two modes:

  `analyze(..., full=False)` — the Phase 1 quick path (projection + delay
                              embedding + recurrence).
  `analyze(..., full=True)`  — adds Lyapunov, correlation dimension, dominant
                              modes, multi-scale emergence, Granger matrix
                              (on top PCA dims), regime changepoints and a
                              Koopman spectrum.

A proper lazy JobGraph replaces this straight-line function in Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from latent_flow.causal import detect_regimes, granger_matrix
from latent_flow.core.store import TrajectoryStore
from latent_flow.core.types import LatentTrajectory, VideoTensor
from latent_flow.dynamics import (
    chaos_verdict,
    correlation_dimension,
    dominant_period,
    embedding,
    identify_attractors,
    lyapunov_rosenstein,
    recurrence,
)
from latent_flow.encoders import build
from latent_flow.geometry import dmd, project
from latent_flow.ingest import CostEstimator, ingest_video
from latent_flow.multiscale import disappear_persist_emerge, dominant_modes


@dataclass
class EncoderResult:
    encoder_name: str
    trajectory: LatentTrajectory
    projection: np.ndarray
    phase_space: dict[str, Any]
    recurrence_rate: float
    determinism: float
    cached: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    video: VideoTensor
    encoders: list[EncoderResult] = field(default_factory=list)


def _full_analysis(traj: LatentTrajectory, fs: float) -> dict[str, Any]:
    z = traj.z
    lyap = lyapunov_rosenstein(z, fs=fs)
    d2 = correlation_dimension(z)
    period = dominant_period(z, fs=fs)
    attractors = identify_attractors(z)
    modes = dominant_modes(z, fs=fs, k=5)
    emergence = disappear_persist_emerge(z, fs=fs)
    causal = granger_matrix(z, lags=4, dims=min(8, z.shape[1]))
    regimes = detect_regimes(z, window=max(8, len(z) // 12))
    koop = dmd(z)
    return {
        "lyapunov": float(lyap["lambda"]),
        "chaos_verdict": chaos_verdict(lyap["lambda"]),
        "correlation_dimension": float(d2.get("D2", float("nan"))),
        "dominant_period": float(period.get("period", float("nan"))),
        "period_score": float(period.get("score", 0.0)),
        "n_attractors": int(attractors["n_attractors"]),
        "attractor_separation": float(attractors["separation_score"]),
        "dominant_modes_hz": modes,
        "multiscale": emergence,
        "granger_matrix": causal,
        "regime_boundaries": [r["index"] for r in regimes],
        "koopman": {
            "rank": koop["rank"],
            "top_frequencies_hz": np.sort(
                koop["frequencies"][np.argsort(np.abs(koop["eigenvalues"]))[::-1]][:5]
            ).tolist(),
            "max_growth_rate": float(np.max(koop["growth_rates"])),
        },
    }


def analyze(
    source: str,
    encoders: list[str],
    target_frames: int = 256,
    max_side: int = 512,
    projection: str = "umap",
    store: TrajectoryStore | None = None,
    full: bool = False,
) -> AnalysisResult:
    """Run the pipeline for one video across several encoders."""
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
        extras = _full_analysis(traj, fs=video.fps) if full else {}

        result.encoders.append(
            EncoderResult(
                encoder_name=name,
                trajectory=traj,
                projection=project(traj.z, method=projection, n_components=3),
                phase_space={"tau": recon["tau"], "dimension": recon["dimension"]},
                recurrence_rate=recurrence.recurrence_rate(rmat),
                determinism=recurrence.determinism(rmat),
                cached=cached is not None,
                extras=extras,
            )
        )
    return result
