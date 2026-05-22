"""Ingestion pipeline: source -> (transcode) -> sample -> VideoTensor."""

from __future__ import annotations

import numpy as np

from latent_flow.core.types import VideoTensor
from latent_flow.ingest.reader import SourceReader
from latent_flow.ingest.sampler import AdaptiveSampler


def _resize(frames: np.ndarray, max_side: int) -> tuple[np.ndarray, dict]:
    """Cheap nearest-neighbour downscale if a frame exceeds max_side.

    Stand-in for the ffmpeg transcoder; Phase 2 swaps in real transcoding.
    """
    _, h, w, _ = frames.shape
    longest = max(h, w)
    if longest <= max_side:
        return frames, {}
    factor = longest / max_side
    nh, nw = int(h / factor), int(w / factor)
    ys = np.linspace(0, h - 1, nh).astype(int)
    xs = np.linspace(0, w - 1, nw).astype(int)
    resized = frames[:, ys][:, :, xs]
    return resized, {"resized_from": [h, w], "resized_to": [nh, nw]}


def ingest_video(
    source: str,
    target_frames: int = 256,
    max_side: int = 512,
    sampling: str = "uniform",
    max_decode_frames: int | None = None,
) -> VideoTensor:
    """Load a video and normalize it into a VideoTensor."""
    raw, fps = SourceReader(source).read(max_frames=max_decode_frames)
    raw, transcode_manifest = _resize(raw, max_side)

    sampler = AdaptiveSampler(strategy=sampling, target_frames=target_frames)
    frames, timestamps, sampling_manifest = sampler.sample(raw, fps)

    eff_fps = len(frames) / max(timestamps[-1], 1e-6) if len(frames) > 1 else fps
    return VideoTensor(
        frames=frames,
        timestamps=timestamps,
        fps=float(eff_fps),
        source=source,
        sampling_manifest=sampling_manifest,
        transcode_manifest=transcode_manifest,
    )
