"""AdaptiveSampler — reduce a raw frame stream to T frames under a budget.

MVP ships the `uniform` strategy. `motion` and `event` are stubbed with the
intended algorithm documented, to be filled in Phase 2.
"""

from __future__ import annotations

import numpy as np


class AdaptiveSampler:
    def __init__(self, strategy: str = "uniform", target_frames: int = 256) -> None:
        if strategy not in {"uniform", "motion", "event"}:
            raise ValueError(f"unknown strategy '{strategy}'")
        self.strategy = strategy
        self.target_frames = target_frames

    def sample(
        self, frames: np.ndarray, fps: float
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Return (sampled frames, timestamps seconds, manifest)."""
        if self.strategy == "uniform":
            return self._uniform(frames, fps)
        # Phase 2: motion-driven densifies where optical-flow norm is high;
        # event-driven cuts at scene boundaries so dynamics never cross a cut.
        raise NotImplementedError(
            f"'{self.strategy}' sampling is a Phase 2 feature"
        )

    def _uniform(
        self, frames: np.ndarray, fps: float
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        n = len(frames)
        if n <= self.target_frames:
            idx = np.arange(n)
        else:
            idx = np.linspace(0, n - 1, self.target_frames).round().astype(int)
        sampled = frames[idx]
        timestamps = idx / fps
        manifest = {
            "strategy": "uniform",
            "original_frames": int(n),
            "sampled_frames": int(len(idx)),
            "stride": float(n / len(idx)),
        }
        return sampled, timestamps, manifest
