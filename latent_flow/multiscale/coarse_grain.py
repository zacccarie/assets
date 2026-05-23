"""Coarse-graining: temporal block averaging and hierarchical pyramids."""

from __future__ import annotations

import numpy as np


def temporal_block_average(z: np.ndarray, scale: int) -> np.ndarray:
    """Average consecutive `scale` frames together. Truncates the tail."""
    if scale <= 1:
        return np.asarray(z, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    n = (len(z) // scale) * scale
    if n == 0:
        return z[:0]
    blocks = z[:n].reshape(n // scale, scale, -1)
    return blocks.mean(axis=1)


def hierarchical(
    z: np.ndarray, scales: tuple[int, ...] = (1, 2, 4, 8, 16)
) -> dict[int, np.ndarray]:
    """Pyramid of coarse-grained trajectories, indexed by scale."""
    return {s: temporal_block_average(z, s) for s in scales}
