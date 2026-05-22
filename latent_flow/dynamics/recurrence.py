"""Recurrence analysis of a latent trajectory.

A recurrence plot marks pairs of times whose latent states are close. Its
texture exposes periodicity (diagonals), stationarity and regime changes —
the entry point to attractor / cycle detection in Phase 2.
"""

from __future__ import annotations

import numpy as np


def _pairwise(z: np.ndarray) -> np.ndarray:
    diff = z[:, None, :] - z[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def recurrence_plot(
    z: np.ndarray, threshold: float | None = None, rate: float = 0.1
) -> tuple[np.ndarray, float]:
    """Binary (T, T) recurrence matrix.

    If `threshold` is None it is chosen so the recurrence rate ~= `rate`.
    Returns (matrix, threshold).
    """
    dist = _pairwise(np.asarray(z, dtype=np.float64))
    if threshold is None:
        off = dist[~np.eye(len(dist), dtype=bool)]
        threshold = float(np.quantile(off, rate))
    return (dist <= threshold).astype(np.uint8), threshold


def recurrence_rate(matrix: np.ndarray) -> float:
    """Fraction of recurrent points (density of the recurrence plot)."""
    return float(matrix.mean())


def determinism(matrix: np.ndarray, min_diag: int = 2) -> float:
    """RQA determinism: fraction of recurrent points on diagonal lines.

    High determinism => predictable / periodic; low => stochastic.
    """
    t = len(matrix)
    on_lines = 0
    total = int(matrix.sum())
    if total == 0:
        return 0.0
    for k in range(-(t - 1), t):
        diag = np.diagonal(matrix, offset=k)
        run = 0
        for v in diag:
            if v:
                run += 1
            else:
                if run >= min_diag:
                    on_lines += run
                run = 0
        if run >= min_diag:
            on_lines += run
    return on_lines / total
