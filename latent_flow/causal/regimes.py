"""Regime segmentation by sliding mean-shift test on the leading PC.

A frame index `t` is a candidate changepoint when the two windows on either
side of it differ in mean by many standard errors (Welch-style score).
Non-maximum suppression keeps only the locally-strongest boundaries.

These boundaries feed the bifurcation analysis: the question "what changes
qualitatively here?" is then answered by recomputing Lyapunov / D2 /
attractor count on each side.
"""

from __future__ import annotations

import numpy as np

from latent_flow.dynamics.embedding import principal_component


def detect_regimes(
    z: np.ndarray, window: int = 20, threshold: float = 2.0
) -> list[dict]:
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    if n < 2 * window + 4:
        return []
    s = principal_component(z)

    scores = np.zeros(n)
    for t in range(window, n - window):
        left, right = s[t - window : t], s[t : t + window]
        ml, mr = left.mean(), right.mean()
        vl, vr = left.var() + 1e-9, right.var() + 1e-9
        scores[t] = abs(ml - mr) / np.sqrt(vl / window + vr / window)

    # local maxima above threshold
    candidates = [
        (int(t), float(scores[t]))
        for t in range(window, n - window)
        if scores[t] > threshold
        and scores[t] >= scores[max(0, t - window) : t + window + 1].max()
    ]
    candidates.sort(key=lambda c: -c[1])
    kept: list[tuple[int, float]] = []
    for t, sc in candidates:
        if all(abs(t - t2) > window for t2, _ in kept):
            kept.append((t, sc))
    kept.sort()
    return [{"index": t, "score": sc} for t, sc in kept]
