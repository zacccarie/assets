"""Cycle detection in a latent trajectory.

Dominant period via FFT of the leading PC plus a periodicity score
(strength of the dominant peak relative to spectral background). A high
score with a stable period is the signature of a limit cycle.
"""

from __future__ import annotations

import numpy as np

from latent_flow.dynamics.embedding import principal_component


def dominant_period(
    z: np.ndarray, fs: float = 1.0, min_period: int = 3
) -> dict:
    s = principal_component(np.asarray(z))
    s = s - s.mean()
    n = len(s)
    if n < 2 * min_period:
        return {"period": float("nan"), "score": 0.0}
    spec = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # skip DC and the very-low end (period must be < n/2)
    valid = (freqs > 0) & (1.0 / np.maximum(freqs, 1e-12) >= min_period)
    if not valid.any():
        return {"period": float("nan"), "score": 0.0}
    peak = int(np.argmax(spec * valid))
    background = np.median(spec[valid])
    score = float(spec[peak] / (background + 1e-12))
    return {
        "period": float(1.0 / freqs[peak]),
        "frequency": float(freqs[peak]),
        "score": score,  # peak prominence; > ~5 => clear cycle
    }
