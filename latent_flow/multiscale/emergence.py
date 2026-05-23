"""Quantify what disappears, persists and emerges across scales.

  disappears  : fraction of spectral energy above the coarse-graining cutoff
                (i.e. content that block-averaging would erase)
  persists    : complementary fraction below the cutoff
  emerges     : gain in temporal regularity (lag-1 autocorrelation of the
                leading PC) when comparing the coarse-grained trajectory to
                the fine one — positive when smoothing reveals structure
                that was buried in the fine-scale noise.

A motif (flow of a crowd, fluid pattern) typical of "emergent" structure
should show both low `disappears` and positive `emerges`.
"""

from __future__ import annotations

import numpy as np

from latent_flow.dynamics.embedding import principal_component
from latent_flow.multiscale.coarse_grain import temporal_block_average
from latent_flow.multiscale.frequency import spectrum


def _lag1(s: np.ndarray) -> float:
    s = s - s.mean()
    den = float((s * s).mean())
    if den <= 1e-12:
        return 0.0
    return float((s[:-1] * s[1:]).mean() / den)


def disappear_persist_emerge(
    z: np.ndarray, fs: float = 1.0, coarse_scale: int = 8
) -> dict:
    z = np.asarray(z, dtype=np.float64)
    freqs, power = spectrum(z, fs)
    total = power.sum()
    if total <= 0:
        return {"disappears": 0.0, "persists": 0.0, "emerges": 0.0}

    cutoff = fs / (2 * coarse_scale)
    high_mask = freqs > cutoff
    disappears = float(power[high_mask].sum() / total)
    persists = float(power[~high_mask].sum() / total)

    z_coarse = temporal_block_average(z, coarse_scale)
    if len(z_coarse) < 4:
        emerges = 0.0
    else:
        emerges = max(
            0.0, _lag1(principal_component(z_coarse)) - _lag1(principal_component(z))
        )
    return {
        "disappears": disappears,
        "persists": persists,
        "emerges": float(emerges),
        "cutoff_hz": float(cutoff),
        "coarse_scale": int(coarse_scale),
    }
