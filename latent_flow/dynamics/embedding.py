"""Phase-space reconstruction by delay embedding (Takens).

For a scalar observable s_t, the embedding is
    v_t = (s_t, s_{t-tau}, ..., s_{t-(m-1)tau}).
Delay tau: first minimum of time-delayed mutual information.
Dimension m: false-nearest-neighbours criterion.
"""

from __future__ import annotations

import numpy as np


def principal_component(z: np.ndarray) -> np.ndarray:
    """Reduce a (T, d) trajectory to a scalar series: the leading PC."""
    if z.ndim == 1:
        return z.astype(np.float64)
    x = z - z.mean(axis=0, keepdims=True)
    # leading principal component via SVD
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return (x @ vt[0]).astype(np.float64)


def estimate_delay(series: np.ndarray, max_lag: int = 50, bins: int = 16) -> int:
    """First minimum of time-delayed mutual information."""
    s = np.asarray(series, dtype=np.float64)
    max_lag = min(max_lag, len(s) // 3)
    mi = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        a, b = s[:-lag], s[lag:]
        hist, _, _ = np.histogram2d(a, b, bins=bins)
        p = hist / hist.sum()
        px = p.sum(axis=1, keepdims=True)
        py = p.sum(axis=0, keepdims=True)
        nz = p > 0
        mi[lag - 1] = np.sum(p[nz] * np.log(p[nz] / (px * py)[nz]))
    for i in range(1, max_lag - 1):
        if mi[i] < mi[i - 1] and mi[i] < mi[i + 1]:
            return i + 1
    return int(np.argmin(mi)) + 1


def estimate_dimension(
    series: np.ndarray, tau: int, max_dim: int = 10, tol: float = 10.0
) -> int:
    """Embedding dimension by false nearest neighbours."""
    s = np.asarray(series, dtype=np.float64)
    prev_frac = 1.0
    for m in range(1, max_dim + 1):
        emb = delay_embedding(s, m=m, tau=tau)
        emb_next = delay_embedding(s, m=m + 1, tau=tau)
        n = min(len(emb), len(emb_next))
        if n < 5:
            return m
        emb, emb_next = emb[:n], emb_next[:n]
        false = 0
        for i in range(n):
            d = np.linalg.norm(emb - emb[i], axis=1)
            d[i] = np.inf
            j = int(np.argmin(d))
            base = d[j]
            if base <= 0:
                continue
            extra = abs(emb_next[i, -1] - emb_next[j, -1])
            if extra / base > tol:
                false += 1
        frac = false / n
        if frac < 0.05 or frac > prev_frac:
            return m
        prev_frac = frac
    return max_dim


def delay_embedding(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Build the (N, m) delay-coordinate matrix."""
    s = np.asarray(series, dtype=np.float64).ravel()
    span = (m - 1) * tau
    n = len(s) - span
    if n <= 0:
        raise ValueError(f"series too short for m={m}, tau={tau}")
    return np.stack([s[i * tau : i * tau + n] for i in range(m)], axis=1)


def reconstruct(z: np.ndarray) -> dict:
    """Full phase-space reconstruction from a latent trajectory."""
    s = principal_component(z)
    tau = estimate_delay(s)
    m = estimate_dimension(s, tau)
    return {
        "observable": s,
        "tau": tau,
        "dimension": m,
        "embedding": delay_embedding(s, m=m, tau=tau),
    }
