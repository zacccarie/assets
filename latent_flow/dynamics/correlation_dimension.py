"""Correlation dimension D2 — Grassberger-Procaccia.

Estimates the fractal dimension of the attractor in the delay-embedding
space. Scales as C(r) ~ r^D2 in the scaling region.
"""

from __future__ import annotations

import numpy as np

from latent_flow.dynamics.embedding import (
    delay_embedding,
    estimate_delay,
    estimate_dimension,
    principal_component,
)


def correlation_dimension(
    series: np.ndarray,
    m: int | None = None,
    tau: int | None = None,
    n_radii: int = 25,
) -> dict:
    """Return D2 and the log-log correlation-sum curve."""
    s = principal_component(np.asarray(series))
    if tau is None:
        tau = estimate_delay(s)
    if m is None:
        m = estimate_dimension(s, tau)
    emb = delay_embedding(s, m=m, tau=tau)

    dist = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=-1)
    off = dist[np.triu_indices_from(dist, k=1)]
    if off.size == 0:
        return {"D2": float("nan")}

    r_lo, r_hi = np.percentile(off[off > 0], [2, 90])
    if r_lo <= 0 or r_hi <= r_lo:
        return {"D2": float("nan")}
    radii = np.logspace(np.log10(r_lo), np.log10(r_hi), n_radii)
    c = np.array([(off < r).mean() for r in radii])

    mask = (c > 0.01) & (c < 0.5)
    if mask.sum() < 4:
        return {"D2": float("nan"), "radii": radii, "C": c, "m": m, "tau": tau}
    slope, _ = np.polyfit(np.log(radii[mask]), np.log(c[mask]), 1)
    return {
        "D2": float(slope),
        "radii": radii,
        "C": c,
        "scaling_mask": mask,
        "m": m,
        "tau": tau,
    }
