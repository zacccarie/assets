"""Largest Lyapunov exponent — Rosenstein's method.

Robust on short, noisy series. Reports the exponent plus the divergence
curve, so the linear-fit region can be inspected. Sign of `lambda`:
  > 0    chaotic (sensitivity to initial conditions)
  ~ 0    periodic / marginally stable
  < 0    contracting
"""

from __future__ import annotations

import numpy as np

from latent_flow.dynamics.embedding import (
    delay_embedding,
    estimate_delay,
    estimate_dimension,
    principal_component,
)


def lyapunov_rosenstein(
    series: np.ndarray,
    m: int | None = None,
    tau: int | None = None,
    fs: float = 1.0,
    theiler: int | None = None,
    max_steps: int = 40,
) -> dict:
    """Largest Lyapunov exponent from a 1D series or a (T, d) trajectory.

    fs : sampling frequency — lambda is reported in 1/time units.
    theiler : window to exclude around each point in the neighbour search
              (avoids temporally-correlated false pairs).
    """
    s = principal_component(np.asarray(series))
    if tau is None:
        tau = estimate_delay(s)
    if m is None:
        m = estimate_dimension(s, tau)
    emb = delay_embedding(s, m=m, tau=tau)
    n = len(emb)
    if theiler is None:
        theiler = max(tau * m, 1)

    dist = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=-1)
    idx = np.arange(n)
    mask = np.abs(np.subtract.outer(idx, idx)) <= theiler
    dist[mask] = np.inf
    nn = np.argmin(dist, axis=1)

    max_k = min(max_steps, n - 1)
    div = []
    for k in range(1, max_k):
        i = idx[(idx + k < n) & (nn + k < n)]
        if len(i) == 0:
            break
        d = np.linalg.norm(emb[i + k] - emb[nn[i] + k], axis=1)
        d = d[d > 0]
        if len(d) == 0:
            break
        div.append(float(np.mean(np.log(d))))
    div = np.asarray(div)
    if len(div) < 4:
        return {"lambda": float("nan"), "m": m, "tau": tau, "divergence_curve": div}

    # Rosenstein-style fit on the *early* region — chaotic divergence saturates
    # quickly, so a wide window dilutes the slope. Use the first ~5 lags.
    fit_end = max(3, min(5, len(div)))
    ks = np.arange(1, fit_end + 1)
    slope, _ = np.polyfit(ks, div[:fit_end], 1)
    return {
        "lambda": float(slope * fs),
        "m": m,
        "tau": tau,
        "fs": fs,
        "divergence_curve": div,
        "fit_end": int(fit_end),
    }


def chaos_verdict(lam: float) -> str:
    if not np.isfinite(lam):
        return "undetermined"
    if lam < 0.005:
        return "regular"
    if lam < 0.05:
        return "weakly chaotic"
    return "chaotic"
