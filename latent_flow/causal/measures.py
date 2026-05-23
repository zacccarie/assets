"""Causal coupling measures: Granger, transfer entropy, CCM.

All operate on pairs of 1-D series. The platform applies them between
latent coordinates (or between region encodings in Phase 3). A causal claim
is only made when all three agree.
"""

from __future__ import annotations

import numpy as np

from latent_flow.dynamics.embedding import delay_embedding


def granger(x: np.ndarray, y: np.ndarray, lags: int = 5) -> dict:
    """Granger causality x -> y: does past of x help predict y beyond y's past?

    Returns the F statistic, a normalized RSS-improvement score in [0, 1],
    and the number of lags.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(y)
    if n != len(x) or n <= 2 * lags + 2:
        raise ValueError("series must have equal length > 2*lags+2")

    Y = y[lags:]
    cols_y = [y[lags - i - 1 : n - i - 1] for i in range(lags)]
    cols_x = [x[lags - i - 1 : n - i - 1] for i in range(lags)]
    A_r = np.column_stack([np.ones(len(Y))] + cols_y)
    A_u = np.column_stack([A_r] + cols_x)

    br, *_ = np.linalg.lstsq(A_r, Y, rcond=None)
    bu, *_ = np.linalg.lstsq(A_u, Y, rcond=None)
    rss_r = float(((Y - A_r @ br) ** 2).sum())
    rss_u = float(((Y - A_u @ bu) ** 2).sum())

    df_u = max(len(Y) - A_u.shape[1], 1)
    f_stat = ((rss_r - rss_u) / lags) / (rss_u / df_u + 1e-12)
    score = max(0.0, (rss_r - rss_u) / max(rss_r, 1e-12))
    return {"F": float(f_stat), "score": float(score), "lags": lags}


def granger_matrix(z: np.ndarray, lags: int = 5, dims: int | None = None) -> np.ndarray:
    """Pairwise Granger score matrix between latent coordinates.

    For high-d latents, restrict to the top `dims` PCA coordinates first to
    keep the O(d^2) loop tractable; this is honest because most encoders
    concentrate variance in few directions.
    """
    z = np.asarray(z, dtype=np.float64)
    if dims is not None and z.shape[1] > dims:
        x = z - z.mean(axis=0)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        z = x @ vt[:dims].T
    d = z.shape[1]
    M = np.zeros((d, d), dtype=np.float32)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            M[i, j] = granger(z[:, i], z[:, j], lags=lags)["score"]
    return M  # M[i, j] = i -> j


def transfer_entropy(
    x: np.ndarray, y: np.ndarray, bins: int = 6, lag: int = 1
) -> float:
    """TE(x -> y): non-linear information flow from x to y, one step ahead.

    Binned plug-in estimator. Cheap, biased upward in short series — used
    here qualitatively (rank/compare), not as an absolute information rate.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y) or len(y) <= lag + 1:
        raise ValueError("series too short or mismatched")

    def disc(a: np.ndarray) -> np.ndarray:
        q = np.quantile(a, np.linspace(0, 1, bins + 1))
        q[-1] += 1e-9
        return np.clip(np.digitize(a, q) - 1, 0, bins - 1)

    yp = disc(y[lag:])
    yc = disc(y[:-lag])
    xc = disc(x[:-lag])

    idx = yp + bins * yc + bins * bins * xc
    counts = np.bincount(idx, minlength=bins ** 3).reshape(bins, bins, bins)
    p_yp_yc_xc = counts / counts.sum()
    p_yc_xc = p_yp_yc_xc.sum(axis=0)
    p_yp_yc = p_yp_yc_xc.sum(axis=2)
    p_yc = p_yc_xc.sum(axis=1)

    te = 0.0
    for a in range(bins):
        for b in range(bins):
            for c in range(bins):
                p = p_yp_yc_xc[a, b, c]
                if p > 0 and p_yc_xc[b, c] > 0 and p_yp_yc[a, b] > 0 and p_yc[b] > 0:
                    te += p * np.log(p * p_yc[b] / (p_yc_xc[b, c] * p_yp_yc[a, b]))
    return float(max(0.0, te))


def ccm(
    x: np.ndarray, y: np.ndarray, m: int = 3, tau: int = 1,
    lib_sizes: tuple[int, ...] | None = None, seed: int = 0,
) -> dict:
    """Sugihara's convergent cross mapping.

    Tests whether x can be recovered from the shadow manifold of y. If yes,
    and the skill *grows* with library size, x is dynamically coupled to y
    (informally: "x causes y").
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("series length mismatch")
    embed = delay_embedding(y, m=m, tau=tau)
    x_target = x[(m - 1) * tau:]
    n = len(embed)
    if lib_sizes is None:
        lib_sizes = tuple(np.linspace(max(10, m + 2), n, 6, dtype=int).tolist())
    rng = np.random.default_rng(seed)

    rhos = []
    for L in lib_sizes:
        L = min(L, n)
        lib = rng.choice(n, size=L, replace=False)
        emb_lib, x_lib = embed[lib], x_target[lib]
        preds = np.empty(n)
        k = min(m + 1, L - 1)
        for i in range(n):
            d = np.linalg.norm(emb_lib - embed[i], axis=1)
            nn = np.argsort(d)[:k]
            d_nn = d[nn]
            if d_nn[0] <= 1e-12:
                preds[i] = x_lib[nn[0]]
            else:
                w = np.exp(-d_nn / d_nn[0])
                w /= w.sum()
                preds[i] = float((w * x_lib[nn]).sum())
        rho = float(np.corrcoef(preds, x_target)[0, 1])
        rhos.append(rho if np.isfinite(rho) else 0.0)

    return {
        "library_sizes": list(lib_sizes),
        "rho": rhos,
        "rho_max": float(max(rhos)),
        "converged": bool(rhos[-1] > rhos[0] + 0.05),
    }
