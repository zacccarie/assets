"""Diffusion maps — embedding that respects the dynamics.

Unlike t-SNE/UMAP, diffusion distance ≈ transition time on the kNN graph.
So the projection preserves attractor structure: points that are quickly
reachable on the trajectory stay close.
"""

from __future__ import annotations

import numpy as np


def diffusion_map(
    z: np.ndarray, n_components: int = 3, n_neighbors: int = 15,
    sigma: float | None = None,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    n = len(z)
    if n_components >= n:
        raise ValueError("n_components must be < number of points")

    d = np.linalg.norm(z[:, None] - z[None, :], axis=-1)
    if sigma is None:
        k = min(n_neighbors + 1, n)
        sigma = float(np.median(np.sort(d, axis=1)[:, 1:k]))
    if sigma <= 0:
        sigma = 1.0

    K = np.exp(-(d ** 2) / (2 * sigma ** 2))
    row = K.sum(axis=1)
    row[row == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(row)
    M = d_inv_sqrt[:, None] * K * d_inv_sqrt[None, :]

    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(eigvals)[::-1]
    # drop the trivial first eigenvector (constant component)
    keep = order[1 : 1 + n_components]
    coords = eigvecs[:, keep] * eigvals[keep]
    return coords.astype(np.float32)
