"""Dimensionality reduction for visualization.

`umap` respects local structure; `pca` is dependency-free and always works.
Phase 2 adds diffusion maps (preferred — diffusion distance encodes
transition time, so it preserves attractor structure).
"""

from __future__ import annotations

import numpy as np


def _pca(z: np.ndarray, n_components: int) -> np.ndarray:
    x = z - z.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return (u[:, :n_components] * s[:n_components]).astype(np.float32)


def project(
    z: np.ndarray, method: str = "umap", n_components: int = 3
) -> np.ndarray:
    """Project a (T, d) trajectory to (T, n_components)."""
    z = np.asarray(z, dtype=np.float64)
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")

    if method == "pca":
        return _pca(z, n_components)

    if method == "umap":
        try:
            import umap  # lazy: optional dependency
        except ImportError:
            # graceful fallback keeps the MVP runnable without umap-learn
            return _pca(z, n_components)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(15, len(z) - 1),
            metric="euclidean",
        )
        return reducer.fit_transform(z).astype(np.float32)

    if method == "tsne":
        raise NotImplementedError("t-SNE is a Phase 2 feature")

    raise ValueError(f"unknown projection method '{method}'")
