"""Attractor identification by clustering of phase-space occupancy.

A coarse but useful first pass: cluster the delay-embedded trajectory and
treat dense, well-separated clusters as candidate attractor regions. The
number of stable clusters across resamples = the number of attractors.
"""

from __future__ import annotations

import numpy as np

from latent_flow.dynamics.embedding import (
    delay_embedding,
    estimate_delay,
    estimate_dimension,
    principal_component,
)


def _kmeans(x: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(iters):
        d = np.linalg.norm(x[:, None] - centers[None], axis=-1)
        labels = np.argmin(d, axis=1)
        new = np.stack(
            [x[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
             for j in range(k)]
        )
        if np.allclose(new, centers):
            break
        centers = new
    return labels


def identify_attractors(
    z: np.ndarray, max_k: int = 6, m: int | None = None, tau: int | None = None
) -> dict:
    """Estimate the number of attractor regions and their occupancy."""
    s = principal_component(np.asarray(z))
    if tau is None:
        tau = estimate_delay(s)
    if m is None:
        m = estimate_dimension(s, tau)
    emb = delay_embedding(s, m=m, tau=tau)

    # silhouette-like criterion: choose k that maximizes inter/intra cluster ratio.
    best = {"k": 1, "score": -np.inf, "labels": np.zeros(len(emb), dtype=int)}
    for k in range(2, max_k + 1):
        labels = _kmeans(emb, k)
        within, between = 0.0, 0.0
        centers = np.stack(
            [emb[labels == j].mean(axis=0) for j in range(k)]
        )
        for j in range(k):
            mask = labels == j
            if mask.any():
                within += np.linalg.norm(emb[mask] - centers[j], axis=1).mean()
        between = np.linalg.norm(
            centers[:, None] - centers[None], axis=-1
        )[np.triu_indices(k, 1)].mean()
        score = between / (within / k + 1e-9)
        if score > best["score"]:
            best = {"k": k, "score": float(score), "labels": labels}

    occupancy = np.bincount(best["labels"], minlength=best["k"]) / len(best["labels"])
    return {
        "n_attractors": int(best["k"]),
        "separation_score": float(best["score"]),
        "occupancy": occupancy.tolist(),
        "labels": best["labels"],
    }
