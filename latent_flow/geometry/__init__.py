"""Geometry: low-dimensional embeddings of the latent trajectory.

MVP: UMAP / PCA projection to 2D-3D for visualization. Phase 2 adds t-SNE,
diffusion maps, persistent homology, spectral operators and Koopman.
"""

from latent_flow.geometry.projection import project

__all__ = ["project"]
