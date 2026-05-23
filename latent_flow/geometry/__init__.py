"""Geometry: low-dimensional & spectral views of the latent trajectory."""

from latent_flow.geometry.diffusion import diffusion_map
from latent_flow.geometry.koopman import dmd
from latent_flow.geometry.projection import project

__all__ = ["project", "diffusion_map", "dmd"]
