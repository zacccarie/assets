"""Dynamics: treat the latent trajectory as a dynamical system."""

from latent_flow.dynamics.attractors import identify_attractors
from latent_flow.dynamics.correlation_dimension import correlation_dimension
from latent_flow.dynamics.cycles import dominant_period
from latent_flow.dynamics.embedding import (
    delay_embedding,
    estimate_delay,
    estimate_dimension,
    principal_component,
    reconstruct,
)
from latent_flow.dynamics.lyapunov import chaos_verdict, lyapunov_rosenstein
from latent_flow.dynamics.recurrence import (
    determinism,
    recurrence_plot,
    recurrence_rate,
)

__all__ = [
    "delay_embedding",
    "estimate_delay",
    "estimate_dimension",
    "principal_component",
    "reconstruct",
    "recurrence_plot",
    "recurrence_rate",
    "determinism",
    "lyapunov_rosenstein",
    "chaos_verdict",
    "correlation_dimension",
    "dominant_period",
    "identify_attractors",
]
