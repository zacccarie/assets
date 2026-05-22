"""Dynamics: treat the latent trajectory as a dynamical system.

MVP: phase-space (delay) embedding + recurrence analysis. Phase 2 adds
attractors, cycles, bifurcations and Lyapunov / correlation-dimension
estimation.
"""

from latent_flow.dynamics.embedding import (
    delay_embedding,
    estimate_delay,
    estimate_dimension,
)
from latent_flow.dynamics.recurrence import recurrence_plot, recurrence_rate

__all__ = [
    "delay_embedding",
    "estimate_delay",
    "estimate_dimension",
    "recurrence_plot",
    "recurrence_rate",
]
