"""Causal analysis — observational, not interventional.

Three families combined, because only their convergence is trustworthy:

  granger          : linear, vector-autoregressive
  transfer_entropy : non-linear, model-free
  ccm              : dynamic coupling (Sugihara), via shadow manifolds

Plus `regimes`: changepoint segmentation of the trajectory, whose
boundaries feed the bifurcation analysis.
"""

from latent_flow.causal.measures import (
    ccm,
    granger,
    granger_matrix,
    transfer_entropy,
)
from latent_flow.causal.regimes import detect_regimes

__all__ = [
    "granger",
    "granger_matrix",
    "transfer_entropy",
    "ccm",
    "detect_regimes",
]
