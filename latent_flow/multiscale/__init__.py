"""Multi-scale analysis: what scale carries what structure.

Operationalizes the design's three-column view:
  disappears  — high-frequency / fine-scale content lost under coarse-graining
  persists    — low-frequency / coarse-scale content invariant to scale
  emerges     — structure invisible at the fine scale, visible at coarse scales
"""

from latent_flow.multiscale.coarse_grain import hierarchical, temporal_block_average
from latent_flow.multiscale.emergence import disappear_persist_emerge
from latent_flow.multiscale.frequency import dominant_modes, spectrum

__all__ = [
    "spectrum",
    "dominant_modes",
    "temporal_block_average",
    "hierarchical",
    "disappear_persist_emerge",
]
