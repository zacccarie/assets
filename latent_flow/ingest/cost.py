"""CostEstimator — predict VRAM / RAM / wall time *before* heavy compute.

Heuristic for the MVP: closed-form formulas with conservative constants.
Phase 2 replaces the time estimate with a 10-frame calibration micro-benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Rough per-encoder cost, calibrated to be pessimistic. Refined in Phase 2.
_ENCODER_PROFILE = {
    "cnn": {"vram_mb_per_frame": 6.0, "ms_per_frame": 4.0, "latent_dim": 512},
    "vit": {"vram_mb_per_frame": 14.0, "ms_per_frame": 11.0, "latent_dim": 768},
}

Verdict = Literal["OK", "DEGRADE", "REFUSE"]


@dataclass
class CostEstimate:
    vram_mb: float
    host_ram_mb: float
    wall_seconds: float
    verdict: Verdict
    suggestion: str

    def render(self) -> str:
        return (
            f"[cost] VRAM~{self.vram_mb:.0f}MB  RAM~{self.host_ram_mb:.0f}MB  "
            f"time~{self.wall_seconds:.1f}s  -> {self.verdict}"
            + (f" ({self.suggestion})" if self.suggestion else "")
        )


class CostEstimator:
    def __init__(self, vram_budget_mb: float = 8000.0) -> None:
        self.vram_budget_mb = vram_budget_mb

    def estimate(
        self,
        num_frames: int,
        resolution: tuple[int, int],
        encoder_name: str,
        batch_size: int = 16,
    ) -> CostEstimate:
        profile = _ENCODER_PROFILE.get(encoder_name, _ENCODER_PROFILE["vit"])
        h, w = resolution
        scale = (h * w) / (224 * 224)

        vram = profile["vram_mb_per_frame"] * batch_size * scale + 1500.0
        latent_mb = num_frames * profile["latent_dim"] * 4 / 1e6
        host_ram = num_frames * h * w * 3 / 1e6 + latent_mb
        wall = num_frames * profile["ms_per_frame"] * scale / 1000.0

        if vram <= self.vram_budget_mb:
            return CostEstimate(vram, host_ram, wall, "OK", "")
        if vram <= self.vram_budget_mb * 2:
            return CostEstimate(
                vram, host_ram, wall, "DEGRADE",
                f"reduce batch_size to ~{max(1, batch_size // 2)} or resolution",
            )
        return CostEstimate(
            vram, host_ram, wall, "REFUSE",
            "downscale the video or pick a lighter encoder",
        )
