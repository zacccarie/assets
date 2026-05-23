"""Per-coordinate power spectrum and dominant temporal modes."""

from __future__ import annotations

import numpy as np


def spectrum(z: np.ndarray, fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum per latent coordinate.

    Returns (freqs, power) where power has shape (F, d).
    """
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    n = len(z)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(z, axis=0)) ** 2 / max(n, 1)
    return freqs, power


def dominant_modes(
    z: np.ndarray, fs: float = 1.0, k: int = 5, skip_dc: bool = True
) -> list[tuple[float, float]]:
    """Top-k frequencies by total energy across the latent."""
    freqs, power = spectrum(z, fs)
    energy = power.sum(axis=1)
    if skip_dc:
        energy = energy.copy()
        energy[0] = 0
    idx = np.argsort(energy)[::-1][:k]
    return [(float(freqs[i]), float(energy[i])) for i in idx]
