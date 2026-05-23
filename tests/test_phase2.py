"""Phase 2 tests: dynamics expansion, multiscale, causal, geometry.

All numpy-only. They exercise the analytic pipeline on synthetic systems
with known ground truth, so test failures point to a real numerical bug,
not just a refactor.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------- fixtures: known dynamical systems ---------------------

def _periodic(t: int = 400, freq: float = 0.05, d: int = 16) -> np.ndarray:
    ts = np.arange(t)
    base = np.stack(
        [np.sin(2 * np.pi * freq * ts), np.cos(2 * np.pi * freq * ts)], axis=1
    )
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((2, d))
    return (base @ mix + 0.01 * rng.standard_normal((t, d))).astype(np.float32)


def _logistic_chaos(t: int = 600, r: float = 3.95) -> np.ndarray:
    """1D chaotic logistic map, embedded into d=8 via random lift."""
    x = np.empty(t, dtype=np.float64)
    x[0] = 0.4
    for i in range(1, t):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    rng = np.random.default_rng(1)
    lift = rng.standard_normal((1, 8))
    return (x[:, None] @ lift).astype(np.float32)


def _two_regimes(t: int = 400) -> np.ndarray:
    """A clear regime switch at t = 200."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal((200, 4)) * 0.5
    b = rng.standard_normal((200, 4)) * 0.5 + 5.0
    return np.concatenate([a, b], axis=0).astype(np.float32)


# ---------------------- dynamics ----------------------

def test_lyapunov_periodic_is_small_chaotic_is_positive():
    from latent_flow.dynamics import lyapunov_rosenstein

    lp = lyapunov_rosenstein(_periodic(), fs=1.0)
    lc = lyapunov_rosenstein(_logistic_chaos(), fs=1.0)
    assert lc["lambda"] > lp["lambda"]  # chaos > periodic — the key invariant
    assert lc["lambda"] > 0.0


def test_correlation_dimension_reasonable():
    from latent_flow.dynamics import correlation_dimension

    d2 = correlation_dimension(_periodic())["D2"]
    # periodic on a 2-torus -> dimension between ~0.5 and ~3
    assert np.isfinite(d2)
    assert 0.3 < d2 < 3.5


def test_dominant_period_recovers_frequency():
    from latent_flow.dynamics import dominant_period

    out = dominant_period(_periodic(freq=0.05), fs=1.0)
    assert abs(out["period"] - 20.0) < 2.0
    assert out["score"] > 5.0


def test_attractor_count_for_two_regimes():
    from latent_flow.dynamics import identify_attractors

    out = identify_attractors(_two_regimes(), max_k=4)
    assert out["n_attractors"] >= 2


# ---------------------- multiscale ----------------------

def test_spectrum_shape_and_emergence():
    from latent_flow.multiscale import disappear_persist_emerge, spectrum

    z = _periodic()
    freqs, power = spectrum(z, fs=1.0)
    assert power.shape == (len(freqs), z.shape[1])

    em = disappear_persist_emerge(z, fs=1.0, coarse_scale=8)
    assert 0.0 <= em["disappears"] <= 1.0
    assert 0.0 <= em["persists"] <= 1.0
    assert abs(em["disappears"] + em["persists"] - 1.0) < 1e-6


def test_block_average_reduces_length():
    from latent_flow.multiscale import temporal_block_average

    z = np.random.randn(100, 5)
    out = temporal_block_average(z, scale=4)
    assert out.shape == (25, 5)


# ---------------------- causal ----------------------

def _coupled_pair(n: int = 500, coupling: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + 0.1 * rng.standard_normal()
        y[t] = 0.5 * y[t - 1] + coupling * x[t - 1] + 0.1 * rng.standard_normal()
    return x, y


def test_granger_detects_coupling_direction():
    from latent_flow.causal import granger

    x, y = _coupled_pair()
    # x drives y, so x->y should score much higher than y->x
    assert granger(x, y, lags=3)["score"] > granger(y, x, lags=3)["score"] + 0.05


def test_transfer_entropy_directional():
    from latent_flow.causal import transfer_entropy

    x, y = _coupled_pair()
    assert transfer_entropy(x, y, bins=5) > transfer_entropy(y, x, bins=5)


def test_ccm_runs_and_converges_on_coupled():
    from latent_flow.causal import ccm

    x, y = _coupled_pair(n=300)
    out = ccm(x, y, m=3, tau=1)
    assert len(out["rho"]) == len(out["library_sizes"])
    assert out["rho_max"] > 0.0


def test_regime_detection_finds_switch():
    from latent_flow.causal import detect_regimes

    regs = detect_regimes(_two_regimes(), window=20, threshold=2.0)
    assert any(180 < r["index"] < 220 for r in regs)


# ---------------------- geometry ----------------------

def test_dmd_recovers_rotation_frequency():
    """A pure rotation z_{t+1} = R(theta) z_t should give freq = theta/(2π).

    With theta = 2π * 0.1 we expect frequency ~ 0.1.
    """
    from latent_flow.geometry import dmd

    theta = 2 * np.pi * 0.1
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    z = np.zeros((200, 2))
    z[0] = [1.0, 0.0]
    for t in range(1, 200):
        z[t] = R @ z[t - 1]
    out = dmd(z)
    assert any(abs(f - 0.1) < 0.02 for f in out["frequencies"])


def test_diffusion_map_shape():
    from latent_flow.geometry import diffusion_map

    coords = diffusion_map(_periodic(), n_components=3, n_neighbors=10)
    assert coords.shape == (400, 3)


def test_projection_diffusion_method():
    from latent_flow.geometry import project

    out = project(_periodic(), method="diffusion", n_components=2)
    assert out.shape == (400, 2)


# ---------------------- encoder registry ----------------------

def test_phase2_encoders_registered():
    from latent_flow.encoders import available

    expected = {"cnn", "vit", "ae", "contrastive", "temporal", "vae", "rssm",
                "world_model", "jepa"}
    assert expected.issubset(set(available()))


def test_stub_encoder_describes_but_refuses_to_encode():
    from latent_flow.core.types import VideoTensor
    from latent_flow.encoders import build

    enc = build("jepa")
    card = enc.describe()
    assert card.family == "jepa"
    assert card.principle  # non-empty
    fake = VideoTensor(
        frames=np.zeros((4, 32, 32, 3), dtype=np.uint8),
        timestamps=np.arange(4, dtype=np.float64) / 30.0,
        fps=30.0,
        source="fake",
    )
    with pytest.raises(NotImplementedError):
        enc.encode(fake)
