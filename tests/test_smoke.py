"""Smoke tests for the Latent Flow MVP skeleton.

These exercise the dependency-light core (types, store, dynamics, geometry,
registry) on synthetic data. Encoder tests that need torch are skipped when
torch is absent.
"""

from __future__ import annotations

import numpy as np
import pytest


def _toy_trajectory(t: int = 200, d: int = 32) -> np.ndarray:
    """A noisy 2-frequency oscillation lifted into d dimensions."""
    ts = np.linspace(0, 8 * np.pi, t)
    base = np.stack([np.sin(ts), np.cos(1.7 * ts)], axis=1)
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((2, d))
    return (base @ mix + 0.02 * rng.standard_normal((t, d))).astype(np.float32)


def test_types_roundtrip(tmp_path):
    from latent_flow.core import LatentTrajectory, TrajectoryStore

    z = _toy_trajectory()
    traj = LatentTrajectory(
        z=z, timestamps=np.arange(len(z)) / 30.0,
        encoder_name="toy", video_fingerprint="abc123",
    )
    store = TrajectoryStore(tmp_path)
    store.put(traj)
    got = store.get("abc123", "toy")
    assert got is not None
    np.testing.assert_allclose(got.z, traj.z)


def test_registry_rejects_duplicates():
    from latent_flow.core import Registry

    reg: Registry[int] = Registry("thing")
    reg.register("a")(lambda: 1)
    assert reg.create("a") == 1
    with pytest.raises(KeyError):
        reg.register("a")(lambda: 2)


def test_delay_embedding_recovers_structure():
    from latent_flow.dynamics import embedding

    recon = embedding.reconstruct(_toy_trajectory())
    assert recon["tau"] >= 1
    assert 1 <= recon["dimension"] <= 10
    assert recon["embedding"].ndim == 2


def test_recurrence_periodic_is_deterministic():
    from latent_flow.dynamics import recurrence

    z = _toy_trajectory()
    mat, thr = recurrence.recurrence_plot(z, rate=0.1)
    assert thr > 0
    assert 0.0 < recurrence.recurrence_rate(mat) < 0.5
    # a periodic signal should be highly deterministic
    assert recurrence.determinism(mat) > 0.5


def test_pca_projection_shapes():
    from latent_flow.geometry import project

    z = _toy_trajectory()
    for n in (2, 3):
        out = project(z, method="pca", n_components=n)
        assert out.shape == (len(z), n)


def test_cost_estimator_verdicts():
    from latent_flow.ingest import CostEstimator

    est = CostEstimator(vram_budget_mb=8000)
    cheap = est.estimate(100, (224, 224), "cnn", batch_size=8)
    assert cheap.verdict == "OK"
    huge = est.estimate(100, (4096, 4096), "vit", batch_size=64)
    assert huge.verdict in {"DEGRADE", "REFUSE"}


def test_encoders_registered():
    from latent_flow.encoders import available

    assert {"cnn", "vit"}.issubset(set(available()))
