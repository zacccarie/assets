"""Dynamic Mode Decomposition / Koopman approximation.

DMD finds the best linear operator A such that z_{t+1} ≈ A z_t.
Its eigenvalues live in the complex plane: |λ| controls growth/decay,
arg(λ) controls oscillation. Together they linearize a nonlinear flow in
the lifted Koopman observable space.
"""

from __future__ import annotations

import numpy as np


def dmd(z: np.ndarray, rank: int | None = None, dt: float = 1.0) -> dict:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or len(z) < 2:
        raise ValueError("z must be (T, d) with T >= 2")
    X = z[:-1].T
    Y = z[1:].T

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    if rank is None:
        cum = np.cumsum(S ** 2) / max((S ** 2).sum(), 1e-12)
        rank = int(np.searchsorted(cum, 0.99) + 1)
    rank = max(1, min(rank, len(S)))

    U_r, S_r, V_r = U[:, :rank], S[:rank], Vt[:rank].T
    A_tilde = U_r.T @ Y @ V_r / S_r  # (r, r)
    eigvals, eigvecs = np.linalg.eig(A_tilde)
    modes = Y @ V_r @ np.diag(1.0 / S_r) @ eigvecs  # (d, r)

    omega = np.log(eigvals.astype(complex)) / dt
    return {
        "eigenvalues": eigvals,
        "omega": omega,
        "modes": modes,
        "rank": int(rank),
        "frequencies": np.abs(np.imag(omega)) / (2 * np.pi),
        "growth_rates": np.real(omega),
    }
