"""Content-addressed store for latent trajectories.

Artifacts are immutable and keyed by (video fingerprint, encoder, params).
Re-running an analysis with a changed parameter recomputes only what depends
on it. MVP backend: .npz files on disk + an in-process dict cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from latent_flow.core.types import LatentTrajectory


def _key(video_fingerprint: str, encoder_name: str, params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    h = hashlib.blake2b(digest_size=12)
    h.update(f"{video_fingerprint}|{encoder_name}|{blob}".encode())
    return h.hexdigest()


class TrajectoryStore:
    def __init__(self, root: str | Path = ".latent_flow_cache") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, LatentTrajectory] = {}

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def get(
        self, video_fingerprint: str, encoder_name: str, params: dict | None = None
    ) -> LatentTrajectory | None:
        key = _key(video_fingerprint, encoder_name, params or {})
        if key in self._mem:
            return self._mem[key]
        path = self._path(key)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        traj = LatentTrajectory(
            z=data["z"],
            timestamps=data["timestamps"],
            encoder_name=str(data["encoder_name"]),
            video_fingerprint=str(data["video_fingerprint"]),
            meta=json.loads(str(data["meta"])),
        )
        self._mem[key] = traj
        return traj

    def put(self, traj: LatentTrajectory, params: dict | None = None) -> str:
        key = _key(traj.video_fingerprint, traj.encoder_name, params or {})
        np.savez(
            self._path(key),
            z=traj.z,
            timestamps=traj.timestamps,
            encoder_name=traj.encoder_name,
            video_fingerprint=traj.video_fingerprint,
            meta=json.dumps(traj.meta, default=str),
        )
        self._mem[key] = traj
        return key
