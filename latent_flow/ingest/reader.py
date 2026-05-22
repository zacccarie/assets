"""SourceReader — abstraction over file / URL / webcam / RTSP.

MVP: local files via PyAV (frame-accurate). Real-time sources are stubbed
with a clear error; they land in Phase 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SourceReader:
    def __init__(self, source: str) -> None:
        self.source = source
        self.is_stream = source.startswith(("rtsp://", "http://", "https://")) or (
            source.isdigit()
        )

    def read(self, max_frames: int | None = None) -> tuple[np.ndarray, float]:
        """Return (frames uint8 (T, H, W, 3) RGB, fps)."""
        if self.is_stream:
            raise NotImplementedError(
                "real-time stream ingestion is a Phase 3 feature; "
                "the MVP supports local files only"
            )
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(self.source)
        return self._read_file(path, max_frames)

    @staticmethod
    def _read_file(path: Path, max_frames: int | None) -> tuple[np.ndarray, float]:
        try:
            import av  # lazy: optional dependency
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "video decoding needs PyAV — install with: pip install 'latent-flow[ingest]'"
            ) from exc

        container = av.open(str(path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if max_frames is not None and len(frames) >= max_frames:
                break
        container.close()
        if not frames:
            raise RuntimeError(f"no frames decoded from {path}")
        return np.stack(frames), fps
