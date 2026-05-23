"""Local demo: run the full Latent Flow pipeline on the 22 images in the repo.

Treats the image sequence as a 22-frame video. Uses the `ae` encoder (PCA,
no torch needed) so it runs on any machine. End-to-end exercise of every
analysis module: dynamics, multiscale, causal, geometry/Koopman.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from latent_flow.analyze import _full_analysis
from latent_flow.core.types import VideoTensor
from latent_flow.dynamics import embedding, recurrence
from latent_flow.encoders import build
from latent_flow.geometry import project

REPO = Path(__file__).resolve().parent.parent
SIZE = (256, 256)


def _nat_key(p: Path) -> int:
    m = re.search(r"img(\d+)", p.name)
    return int(m.group(1)) if m else 0


def _load_images() -> tuple[np.ndarray, list[str]]:
    paths = sorted(
        [*REPO.glob("img*.jpeg"), *REPO.glob("img*.jpg")], key=_nat_key
    )
    if not paths:
        raise SystemExit(f"no images found under {REPO}")
    frames = np.stack(
        [np.array(Image.open(p).convert("RGB").resize(SIZE)) for p in paths]
    )
    return frames, [p.name for p in paths]


def main() -> None:
    frames, names = _load_images()
    print(f"loaded {len(frames)} frames @ {SIZE}  ({names[0]} ... {names[-1]})")

    video = VideoTensor(
        frames=frames,
        timestamps=np.arange(len(frames), dtype=np.float64),
        fps=1.0,
        source=str(REPO),
        sampling_manifest={"strategy": "image-sequence", "n": len(frames)},
    )
    print(f"VideoTensor fingerprint: {video.fingerprint}")

    encoder = build("ae")
    card = encoder.describe()
    print(f"\nencoder: {card.name}  [{card.family}]  dim={card.latent_dim}")
    traj = encoder.encode(video)
    print(f"  latent trajectory: T={traj.num_steps}  d={traj.dim}")
    print(f"  compression ratio: {traj.meta['compression_ratio']:.1f}x")

    print("\nphase-space reconstruction")
    recon = embedding.reconstruct(traj.z)
    print(f"  tau={recon['tau']}  m={recon['dimension']}")
    rmat, thr = recurrence.recurrence_plot(traj.z)
    print(f"  recurrence rate {recurrence.recurrence_rate(rmat):.3f}  "
          f"determinism {recurrence.determinism(rmat):.3f}  threshold {thr:.2f}")

    print("\nfull analysis (Lyapunov + D2 + multiscale + causal + Koopman)")
    extras = _full_analysis(traj, fs=video.fps)
    print(f"  Lyapunov lambda:      {extras['lyapunov']:+.4f}  "
          f"({extras['chaos_verdict']})")
    print(f"  correlation dim D2:   {extras['correlation_dimension']:.2f}")
    print(f"  dominant period:      {extras['dominant_period']:.2f}  "
          f"(score {extras['period_score']:.1f})")
    print(f"  attractors:           {extras['n_attractors']}  "
          f"(separation {extras['attractor_separation']:.2f})")
    ms = extras["multiscale"]
    print(f"  multi-scale:          disappears={ms['disappears']:.2f}  "
          f"persists={ms['persists']:.2f}  emerges={ms['emerges']:.3f}")
    print(f"  regime boundaries:    {extras['regime_boundaries']}")
    kp = extras["koopman"]
    print(f"  Koopman rank {kp['rank']}  top freqs: "
          f"{[f'{f:.3f}' for f in kp['top_frequencies_hz']]}")

    print("\nprojections")
    for method in ("pca", "diffusion"):
        proj = project(traj.z, method=method, n_components=3)
        print(f"  {method:9s} shape={proj.shape}  "
              f"range=[{proj.min():+.2f}, {proj.max():+.2f}]")


if __name__ == "__main__":
    main()
