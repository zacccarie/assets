"""Command-line entry point for the Latent Flow MVP.

    latent-flow encoders
    latent-flow analyze VIDEO --encoder cnn --encoder vit
"""

from __future__ import annotations

import argparse
import sys


def _cmd_encoders(_: argparse.Namespace) -> int:
    from latent_flow.encoders import ENCODERS, build

    print("registered encoders:")
    for name in ENCODERS.names():
        card = build(name).describe()
        print(f"  {name:6s} [{card.family}]  dim={card.latent_dim}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from latent_flow.analyze import analyze

    result = analyze(
        source=args.video,
        encoders=args.encoder,
        target_frames=args.frames,
        max_side=args.max_side,
        projection=args.projection,
    )
    v = result.video
    print(f"video: {v.source}")
    print(f"  {v.num_frames} frames  {v.resolution[0]}x{v.resolution[1]}  "
          f"{v.fps:.1f} fps  fingerprint={v.fingerprint}")
    for enc in result.encoders:
        tag = "cached" if enc.cached else "computed"
        print(f"\nencoder '{enc.encoder_name}' ({tag})")
        print(f"  latent: T={enc.trajectory.num_steps}  d={enc.trajectory.dim}")
        print(f"  phase space: tau={enc.phase_space['tau']}  "
              f"m={enc.phase_space['dimension']}")
        print(f"  recurrence rate: {enc.recurrence_rate:.3f}")
        print(f"  determinism:     {enc.determinism:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="latent-flow",
        description="Analyze video as a trajectory in a latent state space.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("encoders", help="list registered encoders").set_defaults(
        func=_cmd_encoders
    )

    p = sub.add_parser("analyze", help="run the MVP pipeline on a video")
    p.add_argument("video", help="path to a local video file")
    p.add_argument(
        "--encoder", action="append", default=None,
        help="encoder name (repeatable); default: cnn + vit",
    )
    p.add_argument("--frames", type=int, default=256, help="target sampled frames")
    p.add_argument("--max-side", type=int, default=512, help="max frame side (px)")
    p.add_argument(
        "--projection", default="umap", choices=["umap", "pca"],
        help="latent projection method",
    )
    p.set_defaults(func=_cmd_analyze)

    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "analyze" and not args.encoder:
        args.encoder = ["cnn", "vit"]
    try:
        return args.func(args)
    except (RuntimeError, ImportError, FileNotFoundError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
