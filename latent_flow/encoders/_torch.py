"""Shared torch backbone runner for the MVP image encoders.

Lazy imports throughout: the package must import cleanly without torch.
GPU is used when available; batching keeps VRAM bounded.
"""

from __future__ import annotations

import numpy as np


def have_torch() -> bool:
    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
    except ImportError:
        return False
    return True


def run_backbone(
    frames: np.ndarray,
    timm_model: str,
    batch_size: int = 16,
    image_size: int = 224,
) -> np.ndarray:
    """Encode frames (T, H, W, 3 uint8) -> features (T, d) float32.

    Uses the timm model's pooled feature vector (no classification head).
    """
    if not have_torch():
        raise ImportError(
            "this encoder needs torch + timm — "
            "install with: pip install 'latent-flow[encoders]'"
        )
    import timm
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(timm_model, pretrained=True, num_classes=0)
    model = model.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats: list[np.ndarray] = []
    autocast = torch.cuda.amp.autocast if device == "cuda" else _nullcast
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            x = torch.from_numpy(chunk).to(device).float() / 255.0
            x = x.permute(0, 3, 1, 2)  # T,H,W,C -> T,C,H,W
            x = F.interpolate(
                x, size=(image_size, image_size), mode="bilinear",
                align_corners=False,
            )
            x = (x - mean) / std
            with autocast():
                out = model(x)
            feats.append(out.float().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


class _nullcast:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
