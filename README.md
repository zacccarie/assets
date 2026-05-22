# Latent Flow

Experimental research platform that analyzes a **video as a trajectory in a
latent state space** — extracting emergent structure, causal dynamics and
multi-scale patterns rather than recognizing objects.

Full design: [`docs/conception-plateforme.md`](docs/conception-plateforme.md).

## Status — Phase 1 (MVP scaffold)

The end-to-end path *video → latent → dynamics + geometry* is in place. The
three load-bearing abstractions (`VideoTensor`, `LatentTrajectory`, `Encoder`)
are frozen.

| Module | MVP | Later |
|--------|-----|-------|
| `ingest` | local file, uniform sampling, cost estimator | streams, motion/event sampling |
| `encoders` | CNN (ConvNeXt), ViT (DINOv2) | AE, VAE, contrastive, temporal, RSSM, world model, JEPA |
| `dynamics` | delay embedding, recurrence/RQA | attractors, bifurcations, Lyapunov |
| `geometry` | UMAP / PCA projection | t-SNE, diffusion maps, persistent homology, Koopman |
| `core` | trajectory store, plugin registry | lazy JobGraph |

## Install

```bash
pip install -e .                  # core only (numpy)
pip install -e '.[all]'           # + ingestion, encoders, geometry, viz
```

Encoder dependencies (`torch`, `timm`) and ingestion (`av`) are optional so the
package imports cleanly without a GPU.

## Use

```bash
latent-flow encoders                                  # list encoders
latent-flow analyze video.mp4 --encoder cnn --encoder vit
latent-flow analyze video.mp4 --projection pca        # no umap-learn needed
```

```python
from latent_flow.analyze import analyze

result = analyze("video.mp4", encoders=["cnn", "vit"])
for enc in result.encoders:
    print(enc.encoder_name, enc.phase_space, enc.determinism)
```

## Test

```bash
python -m pytest tests/ -q
```

Smoke tests cover the dependency-light core (types, store, dynamics, geometry,
registry) on synthetic data.

## Adding an encoder

Register a factory — the kernel never changes:

```python
from latent_flow.encoders import ENCODERS
from latent_flow.encoders.base import Encoder

@ENCODERS.register("my_encoder")
class MyEncoder(Encoder):
    family = "vae"
    name = "my_encoder"
    def encode(self, video): ...
    def latent_dim(self): ...
    def describe(self): ...
```
