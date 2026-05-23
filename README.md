# Latent Flow

Experimental research platform that analyzes a **video as a trajectory in a
latent state space** — extracting emergent structure, causal dynamics and
multi-scale patterns rather than recognizing objects.

Full design: [`docs/conception-plateforme.md`](docs/conception-plateforme.md).

## Status — Phase 2 (analysis depth)

The end-to-end path *video → latent → dynamics + geometry* is in place, and
the comparison-grade analysis suite is now wired up behind a `--full` flag.
The three load-bearing abstractions (`VideoTensor`, `LatentTrajectory`,
`Encoder`) are frozen.

| Module | In repo | Deferred |
|--------|---------|----------|
| `ingest` | local file, uniform sampling, cost estimator | streams (P3), motion/event sampling |
| `encoders` | CNN, ViT, contrastive, temporal, AE (PCA baseline); VAE/RSSM/world_model/JEPA stubs with frozen contracts | actual training of VAE/RSSM/world_model; V-JEPA wiring |
| `dynamics` | delay embedding, recurrence/RQA, Lyapunov (Rosenstein), correlation dimension, dominant period, attractor clustering | bifurcation diagrams, sliding-window Lyapunov |
| `multiscale` | FFT spectrum, dominant modes, temporal coarse-graining, *disappears / persists / emerges* score | wavelets, multi-resolution DMD |
| `causal` | Granger (+ matrix), transfer entropy, CCM (Sugihara), regime changepoints | regional graphs over patch encodings |
| `geometry` | UMAP, PCA, t-SNE, diffusion maps, DMD/Koopman | persistent homology, sliding-window topology |
| `core` | trajectory store, plugin registry | lazy JobGraph (P3) |

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
latent-flow analyze video.mp4 --full                  # Lyapunov + D2 + multiscale + causal + Koopman
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
