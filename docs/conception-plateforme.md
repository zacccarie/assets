# Latent Flow — Plateforme de recherche sur les dynamiques latentes vidéo

> Outil expérimental d'extraction, d'analyse et de visualisation des structures
> émergentes, dynamiques causales et motifs multi-échelles d'une vidéo, traitée
> comme une **trajectoire dans un espace d'état**.

Nom de travail : **Latent Flow**.

---

## 0. Cadrage : la vidéo comme système dynamique

Hypothèse centrale du projet : une vidéo n'est pas une collection d'images mais
l'**observation partielle d'un système dynamique** sous-jacent. Chaque frame
`x_t` est une mesure bruitée et haute-dimension d'un état latent `z_t` qui évolue
selon une loi `z_{t+1} = F(z_t, u_t)` (avec `u_t` un éventuel contrôle/forçage).

On veut donc estimer trois objets distincts, souvent confondus :

| Objet | Question | Outils |
|-------|----------|--------|
| L'**encodeur** `E : x → z` | Quelle est la bonne carte d'observation ? | CNN, ViT, AE/VAE, contrastif, JEPA |
| Le **flot latent** `F : z_t → z_{t+1}` | Quelle est la loi d'évolution ? | RSSM, world models, Koopman, SINDy |
| La **géométrie de l'espace** `z` | Quelle structure (variété, attracteurs, échelles) ? | UMAP, diffusion maps, homologie persistante |

L'outil ne cherche pas *l'objet* dans l'image, mais la **structure du mouvement
dans l'espace de représentation**. Toute la conception découle de cette
séparation encodeur / dynamique / géométrie.

### Invariant de conception

Tout encodeur, tout estimateur de dynamique, tout réducteur géométrique respecte
une seule interface : il transforme un tenseur de trajectoire
`Z ∈ ℝ^{T×d}` (T pas de temps, d dimensions) en un autre tenseur de trajectoire,
ou en un descripteur. Cette homogénéité est ce qui rend la **comparaison**
possible — qui est l'objet réel du projet.

---

## 1. Architecture détaillée

### 1.1 Vue d'ensemble en couches

```
┌────────────────────────────────────────────────────────────────┐
│  UI  — frontend interactif (latent 2D/3D, trajectoires, causal) │
├────────────────────────────────────────────────────────────────┤
│  API / Orchestration  — sessions, jobs, cache, streaming        │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Ingestion   │  Encodeurs   │  Dynamique   │  Géométrie & Multi- │
│  vidéo       │  (zoo)       │  & Causalité │  échelle            │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│  Noyau dynamique  — tenseurs de trajectoire, registres, métriques│
├────────────────────────────────────────────────────────────────┤
│  Runtime GPU  — torch, mémoire/VRAM, batching, profilage        │
└────────────────────────────────────────────────────────────────┘
```

Principe : **un noyau, des plugins**. Les six familles d'analyse sont des
plugins enregistrés ; le noyau ne connaît que l'interface `TrajectoryTransform`.

### 1.2 Module `ingest` — ingestion vidéo

Responsable de transformer une source hétérogène en un flux de frames normalisé,
avec budget de calcul maîtrisé.

Composants :

- **`SourceReader`** — abstraction sur fichier local, URL, webcam, flux RTSP.
  Backend : `PyAV` (décodage précis frame-accurate) + `decord` (accès aléatoire
  rapide GPU) ; fallback `torchvision.io`.
- **`AdaptiveSampler`** — échantillonnage adaptatif. Trois stratégies :
  - *uniforme* (baseline) ;
  - *piloté par le mouvement* : densifie l'échantillonnage là où la norme du
    flux optique (ou la distance inter-frames) est élevée, raréfie sur les plans
    statiques ;
  - *piloté par l'événement* : détection de coupures de plan (PySceneDetect)
    pour ne jamais interpoler une dynamique à travers un cut.
- **`Transcoder`** — compression automatique si la vidéo dépasse un seuil
  (résolution, débit, durée). Réencodage `ffmpeg` vers une cible
  (ex. 512px, 16 fps) ; conserve un manifeste de la transformation.
- **`CostEstimator`** — *avant* tout calcul lourd, estime :
  - VRAM = f(résolution, taille de batch, encodeur, dtype) ;
  - RAM hôte = f(T, d, nb de représentations mises en cache) ;
  - temps mural ≈ T × coût/frame mesuré sur un micro-benchmark de calibration
    (10 frames) au lancement.
  Renvoie un verdict `OK / DÉGRADER / REFUSER` avec une suggestion de réglages.

Sortie canonique : un objet `VideoTensor` — frames `ℝ^{T×3×H×W}`, timestamps,
manifeste d'échantillonnage, et empreinte (hash) pour le cache.

### 1.3 Module `encoders` — le zoo d'encodeurs

Chaque encodeur implémente :

```python
class Encoder(Protocol):
    name: str
    family: Literal["cnn","vit","ae","vae","contrastive",
                    "temporal","world_model","rssm","jepa"]
    def encode(self, video: VideoTensor) -> LatentTrajectory: ...
    def latent_dim(self) -> int: ...
    def describe(self) -> EncoderCard:   # principe, hypothèses, refs
        ...
```

| Famille | Rôle | Implémentation de référence |
|---------|------|-----------------------------|
| CNN | baseline spatiale, features hiérarchiques | ResNet/ConvNeXt tronqué, features par couche |
| ViT | tokens par patch, attention globale | ViT/DINOv2 (features denses auto-supervisées) |
| Autoencoder | compression reconstructive non contrainte | AE conv. entraîné par tâche |
| VAE | latent probabiliste, régularisé | β-VAE — on expose β comme curseur |
| Contrastif | invariance par augmentation | SimCLR/MoCo — frames vs vues |
| Temporel | encode des fenêtres, pas des frames | VideoMAE, S3D, encodeur 3D-conv |
| World model | latent + prédiction du futur | DreamerV3 (composant observation) |
| RSSM | état récurrent stochastique+déterministe | RSSM séparé de Dreamer, exposé seul |
| JEPA-like | prédiction en espace latent, pas en pixels | V-JEPA / I-JEPA — predictor + target EMA |

Pour **chaque** encodeur, le module produit une `EncoderCard` :
1. *principe* (texte + schéma) ;
2. *espace latent* — projection 2D/3D de la trajectoire ;
3. *visualisation de compression* — taux de compression effectif, courbe
   rate–distortion, énergie spectrale conservée ;
4. *stabilité* — sensibilité aux perturbations (bruit, occlusion, jitter
   temporel), mesurée par la dérive de `z` sous perturbation bornée.

Tous les encodeurs tournent **pré-entraînés** par défaut (l'entraînement n'est
pas le cœur du projet) ; un mode *fine-tune léger* est prévu en V2.

### 1.4 Module `dynamics` — représentation dynamique

Prend une `LatentTrajectory` et reconstruit le système dynamique.

- **Reconstruction d'espace des phases** — plongement par délais (Takens) :
  `z_t → (z_t, z_{t-τ}, …, z_{t-(m-1)τ})`. Choix de `τ` par première
  annulation de l'auto-information mutuelle, de `m` par faux plus proches
  voisins.
- **Trajectoires latentes** — la courbe `t → z_t` projetée, colorée par le
  temps, la vitesse, ou la courbure.
- **Attracteurs / cycles** — détection de récurrence (Recurrence Plots,
  RQA), clustering de l'occupation de l'espace, périodicité par auto-corrélation.
- **Bifurcations** — analyse en fenêtre glissante : on suit comment la
  structure (nb de clusters, exposants) change le long de la vidéo ; un
  changement qualitatif = bifurcation candidate.
- **Estimation du chaos** — plus grand exposant de Lyapunov (Rosenstein),
  dimension de corrélation (Grassberger–Procaccia), entropie (sample/permutation
  entropy). Donne un verdict ordonné/périodique/chaotique avec barres
  d'incertitude.

### 1.5 Module `multiscale` — analyse multi-échelle

- **Coarse-graining / renormalisation** — regroupe progressivement composantes
  latentes ou régions spatiales ; à chaque échelle on mesure ce qui est
  *conservé*. Inspiré du flot de renormalisation : on cherche les quantités
  **invariantes par changement d'échelle**.
- **Compression hiérarchique** — pyramide de représentations
  (fine → grossière), p. ex. par AE empilés ou clustering hiérarchique du
  latent.
- **Décomposition fréquentielle** — sur chaque coordonnée latente : FFT, DWT
  (ondelettes), DMD pour séparer modes lents/rapides.
- **Structures invariantes** — quantités stables à travers les échelles
  (spectre, dimension, modes dominants).

Visualisation cible : **trois colonnes** — *ce qui disparaît* (modes rapides,
détails de texture), *ce qui persiste* (modes lents, structure causale), *les
motifs émergents* (structures absentes à l'échelle fine mais nettes à l'échelle
grossière).

### 1.6 Module `causal` — analyse causale

- **Dépendances temporelles** — causalité de Granger, transfer entropy entre
  coordonnées latentes.
- **Influence entre régions** — on découpe la frame en régions (patches ViT
  ou superpixels), on encode chacune, et on estime le graphe d'influence
  inter-régions.
- **Structures relationnelles** — graphe dynamique dont les nœuds sont des
  régions/objets latents, les arêtes des influences pondérées dans le temps.
- **Causalité émergente** — convergent cross-mapping (CCM, Sugihara) pour
  distinguer corrélation et couplage dynamique.
- **Transitions dynamiques** — segmentation de la trajectoire en régimes
  (HMM/HSMM sur le latent, ou changepoint detection) ; les frontières sont les
  transitions.

> ⚠️ Toute sortie causale est étiquetée **« causalité au sens dynamique /
> observationnelle »** — pas de prétention à de la causalité interventionnelle
> sans intervention. C'est un garde-fou méthodologique explicite dans l'UI.

### 1.7 Module `geometry` — géométrie & topologie

- **Réduction** : UMAP, t-SNE, diffusion maps (cette dernière privilégiée car
  elle respecte la dynamique : la distance de diffusion ≈ temps de transition).
- **Topologie** : homologie persistante (`giotto-tda` / `ripser`) — diagrammes
  de persistance, codes-barres ; détecte trous (cycles) et composantes.
- **Graphes dynamiques** : k-NN dans le latent, évoluant dans le temps.
- **Opérateurs spectraux** : Laplacien du graphe, ses modes propres.
- **Koopman** : approximation de l'opérateur de Koopman (EDMD / DMD) — rend
  *linéaire* une dynamique non linéaire dans un espace de fonctions
  d'observation ; les valeurs propres donnent fréquences et taux de croissance.

### 1.8 Noyau & orchestration

- **`TrajectoryStore`** — tenseurs de trajectoire en mémoire/disque, adressés
  par empreinte (vidéo + encodeur + paramètres).
- **`JobGraph`** — DAG de transformations ; exécution paresseuse, résultats
  intermédiaires mis en cache.
- **`Registry`** — découverte des plugins (encodeurs, analyseurs) par point
  d'entrée ; ajouter un encodeur = déposer une classe, zéro modification du
  noyau.

---

## 2. Stack technique

| Couche | Choix | Justification |
|--------|-------|---------------|
| Langage | Python 3.11+ | écosystème ML/science |
| Tenseurs / GPU | PyTorch 2.x + CUDA | `torch.compile`, AMP, écosystème modèles |
| Décodage vidéo | PyAV, decord, ffmpeg | précision + accès aléatoire GPU |
| Modèles pré-entraînés | `timm`, HuggingFace, DINOv2/V-JEPA officiels | éviter de réentraîner |
| Dynamique | `numpy`, `scipy`, `nolds`, `pyEDM`, `pydmd` | Lyapunov, CCM, DMD/Koopman |
| Topologie | `giotto-tda`, `ripser` | homologie persistante |
| Réduction | `umap-learn`, `scikit-learn`, `pydiffmap` | UMAP/t-SNE/diffusion maps |
| API | FastAPI + WebSocket | jobs async + streaming temps réel |
| Tâches | Job graph maison ; Ray/Dask en V3 si multi-GPU | simplicité d'abord |
| Cache | disque (Parquet/zarr) + LRU mémoire ; clé = hash | reproductibilité |
| Frontend | React + TypeScript ; `deck.gl`/`regl` (2D/3D WebGL), `three.js`, `d3` | rendu de millions de points |
| Paquetage | `uv`/`pip`, Docker (image CUDA), `pyproject.toml` | repro environnement |

**Pipeline GPU** : décodage → (option) décodage GPU NVDEC → resize sur GPU →
batch d'encodage avec AMP (`bfloat16`) → latents rapatriés CPU en `float32` pour
l'analyse dynamique (légère, non parallèle). Le goulot est l'encodage : on le
maintient sur GPU, on garde le reste sur CPU.

**Mémoire / VRAM** : taille de batch auto-réglée par recherche dichotomique au
démarrage (on monte jusqu'à l'OOM, on recule) ; checkpointing d'activations pour
les gros ViT ; latents streamés vers disque (`zarr`) au-delà d'un seuil de T.

**Cache** : chaque artefact (frames échantillonnées, latents, projections,
diagrammes) est immutable et adressé par contenu. Rejouer une analyse avec un
paramètre changé ne recalcule que le sous-arbre affecté du `JobGraph`.

**Temps réel vs offline** :
- *Offline* (défaut) — vidéo entière, toutes les analyses, qualité maximale.
- *Temps réel* — fenêtre glissante de T frames ; seuls les analyseurs à coût
  borné par fenêtre tournent (encodage, trajectoire latente, Koopman incrémental,
  graphe causal glissant). L'homologie persistante et l'UMAP global restent
  offline (recalcul trop coûteux par frame ; en streaming on utilise un UMAP
  *paramétrique* pré-ajusté).

---

## 3. Roadmap technique

### Phase 1 — MVP (« le tube qui marche »)

Objectif : prouver la boucle *vidéo → latent → trajectoire visualisable*.

- `ingest` : fichier local, échantillonnage uniforme, transcodage simple,
  estimateur de coût basique.
- `encoders` : **2 encodeurs** seulement — un CNN (ResNet/ConvNeXt) et un ViT
  (DINOv2). Interface `Encoder` figée ici.
- `dynamics` : trajectoire latente + plongement par délais + un plot de
  récurrence.
- `geometry` : UMAP statique.
- UI : upload, choix d'encodeur, vue latente 2D animée dans le temps.
- Noyau : `TrajectoryStore` + cache disque.

*Critère de sortie* : charger une vidéo de 30 s, voir la trajectoire latente de
deux encodeurs côte à côte, en moins d'une minute sur un seul GPU.

### Phase 2 — Version avancée (« la comparaison »)

Objectif : faire du projet un instrument de **comparaison d'encodeurs et de
dynamiques**.

- `encoders` : compléter le zoo — AE, VAE, contrastif, encodeur temporel
  (VideoMAE), RSSM, world model, JEPA. `EncoderCard` complète (compression,
  stabilité, rate–distortion).
- `dynamics` : attracteurs, cycles, bifurcations, exposants de Lyapunov,
  dimension de corrélation.
- `multiscale` : coarse-graining + décomposition fréquentielle + vue
  « disparaît / persiste / émerge ».
- `causal` : Granger, transfer entropy, CCM, segmentation en régimes.
- `geometry` : t-SNE, diffusion maps, homologie persistante, Koopman/DMD.
- UI : vues 3D, zoom multi-échelle, cartes de causalité, vue spectrale,
  comparaison N encodeurs en grille.
- `JobGraph` paresseux + invalidation fine du cache.

### Phase 3 — Plateforme de recherche

Objectif : reproductibilité, extensibilité, passage à l'échelle.

- Streaming temps réel (webcam/RTSP) avec analyseurs à fenêtre.
- API plugins publique + SDK : tout chercheur ajoute un encodeur ou un
  analyseur sans toucher au noyau.
- Mode batch d'**expériences** : balayage de paramètres, runs nommés,
  comparaison versionnée, export de figures.
- Multi-GPU / multi-nœud (Ray) pour les balayages.
- Datasets de référence intégrés + cartes de métriques.
- Notebooks reproductibles & export (trajectoires, diagrammes, rapports).

---

## 4. Proposition d'interface

Application « atelier » à panneaux. **Une vidéo, plusieurs lentilles
synchronisées par le temps.**

```
┌──────────────────────────────────────────────────────────────────────┐
│  ◀ Source vidéo / playhead ──────────────────────────●──────────────▶ │  ← timeline maître
├───────────────┬──────────────────────────┬───────────────────────────┤
│ ENCODEURS     │  ESPACE LATENT 2D/3D     │  MULTI-ÉCHELLE             │
│ ☑ CNN         │                          │  fine ─────────● grossière│
│ ☑ ViT         │   nuage + trajectoire    │  [disparaît|persiste|émerge]│
│ ☐ VAE   …     │   ⟳ rotation, zoom       │                           │
│ [comparer]    │   ● playhead synchronisé │                           │
├───────────────┼──────────────────────────┼───────────────────────────┤
│ DYNAMIQUE     │  CARTE CAUSALE           │  VUE SPECTRALE / KOOPMAN   │
│ Lyapunov 0.07 │   graphe régions →       │   modes propres, |λ|, freq │
│ dim_corr 2.3  │   influences pondérées   │   attracteurs surlignés    │
│ régime: chaos │                          │                           │
└───────────────┴──────────────────────────┴───────────────────────────┘
```

Principes d'interaction :

- **Tout est lié au temps** — déplacer le playhead met en surbrillance le
  point courant dans *chaque* panneau (latent, graphe causal, spectre).
- **Brushing** — sélectionner une portion de trajectoire dans le latent
  surligne l'intervalle temporel et le segment vidéo correspondants.
- **Curseur multi-échelle** — un slider unique pilote le niveau de
  coarse-graining ; les panneaux se recalculent (depuis le cache).
- **Mode comparaison** — cocher N encodeurs affiche une grille de vues
  latentes alignées, avec un panneau de métriques différentielles.
- **Cartes d'incertitude** — toute estimation (Lyapunov, causalité) affiche
  une barre de confiance ; les sorties causales portent le bandeau
  « observationnel, non interventionnel ».
- **Inspecteur** — clic sur un point latent → frame source, voisins,
  contribution spectrale.

---

## 5. Idées d'expériences

1. **Quel encodeur préserve la dynamique ?** — encoder la même vidéo avec les
   9 familles, comparer exposant de Lyapunov et dimension de corrélation à une
   *vérité-terrain dynamique* (vidéos synthétiques de systèmes connus :
   pendule double, Lorenz rendu, réaction-diffusion). L'encodeur idéal
   *transporte* les invariants dynamiques.
2. **Sous-espace pixel vs sous-espace latent** — la dynamique est-elle plus
   simple (basse dimension, plus linéaire au sens de Koopman) en latent qu'en
   pixels ? Mesure du gain de linéarisation.
3. **JEPA vs VAE sur le bruit** — sur des vidéos à fort bruit
   non prédictible (neige TV, eau), JEPA devrait *ignorer* l'imprévisible
   tandis qu'un VAE le *reconstruit*. Quantifier via l'énergie latente
   allouée aux modes hautes fréquences.
4. **Émergence par coarse-graining** — exhiber une vidéo où un motif
   (ex. flux de foule, banc de poissons) est invisible à l'échelle fine et
   net à l'échelle grossière. Mesurer à quelle échelle il apparaît.
5. **Détection de bifurcation** — vidéo d'une transition de phase
   (ébullition, fonte, fissuration) ; l'analyse en fenêtre détecte-t-elle la
   transition au bon instant ?
6. **Causalité émergente** — scène à deux acteurs couplés (ex. danse,
   trafic) ; CCM retrouve-t-il le sens du couplage ? Tester sur des cas où la
   réponse est connue.
7. **Stabilité sous perturbation** — classer les encodeurs par robustesse
   (occlusion, jitter, compression) de la trajectoire latente.
8. **Universalité multi-échelle** — différentes vidéos d'une même classe de
   phénomène partagent-elles des invariants de renormalisation ?

---

## 6. Suggestions algorithmiques

- **Plongement par délais** — `τ` via premier minimum de l'information
  mutuelle ; `m` via faux plus proches voisins. Robuste, peu de réglages.
- **Lyapunov** — Rosenstein/Kantz (adaptés aux séries courtes et bruitées) ;
  toujours rapporter avec intervalle de confiance par bootstrap.
- **Koopman** — EDMD avec dictionnaire d'observables (RBF ou les coordonnées
  latentes elles-mêmes) ; en streaming, **online DMD** pour mise à jour O(d²)
  par frame.
- **Multi-échelle** — DMD multi-résolution (mrDMD) qui sépare nativement les
  modes par bande temporelle ; complète FFT/ondelettes.
- **Causalité** — combiner Granger (linéaire), transfer entropy
  (non linéaire, modèle-libre) et CCM (couplage dynamique) ; ne conclure que
  sur **convergence des trois**.
- **Réduction respectant la dynamique** — privilégier les diffusion maps : la
  distance de diffusion encode le temps de transition, donc la projection
  préserve la structure d'attracteur, contrairement à t-SNE.
- **Homologie persistante** — sur une *fenêtre glissante* de la trajectoire
  (sliding-window embedding) : détecte la périodicité comme un cycle `H1`
  persistant (approche de l'analyse topologique de séries temporelles).
- **Segmentation en régimes** — HSMM sur le latent, ou détection de
  changepoint bayésienne ; les frontières alimentent l'analyse de bifurcation.
- **Échantillonnage adaptatif** — piloté par la norme du flux optique,
  normalisée par scène pour ne pas sur-échantillonner un simple paning.

---

## 7. Annexes

### 7.1 Datasets suggérés

- **Synthétiques à vérité-terrain** (validation) : rendus de Lorenz/Rössler,
  pendule double, réaction-diffusion, automates — la dynamique est connue.
- **Physique réelle** : vidéos de fluides, pendules, ressorts (jeux type
  *Physics101*), transitions de phase.
- **Naturelles structurées** : nuages, feu, eau, foules, bancs d'animaux —
  émergence multi-échelle.
- **Mouvement humain** : datasets d'action — dynamique articulée, cycles
  (marche).
- **Egocentrique / world models** : navigation, conduite — pour RSSM et
  world models.

### 7.2 Métriques

| Catégorie | Métriques |
|-----------|-----------|
| Compression | taux effectif, courbe rate–distortion, énergie spectrale conservée |
| Fidélité dynamique | erreur sur Lyapunov / dim. de corrélation vs vérité-terrain |
| Linéarisation | qualité de reconstruction Koopman, part de variance des modes |
| Stabilité | dérive latente sous perturbation bornée, constante de Lipschitz empirique |
| Prédictivité | erreur de prédiction `k` pas en avant dans le latent |
| Causalité | accord Granger/TE/CCM, AUC vs couplage connu |
| Topologie | persistance des cycles, stabilité du diagramme (distance bottleneck) |

### 7.3 Liens théoriques (ambition)

- **Systèmes dynamiques** — théorème de plongement de Takens : justifie de
  reconstruire l'état depuis des observations partielles ; attracteurs,
  bifurcations, Lyapunov comme vocabulaire d'analyse.
- **Théorie de l'information** — l'encodage est un goulot d'étranglement
  (information bottleneck) ; le bon latent est *suffisant pour le futur,
  minimal pour le présent* (predictive information).
- **Émergence** — un motif est émergent s'il existe à une échelle de
  coarse-graining et pas en deçà ; opérationnalisable via le flot de
  renormalisation.
- **Compression** — relier rate–distortion et fidélité dynamique :
  comprimer *sans casser la dynamique* est l'objectif, pas comprimer les pixels.
- **Représentation causale** — viser des facteurs latents qui correspondent à
  des mécanismes (causal representation learning) ; la plateforme mesure à
  quel point chaque encodeur s'en approche.
- **Géométrie des espaces latents** — la dynamique vit sur une variété ; la
  courbure, la dimension intrinsèque et les opérateurs spectraux (Laplacien,
  Koopman) en sont les sondes.
- **Multi-échelle** — chercher les invariants stables sous changement
  d'échelle : ce sont eux qui méritent le nom de *loi*.

---

*Document de conception — Latent Flow. Sujet à révision à chaque fin de phase.*
