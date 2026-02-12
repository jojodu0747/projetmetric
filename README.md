# Image Quality Metrics Evaluation Pipeline

A complete pipeline for evaluating image quality metrics based on their ability to detect image degradations monotonically. The system extracts deep features from CNN backbones, models their distribution, and measures how well different distance metrics (FID, KID, CMMD) respond to increasing degradation severity.

## Overview

The pipeline:
1. Extracts features from images using pre-trained CNN backbones (ResNet50, VGG19, DinoV2)
2. Optionally transforms features using Gram matrices and PCA
3. Models the feature distribution using GMM or KDE
4. Evaluates distance metrics (FID, KID, CMMD) for monotonicity under degradations

A good metric should produce **monotonically increasing scores** as image degradation increases.

## Installation

```bash
# Clone the repository
git clone https://github.com/jojodu0747/projetmetric.git
cd projetmetric

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run quick test (50 images, single configuration)
python main.py --mode quick

# Run full evaluation (1000 images, all configurations)
python main.py --mode full
```

## Project Structure

```
projetmetric/
├── config.py                 # All configurable parameters
├── feature_extraction.py     # CNN feature extraction with hooks
├── feature_cache.py          # Persistent disk cache for activations
├── distribution_modeling.py  # GMM and KDE distribution models
├── distance_metrics.py       # Mahalanobis and neg log-likelihood metrics
├── degradation_generator.py  # Image degradation functions
├── evaluation.py             # Monotonicity evaluation
├── main.py                   # Pipeline orchestration
├── requirements.txt          # Python dependencies
├── dataset/                  # Image dataset (VisDrone)
├── cache/                    # Cached activation tensors (auto-generated)
└── results/                  # Output CSV, JSON, and plots
```

## Backbone Models

Three pre-trained backbones are supported:

### ResNet50 (default)
```bash
python main.py --backbone resnet50
```
- Standard CNN architecture
- 2048-dimensional features from layer4
- Fast inference, good baseline

### VGG19
```bash
python main.py --backbone vgg19
```
- Deeper feature maps
- Often used for style/perceptual features
- Slower than ResNet50

### DinoV2 ViT-B/14
```bash
python main.py --backbone dinov2_vitb14
```
- Vision Transformer with self-supervised pre-training
- 768-dimensional patch tokens (averaged)
- Requires 518x518 input images
- State-of-the-art for semantic features

## Configuration

All parameters are in `config.py`:

### Layer Configurations
```python
"layer_configs": [
    {"name": "single_layer_8", "layers": [8]},
    {"name": "single_layer_9", "layers": [9]},
    {"name": "single_layer_10", "layers": [10]},
    {"name": "aggregated_11_to_15", "layers": [11, 12, 13, 14, 15]},
]
```

### Feature Transforms
```python
"feature_transforms": [
    {"name": "raw", "use_gram": False, "use_pca": False},
    {"name": "gram_only", "use_gram": True, "use_pca": False},
    {"name": "gram_pca", "use_gram": True, "use_pca": True, "pca_dim": 10},
]
```

- **raw**: Use features directly (with spatial averaging)
- **gram_only**: Compute Gram matrix for style representation
- **gram_pca**: Gram matrix + PCA reduction to 10 dimensions

### Distribution Models
```python
"distribution_models": [
    {"name": "gmm_diag", "type": "gmm", "covariance_type": "diag", "n_components": 5},
    {"name": "gmm_full", "type": "gmm", "covariance_type": "full", "n_components": 5},
    {"name": "kde", "type": "kde", "bandwidth": "auto"},
]
```

### Distance Metrics
- **FID**: Frechet Inception Distance - compares mean and covariance
- **KID**: Kernel Inception Distance - polynomial kernel MMD
- **CMMD**: MMD with RBF kernel and median heuristic for bandwidth

### Degradation Types
```python
"degradations": {
    "blur": {"type": "gaussian_blur", "levels": [1, 3, 5, 7, 9]},
    "blur_contrast": {"type": "blur_plus_contrast", ...},
    "noise": {"type": "gaussian_noise", "std_levels": [5, 15, 30, 50, 70]},
    "aliasing": {"type": "downsample_upsample", "factors": [2, 4, 6, 8]},
}
```

## Cache d'activations

Le cache permet d'extraire les activations brutes du backbone **une seule fois**, puis d'experimenter librement avec differents parametres (distance, GMM, representation, PCA) sans relancer l'inference GPU.

### Principe

```
Sans cache :  images ──[GPU 5-10min]──> activations ──> Gram/GAP ──> PCA ──> GMM ──> metric
Avec cache :  cache/*.npy ──[disque 1s]──> activations ──> Gram/GAP ──> PCA ──> GMM ──> metric
```

Le cache stocke les tenseurs d'activation **bruts** (sorties des hooks, avant tout post-processing). Ce sont les tenseurs 4D `(N, C, H, W)` captures directement a la sortie d'une couche du reseau.

### Etape 1 : Construire le cache

```bash
# Cacher une layer specifique
python main.py --build-cache --backbone sd_vae --layers 9 --n-inference 1000

# Cacher plusieurs layers d'un coup
python main.py --build-cache --backbone sd_vae --layers 7 9 12 --n-inference 1000

# Avec un nombre d'images d'evaluation specifique
python main.py --build-cache --backbone sd_vae --layers 9 --n-inference 1000 --n-evaluation 50
```

Cela cree l'arborescence suivante :

```
cache/
  sd_vae/
    single_layer_9/
      ref_{hash}/              # Activations des images de reference
        chunk_000.npy          # Images 0-99, shape (100, 512, 32, 32), float32
        chunk_001.npy          # Images 100-199
        ...
        metadata.json          # Chemins d'images, backbone, layer, timestamp
      eval_{hash}/             # Activations des images degradees
        blur/
          level_00.npy         # (50, 512, 32, 32) float32
          level_01.npy
          ...
        noise/
          level_00.npy
          ...
```

### Etape 2 : Experimenter (utilisation automatique)

Ensuite, les commandes habituelles detectent et utilisent le cache automatiquement :

```bash
# Le cache est detecte automatiquement ("CACHE HIT" dans les logs)
python main.py --mode medium --backbone sd_vae
python main.py --mode quick --backbone sd_vae
python main.py --mode full --backbone sd_vae
```

Quand le cache est present :
- Pas de chargement du modele en GPU
- Les activations sont chargees depuis le disque (~1 sec)
- Le post-processing (Gram, GAP, PCA) est applique sur CPU (~2-5 sec)
- Le reste du pipeline (GMM, metriques, evaluation) est identique

### Ce qu'on peut varier sans re-extraire

| Parametre | Exemple | Re-extraction ? |
|-----------|---------|-----------------|
| Distance metric | mahalanobis, neg_loglik | Non |
| Distribution model | gmm_diag, gmm_full, n_components | Non |
| Representation | raw, gram_only, gram_pca | Non |
| PCA dimension | pca_dim=5, 10, 50 | Non |
| Couche (layer) | Seulement si elle est deja en cache | Non |
| Backbone | Toujours | **Oui** |
| Images | Si le set change (hash different) | **Oui** |

### Taille du cache

Pour sd_vae layer 9 (512 channels, spatial 32x32) :
- 1000 images de reference : ~2 GB
- 50 eval images x 40 niveaux de degradation : ~4 GB
- **Total par layer : ~6 GB**

Les couches moins profondes (moins de channels ou plus petite resolution) sont proportionnellement plus legeres.

### Invalidation

Le cache est indexe par un hash MD5 des chemins d'images (tries). Si le dataset change, le hash change et un nouveau cache est cree. Il n'y a pas d'expiration automatique ; supprimer `cache/` pour tout reinitialiser.

## Command Line Options

```bash
python main.py --help

Options:
  --mode {full,medium,quick}  Run mode (default: medium)
  --build-cache               Build activation cache instead of running experiment
  --backbone {resnet50,vgg19,dinov2_vitb14,sd_vae,lpips_vgg,all}
                              Backbone model(s) to use
  --layers N [N ...]          Layer indices to cache (required with --build-cache)
  --n-inference N             Number of reference images
  --n-evaluation N            Number of evaluation images
```

### Examples

```bash
# Quick test with DinoV2
python main.py --mode quick --backbone dinov2_vitb14

# Full evaluation with VGG19
python main.py --mode full --backbone vgg19

# Medium evaluation (recommended) with SD VAE
python main.py --mode medium --backbone sd_vae

# Build cache then experiment
python main.py --build-cache --backbone sd_vae --layers 9 --n-inference 1000
python main.py --mode medium --backbone sd_vae

# Run all backbones
python main.py --mode medium --backbone all

# Custom image counts
python main.py --n-inference 500 --n-evaluation 25
```

## Output

Results are saved to `results/`:

- `results_YYYYMMDD_HHMMSS.csv` - Summary table of all configurations
- `results_YYYYMMDD_HHMMSS.json` - Detailed results with per-level scores
- `monotonicity_top1.png` ... `monotonicity_top5.png` - Plots for best configurations

### Example Output

```
================================================================================
TOP PERFORMING CONFIGURATIONS
================================================================================

1. Mean Monotonicity: 0.9850
   Layers: single_layer_8
   Transform: raw
   Distribution: gmm_diag
   Metric: fid
   Per-degradation scores:
     - blur: 1.0000
     - blur_contrast: 0.9500
     - noise: 1.0000
     - aliasing: 0.9900
```

## How Monotonicity is Measured

For each degradation type:
1. Apply degradation at increasing severity levels (e.g., blur sigma 1, 3, 5, 7, 9)
2. Extract features and compute distance from clean reference
3. Calculate Spearman correlation between severity level and distance score

**Monotonicity = 1.0** means the metric perfectly orders degradation levels (higher degradation = higher score).

## Guide de modification des paramètres

Tous les paramètres sont centralisés dans `config.py`. Voici comment personnaliser chaque aspect du pipeline :

### 1. Modifier les dimensions PCA

Dans `config.py`, section `feature_transforms` (ligne 55) :

```python
"feature_transforms": [
    {"name": "raw", "use_gram": False, "use_pca": False},
    {"name": "gram_only", "use_gram": True, "use_pca": False},
    {"name": "gram_pca", "use_gram": True, "use_pca": True, "pca_dim": 10},  # ← Changer ici
],
```

**Exemple** : Pour tester PCA à 5, 10 et 20 dimensions :
```python
"feature_transforms": [
    {"name": "gram_pca_5", "use_gram": True, "use_pca": True, "pca_dim": 5},
    {"name": "gram_pca_10", "use_gram": True, "use_pca": True, "pca_dim": 10},
    {"name": "gram_pca_20", "use_gram": True, "use_pca": True, "pca_dim": 20},
],
```

**Si le cache existe** : tu peux changer `pca_dim` librement, le cache sera réutilisé automatiquement (pas de ré-extraction nécessaire).

### 2. Modifier le GMM (nombre de composantes, type de covariance)

Dans `config.py`, section `distribution_models` (ligne 62) :

```python
"distribution_models": [
    {"name": "gmm_diag", "type": "gmm", "covariance_type": "diag", "n_components": 5},  # ← Changer ici
    {"name": "gmm_full", "type": "gmm", "covariance_type": "full", "n_components": 5},
],
```

**Options disponibles** :
- `covariance_type` : `"diag"`, `"full"`, `"spherical"`, `"tied"`
- `n_components` : nombre de Gaussiennes (ex: 3, 5, 10)

**Exemple** : Tester différents nombres de composantes :
```python
"distribution_models": [
    {"name": "gmm_diag_3", "type": "gmm", "covariance_type": "diag", "n_components": 3},
    {"name": "gmm_full_5", "type": "gmm", "covariance_type": "full", "n_components": 5},
    {"name": "gmm_full_10", "type": "gmm", "covariance_type": "full", "n_components": 10},
],
```

**Si le cache existe** : tu peux changer le GMM librement, pas de ré-extraction.

### 3. Choisir les métriques de distance

Dans `config.py`, ligne 69 :

```python
"distance_metrics": ["mahalanobis", "neg_loglik"],  # ← Modifier ici
```

**Métriques disponibles** :
- `"mahalanobis"` : Distance de Mahalanobis pondérée aux composantes GMM
- `"neg_loglik"` : Log-vraisemblance négative (plus dégradé = score plus élevé)

**Exemple** : Tester une seule métrique :
```python
"distance_metrics": ["mahalanobis"],
```

**Si le cache existe** : tu peux changer les métriques librement.

### 4. Forcer ou désactiver le cache

**Le cache est automatique** — si les activations existent pour (backbone, layer, image_set), elles sont utilisées.

**Pour forcer une nouvelle extraction** (ignorer le cache) :
```bash
# Supprimer le cache complet
rm -rf cache/

# Supprimer un backbone spécifique
rm -rf cache/sd_vae/

# Supprimer une layer spécifique
rm -rf cache/sd_vae/single_layer_9/
```

**Pour vérifier l'état du cache** :
Le code log automatiquement "CACHE HIT" ou "CACHE MISS" dans les logs pendant l'exécution.

### 5. Changer le nombre d'images

**Via CLI** (temporaire, pour un run unique) :
```bash
python main.py --mode medium --backbone sd_vae --n-inference 500 --n-evaluation 25
```

**Via config.py** (permanent) — ligne 15 :
```python
"n_images_inference": 1000,   # Images pour fitter la distribution
"n_images_evaluation": 50,    # Images pour évaluer les dégradations
```

**Attention** : changer le nombre d'images change le hash du set → nouveau cache créé (l'ancien reste intact).

### 6. Modifier les dégradations

Dans `config.py`, section `degradations` (ligne 71) :

```python
"degradations": {
    "blur": {
        "type": "gaussian_blur",
        "levels": [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.3, 1.6, 2, 2.5, 3],  # ← sigma
    },
    "noise": {
        "type": "gaussian_noise",
        "std_levels": [1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30],  # ← écart-type
    },
    # ...
},
```

**Exemple** : Ajouter une nouvelle dégradation (JPEG) :
```python
"degradations": {
    "blur": {...},
    "jpeg": {
        "type": "jpeg_compression",
        "quality_levels": [90, 70, 50, 30, 10],
    },
},
```

**Attention** : modifier les dégradations nécessite de reconstruire le cache (les activations dégradées changent).

### 7. Sélectionner les layers à tester

**Automatique** : Par défaut, le mode `medium` et `full` testent toutes les layers disponibles pour le backbone.

**Manuel via config.py** — ligne 31 :
```python
"layer_configs": [
    {"name": "single_layer_7", "layers": [7]},
    {"name": "single_layer_9", "layers": [9]},
    {"name": "single_layer_12", "layers": [12]},
    # Supprimer ou commenter les layers non désirées
],
```

**Ou via code** dans `main.py`, les modes `quick` et `medium` ont des listes hardcodées de layers texture-pertinentes. Tu peux les modifier directement.

### 8. Changer le backbone

**Via CLI** :
```bash
python main.py --mode medium --backbone vgg19
python main.py --mode medium --backbone sd_vae
python main.py --mode medium --backbone lpips_vgg
python main.py --mode medium --backbone all  # Tous les backbones
```

**Via config.py** (ligne 28) :
```python
"backbone": "sd_vae",  # Options: resnet50, vgg19, dinov2_vitb14, sd_vae, lpips_vgg
```

**Attention** : changer le backbone nécessite un nouveau cache (les activations sont spécifiques au modèle).

### 9. Workflow typique : expérimentation avec cache

**Étape 1** : Construire le cache une fois (lent, GPU)
```bash
python main.py --build-cache --backbone sd_vae --layers 7 9 12 --n-inference 1000
```

**Étape 2** : Expérimenter librement (rapide, CPU)
```bash
# Test 1 : PCA 10 dimensions, GMM diag 5 composantes
# → Modifier config.py : pca_dim=10, n_components=5
python main.py --mode medium --backbone sd_vae

# Test 2 : PCA 20 dimensions, GMM full 10 composantes
# → Modifier config.py : pca_dim=20, n_components=10
python main.py --mode medium --backbone sd_vae

# Test 3 : Seulement Mahalanobis
# → Modifier config.py : "distance_metrics": ["mahalanobis"]
python main.py --mode medium --backbone sd_vae
```

**Aucune ré-extraction GPU** entre les tests — tout se fait sur CPU à partir du cache.

### 10. Résumé : que peut-on changer sans re-extraire ?

| Paramètre | Re-extraction GPU ? | Fichier | Ligne |
|-----------|---------------------|---------|-------|
| `pca_dim` | ❌ Non | config.py | 59 |
| `n_components` (GMM) | ❌ Non | config.py | 64-65 |
| `covariance_type` (GMM) | ❌ Non | config.py | 64-65 |
| `distance_metrics` | ❌ Non | config.py | 69 |
| `use_gram` (raw/gram_only/gram_pca) | ❌ Non | config.py | 56-60 |
| Layer (si déjà en cache) | ❌ Non | — | — |
| Layer (si pas en cache) | ✅ Oui | — | — |
| Backbone | ✅ Oui | config.py | 28 |
| `n_images_inference` / `n_images_evaluation` | ✅ Oui | config.py | 15-16 |
| Dégradations (levels) | ✅ Oui | config.py | 71-90 |

## Dataset

The pipeline uses the VisDrone dataset by default. To use your own images:

1. Place images in a folder
2. Update `config.py`:
```python
"dataset_path": "/path/to/your/images/"
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy, scipy, scikit-learn
- Pillow, matplotlib

## License

MIT
