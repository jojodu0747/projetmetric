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
├── distribution_modeling.py  # GMM and KDE distribution models
├── distance_metrics.py       # FID, KID, CMMD implementations
├── degradation_generator.py  # Image degradation functions
├── evaluation.py             # Monotonicity evaluation
├── main.py                   # Pipeline orchestration
├── requirements.txt          # Python dependencies
├── dataset/                  # Image dataset (VisDrone)
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

## Command Line Options

```bash
python main.py --help

Options:
  --mode {full,quick}     Run mode (default: full)
  --backbone {resnet50,vgg19,dinov2_vitb14}
                          Override backbone model
  --n-inference N         Number of reference images
  --n-evaluation N        Number of evaluation images
```

### Examples

```bash
# Quick test with DinoV2
python main.py --mode quick --backbone dinov2_vitb14

# Full evaluation with VGG19
python main.py --mode full --backbone vgg19

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
