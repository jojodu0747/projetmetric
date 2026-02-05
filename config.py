"""
Configuration file for Image Quality Metrics Evaluation Pipeline.
All configurable parameters are centralized here.
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Paths
    "dataset_path": os.path.join(BASE_DIR, "dataset/VisDrone2019-DET-train/VisDrone2019-DET-train/images/"),
    "results_path": os.path.join(BASE_DIR, "results/"),
    "n_images_inference": 1000,  # Number of images for distribution inference
    "n_images_evaluation": 50,   # Number of images for degradation evaluation

    # Random seed for reproducibility
    "random_seed": 42,

    # Batch processing
    "batch_size": 32,

    # Image preprocessing
    "image_size": 224,  # Resize images to this size (224 for ResNet/VGG, 518 for DinoV2)

    # Feature extraction - Backbone selection
    "backbone": "resnet50",  # Options: "resnet50", "vgg19", "dinov2_vitb14"

    # Layer configurations for feature extraction
    "layer_configs": [
        {"name": "single_layer_8", "layers": [8]},
        {"name": "single_layer_9", "layers": [9]},
        {"name": "single_layer_10", "layers": [10]},
        {"name": "aggregated_11_to_15", "layers": [11, 12, 13, 14, 15]},
    ],

    # Feature transformation modes
    "feature_transforms": [
        {"name": "raw", "use_gram": False, "use_pca": False},
        {"name": "gram_only", "use_gram": True, "use_pca": False},
        {"name": "gram_pca", "use_gram": True, "use_pca": True, "pca_dim": 10},
    ],

    # Distribution modeling configurations
    "distribution_models": [
        {"name": "gmm_diag", "type": "gmm", "covariance_type": "diag", "n_components": 5},
        {"name": "gmm_full", "type": "gmm", "covariance_type": "full", "n_components": 5},
        {"name": "kde", "type": "kde", "bandwidth": "auto"},  # auto = cross-validation
    ],

    # Distance metrics to evaluate
    "distance_metrics": ["fid", "kid", "cmmd"],

    # Degradation configurations (ordered by increasing severity)
    "degradations": {
        "blur": {
            "type": "gaussian_blur",
            "levels": [1, 3, 5, 7, 9],  # sigma values
        },
        "blur_contrast": {
            "type": "blur_plus_contrast",
            "blur_levels": [1, 3, 5],
            "contrast_factors": [0.8, 0.6, 0.4],
        },
        "noise": {
            "type": "gaussian_noise",
            "std_levels": [5, 15, 30, 50, 70],  # on 0-255 scale
        },
        "aliasing": {
            "type": "downsample_upsample",
            "factors": [2, 4, 6, 8],
        },
    },

    # KID specific settings
    "kid_subsets": 100,      # Number of subsets for KID variance estimation
    "kid_subset_size": 50,   # Size of each subset

    # Logging
    "log_level": "INFO",
    "log_progress_every": 10,  # Log progress every N batches
}

# Backbone-specific configurations
BACKBONE_CONFIGS = {
    "resnet50": {
        "model_name": "resnet50",
        "weights": "IMAGENET1K_V2",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        # Layer indices correspond to ResNet blocks
        "extractable_layers": {
            1: "conv1",
            2: "bn1",
            3: "relu",
            4: "maxpool",
            5: "layer1",
            6: "layer2",
            7: "layer3",
            8: "layer4",
            9: "avgpool",
            10: "fc",
        },
        # Named layers for hook registration
        "layer_names": {
            8: "layer4",      # Last conv block (2048 channels)
            9: "avgpool",     # Global average pooling
            10: "fc",         # Fully connected
            11: "layer4.0",   # Sub-blocks of layer4
            12: "layer4.1",
            13: "layer4.2",
            14: "layer3.5",   # Last block of layer3
            15: "layer3.4",
        },
    },
    "vgg19": {
        "model_name": "vgg19",
        "weights": "IMAGENET1K_V1",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "layer_names": {
            8: "features.25",   # Conv layer
            9: "features.27",   # Conv layer
            10: "features.29",  # Conv layer
            11: "features.31",  # Conv layer
            12: "features.33",  # Conv layer
            13: "features.35",  # Conv layer
            14: "classifier.0", # First FC
            15: "classifier.3", # Second FC
        },
    },
    "dinov2_vitb14": {
        "model_name": "dinov2_vitb14",
        "weights": None,  # From torch.hub
        "input_size": 518,  # DinoV2 uses 518x518
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        # For DinoV2, layers refer to transformer blocks
        "layer_names": {
            8: "blocks.8",
            9: "blocks.9",
            10: "blocks.10",
            11: "blocks.11",
            12: "norm",  # Final layer norm
            13: "blocks.7",
            14: "blocks.6",
            15: "blocks.5",
        },
    },
}


def get_config():
    """Return a copy of the configuration."""
    return CONFIG.copy()


def get_backbone_config(backbone_name: str):
    """Return backbone-specific configuration."""
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone: {backbone_name}. "
                        f"Available: {list(BACKBONE_CONFIGS.keys())}")
    return BACKBONE_CONFIGS[backbone_name]


def update_config(**kwargs):
    """Update configuration with new values."""
    for key, value in kwargs.items():
        if key in CONFIG:
            CONFIG[key] = value
        else:
            raise KeyError(f"Unknown config key: {key}")
