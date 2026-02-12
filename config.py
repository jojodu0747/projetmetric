"""
Configuration file for Image Quality Metrics Evaluation Pipeline.

PIPELINE OVERVIEW:
==================
ÉTAPE 1: Load N_R reference images from VisDrone2019 dataset
ÉTAPE 2: Extract features from a specific backbone model and layer
ÉTAPE 3: Compute Gram matrix for each image → Distribution D_R (N_R vectors)
ÉTAPE 4: Load N_E evaluation images (different from reference)
ÉTAPE 5: Apply progressive degradations → Distributions D_E_k, compute MMD distance with D_R
ÉTAPE 6: Evaluate monotonicity using Spearman correlation
ÉTAPE 7: Save results to CSV with all parameters

CONFIGURATION SECTIONS:
=======================
- CONFIG: Main pipeline parameters (dataset, batch size, degradations, metrics)
- BACKBONE_CONFIGS: Architectural details for each backbone model
- ENABLED_LAYERS: Layer selection for experiments (True/False flags)
  * Use get_enabled_experiment_configs() to get active configurations
  * Modify ENABLED_LAYERS to enable/disable specific layers

All configurable parameters are centralized here.
For detailed layer documentation, see LAYERS_GUIDE.md
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # ========================================================================
    # ÉTAPE 1: DATASET CONFIGURATION
    # ========================================================================
    # Path to VisDrone2019 images
    "dataset_path": os.path.join(BASE_DIR, "dataset/VisDrone2019-DET-train/VisDrone2019-DET-train/images/"),
    "results_path": os.path.join(BASE_DIR, "results/"),

    # N_R: Number of reference images (real distribution)
    "n_images_inference": 100,

    # N_E: Number of evaluation images (to be degraded)
    "n_images_evaluation": 20,

    "random_seed": 42,  # For reproducibility
    "batch_size": 16,   # Batch size for feature extraction

    # Image preprocessing
    "image_size": 256,  # Resize (256 for SD VAE, 224 for VGG/LPIPS, 518 for DinoV2)

    # ========================================================================
    # ÉTAPE 2 & 3: FEATURE EXTRACTION + GRAM MATRIX
    # ========================================================================
    # Backbone: Which model to use (sd_vae, dinov2_vitb14, vgg19, lpips_vgg)
    "backbone": "sd_vae",

    # Experiment configurations: 4 backbones × 6 layers = 24 configs
    # Auto-selected layers (early, mid, late) for each backbone
    "experiment_configs": [
        # SD VAE (17 layers, 0-16)
        {"backbone": "sd_vae", "layer": 0},   # encoder.conv_in
        {"backbone": "sd_vae", "layer": 3},   # encoder.down_blocks.0.downsamplers.0
        {"backbone": "sd_vae", "layer": 6},   # encoder.down_blocks.1.downsamplers.0
        {"backbone": "sd_vae", "layer": 9},   # encoder.down_blocks.2.downsamplers.0
        {"backbone": "sd_vae", "layer": 12},  # encoder.mid_block.resnets.0
        {"backbone": "sd_vae", "layer": 16},  # encoder.conv_out

        # DinoV2 ViT-B/14 (14 layers, 0-13)
        {"backbone": "dinov2_vitb14", "layer": 0},   # patch_embed
        {"backbone": "dinov2_vitb14", "layer": 2},   # blocks.1
        {"backbone": "dinov2_vitb14", "layer": 5},   # blocks.4
        {"backbone": "dinov2_vitb14", "layer": 8},   # blocks.7
        {"backbone": "dinov2_vitb14", "layer": 11},  # blocks.10
        {"backbone": "dinov2_vitb14", "layer": 13},  # norm

        # VGG19 (18 layers, 0-17)
        {"backbone": "vgg19", "layer": 0},   # features.0 (conv1_1)
        {"backbone": "vgg19", "layer": 3},   # features.7 (conv2_2)
        {"backbone": "vgg19", "layer": 7},   # features.16 (conv3_4)
        {"backbone": "vgg19", "layer": 11},  # features.25 (conv4_4)
        {"backbone": "vgg19", "layer": 15},  # features.34 (conv5_4)
        {"backbone": "vgg19", "layer": 17},  # classifier.3 (FC2)

        # LPIPS VGG (16 layers, 0-15)
        {"backbone": "lpips_vgg", "layer": 0},   # net.slice1.0 (conv1_1)
        {"backbone": "lpips_vgg", "layer": 3},   # net.slice2.7 (conv2_2)
        {"backbone": "lpips_vgg", "layer": 7},   # net.slice3.16 (conv3_4)
        {"backbone": "lpips_vgg", "layer": 11},  # net.slice4.25 (conv4_4)
        {"backbone": "lpips_vgg", "layer": 13},  # net.slice5.30 (conv5_2)
        {"backbone": "lpips_vgg", "layer": 15},  # net.slice5.34 (conv5_4)
    ],

    # Legacy layer configs (kept for backward compatibility)
    "layer_configs": [
        {"name": "single_layer_9", "layers": [9]},
    ],

    # Gram matrix computation
    # - use_gram: Apply Gram matrix transformation
    # - gram_averaging: How to average the Gram matrix
    #   * "spatial": Average over all spatial positions (pixels/patches) → (C, C) → upper triangle
    #   * "global": Compute single Gram over entire feature map
    # - use_pca: Apply PCA dimensionality reduction after Gram (False for now)
    "feature_transforms": [
        {
            "name": "gram_spatial",
            "use_gram": True,
            "use_pca": False,
            "gram_patches": False,
            "gram_averaging": "spatial"  # Average Gram over all positions
        },
    ],

    # ========================================================================
    # ÉTAPE 5: DISTANCE METRICS
    # ========================================================================
    # Available metrics for measuring distribution distances:
    # - mmd: Maximum Mean Discrepancy (kernel-based, fast, works well)
    # - fid: Fréchet Inception Distance (assumes Gaussian, good for images)
    # - sinkhorn: Sinkhorn distance (optimal transport, theoretically sound)
    # - energy: Energy distance (metric, no parameters, robust)
    # - adversarial: GAN-style critic (learned, slow but powerful)
    "distance_metrics": [
        # MMD with RBF kernel (default, recommended)
        {"name": "mmd", "kernel": "rbf", "gamma": None},  # Auto gamma via median heuristic

        # Additional metrics (uncomment to enable):
        #{"name": "sinkhorn", "reg": 0.1, "max_iter": 100},  # Optimal transport
        #{"name": "energy", "sample_size": 1000},  # Energy distance
        # {"name": "adversarial", "n_critics": 5, "max_iter": 100},  # GAN-style (slow)
    ],

    # ========================================================================
    # ÉTAPE 5: DEGRADATION CONFIGURATIONS
    # ========================================================================
    # Progressive degradations applied to evaluation images
    # Each degradation has K levels of increasing severity
    # Goal: Verify monotonicity D_E_1 < D_E_2 < ... < D_E_K
    "degradations": {
        "blur": {
            "type": "gaussian_blur",
            "levels": [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.3, 1.6, 2, 2.5, 3],  # sigma values
        },
        "blur_contrast": {
            "type": "blur_plus_contrast",
            "blur_levels": [0.1, 0.3, 0.5, 0.8, 1, 1.5],
            "contrast_factors": [0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
        },
        "noise": {
            "type": "gaussian_noise",
            "std_levels": [1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30],  # on 0-255 scale
        },
        "aliasing": {
            "type": "downsample_upsample",
            "factors": [1.1, 1.2, 1.3, 1.5, 1.7, 2, 2.3, 2.7, 3],
        },
    },

    # ========================================================================
    # ÉTAPE 6: MONOTONICITY EVALUATION (implicit via Spearman in evaluation.py)
    # ÉTAPE 7: RESULTS SAVING (CSV format with all parameters)
    # ========================================================================

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
        "total_layers": 10,
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
        "layer_names": {
            7: "layer3",
            8: "layer4",
            9: "avgpool",
            10: "fc",
            11: "layer4.0",
            12: "layer4.1",
            13: "layer4.2",
            14: "layer3.5",
            15: "layer3.4",
        },
    },
    "vgg19": {
        "model_name": "vgg19",
        "weights": "IMAGENET1K_V1",
        "input_size": 224,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "total_layers": 19,  # 16 conv + 3 FC
        "layer_names": {
            0: "features.0",    # conv1_1 (64 ch)
            1: "features.2",    # conv1_2 (64 ch)
            2: "features.5",    # conv2_1 (128 ch)
            3: "features.7",    # conv2_2 (128 ch)
            4: "features.10",   # conv3_1 (256 ch)
            5: "features.12",   # conv3_2 (256 ch)
            6: "features.14",   # conv3_3 (256 ch)
            7: "features.16",   # conv3_4 (256 ch)
            8: "features.19",   # conv4_1 (512 ch)
            9: "features.21",   # conv4_2 (512 ch)
            10: "features.23",  # conv4_3 (512 ch)
            11: "features.25",  # conv4_4 (512 ch)
            12: "features.28",  # conv5_1 (512 ch)
            13: "features.30",  # conv5_2 (512 ch)
            14: "features.32",  # conv5_3 (512 ch)
            15: "features.34",  # conv5_4 (512 ch)
            16: "classifier.0", # FC1 (4096)
            17: "classifier.3", # FC2 (4096)
        },
    },
    "dinov2_vitb14": {
        "model_name": "dinov2_vitb14",
        "weights": None,  # From torch.hub
        "input_size": 518,  # DinoV2 uses 518x518
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "total_layers": 14,  # patch_embed + 12 blocs transformer + norm
        "layer_names": {
            0: "patch_embed",
            1: "blocks.0",
            2: "blocks.1",
            3: "blocks.2",
            4: "blocks.3",
            5: "blocks.4",
            6: "blocks.5",
            7: "blocks.6",
            8: "blocks.7",
            9: "blocks.8",
            10: "blocks.9",
            11: "blocks.10",
            12: "blocks.11",
            13: "norm",
        },
    },
    "sd_vae": {
        "model_name": "sd_vae",
        "weights": "stabilityai/sd-vae-ft-mse",
        "input_size": 256,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
        "total_layers": 17,
        "layer_names": {
            0: "encoder.conv_in",                       # Conv initiale (128 ch)
            1: "encoder.down_blocks.0.resnets.0",       # DownBlock0 ResNet0
            2: "encoder.down_blocks.0.resnets.1",       # DownBlock0 ResNet1
            3: "encoder.down_blocks.0.downsamplers.0",  # DownBlock0 Downsample
            4: "encoder.down_blocks.1.resnets.0",       # DownBlock1 ResNet0
            5: "encoder.down_blocks.1.resnets.1",       # DownBlock1 ResNet1
            6: "encoder.down_blocks.1.downsamplers.0",  # DownBlock1 Downsample
            7: "encoder.down_blocks.2.resnets.0",       # DownBlock2 ResNet0
            8: "encoder.down_blocks.2.resnets.1",       # DownBlock2 ResNet1
            9: "encoder.down_blocks.2.downsamplers.0",  # DownBlock2 Downsample
            10: "encoder.down_blocks.3.resnets.0",      # DownBlock3 ResNet0
            11: "encoder.down_blocks.3.resnets.1",      # DownBlock3 ResNet1
            12: "encoder.mid_block.resnets.0",          # MidBlock ResNet0
            13: "encoder.mid_block.attentions.0",       # MidBlock Attention
            14: "encoder.mid_block.resnets.1",          # MidBlock ResNet1
            15: "encoder.conv_norm_out",                # GroupNorm finale
            16: "encoder.conv_out",                     # Conv latente (8 ch)
        },
    },
    "lpips_vgg": {
        "model_name": "lpips_vgg",
        "weights": "vgg",
        "input_size": 224,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
        "total_layers": 16,
        "layer_names": {
            0: "net.slice1.0",    # conv1_1 (64 ch)
            1: "net.slice1.2",    # conv1_2 (64 ch)
            2: "net.slice2.5",    # conv2_1 (128 ch)
            3: "net.slice2.7",    # conv2_2 (128 ch)
            4: "net.slice3.10",   # conv3_1 (256 ch)
            5: "net.slice3.12",   # conv3_2 (256 ch)
            6: "net.slice3.14",   # conv3_3 (256 ch)
            7: "net.slice3.16",   # conv3_4 (256 ch)
            8: "net.slice4.19",   # conv4_1 (512 ch)
            9: "net.slice4.21",   # conv4_2 (512 ch)
            10: "net.slice4.23",  # conv4_3 (512 ch)
            11: "net.slice4.25",  # conv4_4 (512 ch)
            12: "net.slice5.28",  # conv5_1 (512 ch)
            13: "net.slice5.30",  # conv5_2 (512 ch)
            14: "net.slice5.32",  # conv5_3 (512 ch)
            15: "net.slice5.34",  # conv5_4 (512 ch)
        },
    },
    "clip_vit_base": {
        "model_name": "clip_vit_base",
        "weights": "openai/clip-vit-base-patch32",
        "input_size": 224,
        "normalize_mean": [0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
        "normalize_std": [0.26862954, 0.26130258, 0.27577711],
        "total_layers": 13,  # 12 transformer blocks + final projection
        "layer_names": {
            0: "vision_model.embeddings",              # Patch embeddings
            1: "vision_model.encoder.layers.0",        # Transformer block 0
            2: "vision_model.encoder.layers.1",        # Transformer block 1
            3: "vision_model.encoder.layers.2",        # Transformer block 2
            4: "vision_model.encoder.layers.3",        # Transformer block 3
            5: "vision_model.encoder.layers.4",        # Transformer block 4
            6: "vision_model.encoder.layers.5",        # Transformer block 5
            7: "vision_model.encoder.layers.6",        # Transformer block 6
            8: "vision_model.encoder.layers.7",        # Transformer block 7
            9: "vision_model.encoder.layers.8",        # Transformer block 8
            10: "vision_model.encoder.layers.9",       # Transformer block 9
            11: "vision_model.encoder.layers.10",      # Transformer block 10
            12: "vision_model.encoder.layers.11",      # Transformer block 11 (final)
        },
    },
}


# ============================================================================
# LAYER SELECTION FOR EXPERIMENTS
# ============================================================================
# Enable/disable specific layers for each backbone
# Set True to include in experiments, False to skip
# Format: {backbone: {layer_index: enabled}}
#
# To configure experiments:
# 1. Find the backbone you want to use below
# 2. Change True → False to disable a layer
# 3. Change False → True to enable a layer
# 4. Run experiments with: python main.py --mode medium
#
# Comments show: layer_name - description (channels, resolution, semantic level)

ENABLED_LAYERS = {
    "sd_vae": {
        0: True,   # encoder.conv_in - Initial conv (128ch, 256×256, early)
        1: False,  # encoder.down_blocks.0.resnets.0 - ResNet block (128ch, 256×256)
        2: False,  # encoder.down_blocks.0.resnets.1 - ResNet block (128ch, 256×256)
        3: True,   # encoder.down_blocks.0.downsamplers.0 - Downsample (128ch→128×128, early-mid)
        4: False,  # encoder.down_blocks.1.resnets.0 - ResNet block (256ch, 128×128)
        5: False,  # encoder.down_blocks.1.resnets.1 - ResNet block (256ch, 128×128)
        6: True,   # encoder.down_blocks.1.downsamplers.0 - Downsample (256ch→64×64, mid)
        7: False,  # encoder.down_blocks.2.resnets.0 - ResNet block (512ch, 64×64)
        8: False,  # encoder.down_blocks.2.resnets.1 - ResNet block (512ch, 64×64)
        9: True,   # encoder.down_blocks.2.downsamplers.0 - Downsample (512ch→32×32, mid-late)
        10: False, # encoder.down_blocks.3.resnets.0 - ResNet block (512ch, 32×32)
        11: False, # encoder.down_blocks.3.resnets.1 - ResNet block (512ch, 32×32)
        12: False,  # encoder.mid_block.resnets.0 - Bottleneck ResNet (512ch, 32×32, late)
        13: False, # encoder.mid_block.attentions.0 - Self-attention (512ch, 32×32)
        14: False, # encoder.mid_block.resnets.1 - Bottleneck ResNet (512ch, 32×32)
        15: False, # encoder.conv_norm_out - GroupNorm (512ch, 32×32)
        16: False,  # encoder.conv_out - Latent space (8ch, 32×32, final)
    },
    "dinov2_vitb14": {
        0: False,   # patch_embed - Patch embedding (768d, 1369 tokens, initial)
        1: False,  # blocks.0 - Transformer block 0 (768d, early)
        2: False,   # blocks.1 - Transformer block 1 (768d, early-mid)
        3: False,  # blocks.2 - Transformer block 2 (768d)
        4: False,  # blocks.3 - Transformer block 3 (768d)
        5: False,   # blocks.4 - Transformer block 4 (768d, mid)
        6: False,  # blocks.5 - Transformer block 5 (768d)
        7: False,  # blocks.6 - Transformer block 6 (768d)
        8: False,   # blocks.7 - Transformer block 7 (768d, mid-late)
        9: False,  # blocks.8 - Transformer block 8 (768d)
        10: False, # blocks.9 - Transformer block 9 (768d)
        11: False,  # blocks.10 - Transformer block 10 (768d, late)
        12: False, # blocks.11 - Transformer block 11 (768d)
        13: False,  # norm - Final layer norm (768d, final)
    },
    "vgg19": {
        0: False,   # features.0 - conv1_1 (64ch, 224×224, very early)
        1: False,  # features.2 - conv1_2 (64ch, 224×224)
        2: False,  # features.5 - conv2_1 (128ch, 112×112)
        3: False,   # features.7 - conv2_2 (128ch, 112×112, early-mid)
        4: False,  # features.10 - conv3_1 (256ch, 56×56)
        5: False,  # features.12 - conv3_2 (256ch, 56×56)
        6: False,  # features.14 - conv3_3 (256ch, 56×56)
        7: False,   # features.16 - conv3_4 (256ch, 56×56, mid)
        8: False,  # features.19 - conv4_1 (512ch, 28×28)
        9: False,  # features.21 - conv4_2 (512ch, 28×28)
        10: False, # features.23 - conv4_3 (512ch, 28×28)
        11: False,  # features.25 - conv4_4 (512ch, 28×28, mid-late)
        12: False, # features.28 - conv5_1 (512ch, 14×14)
        13: False, # features.30 - conv5_2 (512ch, 14×14)
        14: False, # features.32 - conv5_3 (512ch, 14×14)
        15: False,  # features.34 - conv5_4 (512ch, 14×14, late)
        16: False, # classifier.0 - FC1 (4096d)
        17: False,  # classifier.3 - FC2 (4096d, final semantic)
    },
    "lpips_vgg": {
        0: False,   # net.slice1.0 - conv1_1 (64ch, 224×224, very early)
        1: False,  # net.slice1.2 - conv1_2 (64ch, 224×224)
        2: False,  # net.slice2.5 - conv2_1 (128ch, 112×112)
        3: False,   # net.slice2.7 - conv2_2 (128ch, 112×112, early-mid)
        4: False,  # net.slice3.10 - conv3_1 (256ch, 56×56)
        5: False,  # net.slice3.12 - conv3_2 (256ch, 56×56)
        6: False,  # net.slice3.14 - conv3_3 (256ch, 56×56)
        7: False,   # net.slice3.16 - conv3_4 (256ch, 56×56, mid)
        8: False,  # net.slice4.19 - conv4_1 (512ch, 28×28)
        9: False,  # net.slice4.21 - conv4_2 (512ch, 28×28)
        10: False, # net.slice4.23 - conv4_3 (512ch, 28×28)
        11: False,  # net.slice4.25 - conv4_4 (512ch, 28×28, mid-late)
        12: False, # net.slice5.28 - conv5_1 (512ch, 14×14)
        13: False,  # net.slice5.30 - conv5_2 (512ch, 14×14, late)
        14: False, # net.slice5.32 - conv5_3 (512ch, 14×14)
        15: False,  # net.slice5.34 - conv5_4 (512ch, 14×14, final)
    },
    "resnet50": {
        7: False,   # layer3 - Residual stage 3 (1024ch, 14×14, mid)
        8: False,   #layer4.0 - Individual bottleneck 0 (2048ch, 7×7)
        12: False, # layer4.1 - Individual bottleneck 1 (2048ch, 7×7)
        13: False, # layer4.2 - Individual bottleneck 2 (2048ch, 7×7)
        14: False, # layer3.5 - Individual bottleneck 5 (1024ch, 14×14)
        15: False, # layer3.4 - Individual bottleneck 4 (1024ch, 14×14)
    },
    "clip_vit_base": {
        0: False,   # vision_model.embeddings - Patch embed (768d, 50 tokens, initial)
        1: False,  # vision_model.encoder.layers.0 - Transformer 0 (768d, early)
        2: False,  # vision_model.encoder.layers.1 - Transformer 1 (768d)
        3: False,   # vision_model.encoder.layers.2 - Transformer 2 (768d, early-mid)
        4: False,  # vision_model.encoder.layers.3 - Transformer 3 (768d)
        5: False,  # vision_model.encoder.layers.4 - Transformer 4 (768d)
        6: False,   # vision_model.encoder.layers.5 - Transformer 5 (768d, mid)
        7: False,  # vision_model.encoder.layers.6 - Transformer 6 (768d)
        8: False,  # vision_model.encoder.layers.7 - Transformer 7 (768d)
        9: False,   # vision_model.encoder.layers.8 - Transformer 8 (768d, mid-late)
        10: False, # vision_model.encoder.layers.9 - Transformer 9 (768d)
        11: False, # vision_model.encoder.layers.10 - Transformer 10 (768d)
        12: False,  # vision_model.encoder.layers.11 - Transformer 11 (768d, final)
    },
}


def get_enabled_experiment_configs():
    """F,  
    Get list of enabled backbone+layer configs for experiments.

    Returns:
        List of dicts with 'backbone' and 'layer' keys.
        Only includes layers where enabled=True in ENABLED_LAYERS.

    Example:
        >>> configs = get_enabled_experiment_configs()
        >>> print(len(configs))
        32  # If all recommended layers are enabled
        >>> print(configs[0])
        {'backbone': 'sd_vae', 'layer': 0}
    """
    configs = []
    for backbone, layers in ENABLED_LAYERS.items():
        for layer_idx, enabled in layers.items():
            if enabled:
                configs.append({"backbone": backbone, "layer": layer_idx})
    return configs


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
