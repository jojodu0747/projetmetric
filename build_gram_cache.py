"""
Build Gram matrix cache for all backbone/layer configurations.

This script pre-computes and caches Gram matrices for:
- 4 backbones: sd_vae, dinov2_vitb14, vgg19, lpips_vgg
- 6 layers per backbone (24 total configs)
- 100 reference images → 100 individual Gram matrices per config
- 20 evaluation images × all degradation types and levels → 20 Gram matrices per level

The cache stores INDIVIDUAL Gram matrices (not averaged) to enable proper
MMD computation between distributions.

Cache structure: {backbone: {layer_X: {reference: (100, D), degradations: {type: [(20, D), ...]}}}}
"""

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import CONFIG, get_backbone_config
from degradation_generator import DegradationGenerator
from feature_extraction import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_images(image_dir: str, n_images: int, seed: int = 42) -> List[str]:
    """Load n_images paths from directory."""
    np.random.seed(seed)
    all_images = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    if len(all_images) < n_images:
        raise ValueError(f"Not enough images: found {len(all_images)}, need {n_images}")

    selected = np.random.choice(all_images, n_images, replace=False)
    return sorted(selected)


def extract_gram_features(
    extractor: FeatureExtractor,
    image_paths: List[str],
    batch_size: int,
) -> np.ndarray:
    """
    Extract Gram features for all images.

    Args:
        extractor: FeatureExtractor instance
        image_paths: List of image paths
        batch_size: Batch size for extraction

    Returns:
        Gram features (N, D) where D = C*(C+1)/2
    """
    all_grams = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="  Extracting", leave=False):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

        # Extract features (already as Gram matrices)
        features = extractor.extract_batch(batch_images)  # (B, D) where D = C*(C+1)/2
        all_grams.append(features)

    # Concatenate all batches
    all_grams = np.vstack(all_grams)  # (N, D)

    return all_grams


def build_cache_for_config(
    backbone: str,
    layer: int,
    ref_images: List[str],
    eval_images: List[str],
    degradation_generator: DegradationGenerator,
    batch_size: int,
) -> Dict:
    """
    Build cache for a single backbone/layer configuration.

    Returns:
        Dict with 'reference' and 'degradations' keys
    """
    logger.info(f"Building cache for {backbone} layer {layer}")

    # Create extractor for this config
    extractor = FeatureExtractor(
        backbone=backbone,
        layers=[layer],
        transform_config={"use_gram": True, "use_pca": False, "gram_patches": False},
    )

    cache_entry = {
        "reference": None,
        "degradations": {},
    }

    # Extract reference Gram features (individual matrices, not averaged)
    logger.info(f"  Extracting reference Gram features ({len(ref_images)} images)...")
    ref_grams = extract_gram_features(extractor, ref_images, batch_size)
    cache_entry["reference"] = ref_grams
    logger.info(f"  Reference Gram features shape: {ref_grams.shape}")

    # Extract degraded Grams
    for deg_type in CONFIG["degradations"].keys():
        logger.info(f"  Processing degradation: {deg_type}")
        num_levels = degradation_generator.get_num_levels(deg_type)
        cache_entry["degradations"][deg_type] = []

        for level in tqdm(range(num_levels), desc=f"    {deg_type}", leave=False):
            # Apply degradation to all eval images
            degraded_images = []
            for img_path in eval_images:
                img = Image.open(img_path).convert("RGB")
                degraded = degradation_generator.apply_degradation(img, deg_type, level)
                degraded_images.append(degraded)

            # Extract Gram features for this degradation level (individual matrices)
            deg_grams = extract_gram_features_from_pil(
                extractor, degraded_images, batch_size
            )
            cache_entry["degradations"][deg_type].append(deg_grams)

        logger.info(f"    Cached {num_levels} levels for {deg_type}")

    return cache_entry


def extract_gram_features_from_pil(
    extractor: FeatureExtractor,
    images: List[Image.Image],
    batch_size: int,
) -> np.ndarray:
    """
    Extract Gram features from PIL images.

    Args:
        extractor: FeatureExtractor instance
        images: List of PIL Images
        batch_size: Batch size for extraction

    Returns:
        Gram features (N, D) where D = C*(C+1)/2
    """
    all_grams = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        features = extractor.extract_batch(batch_images)  # (B, D)
        all_grams.append(features)

    all_grams = np.vstack(all_grams)  # (N, D)

    return all_grams


def build_full_cache(
    experiment_configs: List[Dict],
    output_path: str,
) -> Dict:
    """
    Build complete Gram cache for all configurations.

    Args:
        experiment_configs: List of {backbone, layer} configs
        output_path: Where to save the cache

    Returns:
        Complete cache dict
    """
    logger.info("="*60)
    logger.info("BUILDING GRAM CACHE")
    logger.info("="*60)

    # Load images
    logger.info(f"Loading {CONFIG['n_images_inference']} reference images...")
    ref_images = load_images(
        CONFIG["dataset_path"],
        CONFIG["n_images_inference"],
        seed=CONFIG["random_seed"],
    )

    logger.info(f"Loading {CONFIG['n_images_evaluation']} evaluation images...")
    eval_images = load_images(
        CONFIG["dataset_path"],
        CONFIG["n_images_evaluation"],
        seed=CONFIG["random_seed"] + 1,  # Different seed to avoid overlap
    )

    # Initialize degradation generator
    degradation_generator = DegradationGenerator()

    # Initialize cache structure
    cache = {
        "metadata": {
            "n_images_inference": CONFIG["n_images_inference"],
            "n_images_evaluation": CONFIG["n_images_evaluation"],
            "degradations": CONFIG["degradations"],
            "timestamp": datetime.now().isoformat(),
            "experiment_configs": experiment_configs,
        }
    }

    # Build cache for each config
    total_configs = len(experiment_configs)
    for i, config in enumerate(experiment_configs, 1):
        backbone = config["backbone"]
        layer = config["layer"]

        logger.info(f"\n[{i}/{total_configs}] {backbone} layer {layer}")

        # Initialize backbone dict if needed
        if backbone not in cache:
            cache[backbone] = {}

        # Build cache for this config
        cache_entry = build_cache_for_config(
            backbone=backbone,
            layer=layer,
            ref_images=ref_images,
            eval_images=eval_images,
            degradation_generator=degradation_generator,
            batch_size=CONFIG["batch_size"],
        )

        cache[backbone][f"layer_{layer}"] = cache_entry

    # Save cache
    logger.info(f"\nSaving cache to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(cache, f, protocol=4)

    # Report cache size
    cache_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Cache saved: {cache_size_mb:.1f} MB")

    logger.info("="*60)
    logger.info("CACHE BUILD COMPLETE")
    logger.info("="*60)

    return cache


def main():
    """Main entry point."""
    # Get experiment configs from CONFIG
    experiment_configs = CONFIG.get("experiment_configs", [])

    if not experiment_configs:
        logger.error("No experiment_configs found in CONFIG!")
        return

    logger.info(f"Building cache for {len(experiment_configs)} configurations:")
    for config in experiment_configs:
        logger.info(f"  - {config['backbone']} layer {config['layer']}")

    # Output path
    cache_dir = os.path.join(CONFIG["results_path"], "cache")
    cache_path = os.path.join(cache_dir, "gram_cache.pkl")

    # Build cache
    cache = build_full_cache(experiment_configs, cache_path)

    logger.info(f"\nCache ready at: {cache_path}")
    logger.info("You can now run experiments with --use_cache flag")


if __name__ == "__main__":
    main()
