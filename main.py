"""
Main pipeline for image quality metrics evaluation.
Orchestrates feature extraction, distribution modeling, and metric evaluation.
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG, BACKBONE_CONFIGS, get_config, update_config, get_enabled_experiment_configs
from feature_extraction import FeatureExtractor
from distribution_modeling import DistributionModel
from distance_metrics import get_distance_metric, get_metric_score
from evaluation import (
    MonotonicityEvaluator,
    evaluate_all_degradations,
    compute_aggregate_monotonicity,
    evaluate_monotonicity,
)
from degradation_generator import BatchDegradationGenerator, DegradationGenerator
from feature_cache import FeatureCache
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def setup_logging(results_path: str):
    """Set up file logging to results directory."""
    log_file = os.path.join(results_path, "experiment.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)


def load_images(
    dataset_path: str,
    n_images: int,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> List[str]:
    """
    Load image paths from a dataset directory.

    Args:
        dataset_path: Path to image directory
        n_images: Number of images to load
        extensions: Valid image extensions

    Returns:
        List of image paths
    """
    logger.info(f"Loading images from {dataset_path}")

    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(dataset_path, f"*{ext}")))
        image_paths.extend(glob(os.path.join(dataset_path, f"*{ext.upper()}")))

    # Remove duplicates and sort for reproducibility
    image_paths = sorted(list(set(image_paths)))

    # Limit to n_images
    np.random.seed(CONFIG["random_seed"])
    if len(image_paths) > n_images:
        indices = np.random.choice(len(image_paths), n_images, replace=False)
        image_paths = [image_paths[i] for i in sorted(indices)]

    logger.info(f"Loaded {len(image_paths)} images")
    return image_paths


def extract_all_degraded_features(
    eval_images: List[str],
    extractor: FeatureExtractor,
    degradation_config: Dict,
) -> Dict[str, List[np.ndarray]]:
    """
    Pre-extract features for all degradation types and levels.

    Returns a cache: {degradation_type: [features_level_0, features_level_1, ...]}
    """
    cache = {}
    batch_generator = BatchDegradationGenerator(degradation_config, num_workers=8)
    deg_generator = DegradationGenerator(degradation_config)

    # Load all images into memory once
    logger.info("Loading evaluation images into memory...")
    pil_images = [Image.open(p).convert("RGB") for p in eval_images]

    for deg_type in degradation_config.keys():
        num_levels = deg_generator.get_num_levels(deg_type)
        cache[deg_type] = []

        logger.info(f"Extracting features for {deg_type} ({num_levels} levels)...")

        for level in range(num_levels):
            # Degrade images in parallel
            degraded_batch = batch_generator.process_image_batch(
                pil_images, deg_type, level
            )
            # Extract features
            features = extractor.extract(degraded_batch, fit_transform=False)
            cache[deg_type].append(features)

    return cache


def normalize_metric_config(metric_config):
    """
    Normalize metric configuration to dict format.

    Args:
        metric_config: Either a string (metric name) or dict with 'name' and params

    Returns:
        Dict with metric parameters
    """
    if isinstance(metric_config, str):
        # String shorthand: just metric name, use defaults
        config = {"name": metric_config}
    else:
        config = metric_config.copy()

    # Set defaults based on metric type
    name = config["name"]

    # MMD defaults
    if name == "mmd":
        config.setdefault("kernel", "rbf")
        config.setdefault("gamma", None)

    # Sinkhorn defaults
    elif name == "sinkhorn":
        config.setdefault("reg", 0.1)
        config.setdefault("max_iter", 100)

    # Energy defaults
    elif name == "energy":
        config.setdefault("sample_size", 1000)

    # Adversarial defaults
    elif name == "adversarial":
        config.setdefault("n_critics", 5)
        config.setdefault("max_iter", 100)

    # FID has no extra params

    return config


def get_metric_kwargs(metric_cfg):
    """
    Extract kwargs for get_distance_metric from normalized config.

    Args:
        metric_cfg: Normalized metric config dict

    Returns:
        Dict of kwargs to pass to get_distance_metric
    """
    name = metric_cfg["name"]
    kwargs = {}

    if name == "mmd":
        kwargs["kernel"] = metric_cfg.get("kernel", "rbf")
        kwargs["gamma"] = metric_cfg.get("gamma")

    elif name == "sinkhorn":
        kwargs["sinkhorn_reg"] = metric_cfg.get("reg", 0.1)
        kwargs["sinkhorn_max_iter"] = metric_cfg.get("max_iter", 100)

    elif name == "energy":
        kwargs["energy_sample_size"] = metric_cfg.get("sample_size", 1000)

    elif name == "adversarial":
        kwargs["adversarial_n_critics"] = metric_cfg.get("n_critics", 5)
        kwargs["adversarial_max_iter"] = metric_cfg.get("max_iter", 100)

    # FID has no extra params

    return kwargs


def get_metric_identifier(metric_config):
    """
    Get a unique identifier string for a metric configuration.

    Args:
        metric_config: Normalized metric config dict

    Returns:
        String identifier like "mmd_rbf_auto", "sinkhorn_reg0.1", etc.
    """
    name = metric_config["name"]

    if name == "mmd":
        kernel = metric_config.get("kernel", "rbf")
        gamma = metric_config.get("gamma")
        if gamma is None:
            gamma_str = "auto"
        else:
            gamma_str = f"{gamma:.4f}".rstrip('0').rstrip('.')
        return f"{name}_{kernel}_{gamma_str}"

    elif name == "sinkhorn":
        reg = metric_config.get("reg", 0.1)
        return f"{name}_reg{reg}"

    elif name == "energy":
        sample_size = metric_config.get("sample_size", 1000)
        return f"{name}_n{sample_size}"

    elif name == "adversarial":
        n_critics = metric_config.get("n_critics", 5)
        max_iter = metric_config.get("max_iter", 100)
        return f"{name}_c{n_critics}_i{max_iter}"

    elif name == "fid":
        return "fid"

    else:
        return name


def evaluate_with_cached_features(
    degraded_features_cache: Dict[str, List[np.ndarray]],
    metric,
) -> Dict[str, Dict]:
    """
    Evaluate metric using pre-extracted cached features.
    For GMM-based metrics: uses internal distribution model.
    For FID/MMD/CMMD: uses internal reference features.
    """
    results = {}

    for deg_type, features_list in degraded_features_cache.items():
        num_levels = len(features_list)
        levels = np.arange(num_levels)
        scores = []

        for features in features_list:
            score = metric.compute(features)
            score = get_metric_score(score)
            scores.append(score)

        mono_score = evaluate_monotonicity(levels, np.array(scores))

        results[deg_type] = {
            "levels": levels.tolist(),
            "scores": scores,
            "monotonicity": mono_score,
        }

    return results


def save_results_csv(results: List[Dict], filepath: str):
    """Save all results to a single comprehensive CSV file.

    Each row contains full context: date, backbone, layer info (indices,
    module names, position/total), transform params, distribution params,
    metric, monotonicity scores, and all hyperparameters.
    """
    if not results:
        return

    backbone_name = CONFIG["backbone"]
    backbone_config = BACKBONE_CONFIGS[backbone_name]
    layer_names_map = backbone_config["layer_names"]
    total_layers = backbone_config.get("total_layers", len(layer_names_map))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    deg_types = list(results[0]["monotonicity"].keys()) if results else []

    # Build header
    fieldnames = [
        "date",
        "modele_backbone",
        "config_couches",
        "indices_couches",
        "modules_extraits",
        "position_couche",
        "nb_couches_modele",
        "transform",
        "use_gram",
        "use_pca",
        "pca_dim",
        "distribution",
        "dist_type",
        "dist_n_components",
        "dist_covariance_type",
        "dist_bandwidth",
        "metrique_distance",
        "monotonie_moyenne",
    ]
    for deg_type in deg_types:
        fieldnames.append(f"mono_{deg_type}")
    fieldnames.extend([
        "image_size",
        "n_images_inference",
        "n_images_evaluation",
        "batch_size",
        "random_seed",
    ])

    rows = []
    for r in results:
        # Resolve layer config
        layer_cfg = None
        for lc in CONFIG["layer_configs"]:
            if lc["name"] == r["layers"]:
                layer_cfg = lc
                break
        layer_indices = layer_cfg["layers"] if layer_cfg else []
        module_names = [layer_names_map.get(idx, f"?(index {idx})") for idx in layer_indices]
        indices_str = " | ".join(str(i) for i in layer_indices)
        position_str = f"{indices_str} / {total_layers}"

        # Resolve transform config
        transform_cfg = None
        for tc in CONFIG["feature_transforms"]:
            if tc["name"] == r["transform"]:
                transform_cfg = tc
                break

        # Resolve distribution config (optional, for backward compatibility)
        dist_cfg = None
        if "distribution_models" in CONFIG:
            for dc in CONFIG["distribution_models"]:
                if dc["name"] == r.get("distribution", ""):
                    dist_cfg = dc
                    break

        row = {
            "date": timestamp,
            "modele_backbone": backbone_name,
            "config_couches": r["layers"],
            "indices_couches": indices_str,
            "modules_extraits": " | ".join(module_names),
            "position_couche": position_str,
            "nb_couches_modele": total_layers,
            "transform": r["transform"],
            "use_gram": transform_cfg.get("use_gram", "") if transform_cfg else "",
            "use_pca": transform_cfg.get("use_pca", "") if transform_cfg else "",
            "pca_dim": transform_cfg.get("pca_dim", "") if transform_cfg else "",
            "distribution": r.get("distribution", "empirical"),
            "dist_type": dist_cfg.get("type", "") if dist_cfg else "",
            "dist_n_components": dist_cfg.get("n_components", "") if dist_cfg else "",
            "dist_covariance_type": dist_cfg.get("covariance_type", "") if dist_cfg else "",
            "dist_bandwidth": dist_cfg.get("bandwidth", "") if dist_cfg else "",
            "metrique_distance": r["metric"],
            "monotonie_moyenne": round(r["mean_monotonicity"], 6),
        }
        for deg_type in deg_types:
            row[f"mono_{deg_type}"] = round(r["monotonicity"].get(deg_type, 0), 6)
        row["image_size"] = CONFIG["image_size"]
        row["n_images_inference"] = CONFIG["n_images_inference"]
        row["n_images_evaluation"] = CONFIG["n_images_evaluation"]
        row["batch_size"] = CONFIG["batch_size"]
        row["random_seed"] = CONFIG["random_seed"]
        rows.append(row)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Résultats sauvegardés dans {filepath}")


def save_results_json(results: List[Dict], filepath: str):
    """Save detailed results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    with open(filepath, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)

    logger.info(f"Detailed results saved to {filepath}")


def print_best_metrics(results: List[Dict], top_n: int = 10):
    """Print the best performing configurations."""
    sorted_results = sorted(results, key=lambda x: x["mean_monotonicity"], reverse=True)

    print("\n" + "=" * 80)
    print("TOP PERFORMING CONFIGURATIONS")
    print("=" * 80)

    for i, r in enumerate(sorted_results[:top_n], 1):
        print(f"\n{i}. Mean Monotonicity: {r['mean_monotonicity']:.4f}")
        print(f"   Layers: {r['layers']}")
        print(f"   Transform: {r['transform']}")
        print(f"   Distribution: {r.get('distribution', 'empirical')}")
        print(f"   Metric: {r['metric']}")
        print(f"   Per-degradation scores:")
        for deg_type, score in r["monotonicity"].items():
            print(f"     - {deg_type}: {score:.4f}")

    print("\n" + "=" * 80)


def plot_monotonicity_curves(
    results: List[Dict],
    results_path: str,
    top_n: int = 5,
):
    """
    Plot metric scores vs degradation levels for top configurations.

    Args:
        results: List of evaluation results
        results_path: Directory to save plots
        top_n: Number of top configurations to plot
    """
    # Sort by mean monotonicity
    sorted_results = sorted(results, key=lambda x: x["mean_monotonicity"], reverse=True)

    for i, r in enumerate(sorted_results[:top_n]):
        if "detailed_scores" not in r:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f"Configuration: {r['layers']} / {r['transform']} / {r['metric']}\n"
            f"Mean Monotonicity: {r['mean_monotonicity']:.4f}",
            fontsize=12,
        )

        for ax, (deg_type, scores) in zip(axes.flat, r["detailed_scores"].items()):
            levels = list(range(len(scores)))
            ax.plot(levels, scores, "o-", linewidth=2, markersize=8)
            ax.set_xlabel("Degradation Level")
            ax.set_ylabel("Metric Score")
            ax.set_title(f"{deg_type} (mono: {r['monotonicity'][deg_type]:.4f})")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(results_path, f"monotonicity_top{i+1}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

    logger.info(f"Plots saved to {results_path}")


def run_experiment():
    """Main experiment runner."""
    logger.info("Starting image quality metrics evaluation experiment")
    logger.info(f"Configuration: {CONFIG}")

    # Set random seed
    np.random.seed(CONFIG["random_seed"])

    # Create results directory
    results_path = CONFIG["results_path"]
    os.makedirs(results_path, exist_ok=True)
    setup_logging(results_path)

    # Load images
    reference_images = load_images(
        CONFIG["dataset_path"], CONFIG["n_images_inference"]
    )
    eval_images = load_images(CONFIG["dataset_path"], CONFIG["n_images_evaluation"])

    # Ensure eval images are different from reference
    eval_set = set(eval_images)
    reference_images = [p for p in reference_images if p not in eval_set]

    if len(reference_images) < 100:
        logger.warning(f"Only {len(reference_images)} reference images available")

    # Get enabled layers for current backbone from ENABLED_LAYERS
    current_backbone = CONFIG["backbone"]
    enabled_layers = [
        cfg["layer"]
        for cfg in get_enabled_experiment_configs()
        if cfg["backbone"] == current_backbone
    ]

    # Create layer configs from enabled layers
    valid_layer_configs = [
        {"name": f"single_layer_{layer_idx}", "layers": [layer_idx]}
        for layer_idx in enabled_layers
    ]

    logger.info(f"Backbone {current_backbone}: {len(valid_layer_configs)} enabled layers: {enabled_layers}")

    results = []
    total_configs = (
        len(valid_layer_configs)
        * len(CONFIG["feature_transforms"])
        * len(CONFIG["distance_metrics"])
    )
    config_idx = 0

    # Activation cache
    cache = FeatureCache()
    backbone = CONFIG["backbone"]
    ref_id = cache.image_set_id(reference_images)
    eval_id = cache.image_set_id(eval_images)
    deg_generator = DegradationGenerator(CONFIG["degradations"])

    # Iterate over all configurations
    for layer_config in valid_layer_configs:
        layer_name = layer_config["name"]
        is_single = layer_name.startswith("single_layer_")

        use_cache = (
            is_single
            and cache.has_reference(backbone, layer_name, ref_id)
            and cache.has_all_degraded(backbone, layer_name, eval_id, CONFIG["degradations"])
        )

        if use_cache:
            # ── CACHE HIT ──
            logger.info(f"CACHE HIT for {backbone}/{layer_name} — skipping extraction")
            ref_dir = cache._ref_dir(backbone, layer_name, ref_id)
            eval_dir = cache._eval_dir(backbone, layer_name, eval_id)
            ref_activations = cache.load_activations(ref_dir)

            for transform_config in CONFIG["feature_transforms"]:
                logger.info(f"Post-processing cached activations: transform={transform_config['name']}")

                reference_features, pca_model = cache.process_activations(
                    ref_activations, transform_config, fit_pca=True
                )
                logger.info(f"Reference features shape: {reference_features.shape}")

                degraded_features_cache = {}
                for deg_type in CONFIG["degradations"].keys():
                    num_levels = deg_generator.get_num_levels(deg_type)
                    features_list = []
                    for level in range(num_levels):
                        deg_act = cache.load_degraded_activations(eval_dir, deg_type, level)
                        deg_feat, _ = cache.process_activations(
                            deg_act, transform_config, pca_model=pca_model
                        )
                        features_list.append(deg_feat)
                    degraded_features_cache[deg_type] = features_list

                # No GMM fitting needed for MMD/FID (distribution-free metrics)
                for metric_config in CONFIG["distance_metrics"]:
                    config_idx += 1

                    # Normalize metric config to dict format
                    metric_cfg = normalize_metric_config(metric_config)
                    metric_name = metric_cfg["name"]
                    metric_id = get_metric_identifier(metric_cfg)

                    logger.info(
                        f"Evaluating config {config_idx}/{total_configs}: "
                        f"metric={metric_id}"
                    )

                    # Create distance metric with reference features
                    metric_kwargs = get_metric_kwargs(metric_cfg)
                    metric = get_distance_metric(
                        metric_name,
                        features_ref=reference_features,
                        **metric_kwargs
                    )

                    eval_results = evaluate_with_cached_features(
                        degraded_features_cache, metric
                    )

                    monotonicity = {
                        deg_type: res["monotonicity"]
                        for deg_type, res in eval_results.items()
                    }
                    detailed_scores = {
                        deg_type: res["scores"]
                        for deg_type, res in eval_results.items()
                    }

                    result = {
                        "layers": layer_config["name"],
                        "transform": transform_config["name"],
                        "distribution": "empirical",  # No distribution model (MMD/FID are distribution-free)
                        "metric": metric_id,
                        "monotonicity": monotonicity,
                        "detailed_scores": detailed_scores,
                        "mean_monotonicity": compute_aggregate_monotonicity(
                            monotonicity
                        ),
                    }
                    results.append(result)
                    logger.info(
                        f"Mean monotonicity: {result['mean_monotonicity']:.4f}"
                    )

        else:
            # ── CACHE MISS: standard extraction ──
            for transform_config in CONFIG["feature_transforms"]:
                logger.info(
                    f"Creating extractor: backbone={backbone}, "
                    f"layers={layer_config['name']}, transform={transform_config['name']}"
                )

                extractor = FeatureExtractor(
                    backbone=backbone,
                    layer_config=layer_config,
                    transform_config=transform_config,
                )

                try:
                    logger.info("Extracting reference features...")
                    reference_features = extractor.extract(
                        reference_images, fit_transform=True
                    )
                    logger.info(f"Reference features shape: {reference_features.shape}")

                    logger.info("Pre-extracting degraded features (cached for all metrics)...")
                    degraded_features_cache = extract_all_degraded_features(
                        eval_images, extractor, CONFIG["degradations"]
                    )

                    # No GMM fitting needed for MMD/FID (distribution-free metrics)
                    for metric_config in CONFIG["distance_metrics"]:
                        config_idx += 1

                        # Normalize metric config to dict format
                        metric_cfg = normalize_metric_config(metric_config)
                        metric_name = metric_cfg["name"]
                        metric_id = get_metric_identifier(metric_cfg)

                        logger.info(
                            f"Evaluating config {config_idx}/{total_configs}: "
                            f"metric={metric_id}"
                        )

                        # Create distance metric with reference features
                        metric_kwargs = get_metric_kwargs(metric_cfg)
                        metric = get_distance_metric(
                            metric_name,
                            features_ref=reference_features,
                            **metric_kwargs
                        )

                        eval_results = evaluate_with_cached_features(
                            degraded_features_cache, metric
                        )

                        monotonicity = {
                            deg_type: res["monotonicity"]
                            for deg_type, res in eval_results.items()
                        }
                        detailed_scores = {
                            deg_type: res["scores"]
                            for deg_type, res in eval_results.items()
                        }

                        result = {
                            "layers": layer_config["name"],
                            "transform": transform_config["name"],
                            "distribution": "empirical",  # No distribution model (MMD/FID are distribution-free)
                            "metric": metric_id,
                            "monotonicity": monotonicity,
                            "detailed_scores": detailed_scores,
                            "mean_monotonicity": compute_aggregate_monotonicity(
                                monotonicity
                            ),
                        }
                        results.append(result)
                        logger.info(
                            f"Mean monotonicity: {result['mean_monotonicity']:.4f}"
                        )

                finally:
                    extractor.cleanup()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_path, f"resultats_{timestamp}.csv")
    save_results_csv(results, csv_path)

    # Print summary
    print_best_metrics(results)

    # Generate plots
    plot_monotonicity_curves(results, results_path)

    logger.info("Experiment completed successfully")
    return results


def run_medium_experiment():
    """Run a medium experiment: all layers & metrics, but skip slow raw features."""
    logger.info("Starting MEDIUM experiment (skipping raw high-dim features)")

    np.random.seed(CONFIG["random_seed"])

    results_path = CONFIG["results_path"]
    os.makedirs(results_path, exist_ok=True)
    setup_logging(results_path)

    # Load images with config values
    reference_images = load_images(
        CONFIG["dataset_path"], CONFIG["n_images_inference"]
    )
    eval_images = load_images(CONFIG["dataset_path"], CONFIG["n_images_evaluation"])

    eval_set = set(eval_images)
    reference_images = [p for p in reference_images if p not in eval_set]

    # Medium mode: only keep transforms that use Gram (works for MMD)
    # raw → 2048 dims (too high for MMD), gram transforms are OK
    medium_transforms = [
        t for t in CONFIG["feature_transforms"] if t.get("use_gram", False)
    ]

    # Get enabled layers for current backbone from ENABLED_LAYERS
    current_backbone = CONFIG["backbone"]
    enabled_layers = [
        cfg["layer"]
        for cfg in get_enabled_experiment_configs()
        if cfg["backbone"] == current_backbone
    ]

    # Create layer configs from enabled layers
    valid_layer_configs = [
        {"name": f"single_layer_{layer_idx}", "layers": [layer_idx]}
        for layer_idx in enabled_layers
    ]

    logger.info(f"Backbone {current_backbone}: {len(valid_layer_configs)} enabled layers: {enabled_layers}")

    total_configs = (
        len(valid_layer_configs)
        * len(medium_transforms)
        * len(CONFIG["distance_metrics"])
    )
    logger.info(f"Medium mode: {total_configs} configurations (MMD/FID only, no GMM)")

    results = []
    config_idx = 0

    # Activation cache
    cache = FeatureCache()
    backbone = CONFIG["backbone"]
    ref_id = cache.image_set_id(reference_images)
    eval_id = cache.image_set_id(eval_images)
    deg_generator = DegradationGenerator(CONFIG["degradations"])

    for layer_config in valid_layer_configs:
        layer_name = layer_config["name"]
        is_single = layer_name.startswith("single_layer_")

        # Check activation cache (only for single-layer configs)
        use_cache = (
            is_single
            and cache.has_reference(backbone, layer_name, ref_id)
            and cache.has_all_degraded(backbone, layer_name, eval_id, CONFIG["degradations"])
        )

        if use_cache:
            # ── CACHE HIT: load raw activations, post-process per transform ──
            logger.info(f"CACHE HIT for {backbone}/{layer_name} — skipping extraction")
            ref_dir = cache._ref_dir(backbone, layer_name, ref_id)
            eval_dir = cache._eval_dir(backbone, layer_name, eval_id)
            ref_activations = cache.load_activations(ref_dir)

            for transform_config in medium_transforms:
                logger.info(f"Post-processing cached activations: transform={transform_config['name']}")

                reference_features, pca_model = cache.process_activations(
                    ref_activations, transform_config, fit_pca=True
                )
                logger.info(f"Reference features shape: {reference_features.shape}")

                # Build degraded features from cached activations
                degraded_features_cache = {}
                for deg_type in CONFIG["degradations"].keys():
                    num_levels = deg_generator.get_num_levels(deg_type)
                    features_list = []
                    for level in range(num_levels):
                        deg_act = cache.load_degraded_activations(eval_dir, deg_type, level)
                        deg_feat, _ = cache.process_activations(
                            deg_act, transform_config, pca_model=pca_model
                        )
                        features_list.append(deg_feat)
                    degraded_features_cache[deg_type] = features_list

                # No GMM fitting needed for MMD/FID (distribution-free metrics)
                for metric_config in CONFIG["distance_metrics"]:
                    config_idx += 1

                    # Normalize metric config to dict format
                    metric_cfg = normalize_metric_config(metric_config)
                    metric_name = metric_cfg["name"]
                    metric_id = get_metric_identifier(metric_cfg)

                    logger.info(
                        f"Evaluating config {config_idx}/{total_configs}: "
                        f"metric={metric_id}"
                    )

                    # Create distance metric with reference features
                    metric_kwargs = get_metric_kwargs(metric_cfg)
                    metric = get_distance_metric(
                        metric_name,
                        features_ref=reference_features,
                        **metric_kwargs
                    )
                    eval_results = evaluate_with_cached_features(
                        degraded_features_cache, metric
                    )

                    monotonicity = {
                        deg_type: res["monotonicity"]
                        for deg_type, res in eval_results.items()
                    }
                    detailed_scores = {
                        deg_type: res["scores"]
                        for deg_type, res in eval_results.items()
                    }

                    result = {
                        "layers": layer_config["name"],
                        "transform": transform_config["name"],
                        "distribution": "empirical",  # No distribution model (MMD/FID are distribution-free)
                        "metric": metric_id,
                        "monotonicity": monotonicity,
                        "detailed_scores": detailed_scores,
                        "mean_monotonicity": compute_aggregate_monotonicity(
                            monotonicity
                        ),
                    }
                    results.append(result)
                    logger.info(
                        f"Mean monotonicity: {result['mean_monotonicity']:.4f}"
                    )

        else:
            # ── CACHE MISS: standard extraction ──
            for transform_config in medium_transforms:
                logger.info(
                    f"Creating extractor: backbone={backbone}, "
                    f"layers={layer_config['name']}, transform={transform_config['name']}"
                )

                extractor = FeatureExtractor(
                    backbone=backbone,
                    layer_config=layer_config,
                    transform_config=transform_config,
                )

                try:
                    logger.info("Extracting reference features...")
                    reference_features = extractor.extract(
                        reference_images, fit_transform=True
                    )
                    logger.info(f"Reference features shape: {reference_features.shape}")

                    logger.info("Pre-extracting degraded features...")
                    degraded_features_cache = extract_all_degraded_features(
                        eval_images, extractor, CONFIG["degradations"]
                    )

                    # No GMM fitting needed for MMD/FID (distribution-free metrics)
                    for metric_config in CONFIG["distance_metrics"]:
                        config_idx += 1

                        # Normalize metric config to dict format
                        metric_cfg = normalize_metric_config(metric_config)
                        metric_name = metric_cfg["name"]
                        metric_id = get_metric_identifier(metric_cfg)

                        logger.info(
                            f"Evaluating config {config_idx}/{total_configs}: "
                            f"metric={metric_id}"
                        )

                        # Create distance metric with reference features
                        metric_kwargs = get_metric_kwargs(metric_cfg)
                        metric = get_distance_metric(
                            metric_name,
                            features_ref=reference_features,
                            **metric_kwargs
                        )
                        eval_results = evaluate_with_cached_features(
                            degraded_features_cache, metric
                        )

                        monotonicity = {
                            deg_type: res["monotonicity"]
                            for deg_type, res in eval_results.items()
                        }
                        detailed_scores = {
                            deg_type: res["scores"]
                            for deg_type, res in eval_results.items()
                        }

                        result = {
                            "layers": layer_config["name"],
                            "transform": transform_config["name"],
                            "distribution": "empirical",  # No distribution model (MMD/FID are distribution-free)
                            "metric": metric_id,
                            "monotonicity": monotonicity,
                            "detailed_scores": detailed_scores,
                            "mean_monotonicity": compute_aggregate_monotonicity(
                                monotonicity
                            ),
                        }
                        results.append(result)
                        logger.info(
                            f"Mean monotonicity: {result['mean_monotonicity']:.4f}"
                        )

                finally:
                    extractor.cleanup()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_path, f"resultats_medium_{timestamp}.csv")
    save_results_csv(results, csv_path)

    print_best_metrics(results)
    plot_monotonicity_curves(results, results_path)

    logger.info("Medium experiment completed successfully")
    return results


def run_quick_test():
    """Run a quick test on texture-relevant layers per backbone, gmm_full only."""
    logger.info("Running quick texture test")

    # Texture-relevant layers per backbone (mid-level = texture/quality, not semantics)
    TEXTURE_LAYERS = {
        "vgg19": [
            {"name": "single_layer_4", "layers": [4]},    # conv3_1 (256ch) — texture onset
            {"name": "single_layer_8", "layers": [8]},    # conv4_1 (512ch) — rich texture
            {"name": "single_layer_10", "layers": [10]},  # conv4_3 (512ch) — texture detail
            {"name": "single_layer_12", "layers": [12]},  # conv5_1 (512ch) — high-level texture
        ],
        "lpips_vgg": [
            {"name": "single_layer_4", "layers": [4]},    # slice3 conv3_1 — texture onset
            {"name": "single_layer_8", "layers": [8]},    # slice4 conv4_1 — rich texture
            {"name": "single_layer_10", "layers": [10]},  # slice4 conv4_3 — texture detail
            {"name": "single_layer_12", "layers": [12]},  # slice5 conv5_1 — high-level texture
        ],
        "sd_vae": [
            {"name": "single_layer_4", "layers": [4]},    # down_blocks.1.resnets.0 — mid features
            {"name": "single_layer_7", "layers": [7]},    # down_blocks.2.resnets.0 — texture compression
            {"name": "single_layer_10", "layers": [10]},  # down_blocks.3.resnets.0 — deep texture
            {"name": "single_layer_12", "layers": [12]},  # mid_block.resnets.0 — bottleneck
        ],
        "dinov2_vitb14": [
            {"name": "single_layer_5", "layers": [5]},    # blocks.4 — mid-level
            {"name": "single_layer_8", "layers": [8]},    # blocks.7 — texture-rich
            {"name": "single_layer_10", "layers": [10]},  # blocks.9 — high-level texture
            {"name": "single_layer_12", "layers": [12]},  # blocks.11 — last block
        ],
        "resnet50": [
            {"name": "single_layer_7", "layers": [7]},    # layer3
            {"name": "single_layer_8", "layers": [8]},    # layer4
        ],
    }

    backbone = CONFIG["backbone"]
    test_layers = TEXTURE_LAYERS.get(backbone, TEXTURE_LAYERS["resnet50"])
    test_transform = {"name": "gram_pca", "use_gram": True, "use_pca": True, "pca_dim": 10}
    test_dist = {"name": "gmm_full", "type": "gmm", "covariance_type": "full", "n_components": 5}

    update_config(n_images_inference=50, n_images_evaluation=20)

    reference_images = load_images(CONFIG["dataset_path"], CONFIG["n_images_inference"])
    eval_images = load_images(CONFIG["dataset_path"], CONFIG["n_images_evaluation"])
    eval_set = set(eval_images)
    reference_images = [p for p in reference_images if p not in eval_set]

    total = len(test_layers) * len(CONFIG["distance_metrics"])
    print(f"\n{'='*70}")
    print(f"QUICK TEST: {backbone} | {len(test_layers)} texture layers | gmm_full | {total} configs")
    print(f"{'='*70}")

    results = []
    config_idx = 0

    # Activation cache
    cache = FeatureCache()
    ref_id = cache.image_set_id(reference_images)
    eval_id = cache.image_set_id(eval_images)
    deg_generator = DegradationGenerator(CONFIG["degradations"])

    for layer_config in test_layers:
        layer_name_str = BACKBONE_CONFIGS[backbone]["layer_names"].get(
            layer_config["layers"][0], f"idx_{layer_config['layers'][0]}"
        )
        lc_name = layer_config["name"]

        use_cache = (
            cache.has_reference(backbone, lc_name, ref_id)
            and cache.has_all_degraded(backbone, lc_name, eval_id, CONFIG["degradations"])
        )

        if use_cache:
            # ── CACHE HIT ──
            logger.info(f"CACHE HIT for {backbone}/{lc_name}")
            ref_dir = cache._ref_dir(backbone, lc_name, ref_id)
            eval_dir = cache._eval_dir(backbone, lc_name, eval_id)
            ref_activations = cache.load_activations(ref_dir)

            reference_features, pca_model = cache.process_activations(
                ref_activations, test_transform, fit_pca=True
            )

            degraded_features_cache = {}
            for deg_type in CONFIG["degradations"].keys():
                num_levels = deg_generator.get_num_levels(deg_type)
                features_list = []
                for level in range(num_levels):
                    deg_act = cache.load_degraded_activations(eval_dir, deg_type, level)
                    deg_feat, _ = cache.process_activations(
                        deg_act, test_transform, pca_model=pca_model
                    )
                    features_list.append(deg_feat)
                degraded_features_cache[deg_type] = features_list
        else:
            # ── CACHE MISS: standard extraction ──
            extractor = FeatureExtractor(
                backbone=backbone,
                layer_config=layer_config,
                transform_config=test_transform,
            )
            try:
                reference_features = extractor.extract(reference_images, fit_transform=True)
                degraded_features_cache = extract_all_degraded_features(
                    eval_images, extractor, CONFIG["degradations"]
                )
            finally:
                extractor.cleanup()

        # Evaluate with distribution-free metrics (FID/MMD/Sinkhorn/Energy/Adversarial)
        for metric_config in CONFIG["distance_metrics"]:
            config_idx += 1
            metric_cfg = normalize_metric_config(metric_config)
            metric_name = metric_cfg["name"]
            metric_id = get_metric_identifier(metric_cfg)

            # Create distance metric with reference features
            metric_kwargs = get_metric_kwargs(metric_cfg)
            metric = get_distance_metric(
                metric_name,
                features_ref=reference_features,
                **metric_kwargs
            )
            eval_results = evaluate_with_cached_features(degraded_features_cache, metric)

            monotonicity = {d: r["monotonicity"] for d, r in eval_results.items()}
            mean_mono = compute_aggregate_monotonicity(monotonicity)

            print(f"\n[{config_idx}/{total}] {layer_name_str} | {metric_id} | mean_mono={mean_mono:+.4f}")
            for deg_type, res in eval_results.items():
                scores_str = " → ".join(f"{s:.2f}" for s in res["scores"][::max(1, len(res["scores"])//4)])
                print(f"  {deg_type:15s}: mono={res['monotonicity']:+.4f}  scores=[{scores_str}]")

            results.append({
                "layers": layer_config["name"],
                "transform": test_transform["name"],
                "distribution": test_dist["name"],
                "metric": metric_id,  # FIXED: was metric_name
                "monotonicity": monotonicity,
                "detailed_scores": {d: r["scores"] for d, r in eval_results.items()},
                "mean_monotonicity": mean_mono,
            })

    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(CONFIG["results_path"], f"resultats_quick_{backbone}_{timestamp}.csv")
    os.makedirs(CONFIG["results_path"], exist_ok=True)
    save_results_csv(results, csv_path)

    print(f"\n{'='*70}")
    print(f"RÉSUMÉ {backbone} — meilleur config:")
    if results:
        best = max(results, key=lambda r: r["mean_monotonicity"])
        layer_idx = int(best["layers"].split("_")[-1])
        layer_name = BACKBONE_CONFIGS[backbone]["layer_names"].get(layer_idx, "?")
        print(f"  {layer_name} | {best['metric']} | mean_mono={best['mean_monotonicity']:+.4f}")
    print(f"{'='*70}\n")


def build_feature_cache(backbone: str, layer_indices: List[int], n_images: int):
    """
    Build persistent activation cache for specified backbone/layers.

    Extracts raw hook activations (before Gram/GAP/PCA) and saves to disk.
    Subsequent experiment runs will detect and use this cache automatically.
    """
    logger.info(f"Building feature cache: backbone={backbone}, layers={layer_indices}, n_images={n_images}")

    update_config(backbone=backbone, n_images_inference=n_images)

    results_path = CONFIG["results_path"]
    os.makedirs(results_path, exist_ok=True)
    setup_logging(results_path)

    cache = FeatureCache()
    np.random.seed(CONFIG["random_seed"])

    # Load images
    reference_images = load_images(CONFIG["dataset_path"], CONFIG["n_images_inference"])
    eval_images = load_images(CONFIG["dataset_path"], CONFIG["n_images_evaluation"])
    eval_set = set(eval_images)
    reference_images = [p for p in reference_images if p not in eval_set]

    ref_id = cache.image_set_id(reference_images)
    eval_id = cache.image_set_id(eval_images)

    logger.info(f"Reference images: {len(reference_images)} (id={ref_id})")
    logger.info(f"Eval images: {len(eval_images)} (id={eval_id})")

    backbone_layers = set(BACKBONE_CONFIGS[backbone]["layer_names"].keys())

    for layer_idx in layer_indices:
        if layer_idx not in backbone_layers:
            logger.warning(f"Layer {layer_idx} not available for {backbone}, skipping")
            continue

        layer_config = {"name": f"single_layer_{layer_idx}", "layers": [layer_idx]}
        layer_name = layer_config["name"]
        module_name = BACKBONE_CONFIGS[backbone]["layer_names"][layer_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"Caching layer {layer_idx} ({module_name})")
        logger.info(f"{'='*60}")

        # Use raw transform (doesn't matter — we use extract_raw_activations)
        extractor = FeatureExtractor(
            backbone=backbone,
            layer_config=layer_config,
            transform_config={"name": "raw", "use_gram": False, "use_pca": False},
        )

        try:
            # ── Reference activations ─────────────────────────────
            if cache.has_reference(backbone, layer_name, ref_id):
                logger.info(f"Reference activations already cached, skipping")
            else:
                logger.info(f"Extracting reference activations ({len(reference_images)} images)...")
                ref_activations = extractor.extract_raw_activations(reference_images)
                ref_dir = cache._ref_dir(backbone, layer_name, ref_id)
                cache.save_activations(
                    ref_dir, ref_activations, reference_images,
                    backbone, module_name,
                )
                logger.info(f"Reference activations cached: shape={ref_activations.shape}")

            # ── Degraded activations ──────────────────────────────
            eval_dir = cache._eval_dir(backbone, layer_name, eval_id)

            if cache.has_all_degraded(backbone, layer_name, eval_id, CONFIG["degradations"]):
                logger.info(f"All degraded activations already cached, skipping")
            else:
                logger.info(f"Extracting degraded activations ({len(eval_images)} eval images)...")
                batch_generator = BatchDegradationGenerator(CONFIG["degradations"], num_workers=8)
                deg_generator = DegradationGenerator(CONFIG["degradations"])

                pil_images = [Image.open(p).convert("RGB") for p in eval_images]

                for deg_type in CONFIG["degradations"].keys():
                    num_levels = deg_generator.get_num_levels(deg_type)
                    logger.info(f"  {deg_type}: {num_levels} levels")

                    for level in range(num_levels):
                        deg_path = eval_dir / deg_type / f"level_{level:02d}.npy"
                        if deg_path.exists():
                            continue

                        degraded_batch = batch_generator.process_image_batch(
                            pil_images, deg_type, level
                        )
                        deg_activations = extractor.extract_raw_activations(degraded_batch)
                        cache.save_degraded_activations(eval_dir, deg_type, level, deg_activations)

                logger.info(f"Degraded activations cached")

        finally:
            extractor.cleanup()

    # Print summary
    print(f"\n{cache.get_cache_summary(backbone)}")
    logger.info("Cache build complete")


def main():
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Image Quality Metrics Evaluation Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "medium", "quick"],
        default="medium",
        help="Run mode: 'full' for complete evaluation, 'medium' for balanced run, 'quick' for test run",
    )
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="Build activation cache for specified backbone/layers instead of running experiment",
    )
    parser.add_argument(
        "--backbone",
        nargs="+",
        choices=["resnet50", "vgg19", "dinov2_vitb14", "sd_vae", "lpips_vgg", "all"],
        default=None,
        help="Backbone model(s) to run. Use 'all' for dinov2+vgg19+sd_vae+lpips_vgg, or list specific ones.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layer indices to cache (for --build-cache). E.g.: --layers 9 or --layers 7 9 12",
    )
    parser.add_argument(
        "--n-inference",
        type=int,
        default=None,
        help="Override number of inference images",
    )
    parser.add_argument(
        "--n-evaluation",
        type=int,
        default=None,
        help="Override number of evaluation images",
    )

    args = parser.parse_args()

    # Apply overrides
    if args.n_inference:
        update_config(n_images_inference=args.n_inference)
    if args.n_evaluation:
        update_config(n_images_evaluation=args.n_evaluation)

    # Determine backbones to run
    if args.backbone and "all" in args.backbone:
        backbones = ["dinov2_vitb14", "vgg19", "sd_vae", "lpips_vgg"]
    elif args.backbone:
        backbones = args.backbone
    else:
        backbones = [CONFIG["backbone"]]

    # Build cache mode
    if args.build_cache:
        if not args.layers:
            parser.error("--build-cache requires --layers (e.g. --layers 9 or --layers 7 9 12)")
        n_images = args.n_inference or CONFIG["n_images_inference"]
        for bb in backbones:
            build_feature_cache(bb, args.layers, n_images)
        return

    # Run for each backbone
    run_fn = {"quick": run_quick_test, "medium": run_medium_experiment, "full": run_experiment}[args.mode]

    for bb in backbones:
        logger.info(f"\n{'='*80}\nRunning backbone: {bb}\n{'='*80}")
        update_config(backbone=bb)
        try:
            run_fn()
        except Exception as e:
            logger.error(f"Backbone {bb} failed: {e}", exc_info=True)
            print(f"\n*** ERREUR pour {bb}: {e} — on continue avec le suivant ***\n")


if __name__ == "__main__":
    main()
