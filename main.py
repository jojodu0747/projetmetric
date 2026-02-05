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

from config import CONFIG, get_config, update_config
from feature_extraction import FeatureExtractor
from distribution_modeling import DistributionModel
from distance_metrics import get_distance_metric, get_metric_score
from evaluation import (
    MonotonicityEvaluator,
    evaluate_all_degradations,
    compute_aggregate_monotonicity,
)

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


def save_results_csv(results: List[Dict], filepath: str):
    """Save results to CSV file."""
    if not results:
        return

    # Flatten results for CSV
    rows = []
    for r in results:
        row = {
            "layers": r["layers"],
            "transform": r["transform"],
            "distribution": r.get("distribution", "empirical"),
            "metric": r["metric"],
            "mean_monotonicity": r["mean_monotonicity"],
        }
        # Add per-degradation monotonicity
        for deg_type, score in r["monotonicity"].items():
            row[f"mono_{deg_type}"] = score
        rows.append(row)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Results saved to {filepath}")


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

    results = []
    total_configs = (
        len(CONFIG["layer_configs"])
        * len(CONFIG["feature_transforms"])
        * len(CONFIG["distribution_models"])
        * len(CONFIG["distance_metrics"])
    )
    config_idx = 0

    # Iterate over all configurations
    for layer_config in CONFIG["layer_configs"]:
        for transform_config in CONFIG["feature_transforms"]:
            # Create feature extractor
            logger.info(
                f"Creating extractor: backbone={CONFIG['backbone']}, "
                f"layers={layer_config['name']}, transform={transform_config['name']}"
            )

            extractor = FeatureExtractor(
                backbone=CONFIG["backbone"],
                layer_config=layer_config,
                transform_config=transform_config,
            )

            try:
                # Extract reference features (fit PCA if needed)
                logger.info("Extracting reference features...")
                reference_features = extractor.extract(
                    reference_images, fit_transform=True
                )
                logger.info(f"Reference features shape: {reference_features.shape}")

                for dist_config in CONFIG["distribution_models"]:
                    # Fit distribution model
                    logger.info(f"Fitting distribution model: {dist_config['name']}")
                    model = DistributionModel(dist_config)
                    model.fit(reference_features)

                    for metric_name in CONFIG["distance_metrics"]:
                        config_idx += 1
                        logger.info(
                            f"Evaluating config {config_idx}/{total_configs}: "
                            f"metric={metric_name}"
                        )

                        metric = get_distance_metric(metric_name)

                        # Evaluate monotonicity across degradations
                        evaluator = MonotonicityEvaluator(
                            extractor, reference_features, CONFIG["degradations"]
                        )
                        eval_results = evaluator.evaluate_all_degradations(
                            eval_images, metric
                        )

                        # Extract monotonicity scores
                        monotonicity = {
                            deg_type: res["monotonicity"]
                            for deg_type, res in eval_results.items()
                        }

                        # Extract detailed scores for plotting
                        detailed_scores = {
                            deg_type: res["scores"]
                            for deg_type, res in eval_results.items()
                        }

                        result = {
                            "layers": layer_config["name"],
                            "transform": transform_config["name"],
                            "distribution": dist_config["name"],
                            "metric": metric_name,
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
    csv_path = os.path.join(results_path, f"results_{timestamp}.csv")
    json_path = os.path.join(results_path, f"results_{timestamp}.json")

    save_results_csv(results, csv_path)
    save_results_json(results, json_path)

    # Print summary
    print_best_metrics(results)

    # Generate plots
    plot_monotonicity_curves(results, results_path)

    logger.info("Experiment completed successfully")
    return results


def run_quick_test():
    """Run a quick test with minimal configuration."""
    logger.info("Running quick test with minimal configuration")

    # Override config for quick test
    update_config(
        n_images_inference=50,
        n_images_evaluation=10,
    )

    # Use only first config of each type
    test_layer_config = CONFIG["layer_configs"][0]
    test_transform_config = CONFIG["feature_transforms"][0]
    test_dist_config = CONFIG["distribution_models"][0]
    test_metric = CONFIG["distance_metrics"][0]

    # Load images
    reference_images = load_images(CONFIG["dataset_path"], 50)
    eval_images = load_images(CONFIG["dataset_path"], 10)

    # Create extractor
    extractor = FeatureExtractor(
        backbone=CONFIG["backbone"],
        layer_config=test_layer_config,
        transform_config=test_transform_config,
    )

    try:
        # Extract features
        reference_features = extractor.extract(reference_images, fit_transform=True)
        logger.info(f"Reference features shape: {reference_features.shape}")

        # Evaluate
        metric = get_distance_metric(test_metric)
        evaluator = MonotonicityEvaluator(
            extractor, reference_features, CONFIG["degradations"]
        )
        results = evaluator.evaluate_all_degradations(eval_images, metric)

        print("\nQuick Test Results:")
        print("-" * 40)
        for deg_type, res in results.items():
            print(f"{deg_type}: monotonicity = {res['monotonicity']:.4f}")
            print(f"  scores: {[f'{s:.4f}' for s in res['scores']]}")

    finally:
        extractor.cleanup()


def main():
    """Entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Image Quality Metrics Evaluation Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "quick"],
        default="full",
        help="Run mode: 'full' for complete evaluation, 'quick' for test run",
    )
    parser.add_argument(
        "--backbone",
        choices=["resnet50", "vgg19", "dinov2_vitb14"],
        default=None,
        help="Override backbone model",
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
    if args.backbone:
        update_config(backbone=args.backbone)
    if args.n_inference:
        update_config(n_images_inference=args.n_inference)
    if args.n_evaluation:
        update_config(n_images_evaluation=args.n_evaluation)

    # Run selected mode
    if args.mode == "quick":
        run_quick_test()
    else:
        run_experiment()


if __name__ == "__main__":
    main()
