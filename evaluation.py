"""
Evaluation module for assessing metric monotonicity under degradations.
Measures how well distance metrics detect increasing image degradation.
"""

import logging
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from PIL import Image
from scipy import stats

from config import CONFIG
from degradation_generator import DegradationGenerator, BatchDegradationGenerator
from distance_metrics import DistanceMetric, get_distance_metric, get_metric_score
from feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)


def compute_spearman_correlation(
    degradation_levels: np.ndarray,
    metric_scores: np.ndarray,
) -> float:
    """
    Compute Spearman correlation between degradation levels and metric scores.

    A perfect monotonic metric would have correlation = 1.0 (higher degradation
    leads to higher distance score).

    Args:
        degradation_levels: Array of severity levels (0, 1, 2, ...)
        metric_scores: Array of corresponding metric scores

    Returns:
        Spearman correlation coefficient (range: -1 to 1)
    """
    if len(degradation_levels) < 2:
        return 0.0

    correlation, p_value = stats.spearmanr(degradation_levels, metric_scores)

    if np.isnan(correlation):
        return 0.0

    return float(correlation)


def compute_pairwise_ordering_score(
    degradation_levels: np.ndarray,
    metric_scores: np.ndarray,
) -> float:
    """
    Compute the percentage of correctly ordered pairs.

    For each pair (i, j) where level[i] < level[j], check if score[i] < score[j].
    A perfect monotonic metric would have score = 1.0.

    Args:
        degradation_levels: Array of severity levels
        metric_scores: Array of corresponding metric scores

    Returns:
        Fraction of correctly ordered pairs (range: 0 to 1)
    """
    n = len(degradation_levels)
    if n < 2:
        return 0.0

    correct_pairs = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if degradation_levels[i] < degradation_levels[j]:
                total_pairs += 1
                if metric_scores[i] < metric_scores[j]:
                    correct_pairs += 1
            elif degradation_levels[i] > degradation_levels[j]:
                total_pairs += 1
                if metric_scores[i] > metric_scores[j]:
                    correct_pairs += 1

    if total_pairs == 0:
        return 0.0

    return correct_pairs / total_pairs


def evaluate_monotonicity(
    degradation_levels: np.ndarray,
    metric_scores: np.ndarray,
    method: str = "spearman",
) -> float:
    """
    Evaluate monotonicity of metric scores with respect to degradation levels.

    Args:
        degradation_levels: Array of severity levels
        metric_scores: Array of corresponding metric scores
        method: Evaluation method ('spearman' or 'pairwise')

    Returns:
        Monotonicity score (higher is better)
    """
    if method == "spearman":
        return compute_spearman_correlation(degradation_levels, metric_scores)
    elif method == "pairwise":
        return compute_pairwise_ordering_score(degradation_levels, metric_scores)
    else:
        raise ValueError(f"Unknown method: {method}")


class MonotonicityEvaluator:
    """
    Evaluates how well distance metrics detect degradations monotonically.
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        reference_features: np.ndarray,
        degradation_config: Dict = None,
    ):
        """
        Initialize the evaluator.

        Args:
            extractor: FeatureExtractor instance
            reference_features: Features from clean reference images
            degradation_config: Degradation configurations
        """
        self.extractor = extractor
        self.reference_features = reference_features
        self.degradation_generator = DegradationGenerator(degradation_config)
        self.degradation_config = degradation_config or CONFIG["degradations"]

    def evaluate_single_image(
        self,
        image: Union[str, Image.Image],
        metric: DistanceMetric,
        degradation_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a single image across all degradation levels.

        Args:
            image: Image path or PIL Image
            metric: Distance metric to use
            degradation_type: Type of degradation

        Returns:
            Tuple of (levels, scores)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Generate degradation sequence
        degraded_sequence = self.degradation_generator.generate_degradation_sequence(
            image, degradation_type
        )

        num_levels = len(degraded_sequence)
        levels = np.arange(num_levels)
        scores = []

        for degraded_img in degraded_sequence:
            # Extract features from degraded image
            features = self.extractor.extract([degraded_img], fit_transform=False)

            # Compute distance from reference
            score = metric.compute(self.reference_features, features)
            score = get_metric_score(score)
            scores.append(score)

        return levels, np.array(scores)

    def evaluate_image_batch(
        self,
        images: List[Union[str, Image.Image]],
        metric: DistanceMetric,
        degradation_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of images across all degradation levels.

        For each level, all images are degraded and their features are
        extracted together, then distance is computed.

        Args:
            images: List of image paths or PIL Images
            metric: Distance metric to use
            degradation_type: Type of degradation

        Returns:
            Tuple of (levels, scores)
        """
        batch_generator = BatchDegradationGenerator(self.degradation_config)
        num_levels = self.degradation_generator.get_num_levels(degradation_type)

        levels = np.arange(num_levels)
        scores = []

        # Load images if paths
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img)

        for level in range(num_levels):
            # Degrade all images at this level
            degraded_batch = batch_generator.process_image_batch(
                pil_images, degradation_type, level
            )

            # Extract features
            features = self.extractor.extract(degraded_batch, fit_transform=False)

            # Compute distance from reference
            score = metric.compute(self.reference_features, features)
            score = get_metric_score(score)
            scores.append(score)

            logger.debug(
                f"Degradation {degradation_type}, level {level}: score = {score:.4f}"
            )

        return levels, np.array(scores)

    def evaluate_all_degradations(
        self,
        images: List[Union[str, Image.Image]],
        metric: DistanceMetric,
        monotonicity_method: str = "spearman",
    ) -> Dict[str, Dict]:
        """
        Evaluate metric monotonicity across all degradation types.

        Args:
            images: List of evaluation images
            metric: Distance metric to use
            monotonicity_method: Method for computing monotonicity score

        Returns:
            Dict mapping degradation type to evaluation results
        """
        results = {}

        for deg_type in self.degradation_config.keys():
            logger.info(f"Evaluating degradation type: {deg_type}")

            levels, scores = self.evaluate_image_batch(images, metric, deg_type)

            mono_score = evaluate_monotonicity(levels, scores, monotonicity_method)

            results[deg_type] = {
                "levels": levels.tolist(),
                "scores": scores.tolist(),
                "monotonicity": mono_score,
            }

        return results


def evaluate_all_degradations(
    eval_images: List[Union[str, Image.Image]],
    extractor: FeatureExtractor,
    reference_features: np.ndarray,
    metric: DistanceMetric,
    degradation_config: Dict = None,
    monotonicity_method: str = "spearman",
) -> Dict[str, float]:
    """
    Convenience function to evaluate all degradation types.

    Args:
        eval_images: List of evaluation images
        extractor: FeatureExtractor instance
        reference_features: Reference feature set
        metric: Distance metric
        degradation_config: Degradation configurations
        monotonicity_method: Method for monotonicity scoring

    Returns:
        Dict mapping degradation type to monotonicity score
    """
    evaluator = MonotonicityEvaluator(
        extractor, reference_features, degradation_config
    )

    full_results = evaluator.evaluate_all_degradations(
        eval_images, metric, monotonicity_method
    )

    # Extract just monotonicity scores
    return {deg_type: res["monotonicity"] for deg_type, res in full_results.items()}


def evaluate_metric_across_degradations(
    eval_images: List[Union[str, Image.Image]],
    extractor: FeatureExtractor,
    reference_features: np.ndarray,
    metric_name: str,
    degradation_config: Dict = None,
) -> Dict[str, Dict]:
    """
    Full evaluation of a metric across all degradation types.

    Args:
        eval_images: List of evaluation images
        extractor: FeatureExtractor instance
        reference_features: Reference feature set
        metric_name: Name of distance metric
        degradation_config: Degradation configurations

    Returns:
        Complete evaluation results with scores and monotonicity
    """
    metric = get_distance_metric(metric_name)
    evaluator = MonotonicityEvaluator(
        extractor, reference_features, degradation_config
    )

    return evaluator.evaluate_all_degradations(eval_images, metric)


def compute_aggregate_monotonicity(
    monotonicity_scores: Dict[str, float],
) -> float:
    """
    Compute aggregate monotonicity score across all degradation types.

    Args:
        monotonicity_scores: Dict mapping degradation type to monotonicity score

    Returns:
        Mean monotonicity score
    """
    if not monotonicity_scores:
        return 0.0

    return np.mean(list(monotonicity_scores.values()))


class ComprehensiveEvaluator:
    """
    Runs comprehensive evaluation across multiple configurations.
    """

    def __init__(
        self,
        reference_images: List[str],
        eval_images: List[str],
    ):
        """
        Initialize comprehensive evaluator.

        Args:
            reference_images: List of reference image paths
            eval_images: List of evaluation image paths
        """
        self.reference_images = reference_images
        self.eval_images = eval_images

    def evaluate_configuration(
        self,
        backbone: str,
        layer_config: Dict,
        transform_config: Dict,
        metric_name: str,
    ) -> Dict:
        """
        Evaluate a single configuration.

        Args:
            backbone: Backbone model name
            layer_config: Layer configuration
            transform_config: Feature transform configuration
            metric_name: Distance metric name

        Returns:
            Evaluation results
        """
        logger.info(
            f"Evaluating: backbone={backbone}, layers={layer_config['name']}, "
            f"transform={transform_config['name']}, metric={metric_name}"
        )

        # Initialize extractor
        extractor = FeatureExtractor(
            backbone=backbone,
            layer_config=layer_config,
            transform_config=transform_config,
        )

        try:
            # Extract reference features
            reference_features = extractor.extract(
                self.reference_images, fit_transform=True
            )

            # Evaluate across degradations
            results = evaluate_metric_across_degradations(
                self.eval_images,
                extractor,
                reference_features,
                metric_name,
            )

            # Compute aggregate score
            monotonicity_scores = {
                deg_type: res["monotonicity"] for deg_type, res in results.items()
            }
            mean_monotonicity = compute_aggregate_monotonicity(monotonicity_scores)

            return {
                "backbone": backbone,
                "layers": layer_config["name"],
                "transform": transform_config["name"],
                "metric": metric_name,
                "degradation_results": results,
                "monotonicity_scores": monotonicity_scores,
                "mean_monotonicity": mean_monotonicity,
            }

        finally:
            extractor.cleanup()

    def run_full_evaluation(self) -> List[Dict]:
        """
        Run evaluation across all configurations specified in CONFIG.

        Returns:
            List of evaluation results for each configuration
        """
        results = []

        backbone = CONFIG["backbone"]

        for layer_config in CONFIG["layer_configs"]:
            for transform_config in CONFIG["feature_transforms"]:
                for metric_name in CONFIG["distance_metrics"]:
                    result = self.evaluate_configuration(
                        backbone,
                        layer_config,
                        transform_config,
                        metric_name,
                    )
                    results.append(result)

        return results
