"""
Distance metrics module for comparing feature distributions.
Implements FID, KID, and CMMD (MMD with RBF kernel).
"""

import logging
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import linalg

from config import CONFIG

logger = logging.getLogger(__name__)


def compute_fid(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Fréchet Inception Distance (FID) between two feature sets.

    FID = ||mu1 - mu2||^2 + Tr(Sigma1 + Sigma2 - 2*sqrt(Sigma1 @ Sigma2))

    Args:
        features_ref: Reference features (N1, D)
        features_test: Test features (N2, D)
        eps: Small constant for numerical stability

    Returns:
        FID score (lower is better, 0 means identical distributions)
    """
    # Compute statistics
    mu1 = np.mean(features_ref, axis=0)
    mu2 = np.mean(features_test, axis=0)

    sigma1 = np.cov(features_ref, rowvar=False)
    sigma2 = np.cov(features_test, rowvar=False)

    # Handle 1D case
    if sigma1.ndim == 0:
        sigma1 = np.array([[sigma1]])
        sigma2 = np.array([[sigma2]])

    # Compute squared difference of means
    diff = mu1 - mu2
    mean_diff_sq = np.sum(diff ** 2)

    # Compute sqrt(Sigma1 @ Sigma2) using matrix square root
    # Add small regularization for numerical stability
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    try:
        # Compute matrix square root
        covmean = linalg.sqrtm(sigma1 @ sigma2)

        # Handle numerical issues - sqrtm may return complex numbers
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                logger.warning("Complex values in sqrtm computation")
            covmean = covmean.real

        # Compute trace term
        trace_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

        # FID score
        fid = mean_diff_sq + trace_term

        # Ensure non-negative (numerical errors can cause small negatives)
        fid = max(0, fid)

    except Exception as e:
        logger.error(f"Error computing FID: {e}")
        # Fallback to simpler approximation
        fid = mean_diff_sq + np.trace(sigma1) + np.trace(sigma2)

    return float(fid)


def compute_fid_from_statistics(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute FID directly from pre-computed statistics.

    Args:
        mu1: Mean of reference distribution (D,)
        sigma1: Covariance of reference distribution (D, D)
        mu2: Mean of test distribution (D,)
        sigma2: Covariance of test distribution (D, D)
        eps: Small constant for numerical stability

    Returns:
        FID score
    """
    diff = mu1 - mu2
    mean_diff_sq = np.sum(diff ** 2)

    # Regularize
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    try:
        covmean = linalg.sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        trace_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        fid = mean_diff_sq + trace_term
        fid = max(0, fid)

    except Exception:
        fid = mean_diff_sq + np.trace(sigma1) + np.trace(sigma2)

    return float(fid)


def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, c: float = 1.0) -> np.ndarray:
    """
    Compute polynomial kernel: k(x, y) = (x.y / d + c)^degree

    Args:
        x: First set of features (N1, D)
        y: Second set of features (N2, D)
        degree: Polynomial degree
        c: Constant term

    Returns:
        Kernel matrix (N1, N2)
    """
    d = x.shape[1]
    return (x @ y.T / d + c) ** degree


def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = None) -> np.ndarray:
    """
    Compute RBF (Gaussian) kernel: k(x, y) = exp(-||x-y||^2 / (2*sigma^2))

    Args:
        x: First set of features (N1, D)
        y: Second set of features (N2, D)
        sigma: Bandwidth parameter (auto-computed if None)

    Returns:
        Kernel matrix (N1, N2)
    """
    # Compute pairwise squared distances
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
    x_sqnorms = np.sum(x ** 2, axis=1, keepdims=True)  # (N1, 1)
    y_sqnorms = np.sum(y ** 2, axis=1, keepdims=True)  # (N2, 1)
    sq_dists = x_sqnorms + y_sqnorms.T - 2 * x @ y.T  # (N1, N2)
    sq_dists = np.maximum(sq_dists, 0)  # Numerical stability

    # Auto-compute sigma using median heuristic if not provided
    if sigma is None:
        # Use median of distances as bandwidth
        median_dist = np.median(np.sqrt(sq_dists + 1e-10))
        sigma = median_dist if median_dist > 0 else 1.0

    return np.exp(-sq_dists / (2 * sigma ** 2))


def compute_mmd_squared(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    kernel_fn: Callable = None,
    biased: bool = False,
) -> float:
    """
    Compute Maximum Mean Discrepancy squared.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    Args:
        features_ref: Reference features (N1, D)
        features_test: Test features (N2, D)
        kernel_fn: Kernel function (default: polynomial)
        biased: Whether to use biased estimator

    Returns:
        MMD^2 score
    """
    if kernel_fn is None:
        kernel_fn = polynomial_kernel

    n1 = features_ref.shape[0]
    n2 = features_test.shape[0]

    # Compute kernel matrices
    K_xx = kernel_fn(features_ref, features_ref)
    K_yy = kernel_fn(features_test, features_test)
    K_xy = kernel_fn(features_ref, features_test)

    if biased:
        # Biased estimator (includes diagonal)
        mmd2 = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
    else:
        # Unbiased estimator (excludes diagonal for K_xx and K_yy)
        # Sum of off-diagonal elements
        sum_xx = np.sum(K_xx) - np.trace(K_xx)
        sum_yy = np.sum(K_yy) - np.trace(K_yy)
        sum_xy = np.sum(K_xy)

        mmd2 = (
            sum_xx / (n1 * (n1 - 1))
            + sum_yy / (n2 * (n2 - 1))
            - 2 * sum_xy / (n1 * n2)
        )

    return float(max(0, mmd2))


def compute_kid(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    num_subsets: int = None,
    subset_size: int = None,
) -> Tuple[float, float]:
    """
    Compute Kernel Inception Distance (KID).

    KID uses polynomial kernel and computes MMD over multiple subsets
    to estimate mean and variance.

    Args:
        features_ref: Reference features (N1, D)
        features_test: Test features (N2, D)
        num_subsets: Number of subsets for variance estimation
        subset_size: Size of each subset

    Returns:
        Tuple of (mean_kid, std_kid)
    """
    num_subsets = num_subsets or CONFIG.get("kid_subsets", 100)
    subset_size = subset_size or CONFIG.get("kid_subset_size", 50)

    # Adjust subset size if necessary
    subset_size = min(subset_size, features_ref.shape[0], features_test.shape[0])

    n1, n2 = features_ref.shape[0], features_test.shape[0]
    scores = []

    rng = np.random.RandomState(CONFIG["random_seed"])

    for _ in range(num_subsets):
        # Random subsets
        idx1 = rng.choice(n1, subset_size, replace=False)
        idx2 = rng.choice(n2, subset_size, replace=False)

        subset_ref = features_ref[idx1]
        subset_test = features_test[idx2]

        # Compute MMD with polynomial kernel
        mmd2 = compute_mmd_squared(subset_ref, subset_test, polynomial_kernel, biased=False)
        scores.append(mmd2)

    mean_kid = np.mean(scores)
    std_kid = np.std(scores)

    return float(mean_kid), float(std_kid)


def compute_cmmd(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    sigma: float = None,
) -> float:
    """
    Compute CMMD (MMD with RBF kernel).

    Uses median heuristic for bandwidth selection if sigma is not provided.

    Args:
        features_ref: Reference features (N1, D)
        features_test: Test features (N2, D)
        sigma: RBF bandwidth (auto-computed if None)

    Returns:
        CMMD score
    """
    # Auto-compute sigma using median heuristic
    if sigma is None:
        # Sample subset for bandwidth estimation if large
        if features_ref.shape[0] > 1000:
            idx = np.random.choice(features_ref.shape[0], 1000, replace=False)
            sample = features_ref[idx]
        else:
            sample = features_ref

        # Compute pairwise distances
        dists = np.sqrt(
            np.sum(sample[:, None, :] ** 2, axis=2)
            + np.sum(sample[None, :, :] ** 2, axis=2)
            - 2 * sample @ sample.T
        )
        sigma = np.median(dists[dists > 0])
        if sigma == 0:
            sigma = 1.0

    def rbf_kernel_with_sigma(x, y):
        return rbf_kernel(x, y, sigma=sigma)

    mmd2 = compute_mmd_squared(features_ref, features_test, rbf_kernel_with_sigma, biased=False)
    return float(mmd2)


class DistanceMetric:
    """
    Wrapper class for distance metrics with consistent interface.
    """

    def __init__(self, metric_name: str):
        """
        Initialize distance metric.

        Args:
            metric_name: Name of the metric ('fid', 'kid', 'cmmd')
        """
        self.metric_name = metric_name.lower()

        if self.metric_name not in ["fid", "kid", "cmmd"]:
            raise ValueError(f"Unknown metric: {metric_name}. "
                           f"Available: fid, kid, cmmd")

    def compute(
        self,
        features_ref: np.ndarray,
        features_test: np.ndarray,
    ) -> Union[float, Tuple[float, float]]:
        """
        Compute the distance metric between two feature sets.

        Args:
            features_ref: Reference features (N1, D)
            features_test: Test features (N2, D)

        Returns:
            Distance score (or tuple for KID with variance)
        """
        if self.metric_name == "fid":
            return compute_fid(features_ref, features_test)
        elif self.metric_name == "kid":
            return compute_kid(features_ref, features_test)
        elif self.metric_name == "cmmd":
            return compute_cmmd(features_ref, features_test)

    def __call__(
        self,
        features_ref: np.ndarray,
        features_test: np.ndarray,
    ) -> Union[float, Tuple[float, float]]:
        """Alias for compute()."""
        return self.compute(features_ref, features_test)


def get_distance_metric(name: str) -> DistanceMetric:
    """
    Factory function to get a distance metric by name.

    Args:
        name: Metric name ('fid', 'kid', 'cmmd')

    Returns:
        DistanceMetric instance
    """
    return DistanceMetric(name)


def get_metric_score(metric_result: Union[float, Tuple[float, float]]) -> float:
    """
    Extract scalar score from metric result (handles KID tuple).

    Args:
        metric_result: Metric output (float or tuple)

    Returns:
        Scalar score
    """
    if isinstance(metric_result, tuple):
        return metric_result[0]  # Mean for KID
    return metric_result


def compute_all_metrics(
    features_ref: np.ndarray,
    features_test: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all available distance metrics.

    Args:
        features_ref: Reference features (N1, D)
        features_test: Test features (N2, D)

    Returns:
        Dict mapping metric names to scores
    """
    results = {}

    # FID
    results["fid"] = compute_fid(features_ref, features_test)

    # KID
    kid_mean, kid_std = compute_kid(features_ref, features_test)
    results["kid"] = kid_mean
    results["kid_std"] = kid_std

    # CMMD
    results["cmmd"] = compute_cmmd(features_ref, features_test)

    return results
