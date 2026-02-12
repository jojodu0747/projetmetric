"""
Distance metrics module.
Implements FID and MMD (Maximum Mean Discrepancy) metrics for distribution comparison.
"""

import logging
from typing import Union, Optional

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


def compute_fid(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    max_dim: int = 10000,
) -> float:
    """
    Compute Fréchet Inception Distance (FID) between reference and test features.

    FID assumes both distributions are Gaussian and measures the distance between
    them using mean and covariance statistics:

    FID = ||μ_ref - μ_test||² + Tr(Σ_ref + Σ_test - 2(Σ_ref Σ_test)^{1/2})

    For very high-dimensional features (> max_dim), automatically applies PCA
    to reduce memory usage.

    Args:
        features_ref: Reference features (N_ref, D)
        features_test: Test features (N_test, D)
        max_dim: Maximum dimension before applying PCA reduction

    Returns:
        FID score (lower = more similar distributions)
    """
    n_ref, dim_orig = features_ref.shape
    n_test, _ = features_test.shape

    # If dimension is too high, apply PCA to make FID computation tractable
    if dim_orig > max_dim:
        logger.warning(
            f"Dimension {dim_orig} too high for FID covariance computation. "
            f"Applying PCA to reduce to {max_dim} dimensions."
        )
        from sklearn.decomposition import PCA
        pca_dim = min(max_dim, n_ref - 1, n_test - 1, dim_orig)
        pca = PCA(n_components=pca_dim)
        features_ref = pca.fit_transform(features_ref)
        features_test = pca.transform(features_test)
        logger.info(f"FID: reduced to {pca_dim} dimensions (explained variance: {pca.explained_variance_ratio_.sum():.4f})")

    # Compute statistics (use float32 to save memory)
    mu_ref = np.mean(features_ref, axis=0, dtype=np.float32)
    mu_test = np.mean(features_test, axis=0, dtype=np.float32)

    # Compute covariance matrices (memory-efficient)
    # Center the data first
    features_ref_centered = features_ref - mu_ref
    features_test_centered = features_test - mu_test

    # Cov = (1/(n-1)) * X^T @ X
    sigma_ref = (features_ref_centered.T @ features_ref_centered) / (n_ref - 1)
    sigma_test = (features_test_centered.T @ features_test_centered) / (n_test - 1)

    # Handle 1D case
    if sigma_ref.ndim == 0:
        sigma_ref = np.array([[sigma_ref]], dtype=np.float32)
    if sigma_test.ndim == 0:
        sigma_test = np.array([[sigma_test]], dtype=np.float32)

    # Add small regularization for numerical stability
    eps = 1e-6
    dim = sigma_ref.shape[0]
    sigma_ref = sigma_ref + np.eye(dim, dtype=np.float32) * eps
    sigma_test = sigma_test + np.eye(dim, dtype=np.float32) * eps

    # Compute squared difference of means
    diff = mu_ref - mu_test
    mean_diff_sq = np.sum(diff ** 2)

    # Compute sqrt of product of covariances: (Σ_ref Σ_test)^{1/2}
    # Use SVD for numerical stability
    try:
        covmean, _ = linalg.sqrtm(sigma_ref @ sigma_test, disp=False)

        # Check for imaginary components (numerical errors)
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                logger.warning("Imaginary component detected in covmean, taking real part")
            covmean = covmean.real
    except Exception as e:
        logger.warning(f"Matrix sqrt failed: {e}, using alternative computation")
        # Fallback: eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(sigma_ref @ sigma_test)
        eigvals = np.maximum(eigvals, 0)  # Ensure non-negative
        covmean = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # FID formula
    fid = mean_diff_sq + np.trace(sigma_ref + sigma_test - 2 * covmean)

    return float(fid)


def compute_mmd(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
    K_ref_ref_cached: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between reference and test features.

    MMD measures the distance between two distributions using kernel embeddings:
    MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]

    Args:
        features_ref: Reference features (N_ref, D)
        features_test: Test features (N_test, D)
        kernel: Kernel type ('rbf' or 'linear')
        gamma: RBF kernel bandwidth. If None, use median heuristic.
        K_ref_ref_cached: Pre-computed K_ref_ref matrix (optional, for speedup)

    Returns:
        MMD² score (lower = more similar distributions)
    """
    n_ref = features_ref.shape[0]
    n_test = features_test.shape[0]

    if kernel == "rbf":
        # Compute gamma using median heuristic if not provided
        if gamma is None:
            # Sample pairs to estimate median distance
            n_samples = min(1000, n_ref + n_test)
            combined = np.vstack([features_ref, features_test])
            indices = np.random.choice(combined.shape[0], n_samples, replace=False)
            sample = combined[indices]

            # Pairwise squared distances
            from scipy.spatial.distance import pdist
            pairwise_sq_dists = pdist(sample, metric="sqeuclidean")
            median_dist_sq = np.median(pairwise_sq_dists)
            gamma = 1.0 / (2 * median_dist_sq + 1e-8)
            logger.debug(f"MMD: using gamma={gamma:.6f} from median heuristic")

        # Compute RBF kernel matrices
        def rbf_kernel(X, Y):
            """RBF kernel: k(x,y) = exp(-gamma * ||x-y||²)"""
            # Compute pairwise squared distances
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)  # (N, 1)
            Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)  # (M, 1)
            sq_dists = X_sq + Y_sq.T - 2 * X @ Y.T  # (N, M)
            return np.exp(-gamma * sq_dists)

        # Use cached K_ref_ref if available (huge speedup!)
        if K_ref_ref_cached is not None:
            K_ref_ref = K_ref_ref_cached
        else:
            K_ref_ref = rbf_kernel(features_ref, features_ref)

        K_test_test = rbf_kernel(features_test, features_test)
        K_ref_test = rbf_kernel(features_ref, features_test)

    elif kernel == "linear":
        # Use cached K_ref_ref if available
        if K_ref_ref_cached is not None:
            K_ref_ref = K_ref_ref_cached
        else:
            K_ref_ref = features_ref @ features_ref.T

        K_test_test = features_test @ features_test.T
        K_ref_test = features_ref @ features_test.T
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # MMD² unbiased estimator (removing diagonal elements)
    # E[k(x,x')] where x ≠ x'
    term1 = (np.sum(K_ref_ref) - np.trace(K_ref_ref)) / (n_ref * (n_ref - 1))
    term2 = (np.sum(K_test_test) - np.trace(K_test_test)) / (n_test * (n_test - 1))
    term3 = 2 * np.sum(K_ref_test) / (n_ref * n_test)

    mmd_sq = term1 + term2 - term3

    # MMD² can be slightly negative due to finite sample bias, clip to 0
    mmd_sq = max(0, mmd_sq)

    return float(mmd_sq)


class DistanceMetric:
    """
    Wrapper class for distance metrics.
    Supports FID and MMD (Maximum Mean Discrepancy) distribution-free metrics.
    """

    def __init__(
        self,
        metric_name: str,
        distribution_model=None,  # Kept for backward compatibility but unused
        features_ref: Optional[np.ndarray] = None,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
    ):
        """
        Initialize distance metric.

        Args:
            metric_name: Name of the metric ('fid' or 'mmd')
            distribution_model: Unused (kept for backward compatibility)
            features_ref: Reference features (required for FID/MMD)
            kernel: Kernel type for MMD ('rbf' or 'linear')
            gamma: RBF kernel bandwidth for MMD (None = auto via median heuristic)
        """
        self.metric_name = metric_name.lower()
        self.distribution_model = distribution_model  # Unused
        self.features_ref = features_ref
        self.kernel = kernel
        self.gamma = gamma

        valid_metrics = ["fid", "mmd"]
        if self.metric_name not in valid_metrics:
            raise ValueError(f"Unknown metric: {metric_name}. "
                           f"Available: {', '.join(valid_metrics)}")

        # Validate requirements
        if features_ref is None:
            raise ValueError(f"{metric_name} requires features_ref")

        # Pre-compute and cache K_ref_ref for MMD (huge speedup for multiple evaluations)
        self.K_ref_ref_cached = None
        if self.metric_name == "mmd" and features_ref is not None:
            logger.info("Pre-computing reference kernel matrix for MMD (one-time cost)...")
            self._precompute_mmd_reference()

    def _precompute_mmd_reference(self):
        """Pre-compute reference kernel matrix K_ref_ref for MMD to avoid recomputation."""
        n_ref = self.features_ref.shape[0]

        # Compute gamma if not provided (median heuristic on reference data)
        if self.kernel == "rbf" and self.gamma is None:
            n_samples = min(1000, n_ref)
            indices = np.random.choice(n_ref, n_samples, replace=False)
            sample = self.features_ref[indices]

            from scipy.spatial.distance import pdist
            pairwise_sq_dists = pdist(sample, metric="sqeuclidean")
            median_dist_sq = np.median(pairwise_sq_dists)
            self.gamma = 1.0 / (2 * median_dist_sq + 1e-8)
            logger.info(f"MMD: using gamma={self.gamma:.6f} from median heuristic")

        # Compute K_ref_ref once
        if self.kernel == "rbf":
            X = self.features_ref
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)
            sq_dists = X_sq + X_sq.T - 2 * X @ X.T
            self.K_ref_ref_cached = np.exp(-self.gamma * sq_dists)
        elif self.kernel == "linear":
            self.K_ref_ref_cached = self.features_ref @ self.features_ref.T

        logger.info(f"Reference kernel matrix cached: shape {self.K_ref_ref_cached.shape}")

    def compute(self, features_test: np.ndarray) -> float:
        """
        Compute the distance metric for test features.

        Args:
            features_test: Test features (N, D)

        Returns:
            Distance score (higher = more different from reference)
        """
        if self.metric_name == "fid":
            return compute_fid(self.features_ref, features_test)
        elif self.metric_name == "mmd":
            return compute_mmd(self.features_ref, features_test,
                             kernel=self.kernel, gamma=self.gamma,
                             K_ref_ref_cached=self.K_ref_ref_cached)
        else:
            raise ValueError(f"Unknown metric: {self.metric_name}")

    def __call__(self, features_test: np.ndarray) -> float:
        """Alias for compute()."""
        return self.compute(features_test)


def get_distance_metric(
    name: str,
    distribution_model=None,  # Kept for backward compatibility
    features_ref: Optional[np.ndarray] = None,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
) -> DistanceMetric:
    """
    Factory function to get a distance metric by name.

    Args:
        name: Metric name ('fid' or 'mmd')
        distribution_model: Unused (kept for backward compatibility)
        features_ref: Reference features (required for FID/MMD)
        kernel: Kernel type for MMD ('rbf' or 'linear')
        gamma: RBF kernel bandwidth for MMD (None = auto via median heuristic)

    Returns:
        DistanceMetric instance
    """
    return DistanceMetric(
        name,
        distribution_model=distribution_model,
        features_ref=features_ref,
        kernel=kernel,
        gamma=gamma,
    )


def get_metric_score(metric_result: Union[float, tuple]) -> float:
    """
    Extract scalar score from metric result.

    Args:
        metric_result: Metric output (float)

    Returns:
        Scalar score
    """
    if isinstance(metric_result, tuple):
        return metric_result[0]
    return metric_result
