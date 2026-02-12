"""
Distance metrics module.
Implements various distribution distance metrics:
- FID (Fréchet Inception Distance)
- MMD (Maximum Mean Discrepancy)
- Sinkhorn Distance (Optimal Transport)
- Energy Distance
- Adversarial Distance (GAN-style critic)
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


def compute_sinkhorn(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    reg: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    Compute Sinkhorn distance (regularized optimal transport) between distributions.

    Uses the Sinkhorn-Knopp algorithm to approximate the Wasserstein distance
    with entropic regularization. More stable and faster than exact OT.

    Distance formula: W_ε(P, Q) = min_{π ∈ Π(P,Q)} <C, π> + ε·H(π)
    where C is the cost matrix, ε is regularization, H is entropy.

    Args:
        features_ref: Reference features (N_ref, D)
        features_test: Test features (N_test, D)
        reg: Entropic regularization parameter (higher = more regularized, faster)
        max_iter: Maximum number of Sinkhorn iterations
        tol: Convergence tolerance

    Returns:
        Sinkhorn distance (lower = more similar distributions)
    """
    n_ref = features_ref.shape[0]
    n_test = features_test.shape[0]

    # Uniform distributions (can be weighted if needed)
    a = np.ones(n_ref, dtype=np.float64) / n_ref
    b = np.ones(n_test, dtype=np.float64) / n_test

    # Compute cost matrix (squared Euclidean distances)
    # C[i,j] = ||x_i - y_j||²
    X_sq = np.sum(features_ref ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(features_test ** 2, axis=1, keepdims=True)
    C = X_sq + Y_sq.T - 2 * features_ref @ features_test.T
    C = np.maximum(C, 0)  # Numerical stability

    # Normalize cost matrix for numerical stability
    C_max = np.max(C)
    if C_max > 0:
        C = C / C_max
    else:
        return 0.0  # Identical distributions

    # Sinkhorn in log-domain for numerical stability
    # log(K) = -C/reg
    log_K = -C / reg
    u = np.zeros(n_ref, dtype=np.float64)
    v = np.zeros(n_test, dtype=np.float64)

    for iteration in range(max_iter):
        u_prev = u.copy()

        # u = log(a) - log_sum_exp(log_K + v)
        u = np.log(a + 1e-16) - np.log(np.sum(np.exp(log_K + v[None, :]), axis=1) + 1e-16)

        # v = log(b) - log_sum_exp(log_K.T + u)
        v = np.log(b + 1e-16) - np.log(np.sum(np.exp(log_K.T + u[None, :]), axis=1) + 1e-16)

        # Check convergence
        if np.max(np.abs(u - u_prev)) < tol:
            break

    # Compute optimal transport cost in log domain
    # P[i,j] = exp(u[i] + log_K[i,j] + v[j])
    log_P = u[:, None] + log_K + v[None, :]
    P = np.exp(log_P)

    # Sinkhorn distance (unnormalized)
    sinkhorn_dist = np.sum(P * C) * C_max  # Restore original scale

    return float(sinkhorn_dist)


def compute_energy_distance(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    sample_size: int = 1000,
) -> float:
    """
    Compute Energy Distance between two distributions.

    Energy distance is based on pairwise Euclidean distances:
    E(X, Y) = 2·E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]

    Properties:
    - Metric (satisfies triangle inequality)
    - Zero iff distributions are identical
    - Works in any dimension without kernel tuning

    For computational efficiency, uses Monte Carlo sampling when n > sample_size.

    Args:
        features_ref: Reference features (N_ref, D)
        features_test: Test features (N_test, D)
        sample_size: Max samples for distance computation (for speed)

    Returns:
        Energy distance (lower = more similar distributions)
    """
    from scipy.spatial.distance import cdist

    n_ref = features_ref.shape[0]
    n_test = features_test.shape[0]

    # Sample if datasets are too large
    if n_ref > sample_size:
        indices = np.random.choice(n_ref, sample_size, replace=False)
        X = features_ref[indices]
    else:
        X = features_ref

    if n_test > sample_size:
        indices = np.random.choice(n_test, sample_size, replace=False)
        Y = features_test[indices]
    else:
        Y = features_test

    # Compute pairwise distances
    # E[||X - Y||]
    dists_XY = cdist(X, Y, metric='euclidean')
    term1 = 2 * np.mean(dists_XY)

    # E[||X - X'||]
    dists_XX = cdist(X, X, metric='euclidean')
    # Remove diagonal (distance to self = 0)
    np.fill_diagonal(dists_XX, 0)
    term2 = np.sum(dists_XX) / (len(X) * (len(X) - 1))

    # E[||Y - Y'||]
    dists_YY = cdist(Y, Y, metric='euclidean')
    np.fill_diagonal(dists_YY, 0)
    term3 = np.sum(dists_YY) / (len(Y) * (len(Y) - 1))

    energy_dist = term1 - term2 - term3

    return float(energy_dist)


def compute_adversarial_distance(
    features_ref: np.ndarray,
    features_test: np.ndarray,
    n_critics: int = 5,
    max_iter: int = 100,
    learning_rate: float = 0.01,
    hidden_dim: int = 128,
) -> float:
    """
    Compute Adversarial Distance using a learned critic (GAN-style discriminator).

    Trains a neural network critic to distinguish between reference and test
    distributions. The critic's loss approximates the Wasserstein distance.

    Based on WGAN (Wasserstein GAN) critic:
    W(P, Q) ≈ max_f E_P[f(x)] - E_Q[f(y)]  where ||f||_L ≤ 1

    We train a simple 2-layer MLP critic with gradient clipping (Lipschitz constraint).

    Args:
        features_ref: Reference features (N_ref, D)
        features_test: Test features (N_test, D)
        n_critics: Number of critic training restarts (ensemble for stability)
        max_iter: Maximum training iterations per critic
        learning_rate: Learning rate for critic training
        hidden_dim: Hidden layer size for critic network

    Returns:
        Adversarial distance (higher = more different distributions)
    """
    n_ref = features_ref.shape[0]
    n_test = features_test.shape[0]
    d = features_ref.shape[1]

    # Normalize features for training stability
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(features_ref)
    Y = scaler.transform(features_test)

    # Train multiple critics and average (reduce variance)
    critic_scores = []

    for _ in range(n_critics):
        # Initialize simple 2-layer critic: f(x) = W2·ReLU(W1·x + b1) + b2
        np.random.seed(None)  # Different initialization each time
        W1 = np.random.randn(d, hidden_dim) * 0.01
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(hidden_dim, 1) * 0.01
        b2 = np.zeros(1)

        # Training loop
        for iteration in range(max_iter):
            # Sample mini-batches
            batch_size = min(32, n_ref, n_test)
            idx_ref = np.random.choice(n_ref, batch_size, replace=False)
            idx_test = np.random.choice(n_test, batch_size, replace=False)
            X_batch = X[idx_ref]
            Y_batch = Y[idx_test]

            # Forward pass
            def critic_forward(x, W1, b1, W2, b2):
                h = np.maximum(0, x @ W1 + b1)  # ReLU
                return h @ W2 + b2

            scores_ref = critic_forward(X_batch, W1, b1, W2, b2)
            scores_test = critic_forward(Y_batch, W1, b1, W2, b2)

            # Wasserstein loss: max E[f(X)] - E[f(Y)]
            # We minimize: -E[f(X)] + E[f(Y)]
            loss = -np.mean(scores_ref) + np.mean(scores_test)

            # Backward pass (simple gradient descent)
            # ∂loss/∂scores_ref = -1/batch_size
            # ∂loss/∂scores_test = +1/batch_size
            grad_scores_ref = -np.ones_like(scores_ref) / batch_size
            grad_scores_test = np.ones_like(scores_test) / batch_size

            # Backprop through network (chain rule)
            grad_W2 = np.zeros_like(W2)
            grad_b2 = np.zeros_like(b2)

            # Ref branch
            h_ref = np.maximum(0, X_batch @ W1 + b1)
            grad_W2 += h_ref.T @ grad_scores_ref
            grad_b2 += np.sum(grad_scores_ref, axis=0)
            grad_h_ref = grad_scores_ref @ W2.T
            grad_h_ref[h_ref <= 0] = 0  # ReLU gradient
            grad_W1_ref = X_batch.T @ grad_h_ref
            grad_b1_ref = np.sum(grad_h_ref, axis=0)

            # Test branch
            h_test = np.maximum(0, Y_batch @ W1 + b1)
            grad_W2 += h_test.T @ grad_scores_test
            grad_b2 += np.sum(grad_scores_test, axis=0)
            grad_h_test = grad_scores_test @ W2.T
            grad_h_test[h_test <= 0] = 0
            grad_W1_test = Y_batch.T @ grad_h_test
            grad_b1_test = np.sum(grad_h_test, axis=0)

            grad_W1 = grad_W1_ref + grad_W1_test
            grad_b1 = grad_b1_ref + grad_b1_test

            # Update weights
            W1 -= learning_rate * grad_W1
            b1 -= learning_rate * grad_b1
            W2 -= learning_rate * grad_W2
            b2 -= learning_rate * grad_b2

            # Weight clipping for Lipschitz constraint (simple version)
            clip_value = 0.01
            W1 = np.clip(W1, -clip_value, clip_value)
            W2 = np.clip(W2, -clip_value, clip_value)

        # Evaluate final critic on full data
        scores_ref_full = critic_forward(X, W1, b1, W2, b2)
        scores_test_full = critic_forward(Y, W1, b1, W2, b2)
        wasserstein_approx = np.mean(scores_ref_full) - np.mean(scores_test_full)
        critic_scores.append(wasserstein_approx)

    # Average over multiple critics
    adversarial_dist = np.mean(critic_scores)

    # Return absolute value (distance should be positive)
    return float(abs(adversarial_dist))


class DistanceMetric:
    """
    Wrapper class for distance metrics.
    Supports: FID, MMD, Sinkhorn, Energy Distance, Adversarial Distance.
    """

    def __init__(
        self,
        metric_name: str,
        distribution_model=None,  # Kept for backward compatibility but unused
        features_ref: Optional[np.ndarray] = None,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        sinkhorn_reg: float = 0.1,
        sinkhorn_max_iter: int = 100,
        energy_sample_size: int = 1000,
        adversarial_n_critics: int = 5,
        adversarial_max_iter: int = 100,
    ):
        """
        Initialize distance metric.

        Args:
            metric_name: Name of metric ('fid', 'mmd', 'sinkhorn', 'energy', 'adversarial')
            distribution_model: Unused (kept for backward compatibility)
            features_ref: Reference features (required for all metrics)
            kernel: Kernel type for MMD ('rbf' or 'linear')
            gamma: RBF kernel bandwidth for MMD (None = auto via median heuristic)
            sinkhorn_reg: Entropic regularization for Sinkhorn distance
            sinkhorn_max_iter: Max iterations for Sinkhorn algorithm
            energy_sample_size: Max samples for Energy distance computation
            adversarial_n_critics: Number of critics for adversarial distance
            adversarial_max_iter: Training iterations for adversarial critic
        """
        self.metric_name = metric_name.lower()
        self.distribution_model = distribution_model  # Unused
        self.features_ref = features_ref
        self.kernel = kernel
        self.gamma = gamma
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.energy_sample_size = energy_sample_size
        self.adversarial_n_critics = adversarial_n_critics
        self.adversarial_max_iter = adversarial_max_iter

        valid_metrics = ["fid", "mmd", "sinkhorn", "energy", "adversarial"]
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
            logger.info(f"MMD: median_dist_sq={median_dist_sq:.2e}, gamma={self.gamma:.10e} (1/gamma={1/self.gamma:.2e})")

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
            return compute_mmd(
                self.features_ref, features_test,
                kernel=self.kernel, gamma=self.gamma,
                K_ref_ref_cached=self.K_ref_ref_cached
            )

        elif self.metric_name == "sinkhorn":
            return compute_sinkhorn(
                self.features_ref, features_test,
                reg=self.sinkhorn_reg,
                max_iter=self.sinkhorn_max_iter
            )

        elif self.metric_name == "energy":
            return compute_energy_distance(
                self.features_ref, features_test,
                sample_size=self.energy_sample_size
            )

        elif self.metric_name == "adversarial":
            return compute_adversarial_distance(
                self.features_ref, features_test,
                n_critics=self.adversarial_n_critics,
                max_iter=self.adversarial_max_iter
            )

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
    sinkhorn_reg: float = 0.1,
    sinkhorn_max_iter: int = 100,
    energy_sample_size: int = 1000,
    adversarial_n_critics: int = 5,
    adversarial_max_iter: int = 100,
) -> DistanceMetric:
    """
    Factory function to get a distance metric by name.

    Args:
        name: Metric name ('fid', 'mmd', 'sinkhorn', 'energy', 'adversarial')
        distribution_model: Unused (kept for backward compatibility)
        features_ref: Reference features (required for all metrics)
        kernel: Kernel type for MMD ('rbf' or 'linear')
        gamma: RBF kernel bandwidth for MMD (None = auto via median heuristic)
        sinkhorn_reg: Entropic regularization for Sinkhorn (lower = closer to true OT)
        sinkhorn_max_iter: Max Sinkhorn iterations
        energy_sample_size: Sample size for Energy distance
        adversarial_n_critics: Number of critics for adversarial distance
        adversarial_max_iter: Critic training iterations

    Returns:
        DistanceMetric instance
    """
    return DistanceMetric(
        name,
        distribution_model=distribution_model,
        features_ref=features_ref,
        kernel=kernel,
        gamma=gamma,
        sinkhorn_reg=sinkhorn_reg,
        sinkhorn_max_iter=sinkhorn_max_iter,
        energy_sample_size=energy_sample_size,
        adversarial_n_critics=adversarial_n_critics,
        adversarial_max_iter=adversarial_max_iter,
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
