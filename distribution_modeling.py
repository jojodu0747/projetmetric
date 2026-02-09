"""
Distribution modeling module for feature space analysis.
Supports Gaussian Mixture Models (GMM) and Kernel Density Estimation (KDE).
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from config import CONFIG

logger = logging.getLogger(__name__)


class DistributionModel:
    """
    Base class for distribution modeling in feature space.

    Supports:
    - GMM with diagonal or full covariance
    - KDE with automatic bandwidth selection
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the distribution model.

        Args:
            config: Model configuration dict with keys:
                - type: 'gmm' or 'kde'
                - For GMM: covariance_type ('diag', 'full'), n_components
                - For KDE: bandwidth ('auto' or float)
        """
        self.config = config or CONFIG["distribution_models"][0]
        self.model_type = self.config.get("type", "gmm")
        self.model = None
        self._fitted = False

        # Cached statistics
        self._mean = None
        self._cov = None
        self._features = None  # Store features for non-parametric access

    def fit(self, features: np.ndarray) -> "DistributionModel":
        """
        Fit the distribution model to feature data.

        Args:
            features: Feature array (N, D)

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting {self.model_type} model on features with shape {features.shape}")

        # Store features for later access
        self._features = features

        # Compute basic statistics
        self._mean = np.mean(features, axis=0)
        self._cov = np.cov(features, rowvar=False)

        # Handle single-dimensional case
        if self._cov.ndim == 0:
            self._cov = np.array([[self._cov]])

        if self.model_type == "gmm":
            self._fit_gmm(features)
        elif self.model_type == "kde":
            self._fit_kde(features)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self._fitted = True
        return self

    def _fit_gmm(self, features: np.ndarray):
        """Fit a Gaussian Mixture Model."""
        n_components = self.config.get("n_components", 5)
        covariance_type = self.config.get("covariance_type", "diag")

        # Adjust n_components if we have few samples
        n_components = min(n_components, features.shape[0] // 2)
        n_components = max(1, n_components)

        # For full covariance, need enough samples per component
        n_samples, n_features = features.shape
        if covariance_type == "full":
            # Need at least n_features+1 samples per component for stable full covariance
            max_components_full = max(1, n_samples // (n_features + 1))
            if n_components > max_components_full:
                n_components = max_components_full
                logger.warning(
                    f"Reduced to {n_components} components for full covariance "
                    f"({n_samples} samples, {n_features} features)"
                )

        logger.info(f"Fitting GMM with {n_components} components, "
                   f"covariance_type={covariance_type}")

        # Use regularization to avoid singular covariance matrices
        reg_covar = 1e-2 if covariance_type == "full" else 1e-6

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=200,
            n_init=3,
            reg_covar=reg_covar,
            random_state=CONFIG["random_seed"],
        )

        try:
            self.model.fit(features)
        except ValueError as e:
            logger.warning(f"GMM full covariance failed: {e}. Falling back to diag.")
            self.model = GaussianMixture(
                n_components=n_components,
                covariance_type="diag",
                max_iter=200,
                n_init=3,
                reg_covar=1e-6,
                random_state=CONFIG["random_seed"],
            )
            self.model.fit(features)

        # Log convergence info
        if not self.model.converged_:
            logger.warning("GMM did not converge")
        logger.info(f"GMM BIC: {self.model.bic(features):.2f}, "
                   f"AIC: {self.model.aic(features):.2f}")

    def _fit_kde(self, features: np.ndarray):
        """Fit a Kernel Density Estimation model."""
        bandwidth = self.config.get("bandwidth", "auto")

        if bandwidth == "auto":
            # Use cross-validation to select bandwidth
            logger.info("Selecting KDE bandwidth via cross-validation")
            bandwidths = np.logspace(-2, 1, 20)

            # Subsample for faster CV if dataset is large
            if features.shape[0] > 500:
                indices = np.random.choice(features.shape[0], 500, replace=False)
                cv_features = features[indices]
            else:
                cv_features = features

            grid = GridSearchCV(
                KernelDensity(kernel="gaussian"),
                {"bandwidth": bandwidths},
                cv=5,
                n_jobs=-1,
            )
            grid.fit(cv_features)
            bandwidth = grid.best_params_["bandwidth"]
            logger.info(f"Selected bandwidth: {bandwidth:.4f}")

        self.model = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        self.model.fit(features)

    def get_params(self) -> Dict:
        """
        Get the parameters of the fitted distribution.

        Returns:
            Dict containing model parameters
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        params = {
            "type": self.model_type,
            "mean": self._mean,
            "covariance": self._cov,
        }

        if self.model_type == "gmm":
            params.update({
                "n_components": self.model.n_components,
                "covariance_type": self.model.covariance_type,
                "weights": self.model.weights_,
                "means": self.model.means_,
                "covariances": self.model.covariances_,
            })
        elif self.model_type == "kde":
            params.update({
                "bandwidth": self.model.bandwidth,
                "kernel": self.model.kernel,
            })

        return params

    def get_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mean and covariance of the fitted distribution.

        Returns:
            Tuple of (mean, covariance)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._mean, self._cov

    def get_gmm_statistics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get GMM-specific statistics (component means and covariances).

        Returns:
            Tuple of (weights, means, covariances)
        """
        if self.model_type != "gmm":
            raise ValueError("This method is only for GMM models")
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return (
            self.model.weights_,
            self.model.means_,
            self.model.covariances_,
        )

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood of samples under the model.

        Args:
            features: Feature array (N, D)

        Returns:
            Log-likelihood scores (N,)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.model_type == "gmm":
            return self.model.score_samples(features)
        elif self.model_type == "kde":
            return self.model.score_samples(features)

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate samples from the fitted distribution.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Sampled features (n_samples, D)
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.model_type == "gmm":
            samples, _ = self.model.sample(n_samples)
        elif self.model_type == "kde":
            samples = self.model.sample(n_samples)

        return samples

    def get_features(self) -> np.ndarray:
        """
        Get the original features used to fit the model.

        Returns:
            Feature array (N, D)
        """
        if self._features is None:
            raise RuntimeError("No features stored. Call fit() first.")
        return self._features


class MultiDistributionModel:
    """
    Manages multiple distribution models for ensemble analysis.
    """

    def __init__(self, configs: List[Dict] = None):
        """
        Initialize multiple distribution models.

        Args:
            configs: List of model configurations
        """
        self.configs = configs or CONFIG["distribution_models"]
        self.models = {}

    def fit(self, features: np.ndarray) -> "MultiDistributionModel":
        """
        Fit all distribution models to the same feature data.

        Args:
            features: Feature array (N, D)

        Returns:
            Self for chaining
        """
        for config in self.configs:
            name = config.get("name", f"{config['type']}_{id(config)}")
            logger.info(f"Fitting distribution model: {name}")

            model = DistributionModel(config)
            model.fit(features)
            self.models[name] = model

        return self

    def get_model(self, name: str) -> DistributionModel:
        """Get a specific model by name."""
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        return self.models[name]

    def get_all_models(self) -> Dict[str, DistributionModel]:
        """Get all fitted models."""
        return self.models


def compute_empirical_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance directly from features.

    Args:
        features: Feature array (N, D)

    Returns:
        Tuple of (mean, covariance)
    """
    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)

    # Handle edge cases
    if cov.ndim == 0:
        cov = np.array([[cov]])

    # Regularize covariance if singular
    min_eig = np.linalg.eigvalsh(cov).min()
    if min_eig < 1e-10:
        cov += np.eye(cov.shape[0]) * 1e-6

    return mean, cov


def fit_optimal_gmm(
    features: np.ndarray,
    max_components: int = 10,
    covariance_type: str = "diag",
) -> GaussianMixture:
    """
    Fit GMM with automatic selection of number of components using BIC.

    Args:
        features: Feature array (N, D)
        max_components: Maximum number of components to try
        covariance_type: Type of covariance ('diag', 'full', 'spherical', 'tied')

    Returns:
        Fitted GaussianMixture model
    """
    max_components = min(max_components, features.shape[0] // 2)

    best_gmm = None
    best_bic = np.inf

    for n in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                max_iter=200,
                n_init=3,
                random_state=CONFIG["random_seed"],
            )
            gmm.fit(features)
            bic = gmm.bic(features)

            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        except Exception as e:
            logger.warning(f"GMM with {n} components failed: {e}")
            continue

    if best_gmm is None:
        raise RuntimeError("Could not fit any GMM model")

    logger.info(f"Optimal GMM: {best_gmm.n_components} components, BIC={best_bic:.2f}")
    return best_gmm
