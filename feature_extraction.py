"""
Feature extraction module using pre-trained CNN backbones.
Supports ResNet50, VGG19, and DinoV2 with configurable layer extraction.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from config import CONFIG, BACKBONE_CONFIGS, get_backbone_config

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """Simple dataset for loading images."""

    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                return torch.zeros(3, 224, 224), img_path
            return Image.new("RGB", (224, 224)), img_path


class FeatureHook:
    """Hook to capture intermediate layer activations."""

    def __init__(self):
        self.features = None

    def __call__(self, module, input, output):
        # Handle different output types
        if isinstance(output, torch.Tensor):
            self.features = output.detach()
        elif isinstance(output, tuple):
            self.features = output[0].detach()
        else:
            self.features = output

    def clear(self):
        self.features = None


class FeatureExtractor:
    """
    Extracts features from images using pre-trained CNN backbones.

    Supports:
    - Multiple backbone architectures (ResNet50, VGG19, DinoV2)
    - Configurable layer extraction
    - Gram matrix computation
    - PCA dimensionality reduction
    """

    def __init__(
        self,
        backbone: str = None,
        layer_config: Dict = None,
        transform_config: Dict = None,
        device: str = None,
    ):
        """
        Initialize the feature extractor.

        Args:
            backbone: Name of the backbone model
            layer_config: Configuration specifying which layers to extract
            transform_config: Configuration for feature transformation (gram, pca)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.backbone_name = backbone or CONFIG["backbone"]
        self.layer_config = layer_config or CONFIG["layer_configs"][0]
        self.transform_config = transform_config or CONFIG["feature_transforms"][0]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone_config = get_backbone_config(self.backbone_name)

        # Initialize model and hooks
        self.model = None
        self.hooks = {}
        self.hook_handles = []

        # PCA model (fitted on reference data)
        self.pca = None
        self.pca_fitted = False

        # Load model
        self._load_model()
        self._register_hooks()

        # Create image transform
        self.image_transform = self._create_transform()

    def _load_model(self):
        """Load the pre-trained backbone model."""
        logger.info(f"Loading backbone: {self.backbone_name}")

        if self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.model = models.resnet50(weights=weights)

        elif self.backbone_name == "vgg19":
            weights = models.VGG19_Weights.IMAGENET1K_V1
            self.model = models.vgg19(weights=weights)

        elif self.backbone_name == "dinov2_vitb14":
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True
            )
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def _get_layer_by_name(self, name: str) -> nn.Module:
        """Get a layer from the model by its name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        layer_names = self.backbone_config["layer_names"]
        layers_to_extract = self.layer_config["layers"]

        for layer_idx in layers_to_extract:
            if layer_idx not in layer_names:
                logger.warning(f"Layer {layer_idx} not found in backbone config")
                continue

            layer_name = layer_names[layer_idx]
            try:
                layer = self._get_layer_by_name(layer_name)
                hook = FeatureHook()
                handle = layer.register_forward_hook(hook)
                self.hooks[layer_idx] = hook
                self.hook_handles.append(handle)
                logger.debug(f"Registered hook on layer {layer_idx}: {layer_name}")
            except Exception as e:
                logger.warning(f"Could not register hook on {layer_name}: {e}")

    def _create_transform(self) -> transforms.Compose:
        """Create the image preprocessing transform."""
        input_size = self.backbone_config["input_size"]
        mean = self.backbone_config["normalize_mean"]
        std = self.backbone_config["normalize_std"]

        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def _process_features(self, features: torch.Tensor) -> np.ndarray:
        """
        Process raw layer activations into a fixed-size vector.

        Args:
            features: Raw tensor from hook (B, C, H, W) or (B, N, D)

        Returns:
            Processed feature vector (B, D)
        """
        # Move to CPU and convert to numpy
        features = features.cpu()

        # Handle 4D CNN features (B, C, H, W)
        if features.dim() == 4:
            # Global average pooling over spatial dimensions
            features = features.mean(dim=[2, 3])  # (B, C)

        # Handle 3D transformer features (B, N, D) - DinoV2 patch tokens
        elif features.dim() == 3:
            # Average over all patch tokens
            features = features.mean(dim=1)  # (B, D)

        # Handle 2D features (B, D) - already processed
        elif features.dim() == 2:
            pass
        else:
            raise ValueError(f"Unexpected feature dimension: {features.dim()}")

        return features.numpy()

    def _aggregate_features(self, features_list: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate features from multiple layers.

        Args:
            features_list: List of feature arrays from different layers

        Returns:
            Aggregated feature array
        """
        if len(features_list) == 1:
            return features_list[0]

        # Normalize each feature set to unit norm before concatenation
        normalized = []
        for f in features_list:
            norms = np.linalg.norm(f, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            normalized.append(f / norms)

        # Concatenate along feature dimension
        aggregated = np.concatenate(normalized, axis=1)

        # Alternative: average (uncomment if preferred)
        # aggregated = np.mean(features_list, axis=0)

        return aggregated

    def _compute_gram_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute Gram matrix for style representation.

        Args:
            features: Feature array (B, D)

        Returns:
            Upper triangular elements of Gram matrix (B, D*(D+1)/2)
        """
        batch_size, dim = features.shape

        # For very high dimensions, subsample to avoid memory issues
        max_dim = 1000
        if dim > max_dim:
            logger.debug(f"Subsampling features from {dim} to {max_dim}")
            indices = np.random.choice(dim, max_dim, replace=False)
            features = features[:, indices]
            dim = max_dim

        gram_features = []
        for i in range(batch_size):
            f = features[i:i+1].T  # (D, 1)
            gram = f @ f.T  # (D, D)
            # Extract upper triangular (including diagonal)
            upper_tri = gram[np.triu_indices(dim)]
            gram_features.append(upper_tri)

        return np.array(gram_features)

    def _apply_pca(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Apply PCA dimensionality reduction.

        Args:
            features: Feature array (B, D)
            fit: Whether to fit the PCA model

        Returns:
            Reduced feature array (B, pca_dim)
        """
        pca_dim = self.transform_config.get("pca_dim", 10)

        if fit or not self.pca_fitted:
            logger.info(f"Fitting PCA with {pca_dim} components on {features.shape}")
            n_components = min(pca_dim, features.shape[0], features.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(features)
            self.pca_fitted = True
            logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")

        return self.pca.transform(features)

    def transform_features(
        self, features: np.ndarray, fit_pca: bool = False
    ) -> np.ndarray:
        """
        Apply configured transformations to features.

        Args:
            features: Raw feature array (B, D)
            fit_pca: Whether to fit PCA on this data

        Returns:
            Transformed feature array
        """
        use_gram = self.transform_config.get("use_gram", False)
        use_pca = self.transform_config.get("use_pca", False)

        if use_gram:
            features = self._compute_gram_matrix(features)
            logger.debug(f"After Gram: shape = {features.shape}")

        if use_pca:
            features = self._apply_pca(features, fit=fit_pca)
            logger.debug(f"After PCA: shape = {features.shape}")

        return features

    def extract_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract features from a batch of preprocessed images.

        Args:
            images: Batch of images (B, C, H, W)

        Returns:
            Feature array (B, D)
        """
        images = images.to(self.device)

        # Clear previous hook outputs
        for hook in self.hooks.values():
            hook.clear()

        # Forward pass
        with torch.no_grad():
            _ = self.model(images)

        # Collect features from all hooks
        features_list = []
        for layer_idx in sorted(self.hooks.keys()):
            hook = self.hooks[layer_idx]
            if hook.features is not None:
                processed = self._process_features(hook.features)
                features_list.append(processed)
            else:
                logger.warning(f"No features captured from layer {layer_idx}")

        if not features_list:
            raise RuntimeError("No features were captured from any layer")

        # Aggregate features from multiple layers
        return self._aggregate_features(features_list)

    def extract(
        self,
        images: Union[List[str], List[Image.Image], torch.Tensor],
        fit_transform: bool = False,
        batch_size: int = None,
    ) -> np.ndarray:
        """
        Extract features from a list of images.

        Args:
            images: List of image paths, PIL images, or tensor batch
            fit_transform: Whether to fit PCA on this data (for reference set)
            batch_size: Batch size for processing

        Returns:
            Feature array (N, D)
        """
        batch_size = batch_size or CONFIG["batch_size"]

        # Handle tensor input
        if isinstance(images, torch.Tensor):
            all_features = []
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i+batch_size]
                features = self.extract_batch(batch)
                all_features.append(features)
            raw_features = np.concatenate(all_features, axis=0)
            return self.transform_features(raw_features, fit_pca=fit_transform)

        # Handle list of paths
        if isinstance(images[0], str):
            dataset = ImageDataset(images, transform=self.image_transform)
        else:
            # List of PIL images
            dataset = [(self.image_transform(img), str(i)) for i, img in enumerate(images)]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.device == "cuda" else False,
        )

        all_features = []
        total_batches = len(dataloader)

        for batch_idx, (batch_images, _) in enumerate(dataloader):
            if batch_idx % CONFIG["log_progress_every"] == 0:
                logger.info(f"Extracting features: batch {batch_idx+1}/{total_batches}")

            features = self.extract_batch(batch_images)
            all_features.append(features)

        raw_features = np.concatenate(all_features, axis=0)
        logger.info(f"Extracted raw features: shape = {raw_features.shape}")

        # Apply transformations
        transformed = self.transform_features(raw_features, fit_pca=fit_transform)
        logger.info(f"Final feature shape: {transformed.shape}")

        return transformed

    def cleanup(self):
        """Remove hooks and free resources."""
        for handle in self.hook_handles:
            handle.remove()
        self.hooks.clear()
        self.hook_handles.clear()


class GramPCATransformer:
    """
    Standalone transformer for Gram matrix + PCA operations.
    Useful when you want to apply transformations separately from extraction.
    """

    def __init__(self, use_gram: bool = True, use_pca: bool = True, pca_dim: int = 10):
        self.use_gram = use_gram
        self.use_pca = use_pca
        self.pca_dim = pca_dim
        self.pca = None
        self._fitted = False

    def _compute_gram(self, features: np.ndarray) -> np.ndarray:
        """Compute Gram matrix upper triangular elements."""
        batch_size, dim = features.shape

        # Subsample for large dimensions
        max_dim = 1000
        if dim > max_dim:
            indices = np.random.choice(dim, max_dim, replace=False)
            features = features[:, indices]
            dim = max_dim

        gram_features = []
        for i in range(batch_size):
            f = features[i:i+1].T
            gram = f @ f.T
            upper_tri = gram[np.triu_indices(dim)]
            gram_features.append(upper_tri)

        return np.array(gram_features)

    def fit(self, features: np.ndarray) -> "GramPCATransformer":
        """Fit the transformer on reference features."""
        transformed = features

        if self.use_gram:
            transformed = self._compute_gram(transformed)

        if self.use_pca:
            n_components = min(self.pca_dim, transformed.shape[0], transformed.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(transformed)

        self._fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted parameters."""
        transformed = features

        if self.use_gram:
            transformed = self._compute_gram(transformed)

        if self.use_pca:
            if not self._fitted:
                raise RuntimeError("Transformer not fitted. Call fit() first.")
            transformed = self.pca.transform(transformed)

        return transformed

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)
