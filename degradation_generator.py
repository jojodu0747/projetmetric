"""
Image degradation generator module.
Generates sequences of degraded images with increasing severity levels.
"""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torchvision import transforms

from config import CONFIG

logger = logging.getLogger(__name__)


class DegradationGenerator:
    """
    Generates various types of image degradations with configurable severity levels.

    Supported degradation types:
    - Gaussian blur
    - Blur + contrast reduction
    - Gaussian noise
    - Aliasing (downsample/upsample)
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the degradation generator.

        Args:
            config: Degradation configurations (defaults to CONFIG["degradations"])
        """
        self.config = config or CONFIG["degradations"]
        self.rng = np.random.RandomState(CONFIG["random_seed"])

    def apply_gaussian_blur(
        self,
        image: Union[Image.Image, np.ndarray],
        sigma: float,
    ) -> Image.Image:
        """
        Apply Gaussian blur to an image.

        Args:
            image: Input image (PIL Image or numpy array)
            sigma: Blur sigma (standard deviation)

        Returns:
            Blurred PIL Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # PIL's GaussianBlur takes radius, which is approximately 2*sigma
        radius = int(sigma * 2)
        radius = max(1, radius)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def apply_contrast(
        self,
        image: Union[Image.Image, np.ndarray],
        factor: float,
    ) -> Image.Image:
        """
        Adjust image contrast.

        Args:
            image: Input image
            factor: Contrast factor (1.0 = original, 0.5 = reduced contrast)

        Returns:
            Contrast-adjusted PIL Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def apply_blur_and_contrast(
        self,
        image: Union[Image.Image, np.ndarray],
        blur_sigma: float,
        contrast_factor: float,
    ) -> Image.Image:
        """
        Apply both blur and contrast reduction.

        Args:
            image: Input image
            blur_sigma: Blur sigma
            contrast_factor: Contrast factor

        Returns:
            Degraded PIL Image
        """
        blurred = self.apply_gaussian_blur(image, blur_sigma)
        return self.apply_contrast(blurred, contrast_factor)

    def apply_gaussian_noise(
        self,
        image: Union[Image.Image, np.ndarray],
        std: float,
    ) -> Image.Image:
        """
        Add Gaussian noise to an image.

        Args:
            image: Input image
            std: Standard deviation of noise (on 0-255 scale)

        Returns:
            Noisy PIL Image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Generate noise
        noise = self.rng.normal(0, std, image.shape).astype(np.float32)

        # Add noise and clip
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy)

    def apply_aliasing(
        self,
        image: Union[Image.Image, np.ndarray],
        factor: int,
    ) -> Image.Image:
        """
        Apply aliasing by downsampling then upsampling.

        Args:
            image: Input image
            factor: Downsampling factor (2, 4, 6, 8)

        Returns:
            Aliased PIL Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        original_size = image.size

        # Downsample (using nearest neighbor for aliasing artifacts)
        small_size = (max(1, original_size[0] // factor), max(1, original_size[1] // factor))
        downsampled = image.resize(small_size, Image.Resampling.NEAREST)

        # Upsample back to original size
        upsampled = downsampled.resize(original_size, Image.Resampling.NEAREST)

        return upsampled

    def apply_degradation(
        self,
        image: Union[Image.Image, np.ndarray],
        degradation_type: str,
        level: int,
    ) -> Image.Image:
        """
        Apply a specific degradation at a given severity level.

        Args:
            image: Input image
            degradation_type: Type of degradation ('blur', 'blur_contrast', 'noise', 'aliasing')
            level: Severity level index (0 = lowest)

        Returns:
            Degraded PIL Image
        """
        if degradation_type not in self.config:
            raise ValueError(f"Unknown degradation type: {degradation_type}")

        deg_config = self.config[degradation_type]

        if degradation_type == "blur":
            levels = deg_config["levels"]
            if level >= len(levels):
                level = len(levels) - 1
            sigma = levels[level]
            return self.apply_gaussian_blur(image, sigma)

        elif degradation_type == "blur_contrast":
            blur_levels = deg_config["blur_levels"]
            contrast_factors = deg_config["contrast_factors"]
            # Combined levels: each blur level paired with decreasing contrast
            if level >= len(blur_levels) * len(contrast_factors):
                level = len(blur_levels) * len(contrast_factors) - 1
            # Sequential pairing
            if level < len(blur_levels):
                blur_idx = level
                contrast_idx = 0
            else:
                blur_idx = len(blur_levels) - 1
                contrast_idx = level - len(blur_levels) + 1
            blur_sigma = blur_levels[min(blur_idx, len(blur_levels) - 1)]
            contrast_factor = contrast_factors[min(contrast_idx, len(contrast_factors) - 1)]
            return self.apply_blur_and_contrast(image, blur_sigma, contrast_factor)

        elif degradation_type == "noise":
            levels = deg_config["std_levels"]
            if level >= len(levels):
                level = len(levels) - 1
            std = levels[level]
            return self.apply_gaussian_noise(image, std)

        elif degradation_type == "aliasing":
            factors = deg_config["factors"]
            if level >= len(factors):
                level = len(factors) - 1
            factor = factors[level]
            return self.apply_aliasing(image, factor)

        else:
            raise ValueError(f"Unknown degradation type: {degradation_type}")

    def get_num_levels(self, degradation_type: str) -> int:
        """
        Get the number of severity levels for a degradation type.

        Args:
            degradation_type: Type of degradation

        Returns:
            Number of levels
        """
        if degradation_type not in self.config:
            raise ValueError(f"Unknown degradation type: {degradation_type}")

        deg_config = self.config[degradation_type]

        if degradation_type == "blur":
            return len(deg_config["levels"])
        elif degradation_type == "blur_contrast":
            # Combined levels
            return max(len(deg_config["blur_levels"]), len(deg_config["contrast_factors"]))
        elif degradation_type == "noise":
            return len(deg_config["std_levels"])
        elif degradation_type == "aliasing":
            return len(deg_config["factors"])
        else:
            return 0

    def generate_degradation_sequence(
        self,
        image: Union[Image.Image, np.ndarray],
        degradation_type: str,
    ) -> List[Image.Image]:
        """
        Generate a sequence of degraded images with increasing severity.

        Args:
            image: Input image
            degradation_type: Type of degradation

        Returns:
            List of degraded images (ordered by increasing severity)
        """
        num_levels = self.get_num_levels(degradation_type)
        sequence = []

        for level in range(num_levels):
            degraded = self.apply_degradation(image, degradation_type, level)
            sequence.append(degraded)

        return sequence

    def generate_all_degradations(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> Dict[str, List[Image.Image]]:
        """
        Generate all degradation types and levels for an image.

        Args:
            image: Input image

        Returns:
            Dict mapping degradation type to list of degraded images
        """
        results = {}

        for deg_type in self.config.keys():
            results[deg_type] = self.generate_degradation_sequence(image, deg_type)

        return results


class BatchDegradationGenerator:
    """
    Efficient batch processing of degradations for evaluation.
    """

    def __init__(self, config: Dict = None):
        self.generator = DegradationGenerator(config)
        self.config = self.generator.config

    def process_image_batch(
        self,
        images: List[Union[str, Image.Image]],
        degradation_type: str,
        level: int,
    ) -> List[Image.Image]:
        """
        Apply degradation to a batch of images.

        Args:
            images: List of image paths or PIL Images
            degradation_type: Type of degradation
            level: Severity level

        Returns:
            List of degraded PIL Images
        """
        results = []

        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")

            degraded = self.generator.apply_degradation(img, degradation_type, level)
            results.append(degraded)

        return results

    def get_degradation_info(self, degradation_type: str, level: int) -> Dict:
        """
        Get human-readable information about a degradation level.

        Args:
            degradation_type: Type of degradation
            level: Severity level

        Returns:
            Dict with degradation parameters
        """
        deg_config = self.config[degradation_type]

        if degradation_type == "blur":
            sigma = deg_config["levels"][min(level, len(deg_config["levels"]) - 1)]
            return {"type": "gaussian_blur", "sigma": sigma}

        elif degradation_type == "blur_contrast":
            blur_levels = deg_config["blur_levels"]
            contrast_factors = deg_config["contrast_factors"]
            blur_idx = min(level, len(blur_levels) - 1)
            contrast_idx = min(level, len(contrast_factors) - 1)
            return {
                "type": "blur_plus_contrast",
                "blur_sigma": blur_levels[blur_idx],
                "contrast_factor": contrast_factors[contrast_idx],
            }

        elif degradation_type == "noise":
            std = deg_config["std_levels"][min(level, len(deg_config["std_levels"]) - 1)]
            return {"type": "gaussian_noise", "std": std}

        elif degradation_type == "aliasing":
            factor = deg_config["factors"][min(level, len(deg_config["factors"]) - 1)]
            return {"type": "downsample_upsample", "factor": factor}

        return {}


def apply_transform_to_tensor(
    image: Image.Image,
    transform: transforms.Compose,
) -> torch.Tensor:
    """
    Apply torchvision transform to a PIL image.

    Args:
        image: PIL Image
        transform: Torchvision transform

    Returns:
        Transformed tensor
    """
    return transform(image)


def degraded_images_to_tensor(
    images: List[Image.Image],
    transform: transforms.Compose,
) -> torch.Tensor:
    """
    Convert list of degraded PIL images to a batched tensor.

    Args:
        images: List of PIL Images
        transform: Torchvision transform

    Returns:
        Batched tensor (N, C, H, W)
    """
    tensors = [transform(img) for img in images]
    return torch.stack(tensors)
