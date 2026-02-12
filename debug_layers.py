"""
Debug script to check if layers 6 and 9 extract different features.
"""
import torch
import numpy as np
from feature_extraction import FeatureExtractor
from main import load_images
from config import CONFIG

# Load a few test images
test_images = load_images(CONFIG["dataset_path"], n_images=5)

print("=" * 80)
print("DEBUGGING LAYER EXTRACTION")
print("=" * 80)

# Test layer 6
print("\n--- Testing Layer 6 ---")
layer_config_6 = {"name": "single_layer_6", "layers": [6]}
transform_config = {"name": "gram_spatial", "use_gram": True, "use_pca": False, "gram_averaging": "spatial"}

extractor_6 = FeatureExtractor(
    backbone="sd_vae",
    layer_config=layer_config_6,
    transform_config=transform_config,
)

print(f"Hooks registered: {list(extractor_6.hooks.keys())}")
print(f"Layer config: {extractor_6.layer_config}")

features_6 = extractor_6.extract(test_images)
print(f"Features shape: {features_6.shape}")
print(f"Features mean: {features_6.mean():.6f}")
print(f"Features std: {features_6.std():.6f}")
print(f"First 10 values: {features_6[0, :10]}")

extractor_6.cleanup()

# Test layer 9
print("\n--- Testing Layer 9 ---")
layer_config_9 = {"name": "single_layer_9", "layers": [9]}

extractor_9 = FeatureExtractor(
    backbone="sd_vae",
    layer_config=layer_config_9,
    transform_config=transform_config,
)

print(f"Hooks registered: {list(extractor_9.hooks.keys())}")
print(f"Layer config: {extractor_9.layer_config}")

features_9 = extractor_9.extract(test_images)
print(f"Features shape: {features_9.shape}")
print(f"Features mean: {features_9.mean():.6f}")
print(f"Features std: {features_9.std():.6f}")
print(f"First 10 values: {features_9[0, :10]}")

extractor_9.cleanup()

# Compare
print("\n--- Comparison ---")
print(f"Shapes equal: {features_6.shape == features_9.shape}")
print(f"Features equal: {np.allclose(features_6, features_9)}")
print(f"Max difference: {np.abs(features_6 - features_9).max():.10f}")
print(f"Mean difference: {np.abs(features_6 - features_9).mean():.10f}")

if np.allclose(features_6, features_9):
    print("\n⚠️  BUG CONFIRMED: Features from layer 6 and 9 are identical!")
else:
    print("\n✓ Features are different, as expected.")
