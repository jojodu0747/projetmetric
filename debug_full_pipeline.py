"""
Debug script to test full pipeline for layers 6 and 9.
"""
import numpy as np
from feature_extraction import FeatureExtractor
from main import load_images, extract_all_degraded_features, evaluate_with_cached_features
from distance_metrics import get_distance_metric
from config import CONFIG

# Use same small dataset as user
reference_images = load_images(CONFIG["dataset_path"], n_images=20)
eval_images = load_images(CONFIG["dataset_path"], n_images=5)

# Make sure eval and ref are disjoint
eval_set = set(eval_images)
reference_images = [p for p in reference_images if p not in eval_set]

transform_config = {
    "name": "gram_spatial",
    "use_gram": True,
    "use_pca": False,
    "gram_averaging": "spatial"
}

print("=" * 80)
print("FULL PIPELINE TEST: Layer 6 vs Layer 9")
print("=" * 80)
print(f"Reference images: {len(reference_images)}")
print(f"Evaluation images: {len(eval_images)}")

results = {}

for layer_idx in [6, 9]:
    print(f"\n{'='*80}")
    print(f"TESTING LAYER {layer_idx}")
    print(f"{'='*80}")

    layer_config = {"name": f"single_layer_{layer_idx}", "layers": [layer_idx]}

    # Create extractor
    extractor = FeatureExtractor(
        backbone="sd_vae",
        layer_config=layer_config,
        transform_config=transform_config,
    )

    # Extract reference features
    print("Extracting reference features...")
    reference_features = extractor.extract(reference_images, fit_transform=True)
    print(f"  Shape: {reference_features.shape}")
    print(f"  Mean: {reference_features.mean():.6f}, Std: {reference_features.std():.6f}")

    # Extract degraded features
    print("Extracting degraded features...")
    degraded_features_cache = extract_all_degraded_features(
        eval_images, extractor, CONFIG["degradations"]
    )

    # Create metric
    metric = get_distance_metric(
        "mmd",
        features_ref=reference_features,
        kernel="rbf",
        gamma=None,
    )

    # Evaluate
    print("Evaluating monotonicity...")
    eval_results = evaluate_with_cached_features(degraded_features_cache, metric)

    # Extract monotonicity scores
    monotonicity = {
        deg_type: res["monotonicity"]
        for deg_type, res in eval_results.items()
    }

    results[layer_idx] = {
        "monotonicity": monotonicity,
        "reference_shape": reference_features.shape,
        "reference_mean": reference_features.mean(),
        "reference_std": reference_features.std(),
    }

    print(f"\nMonotonicity scores:")
    for deg_type, score in monotonicity.items():
        print(f"  {deg_type}: {score:.6f}")

    extractor.cleanup()

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")

print(f"\nFeature shapes:")
print(f"  Layer 6: {results[6]['reference_shape']}")
print(f"  Layer 9: {results[9]['reference_shape']}")
print(f"  Same shape: {results[6]['reference_shape'] == results[9]['reference_shape']}")

print(f"\nFeature statistics:")
print(f"  Layer 6 mean: {results[6]['reference_mean']:.6f}, std: {results[6]['reference_std']:.6f}")
print(f"  Layer 9 mean: {results[9]['reference_mean']:.6f}, std: {results[9]['reference_std']:.6f}")

print(f"\nMonotonicity scores:")
for deg_type in results[6]['monotonicity'].keys():
    score_6 = results[6]['monotonicity'][deg_type]
    score_9 = results[9]['monotonicity'][deg_type]
    diff = abs(score_6 - score_9)
    same = np.isclose(score_6, score_9, atol=1e-6)
    print(f"  {deg_type}:")
    print(f"    Layer 6: {score_6:.6f}")
    print(f"    Layer 9: {score_9:.6f}")
    print(f"    Diff: {diff:.10f}")
    print(f"    Same: {same}")

all_same = all(
    np.isclose(results[6]['monotonicity'][deg_type], results[9]['monotonicity'][deg_type], atol=1e-6)
    for deg_type in results[6]['monotonicity'].keys()
)

if all_same:
    print(f"\n{'='*80}")
    print("⚠️  BUG CONFIRMED: Monotonicity scores are identical!")
    print("This should NOT happen with different layers.")
    print(f"{'='*80}")
else:
    print(f"\n{'='*80}")
    print("✓ Scores are different, as expected.")
    print(f"{'='*80}")
