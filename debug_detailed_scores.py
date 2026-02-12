"""
Debug script to check detailed MMD scores at each degradation level.
"""
import numpy as np
from feature_extraction import FeatureExtractor
from main import load_images, extract_all_degraded_features, evaluate_with_cached_features
from distance_metrics import get_distance_metric
from config import CONFIG
import matplotlib.pyplot as plt

# Use same dataset as user
reference_images = load_images(CONFIG["dataset_path"], n_images=100)
eval_images = load_images(CONFIG["dataset_path"], n_images=20)

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
print("DETAILED SCORES ANALYSIS")
print("=" * 80)

results = {}

for layer_idx in [6, 9]:
    print(f"\n{'='*80}")
    print(f"LAYER {layer_idx}")
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
    print(f"  First 5 values: {reference_features[0, :5]}")

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

    # Store detailed results
    results[layer_idx] = {}

    # Manually evaluate to get detailed scores
    for deg_type, features_list in degraded_features_cache.items():
        print(f"\n  {deg_type}:")
        scores = []
        for level, features in enumerate(features_list):
            score = metric.compute(features)
            scores.append(score)
            print(f"    Level {level}: MMD={score:.8f}")

        results[layer_idx][deg_type] = scores

    extractor.cleanup()

# Compare detailed scores
print(f"\n{'='*80}")
print("DETAILED COMPARISON")
print(f"{'='*80}")

for deg_type in results[6].keys():
    print(f"\n{deg_type}:")
    scores_6 = results[6][deg_type]
    scores_9 = results[9][deg_type]

    print(f"  Layer 6: {[f'{s:.8f}' for s in scores_6]}")
    print(f"  Layer 9: {[f'{s:.8f}' for s in scores_9]}")

    all_same = all(np.isclose(s6, s9, atol=1e-10) for s6, s9 in zip(scores_6, scores_9))
    print(f"  All scores identical: {all_same}")

    if all_same:
        print(f"  ⚠️  BUG: All scores are identical for {deg_type}!")

# Plot scores
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("MMD Scores per Degradation Level", fontsize=14)

for ax, deg_type in zip(axes.flat, results[6].keys()):
    scores_6 = results[6][deg_type]
    scores_9 = results[9][deg_type]
    levels = list(range(len(scores_6)))

    ax.plot(levels, scores_6, 'o-', label='Layer 6', linewidth=2)
    ax.plot(levels, scores_9, 's-', label='Layer 9', linewidth=2)
    ax.set_xlabel('Degradation Level')
    ax.set_ylabel('MMD Score')
    ax.set_title(deg_type)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/jnapolitano/Documents/Métrique/results/debug_scores.png', dpi=150)
print(f"\nPlot saved to results/debug_scores.png")
