"""
Test script for new distance metrics (Sinkhorn, Energy, Adversarial).
"""
import numpy as np
from distance_metrics import compute_sinkhorn, compute_energy_distance, compute_adversarial_distance
from distance_metrics import get_distance_metric

print("=" * 80)
print("TESTING NEW DISTANCE METRICS")
print("=" * 80)

# Create synthetic data: two slightly different distributions
np.random.seed(42)
n_ref = 100
n_test = 100
dim = 50

# Reference: standard normal
features_ref = np.random.randn(n_ref, dim).astype(np.float32)

# Test 1: Similar distribution (small shift)
features_test_similar = features_ref + np.random.randn(n_test, dim) * 0.1

# Test 2: Different distribution (large shift)
features_test_different = features_ref + 2.0 + np.random.randn(n_test, dim) * 0.5

print("\n" + "=" * 80)
print("Test 1: SINKHORN DISTANCE")
print("=" * 80)

sinkhorn_similar = compute_sinkhorn(features_ref, features_test_similar, reg=0.1)
sinkhorn_different = compute_sinkhorn(features_ref, features_test_different, reg=0.1)

print(f"Sinkhorn (similar):    {sinkhorn_similar:.6f}")
print(f"Sinkhorn (different):  {sinkhorn_different:.6f}")
print(f"Ratio (should be > 1): {sinkhorn_different / sinkhorn_similar:.2f}x")

if sinkhorn_different > sinkhorn_similar:
    print("✓ Sinkhorn correctly distinguishes distributions")
else:
    print("✗ Sinkhorn failed to distinguish distributions")

print("\n" + "=" * 80)
print("Test 2: ENERGY DISTANCE")
print("=" * 80)

energy_similar = compute_energy_distance(features_ref, features_test_similar)
energy_different = compute_energy_distance(features_ref, features_test_different)

print(f"Energy (similar):      {energy_similar:.6f}")
print(f"Energy (different):    {energy_different:.6f}")
print(f"Ratio (should be > 1): {energy_different / energy_similar:.2f}x")

if energy_different > energy_similar:
    print("✓ Energy distance correctly distinguishes distributions")
else:
    print("✗ Energy distance failed to distinguish distributions")

print("\n" + "=" * 80)
print("Test 3: ADVERSARIAL DISTANCE")
print("=" * 80)

adversarial_similar = compute_adversarial_distance(
    features_ref, features_test_similar,
    n_critics=3, max_iter=50, learning_rate=0.01
)
adversarial_different = compute_adversarial_distance(
    features_ref, features_test_different,
    n_critics=3, max_iter=50, learning_rate=0.01
)

print(f"Adversarial (similar):    {adversarial_similar:.6f}")
print(f"Adversarial (different):  {adversarial_different:.6f}")
print(f"Ratio (should be > 1):    {adversarial_different / adversarial_similar:.2f}x")

if adversarial_different > adversarial_similar:
    print("✓ Adversarial distance correctly distinguishes distributions")
else:
    print("✗ Adversarial distance failed to distinguish distributions")

print("\n" + "=" * 80)
print("Test 4: DISTANCEMETRIC WRAPPER CLASS")
print("=" * 80)

# Test via DistanceMetric class
metric_sinkhorn = get_distance_metric("sinkhorn", features_ref=features_ref, sinkhorn_reg=0.1)
metric_energy = get_distance_metric("energy", features_ref=features_ref, energy_sample_size=100)
metric_adversarial = get_distance_metric("adversarial", features_ref=features_ref,
                                        adversarial_n_critics=3, adversarial_max_iter=50)

dist_sinkhorn = metric_sinkhorn.compute(features_test_different)
dist_energy = metric_energy.compute(features_test_different)
dist_adversarial = metric_adversarial.compute(features_test_different)

print(f"DistanceMetric (Sinkhorn):     {dist_sinkhorn:.6f}")
print(f"DistanceMetric (Energy):       {dist_energy:.6f}")
print(f"DistanceMetric (Adversarial):  {dist_adversarial:.6f}")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
