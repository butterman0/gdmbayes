"""
Quick test script for spGDMM package.
Run with: python quick_test.py
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

# Import spGDMM components
from spgdmm import (
    spGDMM,
    ModelVariant,
    ModelConfig,
    VarianceType,
    SpatialEffectType,
)

print("=" * 60)
print("Testing spGDMM Package")
print("=" * 60)

# Test 1: Imports
print("\n1. Testing imports...")
assert hasattr(spGDMM, "from_variant")
assert len(ModelVariant) == 9
print("   ✓ Imports successful")

# Test 2: ModelVariant enum
print("\n2. Testing ModelVariant enum...")
for v in ModelVariant:
    assert hasattr(v, "variance_type")
    assert hasattr(v, "spatial_type")
print(f"   ✓ All {len(ModelVariant)} variants available")
print(f"   Variants: {[v.value for v in ModelVariant]}")

# Test 3: ModelConfig
print("\n3. Testing ModelConfig...")
config = ModelConfig(deg=4, knots=3)
assert config.deg == 4
assert config.knots == 3
assert config.variance_type == VarianceType.HOMOGENEOUS
print("   ✓ ModelConfig works")

# Test 4: Config from variant
print("\n4. Testing config from variant...")
config = ModelConfig.from_variant(ModelVariant.MODEL4)
assert config.variance_type == VarianceType.HOMOGENEOUS
assert config.spatial_effect_type == SpatialEffectType.ABS_DIFF
print("   ✓ Config from variant works")

# Test 5: Model initialization
print("\n5. Testing model initialization...")
model = spGDMM.from_variant(ModelVariant.MODEL1)
assert model._config.variance_type == VarianceType.HOMOGENEOUS
assert model._config.deg == 3
print("   ✓ Model initializes correctly")

# Test 6: Generate sample data
print("\n6. Testing with sample data...")
np.random.seed(42)
n_sites = 20

X = pd.DataFrame({
    "xc": np.random.uniform(0, 100, n_sites),
    "yc": np.random.uniform(0, 100, n_sites),
    "time_idx": np.zeros(n_sites, dtype=int),
    "temp": np.random.uniform(5, 20, n_sites),
    "depth": np.random.uniform(0, 200, n_sites),
})

biomass = np.random.exponential(1, (n_sites, 10))
y = pdist(biomass, "braycurtis")
y = np.clip(y, 1e-8, None)

model = spGDMM()
model._generate_and_preprocess_model_data(X, y)

assert model.metadata is not None
assert model.training_metadata is not None
print(f"   ✓ Data preprocessing works")
print(f"   - Sites: {n_sites}")
print(f"   - Pairs: {len(y)}")

# Test 7: Model building
print("\n7. Testing model building...")
model.build_model(model.X_transformed, model.y_transformed)
assert model.model is not None
assert "beta_0" in model.model.free_RVs
print("   ✓ Model builds successfully")

# Test 8: pw_distance
print("\n8. Testing pairwise distance...")
locations = np.array([[0, 0], [1, 0], [0, 1]])
dists = model.pw_distance(locations, distance_measure="euclidean")
assert len(dists) == 3
assert np.all(dists >= 0)
print("   ✓ Distance calculation works")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)