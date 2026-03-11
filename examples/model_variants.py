"""
Model variants example for spGDMM.

This example demonstrates all 9 model variants and their different
combinations of variance structures and spatial effects.
"""

import numpy as np
from spgdmm import ModelVariant, ModelConfig, VarianceType, SpatialEffectType

# -----------------------------------------------------
# 1. Model variant mapping
# -----------------------------------------------------

print("spGDMM Model Variants")
print("="*70)

print("\nVariance Types:")
print("  - HOMOGENEOUS: Constant variance across all predictions")
print("  - COVARIATE_DEPENDENT: Variance depends on covariates")
print("  - POLYNOMIAL: Polynomial variance function of mean")

print("\nSpatial Effect Types:")
print("  - NONE: No spatial random effects")
print("  - ABS_DIFF: Absolute difference in spatial effects")
print("  - SQUARED_DIFF: Squared difference in spatial effects")

print("\nModel Variants Matrix:")
print("-" * 70)
print(f"{'Model':<10} | {'Variance':<20} | {'Spatial Effects':<20}")
print("-" * 70)

for variant in ModelVariant:
    var_type = variant.variance_type.value
    spat_type = variant.spatial_effect_type.value
    print(f"{variant.value:<10} | {var_type:<20} | {spat_type:<20}")

print("-" * 70)

# -----------------------------------------------------
# 2. Using ModelVariant enum (recommended)
# -----------------------------------------------------

print("\n" + "="*70)
print("Usage 1: Creating models with pre-configured variants")
print("="*70)

from spgdmm import spGDMM

# Example: Create model with spatial effects
model_spatial = spGDMM.from_variant(
    ModelVariant.MODEL4,  # Homogeneous variance + abs diff spatial effects
    deg=3,
    knots=2,
)
print(f"\nCreated model: {ModelVariant.MODEL4}")
print(f"  Variance: {model_spatial._config.variance_type}")
print(f"  Spatial: {model_spatial._config.spatial_effect_type}")

# Example: Create model with polynomial variance
model_poly = spGDMM.from_variant(
    ModelVariant.MODEL3,  # Polynomial variance, no spatial effects
    deg=3,
    knots=2,
)
print(f"\nCreated model: {ModelVariant.MODEL3}")
print(f"  Variance: {model_poly._config.variance_type}")
print(f"  Spatial: {model_poly._config.spatial_effect_type}")

# -----------------------------------------------------
# 3. Using ModelConfig dataclass (flexible)
# -----------------------------------------------------

print("\n" + "="*70)
print("Usage 2: Creating models with ModelConfig dataclass")
print("="*70)

# Example: Custom model with spatial effects and homogeneous variance
config_custom = ModelConfig(
    deg=4,
    knots=3,
    mesh_choice="percentile",
    distance_measure="euclidean",
    variance_type=VarianceType.HOMOGENEOUS,
    spatial_effect_type=SpatialEffectType.ABS_DIFF,
    alpha_importance=True,
)

model_custom = spGDMM(config=config_custom)
print(f"\nCreated custom model:")
print(f"  Variance: {model_custom._config.variance_type}")
print(f"  Spatial: {model_custom._config.spatial_effect_type}")
print(f"  Degrees: {model_custom._config.deg}")
print(f"  Knots: {model_custom._config.knots}")

# Example: Variants based on dataclass
config_from_variant = ModelConfig.from_variant(ModelVariant.MODEL6)
print(f"\nModel config from MODEL6:")
print(f"  Variance: {config_from_variant.variance_type}")
print(f"  Spatial: {config_from_variant.spatial_effect_type}")

# Override specific settings
config_overridden = ModelConfig.from_variant(
    ModelVariant.MODEL7,
    deg=5,        # Override degree
    knots=3,       # Override knots
    distance_measure="geodesic",  # Override distance measure
)
print(f"\nModel config from MODEL7 with overrides:")
print(f"  Variance: {config_overridden.variance_type}")
print(f"  Spatial: {config_overridden.spatial_effect_type}")
print(f"  Degrees: {config_overridden.deg}")
print(f"  Distance measure: {config_overridden.distance_measure}")

# -----------------------------------------------------
# 4. Comparing all variants programmatically
# -----------------------------------------------------

print("\n" + "="*70)
print("All Variants Comparison")
print("="*70)

print(f"\n{'Variant':<10} | {'Spatial?':<9} | {'Variance Type':<20}")
print("-" * 70)

for variant in ModelVariant:
    has_spatial = variant.spatial_effect_type != SpatialEffectType.NONE
    print(f"{variant.value:<10} | {'Yes' if has_spatial else 'No':<9} | {variant.variance_type.value:<20}")

print("-" * 70)

# -----------------------------------------------------
# 5. Choosing the right variant
# -----------------------------------------------------

print("\n" + "="*70)
print("Guidelines for Choosing a Model Variant")
print("="*70)

print("""
Model Selection Guide:

1. MODEL1 (Homogeneous, None Spatial)
   - Use as baseline
   - Fastest to fit
   - Good for simple ecological gradients

2. MODEL2 (Covariate-Dependent, None Spatial)
   - When variance varies with predictors
   - Heteroscedasticity suspected

3. MODEL3 (Polynomial, None Spatial)
   - When variance has non-linear relationship with mean
   - Complex heteroscedastic patterns

4-6. Models with ABS_DIFF Spatial Effects
   - Add spatial correlation through absolute differences
   - Use when nearby sites have similar dissimilarities
   - MODEL4/5/6 match MODEL1/2/3 variance types

7-9. Models with SQUARED_DIFF Spatial Effects
   - Add spatial correlation through squared differences
   - Stronger spatial decay
   - MODEL7/8/9 match MODEL1/2/3 variance types
""")

print("\n" + "="*70)
print("Example complete!")
print("="*70)