"""
Model variants example for spGDMM.

This example demonstrates how to configure variance structures and spatial
effects using the unified string/callable API in ModelConfig.
"""

import numpy as np
from spgdmm import ModelConfig, spGDMM, PreprocessorConfig

# -----------------------------------------------------
# 1. Built-in variance and spatial effect strings
# -----------------------------------------------------

print("spGDMM Model Variants")
print("=" * 70)

print("\nVariance options (pass as 'variance' string or callable):")
print("  - 'homogeneous'         : Constant variance across all predictions")
print("  - 'covariate_dependent' : Variance depends on pairwise distance")
print("  - 'polynomial'          : Polynomial variance function of mean")

print("\nSpatial effect options (pass as 'spatial_effect' string or callable):")
print("  - 'none'          : No spatial random effects")
print("  - 'abs_diff'      : Absolute difference in GP spatial effect")
print("  - 'squared_diff'  : Squared difference in GP spatial effect")

# -----------------------------------------------------
# 2. Creating models with string API
# -----------------------------------------------------

print("\n" + "=" * 70)
print("Usage 1: String-based configuration")
print("=" * 70)

model1 = spGDMM(model_config=ModelConfig(variance="homogeneous", spatial_effect="none"))
print(f"\nModel 1: variance={model1._config.variance!r}, spatial_effect={model1._config.spatial_effect!r}")

model2 = spGDMM(model_config=ModelConfig(variance="covariate_dependent", spatial_effect="abs_diff"))
print(f"Model 2: variance={model2._config.variance!r}, spatial_effect={model2._config.spatial_effect!r}")

model3 = spGDMM(model_config=ModelConfig(variance="polynomial", spatial_effect="squared_diff"))
print(f"Model 3: variance={model3._config.variance!r}, spatial_effect={model3._config.spatial_effect!r}")

# -----------------------------------------------------
# 3. Custom variance callable
# -----------------------------------------------------

print("\n" + "=" * 70)
print("Usage 2: Custom callable API")
print("=" * 70)

import pymc as pm

def my_variance(mu, X_sigma):
    """Custom heteroscedastic variance: exponential function of mean."""
    beta_s = pm.HalfNormal("beta_s", sigma=1)
    return beta_s * pm.math.exp(mu)

def my_spatial(psi, row_ind, col_ind):
    """Custom spatial effect: hyperbolic tangent of GP difference."""
    return pm.math.tanh(psi[row_ind] - psi[col_ind])

model_custom = spGDMM(model_config=ModelConfig(variance=my_variance, spatial_effect=my_spatial))
print(f"\nCustom model:")
print(f"  variance callable: {model_custom._config.variance.__name__}")
print(f"  spatial_effect callable: {model_custom._config.spatial_effect.__name__}")

# -----------------------------------------------------
# 4. to_dict / from_dict round-trip
# -----------------------------------------------------

print("\n" + "=" * 70)
print("Serialization")
print("=" * 70)

cfg = ModelConfig(variance="polynomial", spatial_effect="abs_diff")
d = cfg.to_dict()
print(f"\nto_dict(): {d}")

cfg2 = ModelConfig.from_dict(d)
print(f"from_dict(): variance={cfg2.variance!r}, spatial_effect={cfg2.spatial_effect!r}")

# -----------------------------------------------------
# 5. Invalid strings raise ValueError immediately
# -----------------------------------------------------

print("\n" + "=" * 70)
print("Validation")
print("=" * 70)

try:
    ModelConfig(variance="bad_name")
except ValueError as e:
    print(f"\nModelConfig(variance='bad_name') → ValueError: {e}")

try:
    ModelConfig(spatial_effect="bad_name")
except ValueError as e:
    print(f"ModelConfig(spatial_effect='bad_name') → ValueError: {e}")

print("\n" + "=" * 70)
print("Example complete!")
print("=" * 70)
