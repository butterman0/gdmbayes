"""
Basic usage example for spGDMM.

This example demonstrates how to fit a basic spGDMM model
with environmental and spatial predictors.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from spgdmm import spGDMM, ModelVariant, ModelConfig

# -----------------------------------------------------
# 1. Generate synthetic data
# -----------------------------------------------------

# Number of sample sites
n_sites = 50

# Generate random coordinates (simulating spatial locations)
np.random.seed(42)
xc = np.random.uniform(0, 100, n_sites)
yc = np.random.uniform(0, 100, n_sites)
time_idx = np.zeros(n_sites, dtype=int)

# Generate environmental predictors
temp = np.random.uniform(5, 20, n_sites)  # Temperature
salinity = np.random.uniform(30, 35, n_sites)  # Salinity
depth = np.random.uniform(0, 200, n_sites)  # Depth

# Create biomass matrix (simulating species abundances)
n_species = 20
biomass = np.random.exponential(1, (n_sites, n_species))

# Compute pairwise Bray-Curtis dissimilarities
from scipy.spatial.distance import pdist
y = pdist(biomass, "braycurtis")
y = np.clip(y, 1e-8, None)  # Avoid zeros for log transform

# Create predictor DataFrame
X = pd.DataFrame({
    "xc": xc,
    "yc": yc,
    "time_idx": time_idx,
    "temp": temp,
    "salinity": salinity,
    "depth": depth,
})

print(f"Number of sites: {n_sites}")
print(f"Number of pairwise dissimilarities: {len(y)}")
print(f"Dissimilarity range: [{y.min():.3f}, {y.max():.3f}]")

# -----------------------------------------------------
# 2. Fit spGDMM model using pre-configured variant
# -----------------------------------------------------

print("\n" + "="*60)
print("Fitting spGDMM model with MODEL1 (homogeneous variance, no spatial effects)")
print("="*60)

# Create model using pre-configured variant
model = spGDMM.from_variant(
    ModelVariant.MODEL1,
    deg=3,
    knots=2,
    distance_measure="euclidean",
    alpha_importance=True,
)

# Fit the model with minimal sampling for this example
idata = model.fit(
    X,
    y,
    random_seed=42,
    draws=100,  # Reduced for example
    tune=100,   # Reduced for example
    chains=2,   # Reduced for example
    progressbar=True,
)

print("\nModel fitting complete!")
print(f"Posterior samples: {idata.posterior.dims}")
print(f"Variables: {list(idata.posterior.data_vars.keys())}")

# -----------------------------------------------------
# 3. Examine results
# -----------------------------------------------------

# Check convergence
from spgdmm import summarise_sampling
print("\n" + "="*60)
print("Sampling Diagnostics")
print("="*60)
diag = summarise_sampling(idata)

# View parameter summaries
print("\nParameter summaries:")
print(diag.head(10))

# -----------------------------------------------------
# 4. Make predictions on new data
# -----------------------------------------------------

print("\n" + "="*60)
print("Making predictions")
print("="*60)

# New sites to predict
n_pred = 10
X_pred = pd.DataFrame({
    "xc": np.random.uniform(0, 100, n_pred),
    "yc": np.random.uniform(0, 100, n_pred),
    "time_idx": np.zeros(n_pred, dtype=int),
    "temp": np.random.uniform(5, 20, n_pred),
    "salinity": np.random.uniform(30, 35, n_pred),
    "depth": np.random.uniform(0, 200, n_pred),
})

# Get predictive samples
post_pred = model.predict_posterior(
    X_pred,
    extend_idata=False,
    combined=True,
)

print(f"Posterior predictive samples shape: {post_pred.shape}")
print(f"Mean predictions: {post_pred.mean(dim='sample').values}")
print(f"Back-transformed ( Bray-Curtis): {np.exp(post_pred.mean(dim='sample').values)}")

# Save the model
# model.save("spgdmm_model.nc")

print("\n" + "="*60)
print("Example complete!")
print("="*60)