"""
Basic usage example for gdmbayes.

Demonstrates how to fit both the frequentist GDM and the Bayesian spGDMM
with environmental and geographic predictors.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from gdmbayes import GDM, spGDMM, ModelConfig, GDMPreprocessor, SamplerConfig, summarise_sampling

# -----------------------------------------------------------------------
# 1. Generate synthetic data
# -----------------------------------------------------------------------

np.random.seed(42)
n_sites = 30

X = pd.DataFrame({
    "xc": np.random.uniform(0, 100, n_sites),
    "yc": np.random.uniform(0, 100, n_sites),
    "time_idx": np.zeros(n_sites, dtype=int),
    "temp": np.random.uniform(5, 20, n_sites),
    "salinity": np.random.uniform(30, 35, n_sites),
    "depth": np.random.uniform(0, 200, n_sites),
})

biomass = np.random.exponential(1, (n_sites, 20))
y = pdist(biomass, "braycurtis")
y = np.clip(y, 1e-8, 1 - 1e-8)

print(f"Sites: {n_sites}  |  Pairs: {len(y)}  |  y range: [{y.min():.3f}, {y.max():.3f}]")

# -----------------------------------------------------------------------
# 2. Frequentist GDM
# -----------------------------------------------------------------------

print("\n" + "=" * 60)
print("Frequentist GDM")
print("=" * 60)

gdm_model = GDM(geo=True)
gdm_model.fit(X, y)

print(f"Deviance explained: {gdm_model.explained_:.4f}")
print(f"Predictor importance: {gdm_model.predictor_importance_}")
print(f"Coefficients shape: {gdm_model.coef_.shape}")

# Predict pairwise dissimilarities
preds = gdm_model.predict(X)
print(f"Prediction range: [{preds.min():.3f}, {preds.max():.3f}]")

# -----------------------------------------------------------------------
# 3. Bayesian spGDMM (minimal sampling for example speed)
# -----------------------------------------------------------------------

print("\n" + "=" * 60)
print("Bayesian spGDMM")
print("=" * 60)

sampler_cfg = SamplerConfig(draws=100, tune=100, chains=2, nuts_sampler="pymc", progressbar=True)

model = spGDMM(
    preprocessor=GDMPreprocessor(deg=3, knots=2),
    model_config=ModelConfig(variance="homogeneous", spatial_effect="none"),
    sampler_config=sampler_cfg,
)

model.fit(X, y, random_seed=42)
idata = model.idata_

print(f"Posterior samples: {idata.posterior.dims}")
print(f"Variables: {list(idata.posterior.data_vars.keys())}")

# Sampling diagnostics
print("\nSampling Diagnostics")
diag = summarise_sampling(idata)
print(diag.head(10))

# -----------------------------------------------------------------------
# 4. Predict on new data
# -----------------------------------------------------------------------

print("\n" + "=" * 60)
print("Predictions on new data")
print("=" * 60)

n_pred = 10
X_pred = pd.DataFrame({
    "xc": np.random.uniform(0, 100, n_pred),
    "yc": np.random.uniform(0, 100, n_pred),
    "time_idx": np.zeros(n_pred, dtype=int),
    "temp": np.random.uniform(5, 20, n_pred),
    "salinity": np.random.uniform(30, 35, n_pred),
    "depth": np.random.uniform(0, 200, n_pred),
})

post_pred = model.predict_posterior(X_pred, extend_idata=False, combined=True)
print(f"Posterior predictive shape: {post_pred.shape}")

# model.save("spgdmm_model.nc")

print("\n" + "=" * 60)
print("Example complete!")
print("=" * 60)
