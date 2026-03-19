"""
Quick test that seeds control both KFold splits and MCMC reproducibility.

Checks:
  1. Same seed → identical KFold splits
  2. Same seed → identical posterior means (MCMC reproducibility)
  3. Different seed → different KFold splits
  4. Different seed → different posterior means

Uses Panama data (39 sites, fast) with tiny MCMC settings.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.model_selection import KFold

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
env = pd.read_csv(os.path.join(DATA_DIR, "panama_env.csv"))
species = pd.read_csv(os.path.join(DATA_DIR, "panama_species.csv"), index_col=0)

X = pd.DataFrame({
    "xc": env["EW coord"].values,
    "yc": env["NS coord"].values,
    "time_idx": 0,
    "precip": env["precip"].values,
    "elev": env["elev"].values,
})
y = pdist(species.values, metric="braycurtis")
n_sites = len(X)

# ---------------------------------------------------------------------------
# 1. KFold reproducibility
# ---------------------------------------------------------------------------
print("=== 1. KFold split reproducibility ===")

def get_fold0(seed):
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    return next(iter(kf.split(np.arange(n_sites))))

train_a, test_a = get_fold0(seed=42)
train_b, test_b = get_fold0(seed=42)
train_c, test_c = get_fold0(seed=123)

assert np.array_equal(train_a, train_b), "FAIL: same seed gave different train splits"
assert np.array_equal(test_a, test_b),  "FAIL: same seed gave different test splits"
assert not np.array_equal(test_a, test_c), "FAIL: different seeds gave identical test splits"

print(f"  seed=42  fold-0 test sites: {sorted(test_a)}")
print(f"  seed=42  fold-0 test sites: {sorted(test_b)}  (repeat — must match)")
print(f"  seed=123 fold-0 test sites: {sorted(test_c)}  (must differ)")
print("  KFold checks PASSED\n")

# ---------------------------------------------------------------------------
# 2. MCMC reproducibility
# ---------------------------------------------------------------------------
print("=== 2. MCMC reproducibility ===")

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gdmbayes import spGDMM, ModelConfig, SamplerConfig, PreprocessorConfig

def fit_model(seed):
    model = spGDMM(
        preprocessor=PreprocessorConfig(deg=3, knots=1, distance_measure="euclidean"),
        model_config=ModelConfig(variance="homogeneous", spatial_effect="none", alpha_importance=False),
        sampler_config=SamplerConfig(draws=200, tune=200, chains=2, nuts_sampler="nutpie",
                                     progressbar=False, random_seed=seed),
    )
    model.fit(X, y)
    # Return posterior mean of beta (I-spline coefficients); fit() returns self, idata in model.idata
    return float(model.idata.posterior["beta"].values.mean())

print("  Fitting seed=42  (run 1) ...")
mean_42a = fit_model(42)
print(f"    posterior mean coef[0] = {mean_42a:.6f}")

print("  Fitting seed=42  (run 2) ...")
mean_42b = fit_model(42)
print(f"    posterior mean coef[0] = {mean_42b:.6f}")

print("  Fitting seed=123 ...")
mean_123 = fit_model(123)
print(f"    posterior mean coef[0] = {mean_123:.6f}")

assert mean_42a == mean_42b, f"FAIL: same seed gave different posteriors ({mean_42a} vs {mean_42b})"
assert mean_42a != mean_123, f"FAIL: different seeds gave identical posteriors"

print(f"\n  Same seed match:    {mean_42a} == {mean_42b}  ✓")
print(f"  Different seed diff: {mean_42a} != {mean_123}  ✓")
print("  MCMC checks PASSED\n")

print("All reproducibility checks passed.")
