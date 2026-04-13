"""
Panama diagnostic test — 1-fold CV with verbose output.

Tests whether LogNormal(0, sigma=10) beta prior works with:
  - 4000 tuning steps
  - target_accept=0.97
  - nutpie sampler

Prints detailed diagnostics: beta posteriors, divergences, ESS, R-hat,
predictions, and timing.
"""

import argparse
import itertools
import os
import time
import warnings
import numpy as np
import pandas as pd
import arviz as az
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--config_idx", type=int, required=True)
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--tune", type=int, default=3000)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

from gdmbayes import spGDMM, ModelConfig, SamplerConfig, GDMPreprocessor, site_pairs
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
env = pd.read_csv(os.path.join(DATA_DIR, "panama_env.csv"))
species = pd.read_csv(os.path.join(DATA_DIR, "panama_species.csv"), index_col=0)

ENV_PREDICTORS = ["precip", "elev"]
X = pd.DataFrame({
    "xc":       env["EW coord"].values,
    "yc":       env["NS coord"].values,
    "time_idx": 0,
    "precip":   env["precip"].values,
    "elev":     env["elev"].values,
})

y = pdist(species.values, metric="braycurtis")

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"y range: [{y.min():.4f}, {y.max():.4f}], y==1.0: {(y==1.0).sum()}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIGS = [
    dict(spatial_effect="none",          variance="homogeneous"),           # 0
    dict(spatial_effect="none",          variance="covariate_dependent"),   # 1
    dict(spatial_effect="none",          variance="polynomial"),            # 2
    dict(spatial_effect="abs_diff",      variance="homogeneous"),           # 3
    dict(spatial_effect="abs_diff",      variance="covariate_dependent"),   # 4
    dict(spatial_effect="abs_diff",      variance="polynomial"),            # 5
    dict(spatial_effect="squared_diff",  variance="homogeneous"),           # 6
    dict(spatial_effect="squared_diff",  variance="covariate_dependent"),   # 7
    dict(spatial_effect="squared_diff",  variance="polynomial"),            # 8
]

cfg = CONFIGS[args.config_idx]
tag = f"{cfg['spatial_effect']}_{cfg['variance']}"

print(f"\n{'='*60}")
print(f"CONFIG {args.config_idx}: {tag}")
print(f"tune={args.tune}, draws={args.draws}, chains={args.chains}, target_accept=0.97")
print(f"beta prior: LogNormal(mu=0, sigma=10)")
print(f"{'='*60}\n")

def make_spgdmm():
    return spGDMM(
        preprocessor=GDMPreprocessor(
            deg=3, knots=1,
            mesh_choice="percentile",
            distance_measure="euclidean",
        ),
        model_config=ModelConfig(
            variance=cfg["variance"],
            spatial_effect=cfg["spatial_effect"],
            alpha_importance=False,
        ),
        sampler_config=SamplerConfig(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=0.97,
            nuts_sampler="nutpie",
            progressbar=True,
            random_seed=args.seed,
        ),
    )

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def mae(a, b):
    return np.mean(np.abs(a - b))

def print_diagnostics(model, label):
    """Print detailed MCMC diagnostics."""
    idata = model.idata
    print(f"\n--- {label} DIAGNOSTICS ---")

    # Divergences
    if hasattr(idata, "sample_stats"):
        divs = idata.sample_stats.get("diverging", None)
        if divs is not None:
            n_div = int(divs.values.sum())
            n_total = int(divs.values.size)
            print(f"Divergences: {n_div}/{n_total} ({100*n_div/n_total:.1f}%)")

    # Beta posteriors
    if "beta" in idata.posterior:
        beta_vals = idata.posterior["beta"].values  # (chains, draws, n_cols)
        beta_mean = beta_vals.mean(axis=(0, 1))
        beta_std = beta_vals.std(axis=(0, 1))
        beta_median = np.median(beta_vals, axis=(0, 1))
        print(f"\nbeta posteriors (n={len(beta_mean)}):")
        for i in range(len(beta_mean)):
            q5 = np.percentile(beta_vals[:, :, i], 5)
            q95 = np.percentile(beta_vals[:, :, i], 95)
            print(f"  beta[{i}]: mean={beta_mean[i]:.4f}  median={beta_median[i]:.4f}  "
                  f"std={beta_std[i]:.4f}  90%CI=[{q5:.4f}, {q95:.4f}]")

    # beta_0
    if "beta_0" in idata.posterior:
        b0 = idata.posterior["beta_0"].values
        print(f"\nbeta_0: mean={b0.mean():.4f}  std={b0.std():.4f}")

    # sigma2 or beta_sigma
    if "sigma2" in idata.posterior:
        s2 = idata.posterior["sigma2"].values
        print(f"sigma2: mean={s2.mean():.4f}  std={s2.std():.4f}")
    if "beta_sigma" in idata.posterior:
        bs = idata.posterior["beta_sigma"].values
        bs_mean = bs.mean(axis=(0, 1))
        print(f"beta_sigma: {bs_mean}")

    # R-hat and ESS
    try:
        summary = az.summary(idata, var_names=["beta", "beta_0"], round_to=4)
        print(f"\nR-hat range: [{summary['r_hat'].min():.4f}, {summary['r_hat'].max():.4f}]")
        print(f"ESS bulk range: [{summary['ess_bulk'].min():.0f}, {summary['ess_bulk'].max():.0f}]")
        print(f"ESS tail range: [{summary['ess_tail'].min():.0f}, {summary['ess_tail'].max():.0f}]")
    except Exception as e:
        print(f"Summary failed: {e}")

    print(f"--- END {label} DIAGNOSTICS ---\n")


# ---------------------------------------------------------------------------
# 1. Full-data fit with diagnostics
# ---------------------------------------------------------------------------
print("=" * 60)
print("FULL-DATA FIT")
print("=" * 60)
t0 = time.time()
full_model = make_spgdmm()
full_model.fit(X, y)
t_full = time.time() - t0
print(f"\nFull-data fit time: {t_full:.0f}s ({t_full/60:.1f}min)")

print_diagnostics(full_model, "FULL-DATA")

# Quick prediction check
y_pred_full = full_model.predict(X)
print(f"Full-data predictions: min={y_pred_full.min():.4f}, max={y_pred_full.max():.4f}, "
      f"mean={y_pred_full.mean():.4f}")
print(f"RMSE (train): {rmse(y, y_pred_full):.4f}")
print(f"MAE  (train): {mae(y, y_pred_full):.4f}")

# Check for degenerate predictions (all ≈ 1.0)
if y_pred_full.mean() > 0.95:
    print("WARNING: Predictions look degenerate (mean > 0.95) — beta likely exploded!")

# ---------------------------------------------------------------------------
# 2. One-fold CV
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("1-FOLD CV")
print("=" * 60)

n_sites = len(X)
n_pairs = len(y)
kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)

train_sites, test_sites = next(iter(kf.split(np.arange(n_sites))))
train_pair_idx = site_pairs(n_sites, train_sites)
test_pair_idx = site_pairs(n_sites, test_sites)

print(f"Train: {len(train_sites)} sites ({len(train_pair_idx)} pairs)")
print(f"Test:  {len(test_sites)} sites ({len(test_pair_idx)} pairs)")

X_train = X.iloc[train_sites].reset_index(drop=True)
y_train = y[train_pair_idx]
X_test = X.iloc[test_sites].reset_index(drop=True)

t0 = time.time()
cv_model = make_spgdmm()
cv_model.fit(X_train, y_train)
t_cv = time.time() - t0
print(f"\nCV fold fit time: {t_cv:.0f}s ({t_cv/60:.1f}min)")

print_diagnostics(cv_model, "CV-FOLD")

y_pred_cv = cv_model.predict(X_test)

print(f"CV predictions: min={y_pred_cv.min():.4f}, max={y_pred_cv.max():.4f}, "
      f"mean={y_pred_cv.mean():.4f}")
print(f"RMSE (fold 1): {rmse(y[test_pair_idx], y_pred_cv):.4f}")
print(f"MAE  (fold 1): {mae(y[test_pair_idx], y_pred_cv):.4f}")

crps = cv_model.crps(X_test, y[test_pair_idx])
print(f"CRPS (fold 1): {crps:.4f}")

if y_pred_cv.mean() > 0.95:
    print("WARNING: CV predictions look degenerate!")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"SUMMARY — config {args.config_idx} ({tag})")
print(f"{'='*60}")
print(f"Full-data fit: {t_full:.0f}s, CV fold fit: {t_cv:.0f}s")
print(f"Total time: {(t_full + t_cv):.0f}s ({(t_full + t_cv)/60:.1f}min)")
print(f"Train RMSE: {rmse(y, y_pred_full):.4f}")
print(f"CV fold RMSE: {rmse(y[test_pair_idx], y_pred_cv):.4f}")
print(f"CV fold CRPS: {crps:.4f}")
degenerate = y_pred_full.mean() > 0.95 or y_pred_cv.mean() > 0.95
print(f"Degenerate: {'YES — FAILED' if degenerate else 'No — looks good'}")
