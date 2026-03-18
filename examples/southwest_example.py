"""
Southwest Australia GDM example
================================
Fits the frequentist GDM and the Bayesian spGDMM to the canonical southwest
Australia dataset from the R gdm package, then compares performance metrics
against the published benchmarks in White et al. (2024) Table 1.

Dataset
-------
94 sites in SW Australia, 865 plant species, 7 environmental predictors +
geographic distance.  Bray-Curtis dissimilarities pre-computed from the R
community matrix using vegan::vegdist.

White et al. (2024) Table 1 benchmark on this dataset (SW Australia):
  Ferrier (R gdm)   RMSE = 0.0737  MAE = 0.0549
  Best spGDMM model RMSE = 0.0731  MAE = 0.0545

Usage
-----
  # Frequentist only (fast, interactive):
  python southwest_example.py --mode freq

  # Bayesian (slow, meant for sbatch):
  python southwest_example.py --mode bayes

  # Both:
  python southwest_example.py --mode both
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from properscoring import crps_ensemble

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["freq", "bayes", "both"], default="freq")
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--tune", type=int, default=1000)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--spatial", choices=["none", "abs_diff", "squared_diff"], default="none")
parser.add_argument("--output_dir", type=str, default="results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

sites = pd.read_csv(os.path.join(DATA_DIR, "southwest_sites.csv"))
y_df = pd.read_csv(os.path.join(DATA_DIR, "southwest_y.csv"))
y = y_df["y"].values

# Build X: coordinates + 7 environmental predictors (no site ID column)
# Columns: site, awcA, phTotal, sandA, shcA, solumDepth, bio5, bio6, bio15, bio18, bio19, Lat, Long
# We use Lat/Long as coordinates (xc=Long, yc=Lat) and the 7 env predictors.
# White et al. used: bio5, bio6, bio15, bio18, bio19, awcA, phTotal (7 predictors)
ENV_PREDICTORS = ["awcA", "phTotal", "sandA", "shcA", "solumDepth",
                  "bio5", "bio6", "bio15", "bio18", "bio19"]

X = sites.rename(columns={"Long": "xc", "Lat": "yc"}).copy()
X["time_idx"] = 0
# Keep only the columns gdmbayes expects: xc, yc, time_idx, predictors
X = X[["xc", "yc", "time_idx"] + ENV_PREDICTORS]

print(f"X shape: {X.shape}  (94 sites × {len(ENV_PREDICTORS)} predictors + coords)")
print(f"y shape: {y.shape}  (4371 site pairs)")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
print(f"y == 1.0: {(y == 1.0).sum()} pairs\n")

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def crps_point(y_true, y_pred):
    """CRPS for a point forecast equals MAE."""
    return mae(y_true, y_pred)

def crps_samples(y_true, samples_da):
    """CRPS from a DataArray of posterior predictive samples (log scale → exp back)."""
    vals = np.exp(samples_da.values)
    sample_axis = list(samples_da.dims).index("sample")
    if sample_axis == 0:
        vals = vals.T  # → (n_obs, n_samples)
    return crps_ensemble(y_true, vals).mean()

# ---------------------------------------------------------------------------
# 1. Frequentist GDM
# ---------------------------------------------------------------------------
if args.mode in ("freq", "both"):
    from gdmbayes import GDM, PreprocessorConfig
    from sklearn.model_selection import KFold

    print("=" * 60)
    print("FREQUENTIST GDM")
    print("=" * 60)

    def make_gdm():
        return GDM(
            geo=True,
            splines=3,
            knots=2,
            preprocessor_config=PreprocessorConfig(
                deg=3,
                knots=2,
                mesh_choice="percentile",
                distance_measure="geodesic",
            ),
        )

    # --- Full-data fit (training metrics) ---
    gdm = make_gdm()
    gdm.fit(X, y)
    y_pred_train = gdm.predict(X)

    # --- 10-fold CV on site pairs (matches White et al. 2024 methodology) ---
    # Preprocessor is fitted on all sites; only NNLS uses the training pairs.
    n_pairs = len(y)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred_cv = np.full(n_pairs, np.nan)

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_pairs))):
        gdm_cv = make_gdm()
        gdm_cv.fit(X, y, pair_subset=train_idx)
        y_pred_cv[test_idx] = gdm_cv.predict(X)[test_idx]

    r_cv = rmse(y, y_pred_cv)
    m_cv = mae(y, y_pred_cv)
    c_cv = crps_point(y, y_pred_cv)

    r_train = rmse(y, y_pred_train)
    m_train = mae(y, y_pred_train)
    c_train = crps_point(y, y_pred_train)
    corr, _ = pearsonr(y, y_pred_train)

    print(f"\nDeviance explained (train) : {gdm.explained_:.4f}")
    print(f"RMSE (train)               : {r_train:.4f}")
    print(f"MAE  (train)               : {m_train:.4f}")
    print(f"CRPS (train)               : {c_train:.4f}  (= MAE for point forecast)")
    print(f"RMSE (10-fold CV)          : {r_cv:.4f}  (White 2024 Ferrier: 0.0737)")
    print(f"MAE  (10-fold CV)          : {m_cv:.4f}  (White 2024 Ferrier: 0.0549)")
    print(f"CRPS (10-fold CV)          : {c_cv:.4f}  (= MAE for point forecast)")
    print(f"Pearson r (train)          : {corr:.4f}")
    print(f"\nPredictor importance:")
    for name, imp in gdm.predictor_importance_.items():
        print(f"  {name:20s}  {imp:.4f}")

    # Save results
    pd.DataFrame({"y_obs": y, "y_pred_train": y_pred_train, "y_pred_cv": y_pred_cv}).to_csv(
        os.path.join(args.output_dir, "southwest_freq_predictions.csv"), index=False
    )
    pd.DataFrame([{
        "model": "gdmbayes GDM (frequentist)",
        "RMSE_train": r_train, "MAE_train": m_train, "CRPS_train": c_train,
        "RMSE_10fold_CV": r_cv, "MAE_10fold_CV": m_cv, "CRPS_10fold_CV": c_cv,
        "Pearson_r": corr, "deviance_explained": gdm.explained_,
    }]).to_csv(os.path.join(args.output_dir, "southwest_freq_summary.csv"), index=False)
    print(f"\nResults saved to {args.output_dir}/")

# ---------------------------------------------------------------------------
# 2. Bayesian spGDMM
# ---------------------------------------------------------------------------
if args.mode in ("bayes", "both"):
    from gdmbayes import spGDMM, ModelConfig, SamplerConfig, PreprocessorConfig

    print("=" * 60)
    print(f"BAYESIAN spGDMM  (spatial_effect={args.spatial!r})")
    print("=" * 60)

    out_nc = os.path.join(args.output_dir, f"southwest_spgdmm_{args.spatial}.nc")

    if os.path.exists(out_nc):
        print(f"Loading saved model from {out_nc}")
        model = spGDMM.load(out_nc)
    else:
        model = spGDMM(
            preprocessor=PreprocessorConfig(
                deg=3,
                knots=2,
                mesh_choice="percentile",
                distance_measure="geodesic",
            ),
            model_config=ModelConfig(
                variance="homogeneous",
                spatial_effect=args.spatial,
                alpha_importance=True,
            ),
            sampler_config=SamplerConfig(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                target_accept=0.95,
                nuts_sampler="nutpie",
                progressbar=True,
                random_seed=args.seed,
            ),
        )
        model.fit(X, y)
        model.save(out_nc)
        print(f"Model saved to {out_nc}")

    # Full posterior predictive distribution (log scale → exp back to dissimilarity)
    samples_da = model.predict_posterior(X, combined=True)
    y_pred_bayes = np.exp(samples_da.mean(dim="sample").values)

    r_bayes = rmse(y, y_pred_bayes)
    m_bayes = mae(y, y_pred_bayes)
    c_bayes = crps_samples(y, samples_da)
    corr_b, _ = pearsonr(y, y_pred_bayes)

    print(f"\nRMSE (posterior mean): {r_bayes:.4f}")
    print(f"MAE  (posterior mean): {m_bayes:.4f}")
    print(f"CRPS                 : {c_bayes:.4f}")
    print(f"Pearson r            : {corr_b:.4f}")

    bayes_results = pd.DataFrame({"y_obs": y, "y_pred_bayes": y_pred_bayes})
    bayes_results.to_csv(
        os.path.join(args.output_dir, f"southwest_spgdmm_predictions_{args.spatial}.csv"),
        index=False,
    )

print("\nDone.")
