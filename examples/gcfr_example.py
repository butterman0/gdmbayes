"""
Greater Cape Floristic Region (GCFR) GDM example (White et al. 2024 dataset)
=============================================================================
Fits the frequentist GDM and the Bayesian spGDMM to the GCFR plant family
survey data used by White et al. (2024) Table 1.

Dataset
-------
413 sites in the Greater Cape Floristic Region (South Africa), 52 plant
families, 7 environmental predictors.  Coordinates: longitude/latitude.
Source: White et al. (2024) Zenodo repository (https://zenodo.org/records/10091442)
        file: GCFR_family.csv

White et al. (2024) Table 1 benchmarks (10-fold CV):
  Ferrier (R gdm)              RMSE = 0.0786
  Best spGDMM (Model 8)        CRPS = 0.0550  RMSE = 0.0822  MAE = 0.0618
  Best spGDMM (Model 5)        CRPS = 0.0564  RMSE = 0.0859  MAE = 0.0640
  spGDMM no-spatial (Model 1)  CRPS = 0.0590  RMSE = 0.0928  MAE = 0.0685

Usage
-----
  python gcfr_example.py --mode freq
  python gcfr_example.py --mode bayes --spatial abs_diff
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from properscoring import crps_ensemble

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["freq", "bayes", "both"], default="freq")
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--tune", type=int, default=1000)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--spatial", choices=["none", "abs_diff", "squared_diff"], default="none")
parser.add_argument("--output_dir", type=str, default="results/gcfr")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data — White et al. (2024) GCFR dataset
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

gcfr = pd.read_csv(os.path.join(DATA_DIR, "gcfr_family.csv"))

# Match White et al. R code: location_mat = longitude/latitude,
# envr_use = gmap, RFL_CONC, Elevation30m, HeatLoadIndex30m,
#             tmean13c, SoilConductivitymSm, SoilTotalNPercent (7 predictors)
ENV_PREDICTORS = [
    "gmap", "RFL_CONC", "Elevation30m", "HeatLoadIndex30m",
    "tmean13c", "SoilConductivitymSm", "SoilTotalNPercent",
]

FAMILY_COLS = [
    c for c in gcfr.columns
    if c not in ["plot", "longitude", "latitude"] + ENV_PREDICTORS
    and c not in ["tminave01c", "tminave07c"]  # not used by White et al.
]

X = pd.DataFrame({
    "xc":       gcfr["longitude"].values,
    "yc":       gcfr["latitude"].values,
    "time_idx": 0,
})
for col in ENV_PREDICTORS:
    X[col] = gcfr[col].values

# Bray-Curtis dissimilarities from family-level % cover matrix
y = pdist(gcfr[FAMILY_COLS].values, metric="braycurtis")

print(f"X shape: {X.shape}  ({len(gcfr)} sites × {len(ENV_PREDICTORS)} predictors + coords)")
print(f"y shape: {y.shape}  ({len(y)} site pairs)")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
print(f"y == 1.0: {(y == 1.0).sum()} pairs")
print(f"N families: {len(FAMILY_COLS)}")
for col in ENV_PREDICTORS:
    print(f"  {col}: [{gcfr[col].min():.3f}, {gcfr[col].max():.3f}]")
print()


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
    print("FREQUENTIST GDM — GCFR")
    print("=" * 60)

    def make_gdm():
        return GDM(
            geo=True,
            splines=3,
            knots=2,  # White et al. used knots=2 for GCFR (df = deg + knots = 5)
            preprocessor_config=PreprocessorConfig(
                deg=3,
                knots=2,
                mesh_choice="percentile",
                distance_measure="geodesic",
            ),
        )

    gdm = make_gdm()
    gdm.fit(X, y)
    y_pred_train = gdm.predict(X)

    # 10-fold CV on site pairs (matches White et al. 2024 methodology)
    n_pairs = len(y)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred_cv = np.full(n_pairs, np.nan)
    for _, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_pairs))):
        gdm_cv = make_gdm()
        gdm_cv.fit(X, y, pair_subset=train_idx)
        y_pred_cv[test_idx] = gdm_cv.predict(X)[test_idx]

    r_train = rmse(y, y_pred_train)
    m_train = mae(y, y_pred_train)
    c_train = crps_point(y, y_pred_train)
    r_cv = rmse(y, y_pred_cv)
    m_cv = mae(y, y_pred_cv)
    c_cv = crps_point(y, y_pred_cv)
    corr, _ = pearsonr(y, y_pred_train)

    print(f"\nDeviance explained (train) : {gdm.explained_:.4f}")
    print(f"RMSE (train)               : {r_train:.4f}")
    print(f"MAE  (train)               : {m_train:.4f}")
    print(f"CRPS (train)               : {c_train:.4f}  (= MAE for point forecast)")
    print(f"RMSE (10-fold CV)          : {r_cv:.4f}  (White 2024 Ferrier: 0.0786)")
    print(f"MAE  (10-fold CV)          : {m_cv:.4f}")
    print(f"CRPS (10-fold CV)          : {c_cv:.4f}  (= MAE for point forecast)")
    print(f"Pearson r (train)          : {corr:.4f}")
    print(f"\nPredictor importance:")
    for name, imp in gdm.predictor_importance_.items():
        print(f"  {name:30s}  {imp:.4f}")

    pd.DataFrame({"y_obs": y, "y_pred_train": y_pred_train, "y_pred_cv": y_pred_cv}).to_csv(
        os.path.join(args.output_dir, "gcfr_freq_predictions.csv"), index=False
    )

# ---------------------------------------------------------------------------
# 2. Bayesian spGDMM
# ---------------------------------------------------------------------------
if args.mode in ("bayes", "both"):
    from gdmbayes import spGDMM, ModelConfig, SamplerConfig, PreprocessorConfig

    print("=" * 60)
    print(f"BAYESIAN spGDMM — GCFR  (spatial_effect={args.spatial!r})")
    print("=" * 60)

    out_nc = os.path.join(args.output_dir, f"gcfr_spgdmm_{args.spatial}.nc")

    if os.path.exists(out_nc):
        print(f"Loading saved model from {out_nc}")
        model = spGDMM.load(out_nc)
    else:
        model = spGDMM(
            preprocessor=PreprocessorConfig(
                deg=3,
                knots=2,  # White et al. used knots=2 for GCFR (df=5)
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

    r_b = rmse(y, y_pred_bayes)
    m_b = mae(y, y_pred_bayes)
    c_b = crps_samples(y, samples_da)
    corr_b, _ = pearsonr(y, y_pred_bayes)

    print(f"\nRMSE: {r_b:.4f}")
    print(f"MAE:  {m_b:.4f}")
    print(f"CRPS: {c_b:.4f}")
    print(f"r:    {corr_b:.4f}")
    pd.DataFrame({"y_obs": y, "y_pred_bayes": y_pred_bayes}).to_csv(
        os.path.join(args.output_dir, f"gcfr_spgdmm_predictions_{args.spatial}.csv"),
        index=False,
    )

print("\nDone.")
