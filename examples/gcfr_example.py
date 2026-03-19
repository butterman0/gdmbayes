"""
Greater Cape Floristic Region (GCFR) GDM example (White et al. 2024 dataset)
=============================================================================
Fits the frequentist GDM and the Bayesian spGDMM (8 model configurations)
to the GCFR plant family survey data used by White et al. (2024) Table 1.

Dataset
-------
413 sites in the Greater Cape Floristic Region (South Africa), 52 plant
families, 7 environmental predictors.  Coordinates: longitude/latitude.
Source: White et al. (2024) Zenodo repository (https://zenodo.org/records/10091442)
        file: GCFR_family.csv

White et al. (2024) Table 1 benchmarks (10-fold CV, GCFR family-level):
  Naive                                   RMSE = 0.2075  MAE = 0.1711
  Ferrier (R gdm)                         RMSE = 0.1922  MAE = 0.1569
  Model 1  none / homogeneous             CRPS = 0.1104  RMSE = 0.1916  MAE = 0.1566
  Model 2  none / dist-variance           CRPS = 0.1106  RMSE = 0.1919  MAE = 0.1568
  Model 3  none / mean-variance           CRPS = 0.1110  RMSE = 0.1926  MAE = 0.1573
  Model 4  abs_diff / homogeneous         CRPS = 0.0982  RMSE = 0.1701  MAE = 0.1350
  Model 5  abs_diff / dist-variance       CRPS = 0.0972  RMSE = 0.1683  MAE = 0.1340  ← best RMSE/MAE
  Model 6  abs_diff / mean-variance       CRPS = 0.0971  RMSE = 0.1698  MAE = 0.1342  ← best CRPS
  Model 7  squared_diff / homogeneous     CRPS = 0.1016  RMSE = 0.1757  MAE = 0.1401
  Model 8  squared_diff / dist-variance   CRPS = 0.1020  RMSE = 0.1763  MAE = 0.1413
  Model 9  squared_diff / mean-variance   CRPS = 0.1045  RMSE = 0.1823  MAE = 0.1454

Usage
-----
  python gcfr_example.py --mode freq
  python gcfr_example.py --mode bayes
  # Single config for SLURM array job (--array=0-7):
  python gcfr_example.py --mode bayes --config_idx 0
"""

import argparse
import itertools
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
parser.add_argument(
    "--config_idx", type=int, default=None,
    help="Run only this config index (0-8).  Omit to run all 9 configs."
)
parser.add_argument("--output_dir", type=str, default="results/gcfr")
parser.add_argument(
    "--n_folds", type=int, default=5,
    help="Number of CV folds to run (default 5).  Pass 1 to run only the first fold."
)
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
    return crps_ensemble(np.asarray(y_true), vals).mean()


# ---------------------------------------------------------------------------
# 1. Frequentist GDM
# ---------------------------------------------------------------------------
if args.mode in ("freq", "both"):
    from gdmbayes import GDM, PreprocessorConfig, site_pairs
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

    # Site-level CV
    n_sites = len(X)
    n_pairs = len(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = np.full(n_pairs, np.nan)
    for _, (train_sites, test_sites) in enumerate(
        itertools.islice(kf.split(np.arange(n_sites)), args.n_folds)
    ):
        train_pair_idx = site_pairs(n_sites, train_sites)
        test_pair_idx = site_pairs(n_sites, test_sites)
        X_train = X.iloc[train_sites].reset_index(drop=True)
        y_train = y[train_pair_idx]
        X_test = X.iloc[test_sites].reset_index(drop=True)
        gdm_cv = make_gdm()
        gdm_cv.fit(X_train, y_train)
        y_pred_cv[test_pair_idx] = gdm_cv.predict(X_test)

    cv_mask = ~np.isnan(y_pred_cv)
    r_train = rmse(y, y_pred_train)
    m_train = mae(y, y_pred_train)
    c_train = crps_point(y, y_pred_train)
    r_cv = rmse(y[cv_mask], y_pred_cv[cv_mask])
    m_cv = mae(y[cv_mask], y_pred_cv[cv_mask])
    c_cv = crps_point(y[cv_mask], y_pred_cv[cv_mask])
    corr, _ = pearsonr(y, y_pred_train)

    print(f"\nDeviance explained (train) : {gdm.explained_:.4f}")
    print(f"RMSE (train)               : {r_train:.4f}")
    print(f"MAE  (train)               : {m_train:.4f}")
    print(f"CRPS (train)               : {c_train:.4f}  (= MAE for point forecast)")
    print(f"RMSE (10-fold CV)          : {r_cv:.4f}  (White 2024 Ferrier: 0.1922)")
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
# 2. Bayesian spGDMM — 8 model configurations × 5-fold CV
# ---------------------------------------------------------------------------
if args.mode in ("bayes", "both"):
    from gdmbayes import spGDMM, ModelConfig, SamplerConfig, PreprocessorConfig, site_pairs
    from sklearn.model_selection import KFold

    # Model grid matching White et al. (2024) Table 1 (Models 1-9)
    CONFIGS = [
        dict(spatial_effect="none",          variance="homogeneous"),           # Model 1
        dict(spatial_effect="none",          variance="covariate_dependent"),   # Model 2
        dict(spatial_effect="none",          variance="polynomial"),            # Model 3
        dict(spatial_effect="abs_diff",      variance="homogeneous"),           # Model 4
        dict(spatial_effect="abs_diff",      variance="covariate_dependent"),   # Model 5
        dict(spatial_effect="abs_diff",      variance="polynomial"),            # Model 6
        dict(spatial_effect="squared_diff",  variance="homogeneous"),           # Model 7
        dict(spatial_effect="squared_diff",  variance="covariate_dependent"),   # Model 8
        dict(spatial_effect="squared_diff",  variance="polynomial"),            # Model 9
    ]

    configs_to_run = (
        [CONFIGS[args.config_idx]] if args.config_idx is not None else CONFIGS
    )

    n_sites = len(X)
    n_pairs = len(y)
    # 5-fold CV for GCFR: each fit is ~2h so 5 × 8 = 40h total, split into array jobs
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_cv_metrics = []

    for cfg in configs_to_run:
        tag = f"{cfg['spatial_effect']}_{cfg['variance']}"
        print("=" * 60)
        print(f"BAYESIAN spGDMM — GCFR  [{tag}]")
        print("=" * 60)

        def make_spgdmm():
            return spGDMM(
                preprocessor=PreprocessorConfig(
                    deg=3,
                    knots=2,  # White et al. used knots=2 for GCFR (df=5)
                    mesh_choice="percentile",
                    distance_measure="geodesic",
                ),
                model_config=ModelConfig(
                    variance=cfg["variance"],
                    spatial_effect=cfg["spatial_effect"],
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

        # --- Full-data model (saved for response curves / paper figures) ---
        out_nc = os.path.join(args.output_dir, f"gcfr_spgdmm_{tag}.nc")
        if os.path.exists(out_nc):
            print(f"  Full-data model: loading from {out_nc}")
            full_model = spGDMM.load(out_nc)
        else:
            full_model = make_spgdmm()
            full_model.fit(X, y)
            full_model.save(out_nc)
            print(f"  Full-data model saved to {out_nc}")

        # --- Site-level CV ---
        y_pred_cv = np.full(n_pairs, np.nan)
        crps_vals = []

        for fold, (train_sites, test_sites) in enumerate(
            itertools.islice(kf.split(np.arange(n_sites)), args.n_folds)
        ):
            train_pair_idx = site_pairs(n_sites, train_sites)
            test_pair_idx = site_pairs(n_sites, test_sites)
            print(f"  Fold {fold + 1}/{args.n_folds} — {len(train_sites)} train sites "
                  f"({len(train_pair_idx)} pairs), {len(test_sites)} test sites "
                  f"({len(test_pair_idx)} pairs)")
            X_train = X.iloc[train_sites].reset_index(drop=True)
            y_train = y[train_pair_idx]
            X_test = X.iloc[test_sites].reset_index(drop=True)
            cv_model = make_spgdmm()
            cv_model.fit(X_train, y_train)
            samples_da = cv_model.predict_posterior(X_test, combined=True)
            y_pred_cv[test_pair_idx] = np.exp(samples_da.mean(dim="sample").values)
            crps_vals.append(crps_samples(y[test_pair_idx], samples_da))

        cv_mask = ~np.isnan(y_pred_cv)
        r_cv = rmse(y[cv_mask], y_pred_cv[cv_mask])
        m_cv = mae(y[cv_mask], y_pred_cv[cv_mask])
        c_cv = float(np.mean(crps_vals))

        print(f"\n  RMSE ({args.n_folds}-fold CV): {r_cv:.4f}  (White 2024 M5 best RMSE: 0.1683)")
        print(f"  MAE  ({args.n_folds}-fold CV): {m_cv:.4f}  (White 2024 M5 best MAE:  0.1340)")
        print(f"  CRPS ({args.n_folds}-fold CV): {c_cv:.4f}  (White 2024 M6 best CRPS: 0.0971)")

        pd.DataFrame({"y_obs": y, "y_pred_cv": y_pred_cv}).to_csv(
            os.path.join(args.output_dir, f"gcfr_spgdmm_{tag}_cv_predictions.csv"),
            index=False,
        )

        all_cv_metrics.append({
            "dataset": "GCFR",
            "config_tag": tag,
            "spatial_effect": cfg["spatial_effect"],
            "variance": cfg["variance"],
            "RMSE_CV": r_cv,
            "MAE_CV": m_cv,
            "CRPS_CV": c_cv,
            "n_folds_run": len(crps_vals),
            "n_pairs_scored": int(cv_mask.sum()),
            "n_pairs": n_pairs,
        })

    metrics_path = os.path.join(args.output_dir, "gcfr_cv_metrics.csv")
    new_df = pd.DataFrame(all_cv_metrics)
    if os.path.exists(metrics_path) and args.config_idx is not None:
        existing = pd.read_csv(metrics_path)
        existing = existing[~existing["config_tag"].isin(new_df["config_tag"])]
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(metrics_path, index=False)
    print(f"\nCV metrics saved to {metrics_path}")

print("\nDone.")
