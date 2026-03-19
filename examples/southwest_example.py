"""
Southwest Australia GDM example
================================
Fits the frequentist GDM and the Bayesian spGDMM (8 model configurations) to
the canonical southwest Australia dataset from the R gdm package, then
compares performance metrics against the published benchmarks in
White et al. (2024) Table 1.

Dataset
-------
94 sites in SW Australia, 865 plant species, 7 environmental predictors +
geographic distance.  Bray-Curtis dissimilarities pre-computed from the R
community matrix using vegan::vegdist.

White et al. (2024) Table 1 benchmarks (10-fold CV, SW Australia):
  Ferrier (R gdm)                         RMSE = 0.0737  MAE = 0.0549
  Model 1  none / homogeneous             CRPS = 0.0439  RMSE = 0.0790  MAE = 0.0595
  Model 2  none / dist-variance           CRPS = 0.0435  RMSE = 0.0805  MAE = 0.0608
  Model 4  abs_diff / homogeneous         CRPS = 0.0473  RMSE = 0.0840  MAE = 0.0629
  Model 5  abs_diff / dist-variance       CRPS = 0.0454  RMSE = 0.0820  MAE = 0.0626
  Model 7  squared_diff / homogeneous     CRPS = 0.0414  RMSE = 0.0731  MAE = 0.0545  ← best RMSE/MAE
  Model 8  squared_diff / dist-variance   CRPS = 0.0407  RMSE = 0.0748  MAE = 0.0556  ← best CRPS
  (Models 3/6 with polynomial variance not reported for SW Australia in Table 1)

Usage
-----
  # Frequentist only (fast, interactive):
  python southwest_example.py --mode freq

  # Bayesian — all 8 configs × 10-fold CV (meant for sbatch, ~27h):
  python southwest_example.py --mode bayes

  # Single config by index 0-7 (for SLURM array jobs):
  python southwest_example.py --mode bayes --config_idx 0
"""

import argparse
import itertools
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
parser.add_argument(
    "--config_idx", type=int, default=None,
    help="Run only this config index (0-8).  Omit to run all 9 configs."
)
parser.add_argument("--output_dir", type=str, default="results/southwest")
parser.add_argument(
    "--n_folds", type=int, default=10,
    help="Number of CV folds to run (default 10).  Pass 1 to run only the first fold."
)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

sites = pd.read_csv(os.path.join(DATA_DIR, "southwest_sites.csv"))
y_df = pd.read_csv(os.path.join(DATA_DIR, "southwest_y.csv"))
y = y_df["y"].values

# White et al. (2024) R code uses only phTotal, bio5, bio19 (3 predictors)
# with knots=1, deg=3 (df=4 per predictor). Coordinates: Long/Lat.
ENV_PREDICTORS = ["phTotal", "bio5", "bio19"]

X = sites.rename(columns={"Long": "xc", "Lat": "yc"}).copy()
X["time_idx"] = 0
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
    """CRPS from a DataArray of posterior predictive samples (log scale → exp back).

    Parameters
    ----------
    y_true : array-like of shape (n,)
    samples_da : xr.DataArray with a "sample" dimension
    """
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
    print("FREQUENTIST GDM")
    print("=" * 60)

    def make_gdm():
        return GDM(
            geo=True,
            splines=3,
            knots=1,  # White et al. used knots=1 (df = deg + knots = 4)
            preprocessor_config=PreprocessorConfig(
                deg=3,
                knots=1,
                mesh_choice="percentile",
                distance_measure="geodesic",
            ),
        )

    # --- Full-data fit (training metrics) ---
    gdm = make_gdm()
    gdm.fit(X, y)
    y_pred_train = gdm.predict(X)

    # --- Site-level CV ---
    n_sites = len(X)
    n_pairs = len(y)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred_cv = np.full(n_pairs, np.nan)

    for fold, (train_sites, test_sites) in enumerate(
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
    r_cv = rmse(y[cv_mask], y_pred_cv[cv_mask])
    m_cv = mae(y[cv_mask], y_pred_cv[cv_mask])
    c_cv = crps_point(y[cv_mask], y_pred_cv[cv_mask])

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
# 2. Bayesian spGDMM — 8 model configurations × 10-fold CV
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

    # Allow running a single config via --config_idx (for SLURM array jobs)
    configs_to_run = (
        [CONFIGS[args.config_idx]] if args.config_idx is not None else CONFIGS
    )

    n_sites = len(X)
    n_pairs = len(y)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_cv_metrics = []

    for cfg in configs_to_run:
        tag = f"{cfg['spatial_effect']}_{cfg['variance']}"
        print("=" * 60)
        print(f"BAYESIAN spGDMM — SW Australia  [{tag}]")
        print("=" * 60)

        def make_spgdmm():
            return spGDMM(
                preprocessor=PreprocessorConfig(
                    deg=3,
                    knots=1,
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
        out_nc = os.path.join(args.output_dir, f"southwest_spgdmm_{tag}.nc")
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

        print(f"\n  RMSE (10-fold CV): {r_cv:.4f}  (White 2024 M7 best RMSE: 0.0731)")
        print(f"  MAE  (10-fold CV): {m_cv:.4f}  (White 2024 M7 best MAE:  0.0545)")
        print(f"  CRPS (10-fold CV): {c_cv:.4f}  (White 2024 M8 best CRPS: 0.0407)")
        print(f"  CRPS (10-fold CV): {c_cv:.4f}")

        # Save CV predictions
        pd.DataFrame({"y_obs": y, "y_pred_cv": y_pred_cv}).to_csv(
            os.path.join(args.output_dir, f"southwest_spgdmm_{tag}_cv_predictions.csv"),
            index=False,
        )

        # Accumulate metrics
        all_cv_metrics.append({
            "dataset": "SW Australia",
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

    # Save all CV metrics to a single CSV (appending if running one config at a time)
    metrics_path = os.path.join(args.output_dir, "southwest_cv_metrics.csv")
    new_df = pd.DataFrame(all_cv_metrics)
    if os.path.exists(metrics_path) and args.config_idx is not None:
        existing = pd.read_csv(metrics_path)
        # Replace rows with matching config_tag, or append new ones
        existing = existing[~existing["config_tag"].isin(new_df["config_tag"])]
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(metrics_path, index=False)
    print(f"\nCV metrics saved to {metrics_path}")

print("\nDone.")
