"""
Panama GDM example (White et al. 2024 dataset)
===============================================
Fits the frequentist GDM and the Bayesian spGDMM (8 model configurations)
to the Panama tree survey data used by White et al. (2024) Table 1.

Dataset
-------
39 sites spread across Panama (87 km × 83 km), 802 tree species.
Environmental predictors: precipitation (mm/yr), elevation (m).
Coordinates: UTM Zone 17N (EW, NS in metres).
Source: White et al. (2024) Zenodo repository (https://zenodo.org/records/10091442)
        files: Panama_env.csv, Panama_species.csv

White et al. (2024) Table 1 benchmarks (10-fold CV):
  Ferrier (R gdm)              RMSE = 0.0716
  Best spGDMM (Model 8)        CRPS = 0.0450  RMSE = 0.0821  MAE = 0.0618
  Best spGDMM (Model 5)        CRPS = 0.0479  RMSE = 0.0879  MAE = 0.0654
  spGDMM no-spatial (Model 1)  CRPS = 0.0527  RMSE = 0.0954  MAE = 0.0779

Usage
-----
  python panama_example.py --mode freq
  python panama_example.py --mode bayes
  python panama_example.py --mode bayes --config_idx 0
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
parser.add_argument(
    "--config_idx", type=int, default=None,
    help="Run only this config index (0-7).  Omit to run all 8 configs."
)
parser.add_argument("--output_dir", type=str, default="results/panama")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data — White et al. (2024) Panama dataset
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

env = pd.read_csv(os.path.join(DATA_DIR, "panama_env.csv"))
species = pd.read_csv(os.path.join(DATA_DIR, "panama_species.csv"), index_col=0)

# Match White et al. R code: location_mat = cols 2:3, envr_use = cols 4:5
# Columns: site no., EW coord, NS coord, precip, elev, age, geology
ENV_PREDICTORS = ["precip", "elev"]

X = pd.DataFrame({
    "xc":       env["EW coord"].values,
    "yc":       env["NS coord"].values,
    "time_idx": 0,
    "precip":   env["precip"].values,
    "elev":     env["elev"].values,
})

# Bray-Curtis dissimilarities from species abundance matrix
y = pdist(species.values, metric="braycurtis")

print(f"X shape: {X.shape}  ({len(env)} sites × {len(ENV_PREDICTORS)} predictors + coords)")
print(f"y shape: {y.shape}  ({len(y)} site pairs)")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
print(f"y == 1.0: {(y == 1.0).sum()} pairs")
print(f"Precip range: [{env['precip'].min():.1f}, {env['precip'].max():.1f}] mm/yr")
print(f"Elev range:   [{env['elev'].min():.0f}, {env['elev'].max():.0f}] m\n")


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def crps_point(y_true, y_pred):
    """CRPS for a point forecast equals MAE."""
    return mae(y_true, y_pred)

def crps_samples(y_true, samples_da, idx=None):
    """CRPS from a DataArray of posterior predictive samples (log scale → exp back)."""
    vals = np.exp(samples_da.values)
    sample_axis = list(samples_da.dims).index("sample")
    if sample_axis == 0:
        vals = vals.T  # → (n_obs, n_samples)
    if idx is not None:
        return crps_ensemble(np.asarray(y_true)[idx], vals[idx]).mean()
    return crps_ensemble(y_true, vals).mean()


# ---------------------------------------------------------------------------
# 1. Frequentist GDM
# ---------------------------------------------------------------------------
if args.mode in ("freq", "both"):
    from gdmbayes import GDM, PreprocessorConfig
    from sklearn.model_selection import KFold

    print("=" * 60)
    print("FREQUENTIST GDM — Panama")
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
                distance_measure="euclidean",
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
    print(f"RMSE (10-fold CV)          : {r_cv:.4f}  (White 2024 Ferrier: 0.0716)")
    print(f"MAE  (10-fold CV)          : {m_cv:.4f}")
    print(f"CRPS (10-fold CV)          : {c_cv:.4f}  (= MAE for point forecast)")
    print(f"Pearson r (train)          : {corr:.4f}")
    print(f"\nPredictor importance:")
    for name, imp in gdm.predictor_importance_.items():
        print(f"  {name:20s}  {imp:.4f}")

    pd.DataFrame({"y_obs": y, "y_pred_train": y_pred_train, "y_pred_cv": y_pred_cv}).to_csv(
        os.path.join(args.output_dir, "panama_freq_predictions.csv"), index=False
    )

# ---------------------------------------------------------------------------
# 2. Bayesian spGDMM — 8 model configurations × 10-fold CV
# ---------------------------------------------------------------------------
if args.mode in ("bayes", "both"):
    from gdmbayes import spGDMM, ModelConfig, SamplerConfig, PreprocessorConfig
    from sklearn.model_selection import KFold

    # Model grid matching White et al. (2024) Table 1 (Models 1-8)
    CONFIGS = [
        dict(spatial_effect="none",          variance="homogeneous"),           # Model 1
        dict(spatial_effect="none",          variance="covariate_dependent"),   # Model 2
        dict(spatial_effect="none",          variance="polynomial"),            # Model 3
        dict(spatial_effect="abs_diff",      variance="homogeneous"),           # Model 4
        dict(spatial_effect="abs_diff",      variance="covariate_dependent"),   # Model 5
        dict(spatial_effect="abs_diff",      variance="polynomial"),            # Model 6
        dict(spatial_effect="squared_diff",  variance="homogeneous"),           # Model 7
        dict(spatial_effect="squared_diff",  variance="covariate_dependent"),   # Model 8 (best)
    ]

    configs_to_run = (
        [CONFIGS[args.config_idx]] if args.config_idx is not None else CONFIGS
    )

    n_pairs = len(y)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    all_cv_metrics = []

    for cfg in configs_to_run:
        tag = f"{cfg['spatial_effect']}_{cfg['variance']}"
        print("=" * 60)
        print(f"BAYESIAN spGDMM — Panama  [{tag}]")
        print("=" * 60)

        def make_spgdmm():
            return spGDMM(
                preprocessor=PreprocessorConfig(
                    deg=3,
                    knots=1,
                    mesh_choice="percentile",
                    distance_measure="euclidean",
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
        out_nc = os.path.join(args.output_dir, f"panama_spgdmm_{tag}.nc")
        if os.path.exists(out_nc):
            print(f"  Full-data model: loading from {out_nc}")
            full_model = spGDMM.load(out_nc)
        else:
            full_model = make_spgdmm()
            full_model.fit(X, y)
            full_model.save(out_nc)
            print(f"  Full-data model saved to {out_nc}")

        # --- 10-fold CV on site pairs ---
        y_pred_cv = np.full(n_pairs, np.nan)
        crps_vals = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_pairs))):
            print(f"  Fold {fold + 1:2d}/10 — {len(train_idx)} train pairs, "
                  f"{len(test_idx)} test pairs")
            cv_model = make_spgdmm()
            cv_model.fit(X, y, pair_subset=train_idx)

            samples_da = cv_model.predict_posterior(X, combined=True)
            y_pred_all = np.exp(samples_da.mean(dim="sample").values)
            y_pred_cv[test_idx] = y_pred_all[test_idx]
            crps_vals.append(crps_samples(y, samples_da, idx=test_idx))

        r_cv = rmse(y, y_pred_cv)
        m_cv = mae(y, y_pred_cv)
        c_cv = float(np.mean(crps_vals))

        print(f"\n  RMSE (10-fold CV): {r_cv:.4f}  (White 2024 Model 8: 0.0821)")
        print(f"  MAE  (10-fold CV): {m_cv:.4f}  (White 2024 Model 8: 0.0618)")
        print(f"  CRPS (10-fold CV): {c_cv:.4f}  (White 2024 Model 8: 0.0450)")

        pd.DataFrame({"y_obs": y, "y_pred_cv": y_pred_cv}).to_csv(
            os.path.join(args.output_dir, f"panama_spgdmm_{tag}_cv_predictions.csv"),
            index=False,
        )

        all_cv_metrics.append({
            "dataset": "Panama",
            "config_tag": tag,
            "spatial_effect": cfg["spatial_effect"],
            "variance": cfg["variance"],
            "RMSE_CV": r_cv,
            "MAE_CV": m_cv,
            "CRPS_CV": c_cv,
            "n_folds": 10,
            "n_pairs": n_pairs,
        })

    metrics_path = os.path.join(args.output_dir, "panama_cv_metrics.csv")
    new_df = pd.DataFrame(all_cv_metrics)
    if os.path.exists(metrics_path) and args.config_idx is not None:
        existing = pd.read_csv(metrics_path)
        existing = existing[~existing["config_tag"].isin(new_df["config_tag"])]
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(metrics_path, index=False)
    print(f"\nCV metrics saved to {metrics_path}")

print("\nDone.")
