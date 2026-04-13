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

White et al. (2024) Table 1 benchmarks (10-fold CV, BCI/Panama):
  Ferrier (R gdm)                         RMSE = 0.0934  MAE = 0.0716
  Model 1  none / homogeneous             CRPS = 0.0527  RMSE = 0.0954  MAE = 0.0779
  Model 2  none / dist-variance           CRPS = 0.0511  RMSE = 0.0937  MAE = 0.0734
  Model 4  abs_diff / homogeneous         CRPS = 0.0490  RMSE = 0.0878  MAE = 0.0690
  Model 5  abs_diff / dist-variance       CRPS = 0.0479  RMSE = 0.0879  MAE = 0.0654
  Model 7  squared_diff / homogeneous     CRPS = 0.0472  RMSE = 0.0856  MAE = 0.0658
  Model 8  squared_diff / dist-variance   CRPS = 0.0450  RMSE = 0.0821  MAE = 0.0618  ← best CRPS
  (Models 3/6 with polynomial variance not reported for Panama in Table 1)

Usage
-----
  python panama_example.py --mode freq
  python panama_example.py --mode bayes
  python panama_example.py --mode bayes --config_idx 0
"""

import argparse
import datetime
import itertools
import os
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["freq", "bayes", "both"], default="freq")
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--tune", type=int, default=4000)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--config_idx", type=int, default=None,
    help="Run only this config index (0-8).  Omit to run all 9 configs."
)
parser.add_argument("--output_dir", type=str, default="results/panama")
parser.add_argument(
    "--n_folds", type=int, default=10,
    help="Number of CV folds to run (default 10).  Pass 1 to run only the first fold."
)
parser.add_argument(
    "--skip_full_model", action="store_true",
    help="Skip fitting/saving the full-data model (CV only).  Use for smoke tests."
)
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

# ---------------------------------------------------------------------------
# 1. Frequentist GDM
# ---------------------------------------------------------------------------
if args.mode in ("freq", "both"):
    from sklearn.model_selection import KFold

    from gdmbayes import GDM, site_pairs

    print("=" * 60)
    print("FREQUENTIST GDM — Panama")
    print("=" * 60)

    def make_gdm():
        return GDM(
            geo=True,
            splines=3,
            knots=1,  # White et al. used knots=1 (df = deg + knots = 4)
        )

    gdm = make_gdm()
    gdm.fit(X, y)
    y_pred_train = gdm.predict(X)

    # Site-level CV
    n_sites = len(X)
    n_pairs = len(y)
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
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
    print(f"RMSE (10-fold CV)          : {r_cv:.4f}  (White 2024 Ferrier: 0.0934)")
    print(f"MAE  (10-fold CV)          : {m_cv:.4f}")
    print(f"CRPS (10-fold CV)          : {c_cv:.4f}  (= MAE for point forecast)")
    print(f"Pearson r (train)          : {corr:.4f}")
    print("\nPredictor importance:")
    for name, imp in gdm.predictor_importance_.items():
        print(f"  {name:20s}  {imp:.4f}")

    pd.DataFrame({"y_obs": y, "y_pred_train": y_pred_train, "y_pred_cv": y_pred_cv}).to_csv(
        os.path.join(args.output_dir, "panama_freq_predictions.csv"), index=False
    )

# ---------------------------------------------------------------------------
# 2. Bayesian spGDMM — 8 model configurations × 10-fold CV
# ---------------------------------------------------------------------------
if args.mode in ("bayes", "both"):
    from sklearn.model_selection import KFold

    from gdmbayes import ModelConfig, GDMPreprocessor, SamplerConfig, site_pairs, spGDMM

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
    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)

    all_cv_metrics = []

    for cfg in configs_to_run:
        tag = f"{cfg['spatial_effect']}_{cfg['variance']}"
        print("=" * 60)
        print(f"BAYESIAN spGDMM — Panama  [{tag}]")
        print("=" * 60)

        def make_spgdmm():
            return spGDMM(
                preprocessor=GDMPreprocessor(
                    deg=3,
                    knots=1,
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

        # --- Full-data model (saved for response curves / paper figures) ---
        if not args.skip_full_model:
            out_nc = os.path.join(args.output_dir, f"panama_spgdmm_{tag}.nc")
            lock_path = out_nc + ".lock"
            with open(lock_path, "w") as _lock_f:
                import fcntl
                fcntl.flock(_lock_f, fcntl.LOCK_EX)
                if os.path.exists(out_nc):
                    print(f"  Full-data model: loading from {out_nc}")
                    spGDMM.load(out_nc)
                else:
                    full_model = make_spgdmm()
                    full_model.fit(X, y)
                    full_model.save(out_nc)
                    print(f"  Full-data model saved to {out_nc}")

        # --- Standard sklearn CV — fit on training sites, predict on test sites ---
        fold_metrics = []

        for fold, (train_sites, test_sites) in enumerate(
            itertools.islice(kf.split(np.arange(n_sites)), args.n_folds)
        ):
            train_pair_idx = site_pairs(n_sites, train_sites)
            test_pair_idx  = site_pairs(n_sites, test_sites)
            print(f"  Fold {fold + 1}/{args.n_folds} — {len(train_sites)} train sites, "
                  f"{len(test_sites)} test sites, {len(test_pair_idx)} test pairs")
            cv_model = make_spgdmm()
            cv_model.fit(
                X.iloc[train_sites].reset_index(drop=True),
                y[train_pair_idx],
            )
            import arviz as az
            rhat = az.rhat(cv_model.idata_)
            rhat_max = float(rhat.to_array().max())
            rhat_vars = {v: float(rhat[v].max()) for v in rhat.data_vars}
            print(f"  R-hat max: {rhat_max:.4f}  per-var: "
                  + "  ".join(f"{k}={v:.3f}" for k, v in rhat_vars.items()))
            X_test = X.iloc[test_sites].reset_index(drop=True)
            y_test = y[test_pair_idx]
            y_pred_mean = cv_model.predict(X_test)
            fold_metrics.append({
                "rmse": rmse(y_test, y_pred_mean),
                "mae":  mae(y_test, y_pred_mean),
                "crps": cv_model.crps(X_test, y_test),
            })

        r_cv = np.mean([m["rmse"] for m in fold_metrics])
        m_cv = np.mean([m["mae"] for m in fold_metrics])
        c_cv = np.mean([m["crps"] for m in fold_metrics])

        print(f"\n  RMSE (10-fold CV): {r_cv:.4f}  (White 2024 Model 8: 0.0821)")
        print(f"  MAE  (10-fold CV): {m_cv:.4f}  (White 2024 Model 8: 0.0618)")
        print(f"  CRPS (10-fold CV): {c_cv:.4f}  (White 2024 Model 8: 0.0450)")

        all_cv_metrics.append({
            "dataset": "Panama",
            "config_tag": tag,
            "seed": args.seed,
            "spatial_effect": cfg["spatial_effect"],
            "variance": cfg["variance"],
            "RMSE_CV": r_cv,
            "MAE_CV": m_cv,
            "CRPS_CV": c_cv,
            "n_folds": args.n_folds,
            "n_pairs": n_pairs,
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            "commit": _git_hash(),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        })

    metrics_path = os.path.join(args.output_dir, "panama_cv_metrics.csv")
    new_df = pd.DataFrame(all_cv_metrics)
    import fcntl
    with open(metrics_path + ".lock", "w") as _csv_lock:
        fcntl.flock(_csv_lock, fcntl.LOCK_EX)
        if os.path.exists(metrics_path):
            existing = pd.read_csv(metrics_path)
            mask = ~(
                existing["config_tag"].isin(new_df["config_tag"]) &
                (existing["seed"] == args.seed) &
                (existing["n_folds"] == args.n_folds)
            )
            existing = existing[mask]
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_csv(metrics_path, index=False)
    print(f"\nCV metrics saved to {metrics_path}")

print("\nDone.")
