"""
Barro Colorado Island (BCI) GDM example
=========================================
Fits the frequentist GDM and the Bayesian spGDMM to the BCI tree census data
(Condit et al. 2002; 50 1-ha plots, 225 tree species) as used in Ferrier et al.
(2007) and White et al. (2024).

Only continuous/numeric environmental predictors are used:
  UTM.EW, UTM.NS, Precipitation, Elevation, EnvHet

Usage
-----
  python bci_example.py --mode freq
  python bci_example.py --mode bayes --spatial abs_diff
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

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["freq", "bayes", "both"], default="freq")
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--tune", type=int, default=1000)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--spatial", choices=["none", "abs_diff", "squared_diff"], default="none")
parser.add_argument("--output_dir", type=str, default="results/bci")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

sites = pd.read_csv(os.path.join(DATA_DIR, "bci_sites.csv"))
y_df = pd.read_csv(os.path.join(DATA_DIR, "bci_y.csv"))
y = y_df["y"].values

# Numeric predictors only (drop categorical Age.cat, Geology, Habitat, Stream)
# Drop predictors with zero variance (e.g. Precipitation is constant across all BCI plots)
CANDIDATE_ENV = ["Precipitation", "Elevation", "EnvHet"]
ENV_PREDICTORS = [c for c in CANDIDATE_ENV if sites[c].std() > 0]

# bci_sites.csv has UTM.EW/UTM.NS (real coordinates) and xc/yc (grid approximation)
# Use UTM coordinates for accurate geographic distances
X = sites[["UTM.EW", "UTM.NS"]].rename(columns={"UTM.EW": "xc", "UTM.NS": "yc"}).copy()
X["time_idx"] = 0
for col in ENV_PREDICTORS:
    X[col] = sites[col].values

print(f"X shape: {X.shape}  ({len(sites)} sites × {len(ENV_PREDICTORS)} predictors)")
print(f"y shape: {y.shape}  ({len(y)} site pairs)")
print(f"y range: [{y.min():.4f}, {y.max():.4f}]\n")


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
    # az.extract returns (..., sample) or (sample, ...) — put sample last for crps_ensemble
    sample_axis = list(samples_da.dims).index("sample")
    if sample_axis == 0:
        vals = vals.T  # → (n_obs, n_samples)
    return crps_ensemble(y_true, vals).mean()


# ---------------------------------------------------------------------------
# 1. Frequentist GDM
# ---------------------------------------------------------------------------
if args.mode in ("freq", "both"):
    from gdmbayes import GDM, PreprocessorConfig

    print("=" * 60)
    print("FREQUENTIST GDM — BCI")
    print("=" * 60)

    gdm = GDM(
        geo=True,
        splines=3,
        knots=2,
        preprocessor_config=PreprocessorConfig(
            deg=3,
            knots=2,
            mesh_choice="percentile",
            distance_measure="euclidean",
        ),
    )
    gdm.fit(X, y)
    y_pred = gdm.predict(X)

    r = rmse(y, y_pred)
    m = mae(y, y_pred)
    c = crps_point(y, y_pred)
    corr, _ = pearsonr(y, y_pred)

    print(f"\nDeviance explained : {gdm.explained_:.4f}")
    print(f"RMSE               : {r:.4f}")
    print(f"MAE                : {m:.4f}")
    print(f"CRPS               : {c:.4f}  (= MAE for point forecast)")
    print(f"Pearson r          : {corr:.4f}")
    print(f"\nPredictor importance:")
    for name, imp in gdm.predictor_importance_.items():
        print(f"  {name:20s}  {imp:.4f}")

    pd.DataFrame({"y_obs": y, "y_pred_freq": y_pred}).to_csv(
        os.path.join(args.output_dir, "bci_freq_predictions.csv"), index=False
    )

# ---------------------------------------------------------------------------
# 2. Bayesian spGDMM
# ---------------------------------------------------------------------------
if args.mode in ("bayes", "both"):
    from gdmbayes import spGDMM, ModelConfig, SamplerConfig, PreprocessorConfig

    print("=" * 60)
    print(f"BAYESIAN spGDMM — BCI  (spatial_effect={args.spatial!r})")
    print("=" * 60)

    out_nc = os.path.join(args.output_dir, f"bci_spgdmm_{args.spatial}.nc")

    if os.path.exists(out_nc):
        print(f"Loading saved model from {out_nc}")
        model = spGDMM.load(out_nc)
    else:
        model = spGDMM(
            preprocessor=PreprocessorConfig(
                deg=3,
                knots=2,
                mesh_choice="percentile",
                distance_measure="euclidean",
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
    y_pred_bayes = np.exp(samples_da).mean(dim="sample").values

    r_b = rmse(y, y_pred_bayes)
    m_b = mae(y, y_pred_bayes)
    c_b = crps_samples(y, samples_da)
    corr_b, _ = pearsonr(y, y_pred_bayes)

    print(f"\nRMSE: {r_b:.4f}")
    print(f"MAE:  {m_b:.4f}")
    print(f"CRPS: {c_b:.4f}")
    print(f"r:    {corr_b:.4f}")
    pd.DataFrame({"y_obs": y, "y_pred_bayes": y_pred_bayes}).to_csv(
        os.path.join(args.output_dir, f"bci_spgdmm_predictions_{args.spatial}.csv"), index=False
    )

print("\nDone.")
