"""Our Bayesian spGDMM on the single shared Panama fold.

Runs the 4 model configurations that correspond to White's nimble_code
{1, 2, 4, 7}:

    model 1 : spatial_effect='none'          variance='homogeneous'
    model 2 : spatial_effect='none'          variance='covariate_dependent'
    model 4 : spatial_effect='abs_diff'      variance='homogeneous'
    model 7 : spatial_effect='squared_diff'  variance='homogeneous'

Fits on 36 training sites and predicts on the full 39-site design,
then extracts the 111 held-out pair predictions (those touching at least
one of the 3 held-out sites).  Predictions clipped at 1.0.
"""

import json
import os
import sys
import time

import arviz as az
import numpy as np
import pandas as pd
from properscoring import crps_ensemble
from scipy.spatial.distance import pdist

sys.path.insert(0, "/cluster/home/haroldh/spgdmm/src")
from gdmbayes import (  # noqa: E402
    GDMPreprocessor,
    ModelConfig,
    SamplerConfig,
    site_pairs,
    spGDMM,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FOLD = os.path.join(HERE, "fold")
DATA = "/cluster/home/haroldh/spgdmm/external/spGDMM-code/data"
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def touches_mask(n_sites: int, test_sites) -> np.ndarray:
    row, col = np.triu_indices(n_sites, k=1)
    s = set(np.asarray(test_sites).tolist())
    return np.array([(r in s) or (c in s) for r, c in zip(row, col)])


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


fold_meta = json.load(open(os.path.join(FOLD, "fold.json")))
test_sites_0 = pd.read_csv(os.path.join(FOLD, "test_sites_py.csv"))["site"].values

env = pd.read_csv(os.path.join(DATA, "Panama_env.csv"))
species = pd.read_csv(os.path.join(DATA, "Panama_species.csv"), index_col=0)
X = pd.DataFrame(
    {
        "xc": env["EW coord"].values,
        "yc": env["NS coord"].values,
        "time_idx": 0,
        "precip": env["precip"].values,
        "elev": env["elev"].values,
    }
)
n_sites = len(X)
assert n_sites == fold_meta["n_sites"]
Z = pdist(species.values, metric="braycurtis")

test_mask = touches_mask(n_sites, test_sites_0)
all_sites = np.arange(n_sites)
train_sites = np.setdiff1d(all_sites, test_sites_0)
train_pair_idx = site_pairs(n_sites, train_sites)

print(f"n_sites={n_sites}  train={len(train_sites)}  test={len(test_sites_0)}  "
      f"test_pairs={int(test_mask.sum())}")

CONFIGS = [
    (1, dict(spatial_effect="none",         variance="homogeneous")),
    (2, dict(spatial_effect="none",         variance="covariate_dependent")),
    (4, dict(spatial_effect="abs_diff",     variance="homogeneous")),
    (7, dict(spatial_effect="squared_diff", variance="homogeneous")),
]

DRAWS = int(os.environ.get("DRAWS", 1000))
TUNE = int(os.environ.get("TUNE", 1000))
CHAINS = int(os.environ.get("CHAINS", 4))
TARGET_ACCEPT = float(os.environ.get("TARGET_ACCEPT", 0.97))
SEED = 42

rows = []
for model_num, cfg in CONFIGS:
    tag = f"model_{model_num}"
    print(f"\n=== {tag}  spatial={cfg['spatial_effect']}  variance={cfg['variance']} ===")

    spg = spGDMM(
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
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            nuts_sampler="nutpie",
            progressbar=False,
            random_seed=SEED,
        ),
    )

    t0 = time.time()
    spg.fit(X.iloc[train_sites].reset_index(drop=True), Z[train_pair_idx])
    fit_s = time.time() - t0
    rhat = az.rhat(spg.idata_)
    rhat_max = float(rhat.to_array().max())
    print(f"  fit time={fit_s:.1f}s   r-hat max={rhat_max:.4f}")

    # Predict on full 39-site design; extract held-out pairs.
    y_post = spg.predict_posterior(X, combined=True, extend_idata=False)
    # y_post dims include "sample" and a pair dim; find the pair dim.
    pair_dim = next(d for d in y_post.dims if d != "sample")
    y_samples_full = y_post.transpose(pair_dim, "sample").values   # (n_pairs, n_samples)
    assert y_samples_full.shape[0] == test_mask.size, (y_samples_full.shape, test_mask.size)
    y_samples_hold = y_samples_full[test_mask]
    pred_mean = y_samples_hold.mean(axis=-1)
    Z_true = Z[test_mask]
    r = rmse(Z_true, pred_mean)
    m = mae(Z_true, pred_mean)
    c = float(crps_ensemble(Z_true, y_samples_hold).mean())
    print(f"  RMSE={r:.4f}  MAE={m:.4f}  CRPS={c:.4f}")

    rows.append(
        {
            "dataset": "Panama",
            "implementation": "gdmbayes_bayes",
            "model": tag,
            "deg": 3,
            "knots": 1,
            "df": 4,
            "RMSE": r,
            "MAE": m,
            "CRPS": c,
            "n_train_sites": int(len(train_sites)),
            "n_test_sites": int(len(test_sites_0)),
            "n_test_pairs": int(test_mask.sum()),
            "mcmc_time_s": fit_s,
            "rhat_max": rhat_max,
            "n_chains": CHAINS,
            "tune": TUNE,
            "draws": DRAWS,
        }
    )

out_csv = os.path.join(RESULTS, "bayes_ours.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")
