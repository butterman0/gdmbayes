"""Re-run only models 4 and 7 with more tuning for R-hat.

Original 24304190 gave R-hat ≈ 1.74 on the spatial models after 1000 tune;
bump to 3000 tune / 1000 draws / 4 chains and overwrite only rows 4 and 7 in
results/bayes_ours.csv.
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


def touches_mask(n_sites: int, test_sites) -> np.ndarray:
    row, col = np.triu_indices(n_sites, k=1)
    s = set(np.asarray(test_sites).tolist())
    return np.array([(r in s) or (c in s) for r, c in zip(row, col)])


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p): return float(np.mean(np.abs(y - p)))


fold_meta = json.load(open(os.path.join(FOLD, "fold.json")))
test_sites_0 = pd.read_csv(os.path.join(FOLD, "test_sites_py.csv"))["site"].values

env = pd.read_csv(os.path.join(DATA, "Panama_env.csv"))
species = pd.read_csv(os.path.join(DATA, "Panama_species.csv"), index_col=0)
X = pd.DataFrame({
    "xc": env["EW coord"].values,
    "yc": env["NS coord"].values,
    "time_idx": 0,
    "precip": env["precip"].values,
    "elev": env["elev"].values,
})
n_sites = len(X)
Z = pdist(species.values, metric="braycurtis")
test_mask = touches_mask(n_sites, test_sites_0)
train_sites = np.setdiff1d(np.arange(n_sites), test_sites_0)
train_pair_idx = site_pairs(n_sites, train_sites)

CONFIGS = [
    (4, dict(spatial_effect="abs_diff",     variance="homogeneous")),
    (7, dict(spatial_effect="squared_diff", variance="homogeneous")),
]

DRAWS = int(os.environ.get("DRAWS", 1000))
TUNE = int(os.environ.get("TUNE", 3000))
CHAINS = int(os.environ.get("CHAINS", 4))
TARGET_ACCEPT = float(os.environ.get("TARGET_ACCEPT", 0.97))
SEED = 42

new_rows = []
for model_num, cfg in CONFIGS:
    tag = f"model_{model_num}"
    print(f"\n=== {tag}  spatial={cfg['spatial_effect']}  variance={cfg['variance']} "
          f"(tune={TUNE}, draws={DRAWS}, chains={CHAINS}) ===")
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
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            nuts_sampler="nutpie", progressbar=False,
            random_seed=SEED,
        ),
    )
    t0 = time.time()
    spg.fit(X.iloc[train_sites].reset_index(drop=True), Z[train_pair_idx])
    fit_s = time.time() - t0
    rhat = az.rhat(spg.idata_)
    rhat_max = float(rhat.to_array().max())
    rhat_vars = {v: float(rhat[v].max()) for v in rhat.data_vars}
    print(f"  fit time={fit_s:.1f}s   r-hat max={rhat_max:.4f}")
    print(f"  r-hat per var: {', '.join(f'{k}={v:.3f}' for k, v in rhat_vars.items())}")

    y_post = spg.predict_posterior(X, combined=True, extend_idata=False)
    pair_dim = next(d for d in y_post.dims if d != "sample")
    y_samples_full = y_post.transpose(pair_dim, "sample").values
    y_samples_hold = y_samples_full[test_mask]
    pred_mean = y_samples_hold.mean(axis=-1)
    Z_true = Z[test_mask]
    r = rmse(Z_true, pred_mean)
    m = mae(Z_true, pred_mean)
    c = float(crps_ensemble(Z_true, y_samples_hold).mean())
    print(f"  RMSE={r:.4f}  MAE={m:.4f}  CRPS={c:.4f}")
    new_rows.append({
        "dataset": "Panama",
        "implementation": "gdmbayes_bayes",
        "model": tag, "deg": 3, "knots": 1, "df": 4,
        "RMSE": r, "MAE": m, "CRPS": c,
        "n_train_sites": int(len(train_sites)),
        "n_test_sites": int(len(test_sites_0)),
        "n_test_pairs": int(test_mask.sum()),
        "mcmc_time_s": fit_s, "rhat_max": rhat_max,
        "n_chains": CHAINS, "tune": TUNE, "draws": DRAWS,
    })

existing = pd.read_csv(os.path.join(RESULTS, "bayes_ours.csv"))
existing = existing[~existing["model"].isin(["model_4", "model_7"])]
updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
updated = updated.sort_values("model").reset_index(drop=True)
updated.to_csv(os.path.join(RESULTS, "bayes_ours.csv"), index=False)
print(f"\nWrote results/bayes_ours.csv with updated models 4, 7.")
