"""Diagnostic rerun of models 4 & 7: is the psi R-hat=1.74 a real mixing
failure or just label symmetry?

For each model, after fitting we:
  1. Run az.rhat on parameters as before.
  2. Predict the full 741-pair vector *with chain dimension preserved*.
  3. Compute per-chain RMSE/MAE/CRPS on the held-out pairs.
  4. Compute az.rhat on the predicted log_y for held-out pairs.
  5. Save idata_ to results/idata_<tag>.nc for future use.

If only psi has high R-hat but per-chain metrics agree and log_y R-hat ≈ 1,
the likelihood-level posterior is actually well-converged and the raw psi
R-hat reflects the shift/sign symmetry in |psi[i]-psi[j]| / (psi[i]-psi[j])^2.
"""

import json
import os
import sys
import time

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
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

diag_rows = []
for model_num, cfg in CONFIGS:
    tag = f"model_{model_num}"
    print(f"\n=== {tag}  spatial={cfg['spatial_effect']} "
          f"(tune={TUNE}, draws={DRAWS}, chains={CHAINS}) ===")
    idata_path = os.path.join(RESULTS, f"idata_{tag}.nc")

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
    if os.path.exists(idata_path):
        # Reload previously-sampled idata and rebuild the PyMC model so
        # predict_posterior() works without any fresh MCMC.
        print(f"  loading cached {idata_path} (skipping refit)")
        spg.idata_ = az.from_netcdf(idata_path)
        spg._generate_and_preprocess_model_data(
            X.iloc[train_sites].reset_index(drop=True), Z[train_pair_idx]
        )
        spg.build_model()
        fit_s = float("nan")
    else:
        t0 = time.time()
        spg.fit(X.iloc[train_sites].reset_index(drop=True), Z[train_pair_idx])
        fit_s = time.time() - t0
        print(f"  fit time = {fit_s:.1f}s")
        spg.idata_.to_netcdf(idata_path)
        print(f"  wrote {idata_path}")

    # Parameter R-hat (per variable)
    rhat = az.rhat(spg.idata_)
    rhat_vars = {v: float(rhat[v].max()) for v in rhat.data_vars}
    rhat_max = max(rhat_vars.values())
    print(f"  param r-hat max = {rhat_max:.4f}")
    for k, v in rhat_vars.items():
        print(f"    {k} = {v:.3f}")

    # Predict with chain dimension preserved
    y_post = spg.predict_posterior(X, combined=False, extend_idata=False)
    print(f"  y dims = {y_post.dims}  shape = {y_post.shape}")
    non_cd = [d for d in y_post.dims if d not in ("chain", "draw")]
    pair_dim = non_cd[0]
    y_samples = y_post.transpose("chain", "draw", pair_dim).values  # (chain, draw, n_pairs)
    assert y_samples.shape[-1] == test_mask.size

    Z_true = Z[test_mask]

    # Per-chain predictive metrics
    per_chain = []
    for c in range(y_samples.shape[0]):
        hold = y_samples[c][:, test_mask]  # (draw, pair_hold)
        pm = hold.mean(axis=0)
        r = rmse(Z_true, pm)
        m = mae(Z_true, pm)
        crps = float(crps_ensemble(Z_true, hold.T).mean())
        per_chain.append((r, m, crps))
        print(f"  chain {c}: RMSE={r:.4f}  MAE={m:.4f}  CRPS={crps:.4f}")

    per_chain_arr = np.array(per_chain)
    rmse_min = float(per_chain_arr[:, 0].min())
    rmse_max = float(per_chain_arr[:, 0].max())
    print(f"  chain RMSE range: {rmse_min:.4f} - {rmse_max:.4f}  "
          f"(spread {rmse_max - rmse_min:.4f})")

    # R-hat on predicted held-out y (dissimilarity scale)
    y_hold = y_samples[:, :, test_mask]
    da = xr.DataArray(y_hold, dims=["chain", "draw", "pair"])
    rhat_pred = az.rhat(xr.Dataset({"y": da}))
    rhat_pred_max = float(rhat_pred["y"].max())
    rhat_pred_mean = float(rhat_pred["y"].mean())
    print(f"  predicted y R-hat: max = {rhat_pred_max:.4f}  "
          f"mean = {rhat_pred_mean:.4f}")

    # Pooled metrics (all chains)
    pooled = y_samples.reshape(-1, y_samples.shape[-1])[:, test_mask]
    pm_pool = pooled.mean(axis=0)
    r_pool = rmse(Z_true, pm_pool)
    m_pool = mae(Z_true, pm_pool)
    crps_pool = float(crps_ensemble(Z_true, pooled.T).mean())
    print(f"  pooled: RMSE={r_pool:.4f}  MAE={m_pool:.4f}  CRPS={crps_pool:.4f}")

    diag_rows.append({
        "model": tag,
        "rhat_param_max": rhat_max,
        "rhat_pred_logy_max": rhat_pred_max,
        "rhat_pred_logy_mean": rhat_pred_mean,
        "chain_rmse_min": rmse_min,
        "chain_rmse_max": rmse_max,
        "chain_rmse_spread": rmse_max - rmse_min,
        "pooled_rmse": r_pool,
        "pooled_mae": m_pool,
        "pooled_crps": crps_pool,
    })

out = pd.DataFrame(diag_rows)
out_csv = os.path.join(RESULTS, "bayes_ours_diagnose_47.csv")
out.to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")
print(out.to_string(index=False))
