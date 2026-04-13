"""Frequentist GDM on the single shared Panama fold.

Runs our GDM at df=3 and df=4 against the fixed test-site list written
by prepare_fold.R. Predictions clipped at 1.0 to match White's metric
(model_1_CV.R:220-227).

Note on our GDM API: `splines=` is the polynomial *degree* (forwarded as
`deg=` to GDMPreprocessor), and total basis per predictor = splines + knots.
We fix deg=3 (cubic I-spline, matching White's splines2::iSpline(degree=2))
and vary knots ∈ {0, 1} to get df ∈ {3, 4}.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

sys.path.insert(0, "/cluster/home/haroldh/spgdmm/src")
from gdmbayes import GDM  # noqa: E402


def touches_mask(n_sites: int, test_sites) -> np.ndarray:
    """Boolean mask over the condensed pair vector — True for pairs that
    include at least one site in ``test_sites``.  Uses scipy's row-major
    upper-triangle order (same as ``scipy.spatial.distance.pdist``)."""
    row, col = np.triu_indices(n_sites, k=1)
    s = set(np.asarray(test_sites).tolist())
    return np.array([(r in s) or (c in s) for r, c in zip(row, col)])

HERE = os.path.dirname(os.path.abspath(__file__))
FOLD = os.path.join(HERE, "fold")
DATA = "/cluster/home/haroldh/spgdmm/external/spGDMM-code/data"
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

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

# Python-native dissimilarity, matches our existing Panama pipeline.
Z = pdist(species.values, metric="braycurtis")

# Cross-check: R exported its own Z via vegan::vegdist in upper.tri order.
Z_R = pd.read_csv(os.path.join(FOLD, "Z.csv"))["Z"].values
assert len(Z) == len(Z_R), (len(Z), len(Z_R))
# Values should match as a multiset regardless of pair ordering
assert np.allclose(np.sort(Z), np.sort(Z_R)), "Python vs R Bray-Curtis disagree on values"
print(f"OK: Python & R Bray-Curtis agree on all {len(Z)} pair values (as multiset).")

test_mask = touches_mask(n_sites, test_sites_0)
train_mask = ~test_mask
print(f"n_sites={n_sites}  test_sites={len(test_sites_0)}  "
      f"test_pairs={int(test_mask.sum())}  train_pairs={int(train_mask.sum())}")
assert int(test_mask.sum()) == fold_meta["n_held_pairs"]

all_sites = np.arange(n_sites)
train_sites = np.setdiff1d(all_sites, test_sites_0)

# Within-train pair indices (scipy condensed order) for fit; full row/col for predict.
from gdmbayes import site_pairs  # noqa: E402
train_pair_idx = site_pairs(n_sites, train_sites)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


rows = []
for df in (3, 4):
    knots = df - 3  # deg=3 (cubic), df = deg + knots
    gdm = GDM(splines=3, knots=knots, geo=True)
    gdm.fit(X.iloc[train_sites].reset_index(drop=True), Z[train_pair_idx])
    y_pred_full = gdm.predict(X)          # full 741-pair vector (scipy order)
    y_pred_hold = np.minimum(1.0, y_pred_full[test_mask])
    Z_true = Z[test_mask]
    r, m = rmse(Z_true, y_pred_hold), mae(Z_true, y_pred_hold)
    print(f"  df={df} (deg=3, knots={knots})  "
          f"RMSE={r:.4f}  MAE={m:.4f}  CRPS(point)={m:.4f}")
    rows.append(
        {
            "dataset": "Panama",
            "implementation": "gdmbayes_freq",
            "model": f"GDM_df{df}",
            "deg": 3,
            "knots": knots,
            "df": df,
            "RMSE": r,
            "MAE": m,
            "CRPS": m,
            "n_train_sites": int(len(train_sites)),
            "n_test_sites": int(len(test_sites_0)),
            "n_test_pairs": int(test_mask.sum()),
        }
    )

out_csv = os.path.join(RESULTS, "freq_gdm.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")
