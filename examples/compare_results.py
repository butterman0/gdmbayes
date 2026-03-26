"""
Compare gdmbayes results across datasets and model configurations
against White et al. (2024) Table 1 benchmarks.

Loads ``*_cv_metrics.csv`` files written by each example script and prints
a unified comparison table.  White et al. benchmarks are loaded from
``benchmarks/white2024_table1.csv`` (10-fold CV, posterior-mean predictions).

Usage
-----
  python compare_results.py
  python compare_results.py --results_dir /path/to/results
"""

import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_dir", type=str,
    default=os.path.join(os.path.dirname(__file__), "results"),
)
parser.add_argument(
    "--no_poly", action="store_true",
    help="Exclude polynomial variance models (3/6/9) from output.",
)
args = parser.parse_args()

RESULTS_DIR = args.results_dir


# ---------------------------------------------------------------------------
# White et al. (2024) Table 1 benchmarks (10-fold CV)
# Loaded from benchmarks/white2024_table1.csv.
# ---------------------------------------------------------------------------
_BENCHMARK_CSV = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "white2024_table1.csv")
_bench = pd.read_csv(_BENCHMARK_CSV)

# Map CSV dataset names to the short names used in our results
_DATASET_MAP = {"GCFR Family": "GCFR", "GCFR Species": "GCFR Species"}

# Map CSV model numbers to config_tags and display labels
_SPATIAL_LABELS = {"none": "none", "abs_diff": "abs_diff", "squared_diff": "sq_diff"}
_VARIANCE_LABELS = {
    "homogeneous": "hom", "covariate_dependent": "cov_dep", "polynomial": "poly",
}

WHITE_BENCHMARKS = []
for _, r in _bench.iterrows():
    ds = _DATASET_MAP.get(r["dataset"], r["dataset"])
    mn = str(r["model_number"])
    if mn in ("naive", "ferrier"):
        config_tag = mn
        label = f"White: {r['model_name']}"
    else:
        se = r["spatial_effect"]
        var = r["variance"]
        if args.no_poly and var == "polynomial":
            continue
        config_tag = f"{se}_{var}"
        se_short = _SPATIAL_LABELS.get(se, se)
        var_short = _VARIANCE_LABELS.get(var, var)
        label = f"White: Model {mn} ({se_short}/{var_short})"
    crps = r["CRPS"] if pd.notna(r.get("CRPS")) else None
    rmse = r["RMSE"] if pd.notna(r.get("RMSE")) else None
    mae = r["MAE"] if pd.notna(r.get("MAE")) else None
    WHITE_BENCHMARKS.append(dict(
        dataset=ds, model=label, config_tag=config_tag,
        RMSE_CV=rmse, MAE_CV=mae, CRPS_CV=crps,
    ))

rows = list(WHITE_BENCHMARKS)


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   —  "
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Load R gdm (Ferrier re-run) results
# ---------------------------------------------------------------------------
_R_GDM_CSV = os.path.join(RESULTS_DIR, "r_gdm_results.csv")
if os.path.exists(_R_GDM_CSV):
    r_gdm_df = pd.read_csv(_R_GDM_CSV)
    for _, r in r_gdm_df.iterrows():
        rows.append(dict(
            dataset=r["dataset"],
            model=str(r["model"]),
            config_tag="r_gdm_ferrier",
            RMSE_CV=r.get("RMSE_CV"), MAE_CV=r.get("MAE_CV"), CRPS_CV=None,
        ))

# ---------------------------------------------------------------------------
# Load frequentist CV summaries
# ---------------------------------------------------------------------------
FREQ_FILES = {
    "SW Australia": os.path.join(RESULTS_DIR, "southwest", "southwest_freq_predictions.csv"),
    "Panama":       os.path.join(RESULTS_DIR, "panama",    "panama_freq_predictions.csv"),
    "GCFR":         os.path.join(RESULTS_DIR, "gcfr",      "gcfr_freq_predictions.csv"),
}

FREQ_PRED_COLS = {
    "SW Australia": ("y_obs", "y_pred_cv"),
    "Panama":       ("y_obs", "y_pred_cv"),
    "GCFR":         ("y_obs", "y_pred_cv"),
}

for ds, fpath in FREQ_FILES.items():
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath)
    y_col, p_col = FREQ_PRED_COLS[ds]
    if p_col not in df.columns:
        continue
    mask = ~np.isnan(df[p_col].values)
    y_obs = df[y_col].values[mask]
    y_pred = df[p_col].values[mask]
    r = float(np.sqrt(np.mean((y_obs - y_pred) ** 2)))
    m = float(np.mean(np.abs(y_obs - y_pred)))
    rows.append(dict(
        dataset=ds,
        model="gdmbayes GDM (freq, 10-fold CV)",
        config_tag="gdmbayes_freq",
        RMSE_CV=r, MAE_CV=m, CRPS_CV=m,  # CRPS = MAE for point forecast
    ))

# ---------------------------------------------------------------------------
# Load Bayesian CV metrics CSVs
# ---------------------------------------------------------------------------
BAYES_FILES = {
    "SW Australia": os.path.join(RESULTS_DIR, "southwest", "southwest_cv_metrics.csv"),
    "Panama":       os.path.join(RESULTS_DIR, "panama",    "panama_cv_metrics.csv"),
    "GCFR":         os.path.join(RESULTS_DIR, "gcfr",      "gcfr_cv_metrics.csv"),
}

MODEL_LABELS = {
    "none_homogeneous":          "Model 1: none / homogeneous",
    "none_covariate_dependent":  "Model 2: none / cov_dep",
    "none_polynomial":           "Model 3: none / polynomial",
    "abs_diff_homogeneous":      "Model 4: abs_diff / homogeneous",
    "abs_diff_covariate_dependent": "Model 5: abs_diff / cov_dep",
    "abs_diff_polynomial":       "Model 6: abs_diff / polynomial",
    "squared_diff_homogeneous":  "Model 7: sq_diff / homogeneous",
    "squared_diff_covariate_dependent": "Model 8: sq_diff / cov_dep",
    "squared_diff_polynomial":          "Model 9: sq_diff / polynomial",
}

for ds, fpath in BAYES_FILES.items():
    if not os.path.exists(fpath):
        continue
    cv_df = pd.read_csv(fpath)
    for _, row in cv_df.iterrows():
        tag = row["config_tag"]
        if args.no_poly and "polynomial" in tag:
            continue
        label = MODEL_LABELS.get(tag, tag)
        n_folds = int(row.get("n_folds_run", row.get("n_folds", 0)))
        rows.append(dict(
            dataset=ds,
            model=f"gdmbayes spGDMM {label} ({n_folds}-fold CV)",
            config_tag=tag,
            RMSE_CV=row.get("RMSE_CV"),
            MAE_CV=row.get("MAE_CV"),
            CRPS_CV=row.get("CRPS_CV"),
        ))

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
df = pd.DataFrame(rows)

# Sort so each dataset's White benchmarks appear next to gdmbayes results
_ds_order = {ds: i for i, ds in enumerate(df["dataset"].unique())}
df["_ds_rank"] = df["dataset"].map(_ds_order)
df["_is_white"] = df["model"].str.startswith("White:").map({True: 0, False: 1})
df = df.sort_values(["_ds_rank", "_is_white", "model"]).drop(columns=["_ds_rank", "_is_white"])

print("\n" + "=" * 100)
print(f"{'Dataset':<16} {'Model':<52} {'RMSE':>7} {'MAE':>7} {'CRPS':>7}")
print("=" * 100)

current_ds = None
for _, row in df.iterrows():
    if row["dataset"] != current_ds:
        if current_ds is not None:
            print()
        current_ds = row["dataset"]
    print(f"{row['dataset']:<16} {str(row['model']):<52} "
          f"{fmt(row['RMSE_CV']):>7} {fmt(row['MAE_CV']):>7} {fmt(row['CRPS_CV']):>7}")

print("=" * 100)
print("Notes:")
print("  White et al. metrics: 10-fold CV on pairs; posterior-mean point prediction.")
print("  gdmbayes Bayesian:    10-fold CV on pairs.")
print("  CRPS for frequentist = MAE (point forecast is a degenerate distribution).")
print()

# Save to CSV
out_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Table saved to {out_path}")
