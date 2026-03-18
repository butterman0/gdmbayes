"""
Compare gdmbayes results across datasets and model configurations
against White et al. (2024) Table 1 benchmarks.

Loads ``*_cv_metrics.csv`` files written by each example script and prints
a unified comparison table.  White et al. benchmarks are hardcoded from
Table 1 of the published paper (10-fold CV, posterior-mean predictions).

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
args = parser.parse_args()

RESULTS_DIR = args.results_dir


# ---------------------------------------------------------------------------
# White et al. (2024) Table 1 benchmarks (10-fold CV)
# Model numbering follows White et al.; CRPS is the proper scoring rule.
# CRPS not reported for Ferrier (R gdm) — it returns only point predictions.
# ---------------------------------------------------------------------------
WHITE_BENCHMARKS = [
    # SW Australia (94 sites, 3 predictors, knots=1)
    dict(dataset="SW Australia", model="White: Ferrier (R gdm)", config_tag="ferrier",
         RMSE_CV=0.0737, MAE_CV=0.0549, CRPS_CV=None),
    dict(dataset="SW Australia", model="White: spGDMM best", config_tag="best",
         RMSE_CV=0.0731, MAE_CV=0.0545, CRPS_CV=None),

    # Panama (39 sites, 2 predictors, knots=1)
    dict(dataset="Panama", model="White: Ferrier (R gdm)", config_tag="ferrier",
         RMSE_CV=0.0716, MAE_CV=None, CRPS_CV=None),
    dict(dataset="Panama", model="White: spGDMM Model 1 (none/hom)", config_tag="none_homogeneous",
         RMSE_CV=0.0954, MAE_CV=0.0779, CRPS_CV=0.0527),
    dict(dataset="Panama", model="White: spGDMM Model 5 (sq_diff/hom)", config_tag="squared_diff_homogeneous",
         RMSE_CV=0.0879, MAE_CV=0.0654, CRPS_CV=0.0479),
    dict(dataset="Panama", model="White: spGDMM Model 8 (sq_diff/cov_dep)", config_tag="best",
         RMSE_CV=0.0821, MAE_CV=0.0618, CRPS_CV=0.0450),

    # GCFR (413 sites, 7 predictors, knots=2)
    dict(dataset="GCFR", model="White: Ferrier (R gdm)", config_tag="ferrier",
         RMSE_CV=0.0786, MAE_CV=None, CRPS_CV=None),
    dict(dataset="GCFR", model="White: spGDMM Model 1 (none/hom)", config_tag="none_homogeneous",
         RMSE_CV=0.0928, MAE_CV=0.0685, CRPS_CV=0.0590),
    dict(dataset="GCFR", model="White: spGDMM Model 5 (sq_diff/hom)", config_tag="squared_diff_homogeneous",
         RMSE_CV=0.0859, MAE_CV=0.0640, CRPS_CV=0.0564),
    dict(dataset="GCFR", model="White: spGDMM Model 8 (sq_diff/cov_dep)", config_tag="best",
         RMSE_CV=0.0822, MAE_CV=0.0618, CRPS_CV=0.0550),
]

rows = list(WHITE_BENCHMARKS)


def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   —  "
    return f"{v:.4f}"


# ---------------------------------------------------------------------------
# Load frequentist CV summaries
# ---------------------------------------------------------------------------
FREQ_FILES = {
    "SW Australia": os.path.join(RESULTS_DIR, "southwest", "southwest_freq_summary.csv"),
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
}

for ds, fpath in BAYES_FILES.items():
    if not os.path.exists(fpath):
        continue
    cv_df = pd.read_csv(fpath)
    for _, row in cv_df.iterrows():
        tag = row["config_tag"]
        label = MODEL_LABELS.get(tag, tag)
        n_folds = int(row.get("n_folds", "?"))
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
print("  gdmbayes Bayesian:    10-fold CV (SW/Panama) or 5-fold (GCFR) on pairs.")
print("  CRPS for frequentist = MAE (point forecast is a degenerate distribution).")
print()

# Save to CSV
out_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Table saved to {out_path}")
