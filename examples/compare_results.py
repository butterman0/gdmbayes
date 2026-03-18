"""
Compare gdmbayes results across datasets and spatial configurations
against White et al. (2024) Table 1 benchmarks.

Usage
-----
  python compare_results.py
"""

import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# ---------------------------------------------------------------------------
# White et al. (2024) Table 1 benchmarks (10-fold CV, posterior mean for Bayes)
# ---------------------------------------------------------------------------
# Model 1 = no spatial RE; Model 5 = abs_diff; Model 8 = squared_diff + dist-dep variance
# CRPS not reported for Ferrier (R gdm); RMSE/MAE reported.
WHITE_BENCHMARKS = [
    # SW Australia (94 sites, 3 predictors, knots=1)
    dict(dataset="SW Australia", model="White: Ferrier (R gdm)", spatial="—",
         RMSE_CV=0.0737, MAE_CV=0.0549, CRPS_CV=None),
    dict(dataset="SW Australia", model="White: spGDMM (best)", spatial="—",
         RMSE_CV=0.0731, MAE_CV=0.0545, CRPS_CV=None),

    # Panama (39 sites, 2 predictors, knots=1)
    dict(dataset="Panama", model="White: Ferrier (R gdm)", spatial="—",
         RMSE_CV=0.0716, MAE_CV=None, CRPS_CV=None),
    dict(dataset="Panama", model="White: spGDMM Model 1 (none)", spatial="none",
         RMSE_CV=0.0954, MAE_CV=0.0779, CRPS_CV=0.0527),
    dict(dataset="Panama", model="White: spGDMM Model 5 (sq_diff)", spatial="squared_diff",
         RMSE_CV=0.0879, MAE_CV=0.0654, CRPS_CV=0.0479),
    dict(dataset="Panama", model="White: spGDMM Model 8 (best)", spatial="—",
         RMSE_CV=0.0821, MAE_CV=0.0618, CRPS_CV=0.0450),

    # GCFR (413 sites, 7 predictors, knots=2)
    dict(dataset="GCFR", model="White: Ferrier (R gdm)", spatial="—",
         RMSE_CV=0.0786, MAE_CV=None, CRPS_CV=None),
    dict(dataset="GCFR", model="White: spGDMM Model 1 (none)", spatial="none",
         RMSE_CV=0.0928, MAE_CV=0.0685, CRPS_CV=0.0590),
    dict(dataset="GCFR", model="White: spGDMM Model 5 (sq_diff)", spatial="squared_diff",
         RMSE_CV=0.0859, MAE_CV=0.0640, CRPS_CV=0.0564),
    dict(dataset="GCFR", model="White: spGDMM Model 8 (best)", spatial="—",
         RMSE_CV=0.0822, MAE_CV=0.0618, CRPS_CV=0.0550),
]

rows = list(WHITE_BENCHMARKS)

# ---------------------------------------------------------------------------
# Load gdmbayes results
# ---------------------------------------------------------------------------

def load_freq(path, y_col="y_obs", pred_col="y_pred_cv"):
    """Load frequentist predictions CSV and compute RMSE/MAE."""
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    if pred_col not in df.columns:
        return None, None
    mask = ~np.isnan(df[pred_col].values)
    r = rmse(df[y_col].values[mask], df[pred_col].values[mask])
    m = mae(df[y_col].values[mask], df[pred_col].values[mask])
    return r, m


def load_bayes(path, y_col="y_obs", pred_col="y_pred_bayes"):
    """Load Bayesian predictions CSV (posterior mean); no CRPS here (need samples)."""
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    r = rmse(df[y_col].values, df[pred_col].values)
    m = mae(df[y_col].values, df[pred_col].values)
    return r, m


# SW Australia — frequentist
r, m = load_freq(
    os.path.join(RESULTS_DIR, "southwest", "southwest_freq_predictions.csv"),
    pred_col="y_pred_cv",
)
rows.append(dict(dataset="SW Australia", model="gdmbayes GDM (freq, 10-fold CV)",
                 spatial="—", RMSE_CV=r, MAE_CV=m, CRPS_CV=m))  # CRPS=MAE for point forecast

# SW Australia — Bayesian
for spatial in ("none", "abs_diff", "squared_diff"):
    r, m = load_bayes(
        os.path.join(RESULTS_DIR, "southwest", f"southwest_spgdmm_predictions_{spatial}.csv"),
    )
    rows.append(dict(dataset="SW Australia", model=f"gdmbayes spGDMM ({spatial})",
                     spatial=spatial, RMSE_CV=r, MAE_CV=m, CRPS_CV=None))

# Panama — frequentist
r, m = load_freq(
    os.path.join(RESULTS_DIR, "panama", "panama_freq_predictions.csv"),
    pred_col="y_pred_cv",
)
rows.append(dict(dataset="Panama", model="gdmbayes GDM (freq, 10-fold CV)",
                 spatial="—", RMSE_CV=r, MAE_CV=m, CRPS_CV=m))

# Panama — Bayesian
for spatial in ("none", "abs_diff", "squared_diff"):
    r, m = load_bayes(
        os.path.join(RESULTS_DIR, "panama", f"panama_spgdmm_predictions_{spatial}.csv"),
    )
    rows.append(dict(dataset="Panama", model=f"gdmbayes spGDMM ({spatial})",
                     spatial=spatial, RMSE_CV=r, MAE_CV=m, CRPS_CV=None))

# GCFR — frequentist
r, m = load_freq(
    os.path.join(RESULTS_DIR, "gcfr", "gcfr_freq_predictions.csv"),
    pred_col="y_pred_cv",
)
rows.append(dict(dataset="GCFR", model="gdmbayes GDM (freq, 10-fold CV)",
                 spatial="—", RMSE_CV=r, MAE_CV=m, CRPS_CV=m))

# GCFR — Bayesian
for spatial in ("none", "abs_diff", "squared_diff"):
    r, m = load_bayes(
        os.path.join(RESULTS_DIR, "gcfr", f"gcfr_spgdmm_predictions_{spatial}.csv"),
    )
    rows.append(dict(dataset="GCFR", model=f"gdmbayes spGDMM ({spatial})",
                     spatial=spatial, RMSE_CV=r, MAE_CV=m, CRPS_CV=None))

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
df = pd.DataFrame(rows)


def fmt(v):
    return f"{v:.4f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—"


print("\n" + "=" * 90)
print(f"{'Dataset':<16} {'Model':<42} {'RMSE':>7} {'MAE':>7} {'CRPS':>7}")
print("=" * 90)

current_ds = None
for _, row in df.iterrows():
    if row["dataset"] != current_ds:
        if current_ds is not None:
            print()
        current_ds = row["dataset"]
    print(f"{row['dataset']:<16} {row['model']:<42} {fmt(row['RMSE_CV']):>7} "
          f"{fmt(row['MAE_CV']):>7} {fmt(row['CRPS_CV']):>7}")

print("=" * 90)
print("Notes: gdmbayes CRPS for Bayesian models requires re-running with predict_posterior.")
print("       White et al. CRPS computed using proper scoring rule (properscoring).")
print()

# Save to CSV
out_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
df.to_csv(out_path, index=False)
print(f"Table saved to {out_path}")
