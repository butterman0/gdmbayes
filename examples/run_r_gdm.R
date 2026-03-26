#!/usr/bin/env Rscript
# R gdm (Ferrier) 10-fold CV comparison script
#
# Replicates White et al. (2024) "Ferrier (R gdm)" baseline using the R gdm
# package. Settings match White's code:
#   SW Australia: formatsitepair + Long/Lat coords, splines=4 (df=4), geo=TRUE
#   Panama: manual site-pair table + UTM km coords, splines=4, geo=TRUE
#   10-fold site-level CV, seed=42
#
# Usage (from examples/ directory):
#   Rscript run_r_gdm.R
#   Rscript run_r_gdm.R --output_dir results

Sys.setenv(PROJ_DATA = "/cluster/home/haroldh/miniforge3/pkgs/proj-9.8.0-he0df7b0_0/share/proj")
suppressPackageStartupMessages({ library(gdm); library(vegan) })

args = commandArgs(trailingOnly = TRUE)
output_dir = "results"
for (i in seq_along(args)) {
  if (args[i] == "--output_dir" && i < length(args)) output_dir = args[i+1]
}
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(42)

data_dir = "/cluster/home/haroldh/spgdmm/examples/data"
rmse_fn  = function(o, p) sqrt(mean((o - p)^2))
mae_fn   = function(o, p) mean(abs(o - p))

# Build gdm site-pair table manually.
# site_df: must have columns x, y, and all pred_cols.
# dist_vec: condensed BC dissimilarities (upper-triangle row order).
# Column layout: distance, weights, s1.xCoord, s1.yCoord, s2.xCoord, s2.yCoord,
#   s1.pred1, ..., s1.predK, s2.pred1, ..., s2.predK  (all s1.* then all s2.*)
make_sitepair_df = function(site_df, dist_vec, pred_cols) {
  ns  = nrow(site_df)
  idx = which(upper.tri(matrix(0, ns, ns)), arr.ind = TRUE)
  s1  = idx[,1]; s2 = idx[,2]
  df  = data.frame(distance = dist_vec, weights = 1.0,
                   s1.xCoord = site_df$x[s1], s1.yCoord = site_df$y[s1],
                   s2.xCoord = site_df$x[s2], s2.yCoord = site_df$y[s2])
  for (col in pred_cols) df[[paste0("s1.", col)]] = site_df[[col]][s1]
  for (col in pred_cols) df[[paste0("s2.", col)]] = site_df[[col]][s2]
  class(df) = c("gdmData", "data.frame")
  df
}

# 10-fold site-level CV using site-pair tables.
# sp_tab_full: full site-pair table (gdmData) for all pairs
# site_x, site_y: coordinate vectors (length ns) for site identification
# splines_n, geo: gdm() arguments
run_cv = function(sp_tab_full, site_x, site_y, pred_cols, splines_n, k = 10, geo = TRUE) {
  ns     = length(site_x)
  folds  = sample(rep_len(1:k, ns))   # balanced fold assignment over sites
  s1x    = sp_tab_full$s1.xCoord; s1y = sp_tab_full$s1.yCoord
  s2x    = sp_tab_full$s2.xCoord; s2y = sp_tab_full$s2.yCoord

  # Map each pair row to its two site indices (1-based)
  match_site = function(xs, ys) {
    sapply(seq_along(xs), function(i) which(site_x == xs[i] & site_y == ys[i])[1L])
  }
  si1 = match_site(s1x, s1y)
  si2 = match_site(s2x, s2y)

  n_pred_total = length(pred_cols) + as.integer(geo)
  rmse_v = mae_v = numeric(k)
  for (fi in seq_len(k)) {
    train_idx = which(folds != fi); test_idx = which(folds == fi)
    tr_mask = (si1 %in% train_idx) & (si2 %in% train_idx)
    te_mask = (si1 %in% test_idx)  & (si2 %in% test_idx)
    tr_sp = sp_tab_full[tr_mask,]; class(tr_sp) = c("gdmData","data.frame")
    te_sp = sp_tab_full[te_mask,]; class(te_sp) = c("gdmData","data.frame")
    tryCatch({
      mod = gdm(tr_sp, geo = geo, splines = rep(splines_n, n_pred_total))
      if (!is.null(mod)) {
        y_pred = predict(mod, data = te_sp)
        y_obs  = te_sp$distance
        rmse_v[fi] = rmse_fn(y_obs, y_pred)
        mae_v[fi]  = mae_fn(y_obs, y_pred)
      }
    }, error = function(e) cat(sprintf("  Fold %d error: %s\n", fi, conditionMessage(e))))
  }
  c(RMSE = mean(rmse_v[rmse_v > 0]), MAE = mean(mae_v[mae_v > 0]))
}

all_results = list()

# =============================================================================
# 1. SW Australia — use formatsitepair with raw Long/Lat (matching White's gdm call)
# =============================================================================
cat("=== R gdm 10-fold CV — SW Australia ===\n")
sw_long = read.csv(file.path(data_dir, "southwest.csv"), check.names = FALSE)
env_cols_sw = c("phTotal", "bio5", "bio19")
sp_tab_sw   = suppressWarnings(
  formatsitepair(sw_long, bioFormat = 2,
                 XColumn = "Long", YColumn = "Lat",
                 sppColumn = "species", siteColumn = "site",
                 predData = sw_long[!duplicated(sw_long$site),
                                    c("site","Long","Lat", env_cols_sw)])
)
site_df_sw  = sw_long[!duplicated(sw_long$site), c("Long","Lat",env_cols_sw)]
site_df_sw  = site_df_sw[order(site_df_sw$Long, site_df_sw$Lat),]
site_x_sw   = site_df_sw$Long; site_y_sw = site_df_sw$Lat

m_sw = run_cv(sp_tab_sw, site_x_sw, site_y_sw, env_cols_sw, splines_n = 4, k = 10)
cat(sprintf("  RMSE (10-fold CV): %.4f  (White 2024 Ferrier: 0.0737)\n", m_sw["RMSE"]))
cat(sprintf("  MAE  (10-fold CV): %.4f  (White 2024 Ferrier: 0.0549)\n", m_sw["MAE"]))
all_results[["SW Australia"]] = data.frame(
  dataset="SW Australia", model="R gdm (Ferrier, re-run)", splines=4, n_folds=10,
  RMSE_CV=m_sw["RMSE"], MAE_CV=m_sw["MAE"])

# =============================================================================
# 2. Panama — raw UTM/1000 km coords (matching White's rdist()/1e3 approach)
# =============================================================================
cat("=== R gdm 10-fold CV — Panama ===\n")
env_pan     = read.csv(file.path(data_dir, "panama_env.csv"), check.names = FALSE)
species_pan = read.csv(file.path(data_dir, "panama_species.csv"), row.names = 1)
dist_pan    = as.matrix(vegdist(species_pan, method = "bray"))
y_pan       = dist_pan[upper.tri(dist_pan)]
env_cols_pan = c("precip", "elev")

site_df_pan = data.frame(
  x      = env_pan[["EW coord"]] / 1000,
  y      = env_pan[["NS coord"]] / 1000,
  precip = env_pan$precip,
  elev   = env_pan$elev
)
sp_tab_pan = make_sitepair_df(site_df_pan, y_pan, env_cols_pan)

m_pan = run_cv(sp_tab_pan, site_df_pan$x, site_df_pan$y, env_cols_pan, splines_n = 4, k = 10)
cat(sprintf("  RMSE (10-fold CV): %.4f  (White 2024 Ferrier: 0.0934)\n", m_pan["RMSE"]))
cat(sprintf("  MAE  (10-fold CV): %.4f  (White 2024 Ferrier: 0.0716)\n", m_pan["MAE"]))
all_results[["Panama"]] = data.frame(
  dataset="Panama", model="R gdm (Ferrier, re-run)", splines=4, n_folds=10,
  RMSE_CV=m_pan["RMSE"], MAE_CV=m_pan["MAE"])

# =============================================================================
# Save
# =============================================================================
out_df   = do.call(rbind, all_results); rownames(out_df) = NULL
out_path = file.path(output_dir, "r_gdm_results.csv")
write.csv(out_df, out_path, row.names = FALSE)
cat(sprintf("\nSaved to %s\n", out_path))
