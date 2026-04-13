#!/usr/bin/env Rscript
# R gdm on the single shared Panama fold.
#
# Fits gdm::gdm(splines=3) and gdm::gdm(splines=4) on the 36 training sites
# and predicts dissimilarity for the 111 held-out pairs that touch the
# 3 test sites. Metrics clipped at 1.0 to match White's convention.

suppressPackageStartupMessages({
  library(vegan)
  library(gdm)
})

HERE    <- "/cluster/home/haroldh/spgdmm/experiments/panama_single_fold"
DATA    <- "/cluster/home/haroldh/spgdmm/external/spGDMM-code/data"
FOLD    <- file.path(HERE, "fold")
RESULTS <- file.path(HERE, "results")
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

test_sites <- read.csv(file.path(FOLD, "test_sites_r.csv"))$site
panama_data <- read.csv(file.path(DATA, "Panama_species.csv"))[, -1]
panama_env  <- read.csv(file.path(DATA, "Panama_env.csv"))

ns <- nrow(panama_env)
all_sites <- 1:ns
train_sites <- setdiff(all_sites, test_sites)

# Build the full site-pair table with pre-computed Bray-Curtis (bioFormat = 3).
dist_mat <- as.matrix(vegdist(panama_data, "bray"))
gdm_dissim <- data.frame(site = 1:ns, dist_mat)
colnames(gdm_dissim) <- c("site", 1:ns)
pred_data <- data.frame(
  site = 1:ns,
  x    = panama_env$EW.coord,
  y    = panama_env$NS.coord,
  precip = panama_env$precip,
  elev   = panama_env$elev
)
sitePairAll <- formatsitepair(
  bioData = gdm_dissim, bioFormat = 3,
  XColumn = "x", YColumn = "y", siteColumn = "site",
  predData = pred_data
)
cat(sprintf("Full site-pair table: %d rows\n", nrow(sitePairAll)))

# formatsitepair drops the site column and emits pairs in row-major order:
# (1,2), (1,3), ..., (1,ns), (2,3), (2,4), ...
pairs_mat <- do.call(rbind, lapply(1:(ns - 1), function(i) {
  cbind(i, (i + 1):ns)
}))
s1 <- pairs_mat[, 1]
s2 <- pairs_mat[, 2]
stopifnot(length(s1) == nrow(sitePairAll))
stopifnot(isTRUE(all.equal(sitePairAll$distance, dist_mat[cbind(s1, s2)], tolerance = 1e-10)))
touches_test <- (s1 %in% test_sites) | (s2 %in% test_sites)
train_only   <- (s1 %in% train_sites) & (s2 %in% train_sites)
cat(sprintf("train-only pairs: %d   held-out pairs: %d\n",
            sum(train_only), sum(touches_test)))
stopifnot(sum(train_only) + sum(touches_test) == nrow(sitePairAll))

rows <- list()
for (splines in c(3, 4)) {
  gdm_in <- sitePairAll[train_only, , drop = FALSE]
  fit <- gdm(gdm_in, geo = TRUE, splines = rep(splines, 3))
  # Predict held-out pairs with the fitted model
  pred <- predict(fit, sitePairAll[touches_test, , drop = FALSE])
  pred <- pmin(pred, 1.0)
  obs  <- sitePairAll$distance[touches_test]
  r    <- sqrt(mean((obs - pred)^2))
  m    <- mean(abs(obs - pred))
  cat(sprintf("  splines=%d  RMSE=%.4f  MAE=%.4f  CRPS(point)=%.4f\n",
              splines, r, m, m))
  rows[[length(rows) + 1]] <- data.frame(
    dataset        = "Panama",
    implementation = "R_gdm",
    model          = sprintf("gdm_splines%d", splines),
    deg            = 2,          # DoSplineCalc is piecewise quadratic
    knots          = splines - 2,
    df             = splines,
    RMSE           = r,
    MAE            = m,
    CRPS           = m,
    n_train_sites  = length(train_sites),
    n_test_sites   = length(test_sites),
    n_test_pairs   = sum(touches_test),
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, rows)
out_csv <- file.path(RESULTS, "r_gdm.csv")
write.csv(out, out_csv, row.names = FALSE)
cat(sprintf("\nWrote %s\n", out_csv))
