#!/usr/bin/env Rscript
# Regenerate r_gdm_southwest.json: reference fit of the R `gdm` package on the
# shipped Southwest Australia dataset, consumed by
# src/gdmbayes/tests/test_r_gdm_regression.py.
#
# Usage (from the repo root; the R env with gdm installed is a conda env):
#   /cluster/home/haroldh/miniforge3/envs/R/bin/Rscript \
#       src/gdmbayes/tests/data/make_r_gdm_fixture.R
#
# Settings mirror examples/southwest_running_example.ipynb:
#   predictors phTotal, bio5, bio19; coordinates Long/Lat; geo = TRUE;
#   R gdm default splines (3 per predictor, knots at the 0/50/100 % quantiles).
# The Python equivalent is GDM(geo=True, splines=2, knots=1): R's "3 splines"
# are quadratic I-splines (dms_variants order 2) on a 3-point mesh.

suppressPackageStartupMessages(library(gdm))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
here <- if (length(script)) dirname(normalizePath(script)) else getwd()
repo <- normalizePath(file.path(here, "..", "..", "..", ".."))
data_dir <- file.path(repo, "examples", "data", "southwest")

sites <- read.csv(file.path(data_dir, "southwest_sites.csv"))
y <- read.csv(file.path(data_dir, "southwest_y.csv"))$y
n <- nrow(sites)
stopifnot(length(y) == n * (n - 1) / 2)

# southwest_y.csv is a condensed dissimilarity vector in scipy pdist order
# (row-major upper triangle: (1,2), (1,3), ..., (1,n), (2,3), ...).
D <- matrix(0, n, n)
k <- 1
for (i in 1:(n - 1)) {
  for (j in (i + 1):n) {
    D[i, j] <- y[k]
    D[j, i] <- y[k]
    k <- k + 1
  }
}

env_predictors <- c("phTotal", "bio5", "bio19")
bio <- data.frame(site = sites$site, D)
pred <- sites[, c("site", "Long", "Lat", env_predictors)]

sp <- formatsitepair(bio, bioFormat = 3, XColumn = "Long", YColumn = "Lat",
                     predData = pred, siteColumn = "site")
stopifnot(nrow(sp) == length(y))

# Recover (s1, s2) 0-based site indices for every pair by matching coordinates,
# and check that formatsitepair kept the row-major upper-triangle order.
key <- paste(sites$Long, sites$Lat)
s1 <- match(paste(sp$s1.xCoord, sp$s1.yCoord), key) - 1L
s2 <- match(paste(sp$s2.xCoord, sp$s2.yCoord), key) - 1L
stopifnot(!anyNA(s1), !anyNA(s2))
iu <- which(upper.tri(D), arr.ind = TRUE)
iu <- iu[order(iu[, "row"], iu[, "col"]), ]
stopifnot(all(s1 == iu[, "row"] - 1L), all(s2 == iu[, "col"] - 1L))
stopifnot(isTRUE(all.equal(sp$distance, y)))

m <- gdm(sp, geo = TRUE)

out <- list(
  r_version = R.version.string,
  gdm_version = as.character(packageVersion("gdm")),
  settings = list(
    geo = TRUE,
    splines = m$splines,
    env_predictors = env_predictors,
    x_column = "Long",
    y_column = "Lat",
    pair_order = "upper_triangle_row_major",
    n_sites = n,
    n_pairs = length(y)
  ),
  predictors = m$predictors,
  knots = m$knots,
  intercept = m$intercept,
  coefficients = m$coefficients,
  sum_coeff = m$sumCoeff,
  explained_percent = m$explained,
  gdm_deviance = m$gdmdeviance,
  null_deviance = m$nulldeviance,
  n_pairs_with_ids = 200L,
  pair_s1 = s1[1:200],
  pair_s2 = s2[1:200],
  predicted = round(m$predicted, 8)
)

out_path <- file.path(here, "r_gdm_southwest.json")
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = FALSE), out_path)
cat(sprintf("wrote %s (gdm %s, explained = %.4f%%)\n", out_path,
            out$gdm_version, out$explained_percent))
