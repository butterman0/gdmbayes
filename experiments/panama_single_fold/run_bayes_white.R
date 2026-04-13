#!/usr/bin/env Rscript
# White NIMBLE models 1, 2, 4, 7 on the single shared Panama fold.
#
# Structure follows nimble_code/Panama/model_1_CV.R exactly (same priors,
# same I-spline basis, same X_sigma = [1, poly(dist,3)], same rho_fix,
# same dinterval censoring for Z==1, same AF_slice block sampler),
# but:
#   - holds out the sites listed in fold/test_sites_r.csv (no other folds)
#   - runs nimble_code{1,2,4,7} in sequence
#   - uses 2 chains to enable R-hat
#
# Outputs: results/bayes_white.csv with RMSE/MAE/CRPS + r-hat diagnostics.

suppressPackageStartupMessages({
  library(splines)
  library(fields)
  library(splines2)
  library(nimble)
  library(vegan)
  library(scoringRules)
  library(dplyr)
  library(coda)
})

HERE    <- "/cluster/home/haroldh/spgdmm/experiments/panama_single_fold"
DATA    <- "/cluster/home/haroldh/spgdmm/external/spGDMM-code/data"
FOLD    <- file.path(HERE, "fold")
RESULTS <- file.path(HERE, "results")
WHITE   <- "/cluster/home/haroldh/spgdmm/external/spGDMM-code/nimble_code"
dir.create(RESULTS, showWarnings = FALSE, recursive = TRUE)

# --- Load shared fold ------------------------------------------------------
test_sites <- read.csv(file.path(FOLD, "test_sites_r.csv"))$site
cat(sprintf("Shared test sites: %s\n", paste(test_sites, collapse = ", ")))

# --- Load data (verbatim from model_1_CV.R) --------------------------------
panama_data <- read.csv(file.path(DATA, "Panama_species.csv"))[, -1]
panama_env  <- read.csv(file.path(DATA, "Panama_env.csv"))

location_mat <- panama_env[, 2:3]
envr_use     <- panama_env[, 4:5]
species_mat  <- panama_data
ns <- nrow(location_mat)

dist_use <- as.matrix(vegdist(species_mat, "bray"))
Z <- dist_use[upper.tri(dist_use)]
N <- length(Z)

# --- Covariates (verbatim) -------------------------------------------------
dist_mat <- as.matrix(rdist(cbind(location_mat$EW.coord, location_mat$NS.coord)) / 1e3)
vec_distance <- dist_mat[upper.tri(dist_mat)]

X <- envr_use
deg <- 3; knots <- 1; df_use <- deg + knots
formula_use <- as.formula(paste("~ 0 +", paste(
  paste("iSpline(`", colnames(X), "`,degree=", deg - 1,
        ",df = ", df_use, " ,intercept = TRUE)", sep = ""), collapse = "+")))
I_spline_bases <- model.matrix(formula_use, data = X)

X_GDM <- cbind(
  sapply(seq_len(ncol(I_spline_bases)), function(i) {
    dist_temp <- rdist(I_spline_bases[, i])
    dist_temp[upper.tri(dist_temp)]
  }),
  iSpline(vec_distance, degree = deg - 1, df = df_use, intercept = TRUE)
)
p <- ncol(X_GDM)
colnames(X_GDM) <- c(
  paste(rep(colnames(X), each = df_use), "I", rep(1:df_use, times = ncol(X)), sep = ""),
  paste("dist", "I", 1:df_use, sep = "")
)

tmp <- matrix(rep(1:nrow(dist_use), each  = nrow(dist_use)), nrow = nrow(dist_use))
col_ind <- tmp[upper.tri(tmp)]
tmp <- matrix(rep(1:nrow(dist_use), times = nrow(dist_use)), nrow = nrow(dist_use))
row_ind <- tmp[upper.tri(tmp)]

# --- Initial values via BFGS (verbatim per-model, from model_{N}_AFS.R) ----
lm_mod <- lm(log(Z) ~ X_GDM)
set.seed(42)  # only affects psi jitter in the init

bfgs_init <- function(model_num) {
  par0 <- c(.3,
            ifelse(coef(lm_mod)[-1] > 0, log(coef(lm_mod)[-1]), -10),
            rnorm(ns))
  if (model_num %in% c(1, 2)) {
    obj <- function(par) {
      sum((log(Z) - par[1] - X_GDM %*% exp(par[2:(p + 1)]))^2)
    }
  } else if (model_num == 4) {
    obj <- function(par) {
      sum((log(Z) - par[1] - X_GDM %*% exp(par[2:(p + 1)]) -
             abs(par[p + 1 + row_ind] - par[p + 1 + col_ind]))^2)
    }
  } else if (model_num == 7) {
    obj <- function(par) {
      sum((log(Z) - par[1] - X_GDM %*% exp(par[2:(p + 1)]) -
             (par[p + 1 + row_ind] - par[p + 1 + col_ind])^2)^2)
    }
  } else {
    stop(sprintf("bfgs_init not implemented for model %d", model_num))
  }
  optim(par0, obj, method = "BFGS")
}

# --- Spatial range parameter (verbatim) ------------------------------------
rho_fix <- max(dist_mat) / 10
R_spat  <- exp(-dist_mat / rho_fix)
R_inv   <- solve(R_spat)

# --- Polynomial log-variance design (verbatim) -----------------------------
X_sigma <- cbind(1, poly(vec_distance, degree = 3))
p_sigma <- ncol(X_sigma)

source(file.path(WHITE, "nimble_models.R"))

# --- Held-out mask from shared test sites ----------------------------------
idx_hold <- (col_ind %in% test_sites) | (row_ind %in% test_sites)
cat(sprintf("Held-out pairs: %d / %d\n", sum(idx_hold), N))

Z_use  <- Z
Z_hold <- Z[idx_hold]
Z_use[idx_hold] <- NA

data_list <- list(
  log_V    = ifelse(Z_use == 1, NA, log(Z_use)),
  censored = ifelse(idx_hold, NA, 1 * (Z_use == 1 & !idx_hold)),
  c        = rep(0, N)
)

constants <- list(
  n = N, p = p, x = X_GDM, n_loc = ns,
  p_sigma = p_sigma, X_sigma = X_sigma, R_inv = R_inv,
  zeros = rep(0, ns), row_ind = row_ind, col_ind = col_ind
)

# --- MCMC budget -----------------------------------------------------------
# Match White's Panama CV defaults (10e3 burn + 10e3 post), 2 chains for R-hat.
N_CHAINS <- 2
N_BURN   <- 10000
N_POST   <- 10000
N_TOT    <- N_BURN + N_POST

# --- Model configurations --------------------------------------------------
MODEL_CODES   <- list("1" = nimble_code1, "2" = nimble_code2,
                      "4" = nimble_code4, "7" = nimble_code7)
HAS_SPATIAL   <- list("1" = FALSE, "2" = FALSE, "4" = TRUE, "7" = TRUE)
HAS_BETASIGMA <- list("1" = FALSE, "2" = TRUE,  "4" = FALSE, "7" = FALSE)

run_model <- function(model_num) {
  tag <- as.character(model_num)
  code <- MODEL_CODES[[tag]]
  cat(sprintf("\n=== NIMBLE model %d ===\n", model_num))

  lm_out <- bfgs_init(model_num)
  inits <- list(
    beta_0    = lm_out$par[1],
    log_V     = ifelse(is.na(data_list$log_V), log(Z), NA),
    log_beta  = lm_out$par[2:(p + 1)]
  )
  if (HAS_SPATIAL[[tag]]) {
    inits$sig2_psi <- 1
    inits$psi      <- lm_out$par[-(1:(p + 1))]
  }
  if (HAS_BETASIGMA[[tag]]) {
    inits$beta_sigma <- c(-5, -20, 12, 2)
  }

  model <- nimbleModel(code, constants = constants, data = data_list, inits = inits)
  mcmcConf <- configureMCMC(model)

  # Block sampler for beta_0, log(beta_jk), and either sigma2 or beta_sigma
  block_targets <- c("beta_0", "log_beta")
  if (HAS_BETASIGMA[[tag]]) {
    block_targets <- c(block_targets, "beta_sigma")
  } else {
    block_targets <- c(block_targets, "sigma2")
  }
  mcmcConf$removeSamplers(block_targets)
  mcmcConf$addSampler(target = block_targets, type = "AF_slice")

  monitor_vars <- c("beta_0", "beta",
                    if (HAS_BETASIGMA[[tag]]) "beta_sigma" else "sigma2",
                    paste0("log_V[", which(idx_hold), "]"))
  if (HAS_SPATIAL[[tag]]) {
    monitor_vars <- c(monitor_vars, "sig2_psi")
  }
  mcmcConf$addMonitors(monitor_vars)

  codeMCMC <- buildMCMC(mcmcConf)
  Cmodel   <- compileNimble(codeMCMC, model)

  st <- proc.time()
  post_samples <- runMCMC(
    Cmodel$codeMCMC,
    niter = N_TOT, nburnin = N_BURN, thin = 1,
    nchains = N_CHAINS, setSeed = c(101, 202),
    samplesAsCodaMCMC = TRUE
  )
  elapsed <- (proc.time() - st)[3]
  cat(sprintf("  MCMC time: %.1f s\n", elapsed))

  # R-hat across the non-log_V scalar parameters
  scalar_vars <- setdiff(varnames(post_samples),
                         grep("^log_V\\[", varnames(post_samples), value = TRUE))
  rhat <- tryCatch(
    gelman.diag(post_samples[, scalar_vars, drop = FALSE], multivariate = FALSE)$psrf[, 1],
    error = function(e) { cat("gelman.diag error:", conditionMessage(e), "\n"); NA }
  )
  rhat_max <- suppressWarnings(max(rhat, na.rm = TRUE))
  cat(sprintf("  R-hat max (scalar vars): %.4f\n", rhat_max))

  # Pool both chains, extract log_V predictions for held-out pairs.
  post_mat <- do.call(rbind, lapply(post_samples, as.matrix))
  cat(sprintf("  post_mat dim: %d x %d\n", nrow(post_mat), ncol(post_mat)))
  cat(sprintf("  first 6 colnames: %s\n",
              paste(head(colnames(post_mat), 6), collapse = ", ")))
  lv_any <- grep("log_V", colnames(post_mat), value = TRUE)
  cat(sprintf("  # columns mentioning log_V: %d (first 3: %s)\n",
              length(lv_any), paste(head(lv_any, 3), collapse = ", ")))

  # Match any of these possible naming conventions:
  # "log_V[3]", "log_V.3.", "log_V.3", "log_V3"
  hold_idx <- which(idx_hold)
  candidate_patterns <- list(
    paste0("log_V[", hold_idx, "]"),
    paste0("log_V.", hold_idx, "."),
    paste0("log_V.", hold_idx),
    paste0("log_V", hold_idx)
  )
  col_order <- NA
  for (patt in candidate_patterns) {
    m <- match(patt, colnames(post_mat))
    if (!any(is.na(m))) { col_order <- m; break }
  }
  if (any(is.na(col_order))) {
    stop(sprintf("Could not match held-out log_V column names. Tried patterns: %s",
                 paste(sapply(candidate_patterns, function(x) x[1]), collapse = ", ")))
  }
  log_V_pred <- post_mat[, col_order, drop = FALSE]
  cat(sprintf("  log_V_pred dim: %d x %d\n", nrow(log_V_pred), ncol(log_V_pred)))

  # Clip log_V > 0 to 0 then exp (matches White: ifelse(x>0,1,exp(x))).
  # Matrix indexing preserves dim; pmin(scalar, matrix) returns a vector.
  log_V_clip <- log_V_pred
  log_V_clip[log_V_clip > 0] <- 0
  Z_preds <- exp(log_V_clip)
  stopifnot(is.matrix(Z_preds))
  pred_mean <- colMeans(Z_preds)

  MSE <- mean((Z_hold - pred_mean)^2)
  MAE <- mean(abs(Z_hold - pred_mean))
  CRPS <- mean(crps_sample(Z_hold, t(Z_preds)))
  cat(sprintf("  RMSE=%.4f  MAE=%.4f  CRPS=%.4f\n", sqrt(MSE), MAE, CRPS))

  data.frame(
    dataset        = "Panama",
    implementation = "White_NIMBLE",
    model          = sprintf("model_%d", model_num),
    deg            = 3,
    knots          = 1,
    df             = df_use,
    RMSE           = sqrt(MSE),
    MAE            = MAE,
    CRPS           = CRPS,
    n_train_sites  = ns - length(test_sites),
    n_test_sites   = length(test_sites),
    n_test_pairs   = length(Z_hold),
    mcmc_time_s    = as.numeric(elapsed),
    rhat_max       = rhat_max,
    n_chains       = N_CHAINS,
    n_burn         = N_BURN,
    n_post         = N_POST,
    stringsAsFactors = FALSE
  )
}

all_rows <- list()
for (m in c(1, 2, 4, 7)) {
  all_rows[[length(all_rows) + 1]] <- run_model(m)
}
out <- do.call(rbind, all_rows)
out_csv <- file.path(RESULTS, "bayes_white.csv")
write.csv(out, out_csv, row.names = FALSE)
cat(sprintf("\nWrote %s\n", out_csv))
print(out)
