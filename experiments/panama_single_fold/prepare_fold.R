#!/usr/bin/env Rscript
# Build a single shared Panama train/test split used by every runner.
#
# Re-implements White et al. (2024) fold logic exactly:
#   set.seed(1); tmp = sample(1:ns); idx_keep = which((1:ns) %% 10 == 0);
#   test_sites = sort(tmp[idx_keep])
# and picks fold 1.
#
# Outputs (written next to this script under ./fold/):
#   fold.json            — machine-readable metadata (n_sites, test site indices)
#   test_sites_r.csv     — 1-indexed site numbers (for R runners)
#   test_sites_py.csv    — 0-indexed site numbers (for Python runners)
#   Z.csv                — condensed Bray-Curtis vector (length N = ns*(ns-1)/2)
#                          using upper.tri order matching NIMBLE code
#   row_ind.csv/col_ind.csv — 1-indexed site pair indices (R upper.tri order)
#   idx_hold.csv         — boolean mask into Z (TRUE for held-out pairs)

suppressPackageStartupMessages({
  library(vegan)
})

DATA_DIR <- "/cluster/home/haroldh/spgdmm/external/spGDMM-code/data"
OUT_DIR  <- "/cluster/home/haroldh/spgdmm/experiments/panama_single_fold/fold"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

panama_data <- read.csv(file.path(DATA_DIR, "Panama_species.csv"))[, -1]
panama_env  <- read.csv(file.path(DATA_DIR, "Panama_env.csv"))

species_mat <- panama_data
ns <- nrow(panama_env)

dist_use <- as.matrix(vegdist(species_mat, "bray"))
Z <- dist_use[upper.tri(dist_use)]
N <- length(Z)

# Match NIMBLE code exactly for pair → site indices
tmpC <- matrix(rep(1:nrow(dist_use), each  = nrow(dist_use)), nrow = nrow(dist_use))
col_ind <- tmpC[upper.tri(tmpC)]
tmpR <- matrix(rep(1:nrow(dist_use), times = nrow(dist_use)), nrow = nrow(dist_use))
row_ind <- tmpR[upper.tri(tmpR)]

# White's fold construction
K_fold <- 10
set.seed(1)
tmp <- sample(1:ns)
ind_loc_hold <- lapply(1:K_fold, function(i) {
  idx_keep <- which((1:ns) %% K_fold == (i - 1))
  sort(tmp[idx_keep])
})

FOLD <- 1
test_sites_r <- ind_loc_hold[[FOLD]]
test_sites_py <- test_sites_r - 1L

idx_hold <- (col_ind %in% test_sites_r) | (row_ind %in% test_sites_r)

cat(sprintf("ns=%d  N=%d  test_sites=%d  held pairs=%d / %d\n",
            ns, N, length(test_sites_r), sum(idx_hold), N))
cat(sprintf("test sites (1-indexed): %s\n",
            paste(test_sites_r, collapse = ", ")))

write.csv(data.frame(site = test_sites_r),  file.path(OUT_DIR, "test_sites_r.csv"),  row.names = FALSE)
write.csv(data.frame(site = test_sites_py), file.path(OUT_DIR, "test_sites_py.csv"), row.names = FALSE)
write.csv(data.frame(Z = Z),                file.path(OUT_DIR, "Z.csv"),             row.names = FALSE)
write.csv(data.frame(row_ind = row_ind, col_ind = col_ind, idx_hold = as.integer(idx_hold)),
          file.path(OUT_DIR, "pairs.csv"), row.names = FALSE)

json <- sprintf('{"n_sites": %d, "n_pairs": %d, "fold": %d, "seed": 1, "K_fold": %d, "n_held_pairs": %d, "test_sites_1idx": [%s]}',
                ns, N, FOLD, K_fold, sum(idx_hold),
                paste(test_sites_r, collapse = ", "))
writeLines(json, file.path(OUT_DIR, "fold.json"))

cat(sprintf("Wrote fold metadata to %s\n", OUT_DIR))
