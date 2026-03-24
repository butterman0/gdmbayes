# Validation Against White et al. (2024)

Ground-truth benchmarks from White et al. (2024) Table 1 are stored in `benchmarks/white2024_table1.csv` (all 4 datasets, all models + Naive/Ferrier baselines). This is the reference for all validation work.

## Datasets

| Dataset | Sites | Predictors | Knots | Source |
|---------|-------|------------|-------|--------|
| Panama (BCI) | 39 | 2 (precip, elev) | 1 | White et al. repo |
| SW Australia | 94 | 3 (phTotal, bio5, bio19) | 1 | R `gdm` package |
| GCFR Family | 412 | 7 | 2 | White et al. repo |
| GCFR Species | 412 | 7 | 2 | White et al. repo |

All example datasets live in `examples/data/`. Panama, GCFR Family, and GCFR Species files are byte-identical to White et al.'s originals from `philawhite/spGDMM-code`.

## Required Priors and Settings

For fair comparison against White et al., all validation runs **must use identical priors and settings**:
- `alpha_importance=False` (White et al. uses LogNormal priors, no Dirichlet/alpha structure)
- `LogNormal(mu=0, sigma=10)` prior on beta coefficients — **never change priors to work around sampler issues**
- `Normal(mu=0, sigma=10)` prior on beta_sigma coefficients
- 10-fold site-level CV for all datasets
- `compare_results.py` loads CV metrics CSVs and prints a unified comparison table
- Validate on Panama first (smallest dataset, fastest iteration), then SW Australia, then GCFR

## Experiment Tracking

CV metrics CSVs (`*_cv_metrics.csv`) deduplicate by `(config_tag, seed, n_folds)` — rerunning a config automatically overwrites the old row. **Do not delete result files before resubmitting jobs**; the dedup logic handles it. Each row also records the git commit hash and timestamp for provenance.

## SLURM Array Jobs

**Never run MCMC on the login node.** Always submit via `sbatch`. Use `--array=3-8` to run a subset of configs.

```bash
# Submit Bayesian CV jobs (one array task per model config)
sbatch examples/run_bayes_panama.sh   # 9 configs x 10 folds, ~4h
sbatch examples/run_bayes_sw.sh       # 9 configs x 10 folds, ~4h
sbatch examples/run_bayes_gcfr.sh     # 9 configs x 10 folds, ~48h
```
