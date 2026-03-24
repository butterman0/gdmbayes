# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Workflow

After enacting a plan, immediately:
1. Create a git commit. Never add Claude as a co-author or include any Claude attribution in commit messages or git logs.
2. Update any relevant documentation (docstrings, CLAUDE.md, or other docs) to reflect the changes made.

## Commands

```bash
# Install in editable mode (Python ≥ 3.10 required)
pip install -e ".[dev]"

# Run all tests
pytest src/gdmbayes/tests/

# Run a single test file or test
pytest src/gdmbayes/tests/test_models.py -v
pytest src/gdmbayes/tests/test_models.py::TestSpGDMM::test_fit -v

# Run with coverage
pytest src/gdmbayes/tests/ --cov=src/gdmbayes

# Lint and format
ruff check src/ --fix

# Type check
mypy src/gdmbayes/

# Run examples (frequentist or Bayesian CV)
python examples/panama_example.py --mode freq
python examples/panama_example.py --mode bayes --config_idx 0
python examples/gcfr_example.py --mode freq
python examples/southwest_example.py --mode freq

# Compare results against White et al. (2024) Table 1
python examples/compare_results.py
```

## Architecture

**gdmbayes** (formerly spGDMM) is a Python package for modeling ecological dissimilarities using I-spline basis functions with spatial and environmental predictors. It provides both a frequentist GDM estimator and a full Bayesian backend via PyMC/nutpie, and is the only Python GDM implementation.

### Class Hierarchy

```
spGDMM (models/_spgdmm.py)               ← pure Bayesian estimator (sklearn-style)
    └── GDMModel (models/_gdm_model.py)  ← Bayesian, returns GDMResult
            └── gdm() convenience function

GDM (models/_gdm.py)                     ← frequentist sklearn estimator
    uses: GDMPreprocessor internally
    fit(X, y) → NNLS on I-spline features → coef_, predictor_importance_, explained_

GDMPreprocessor (preprocessing/_preprocessor.py)  ← sklearn transformer
    owns: I-spline mesh computation, pairwise distance, feature matrix assembly
    used by: spGDMM (as self.preprocessor) and GDM (as self.preprocessor_)
```

- **`GDM`** (`models/_gdm.py`): Frequentist sklearn estimator implementing the R GDM algorithm. `fit(X, y)` applies the cloglog link to `y`, runs NNLS on the I-spline feature matrix, and stores coefficients. `predict(X)` returns pairwise dissimilarities via `1 - exp(-X_GDM @ coef_)`.
- **`GDMPreprocessor`** (`preprocessing/_preprocessor.py`): sklearn-style transformer. Owns all data-transformation logic: I-spline mesh construction, geographic distance computation, and pairwise feature matrix assembly. Fitted state is saved/loaded via `to_xarray()` / `from_xarray()`.
- **`spGDMM`** (`models/_spgdmm.py`): Pure Bayesian estimator. Delegates preprocessing to `self.preprocessor` (a `GDMPreprocessor`). `fit()` calls `preprocessor.fit()`, builds a PyMC model, runs MCMC, and returns `InferenceData`. `_transform_for_prediction()` delegates to `preprocessor.transform()`.
- **`GDMModel`** (`models/_gdm_model.py`): Wraps `spGDMM` and returns `GDMResult`, an R-compatible dataclass (deviance, coefficients, knots, splines, idata, etc.). Accepts site-level `(X, y)` input.
- **`plotting._plots`** (`plotting/_plots.py`): `plot_isplines()` and related visualisation helpers for fitted spGDMM models.
- **`utils.site_pairs`** (`utils.py`): Converts a site-index subset to condensed pair indices into a `y` vector. Used in all CV loops to slice train/test pairs from the full pairwise dissimilarity vector.

### Configuration

- **`PreprocessorConfig`** (`preprocessing/_config.py`): I-spline settings and distance measure — `deg`, `knots`, `mesh_choice`, `distance_measure`, `custom_dist_mesh`, `custom_predictor_mesh`, `extrapolation`.
- **`ModelConfig`** (`models/_config.py`): Bayesian model structure — variance (homogeneous/covariate_dependent/polynomial/custom), spatial_effect (none/abs_diff/squared_diff/custom), `alpha_importance`, and custom callable fields. **No longer contains preprocessing fields.**
- **`SamplerConfig`** (`models/_config.py`): MCMC settings (draws, tune, chains, target_accept, nuts_sampler).

### Data Flow

1. Raw site coordinates + environmental predictors + condensed pairwise dissimilarities `y`
2. **Frequentist path**: `GDM.fit(X, y)` → `GDMPreprocessor.fit(X)` computes I-spline meshes → `GDMPreprocessor.transform(X)` → NNLS → `coef_`, `predictor_importance_`, `explained_`
3. **Bayesian path**: `spGDMM.fit(X, y)` → `GDMPreprocessor.fit()` computes I-spline meshes and bases → PyMC model → MCMC → `InferenceData`
4. `GDMModel.fit(X, y)` wraps spGDMM and returns `GDMResult`
5. Prediction: `GDM.predict(X)` or `spGDMM._transform_for_prediction()` delegates to `GDMPreprocessor.transform()` using the stored meshes

### Key Design Decisions

- Preprocessing is separated from the Bayesian estimator: `GDMPreprocessor` is passed as a constructor arg to `spGDMM` (not as a pipeline step) because `build_model()` needs scalar metadata from it (`location_values_train_`, `length_scale_`, dimension counts) that cannot travel through a standard `Pipeline`. This follows the same idiom as `GaussianProcessRegressor(kernel=RBF())`.
- Transformation state (spline knots, column indices, spatial metadata) is saved into `idata` via `GDMPreprocessor.to_xarray()` (called by `spGDMM._save_input_params()`) and reconstructed via `GDMPreprocessor.from_xarray()`.
- `ModelConfig` now contains only Bayesian model-structure fields. Legacy preprocessing keys passed via `model_config` trigger a `DeprecationWarning`.
- `ModelMetadata` stores `d_mean` / `d_std` for standardising pairwise distances, and `poly_transform` (R⁻¹ from QR decomposition) for converting raw monomials `[1, d_z, d_z², d_z³]` to an orthogonal polynomial basis. Both `_generate_and_preprocess_model_data()` (training) and `_data_setter()` (prediction) apply the same QR transform so train/predict are consistent.
- **MCMC initialisation**: `_compute_initvals()` runs a multi-stage BFGS matching White et al.: (1) squared-error for beta_0/beta, (1b) joint re-optimisation of [beta_0, log_beta, psi] including the spatial effect term when spatial models are configured, (2) profile Gaussian NLL for beta_sigma given fixed mu+spatial. Psi init is critical — without it NUTS starts with zero spatial contribution and struggles to discover GP structure, especially through the non-differentiable `abs()` in abs_diff models.
- **nutpie initvals**: nutpie 0.16.x ignores `initvals` passed via `pm.sample()`. The workaround is `model.set_initval(rv, value)` which modifies PyMC's `rvs_to_initial_values` dict before nutpie compiles the model — nutpie's `compile_pymc_model()` picks them up via `make_initial_point_fn()`. This avoids custom compilation/sampling code.
- **Masked-holdout CV**: White et al. (2024) fits on ALL sites with held-out pairs masked as NA, so the GP samples `psi` at test-site locations. We replicate this by splitting the likelihood: `pm.Censored` for observed train pairs + `pm.Normal` for held-out pairs (free latent RVs). Use `spGDMM.fit(X, y, holdout_mask=mask)` and `extract_holdout_predictions()` to retrieve posterior samples. The `holdout_pairs(n_sites, test_sites)` utility returns indices where EITHER site is a test site (matching White et al.'s masking strategy).
- Full-data `.nc` files in the Panama example use `fcntl.flock` to avoid HDF5 race conditions when array jobs write concurrently.
- Distance calculations (`distances/_general.py`, `distances/_ocean.py`) are modular and support Euclidean, geodesic, and grid-based ocean path distances.
- `ruff` line length is 100; rule E501 (line too long) is ignored.
- Tests live inside the package at `src/gdmbayes/tests/`.

## Validation Against White et al. (2024)

Ground-truth benchmarks from White et al. (2024) Table 1 are stored in `benchmarks/white2024_table1.csv` (all 4 datasets, all models + Naive/Ferrier baselines). This is the reference for all validation work.

### Datasets

| Dataset | Sites | Predictors | Knots | Source |
|---------|-------|------------|-------|--------|
| Panama (BCI) | 39 | 2 (precip, elev) | 1 | White et al. repo |
| SW Australia | 94 | 3 (phTotal, bio5, bio19) | 1 | R `gdm` package |
| GCFR Family | 412 | 7 | 2 | White et al. repo |
| GCFR Species | 412 | 7 | 2 | White et al. repo |

All example datasets live in `examples/data/`. Panama, GCFR Family, and GCFR Species files are byte-identical to White et al.'s originals from `philawhite/spGDMM-code`.

### Validation settings

For fair comparison against White et al., all validation runs **must use identical priors and settings**:
- `alpha_importance=False` (White et al. uses LogNormal priors, no Dirichlet/alpha structure)
- `LogNormal(mu=0, sigma=10)` prior on beta coefficients — **never change priors to work around sampler issues**
- `Normal(mu=0, sigma=10)` prior on beta_sigma coefficients
- 10-fold site-level CV for all datasets
- `compare_results.py` loads CV metrics CSVs and prints a unified comparison table
- Validate on Panama first (smallest dataset, fastest iteration), then SW Australia, then GCFR

### Experiment tracking

CV metrics CSVs (`*_cv_metrics.csv`) deduplicate by `(config_tag, seed, n_folds)` — rerunning a config automatically overwrites the old row. **Do not delete result files before resubmitting jobs**; the dedup logic handles it. Each row also records the git commit hash and timestamp for provenance.

### SLURM array jobs

```bash
# Submit Bayesian CV jobs (one array task per model config)
sbatch examples/run_bayes_panama.sh   # 9 configs × 10 folds, ~4h
sbatch examples/run_bayes_sw.sh       # 9 configs × 10 folds, ~4h
sbatch examples/run_bayes_gcfr.sh     # 9 configs × 10 folds, ~48h
```

## Manuscript

Target journal and submission planning: see `paper/journal_planning.md`.
