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

### Class Hierarchy

```
spGDMM (models/spgdmm.py)               ← Bayesian estimator (sklearn-style)
    methods: fit, predict, gdm_transform, ispline_extract

GDM (models/gdm.py)                     ← frequentist sklearn estimator
    uses: GDMPreprocessor internally
    fit(X, y) → NNLS on I-spline features → coef_, predictor_importance_, explained_

GDMPreprocessor (preprocessing/preprocessor.py)  ← sklearn transformer
    params: deg, knots, mesh_choice, distance_measure, extrapolation, custom_*_mesh
    owns: I-spline mesh computation, pairwise distance, feature matrix assembly
    used by: spGDMM (as self.preprocessor) and GDM (as self.preprocessor_)
```

> **Details:** class descriptions, configs, data flow in [docs/architecture.md](docs/architecture.md)

### Key Design Decisions

- **nutpie initvals**: nutpie 0.16.x ignores `initvals` passed via `pm.sample()`. The workaround is `model.set_initval(rv, value)` which modifies PyMC's `rvs_to_initial_values` dict before nutpie compiles the model. See `_apply_initvals()`.
- **MCMC initialisation**: `_compute_initvals()` runs a multi-stage BFGS matching White et al.: (1) squared-error for beta_0/beta, (1b) joint re-optimisation of [beta_0, log_beta, psi] including the spatial effect term, (2) profile Gaussian NLL for beta_sigma given fixed mu+spatial. Psi init is critical — without it NUTS starts with zero spatial contribution and struggles to discover GP structure.
- **GP coordinate units**: The GP receives coordinates in km (raw coords ÷ 1000 for euclidean) so they match the `length_scale_` (also in km). PyMC's `Exponential` kernel uses `exp(-d/(2*ls))`, while White uses `exp(-d/rho)`, so `ls = rho/2`.
- **`build_model(X, y)`**: Single entry point that preprocesses data (via `_generate_and_preprocess_model_data`) then builds the PyMC model. Called by `fit()` and `load()`. Can also be called with no args if preprocessing was already done.
- **Orthogonal polynomial basis**: `poly_fit` / `poly_predict` in `variance.py` replicate R's `poly()` (QR on centered Vandermonde, three-term recurrence for prediction). Used only for `variance="covariate_dependent"` to build `X_sigma = [1, poly(distance, 3)]`.
- **Standard sklearn CV with GP conditional**: `fit(X_train, y_train)` / `predict(X_test)`. When `n_pred != n_train` and a spatial effect is active, `_predict_gp_conditional()` fires inside `predict()`: samples `psi_pred` via `gp.conditional()` in PyMC, then assembles the full linear predictor in NumPy. `holdout_pairs` remains a utility export (complements `site_pairs`). See [docs/design_decisions.md](docs/design_decisions.md).
- `ruff` line length is 100; rule E501 (line too long) is ignored.
- Tests live inside the package at `src/gdmbayes/tests/`.

> **Details:** preprocessor separation, serialization, file locking in [docs/design_decisions.md](docs/design_decisions.md)

## Validation Against White et al. (2024)

For fair comparison against White et al., all validation runs **must use identical priors and settings**:
- `alpha_importance=False` (White et al. uses LogNormal priors, no Dirichlet/alpha structure)
- `LogNormal(mu=0, sigma=10)` prior on beta coefficients — **never change priors to work around sampler issues**
- `Normal(mu=0, sigma=10)` prior on beta_sigma coefficients
- 10-fold site-level CV for all datasets (final results)
- **Never run MCMC on the login node.** Always submit via `sbatch`.

### Quick smoke tests

For fast iteration (testing code changes, not final results), run **3 folds of the 10-fold split** (`--n_folds 3`). This uses the same 90/10 train/test ratio as the full run (~12 test sites across 3 folds for Panama) and produces reasonably stable metric estimates at 30% of the cost. Use the same `--seed 42` so fold assignments match the eventual full 10-fold run.

> **Details:** dataset table, experiment tracking, SLURM scripts in [docs/validation.md](docs/validation.md)

## Manuscript

Target journal and submission planning: see `paper/journal_planning.md`.
