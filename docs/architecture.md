# Architecture

**gdmbayes** (formerly spGDMM) is a Python package for modeling ecological dissimilarities using I-spline basis functions with spatial and environmental predictors. It provides both a frequentist GDM estimator and a full Bayesian backend via PyMC/nutpie, and is the only Python GDM implementation.

## Class Descriptions

- **`GDM`** (`models/gdm.py`): Frequentist sklearn estimator implementing the R GDM algorithm. `fit(X, y)` applies the cloglog link to `y`, runs NNLS on the I-spline feature matrix, and stores coefficients. `predict(X)` returns pairwise dissimilarities via `1 - exp(-X_GDM @ coef_)`.
- **`GDMPreprocessor`** (`preprocessor.py`): sklearn-style transformer. Init params: `deg`, `knots`, `mesh_choice`, `distance_measure`, `extrapolation`, `custom_dist_mesh`, `custom_predictor_mesh`. Owns all data-transformation logic: I-spline mesh construction, geographic distance computation, and pairwise feature matrix assembly. Fitted state is saved/loaded via `to_xarray()` / `from_xarray()`.
- **`spGDMM`** (`models/spgdmm.py`): Bayesian estimator. Delegates preprocessing to `self.preprocessor` (a `GDMPreprocessor`). `fit()` calls `preprocessor.fit()`, builds a PyMC model, runs MCMC, and returns `InferenceData`. Includes `gdm_transform()` and `ispline_extract()` methods.
- **`plotting`** (`plotting/`): One module per plot — `isplines.py` (`plot_isplines`), `ppc.py` (`plot_ppc`), `scoring.py` (`crps_boxplot`).
- **`diagnostics`** (`diagnostics.py`): `summarise_sampling()` for ESS/R-hat/divergence summaries.
- **`maps`** (`maps.py`): `rgb_biological_space()` / `rgb_from_biological_space()` for PCA-based RGB community-composition maps.
- **`utils.site_pairs`** (`utils.py`): Converts a site-index subset to condensed pair indices into a `y` vector. Used in all CV loops to slice train/test pairs from the full pairwise dissimilarity vector.

## Configuration

- **`ModelConfig`** (`models/config.py`): Bayesian model structure — variance (homogeneous/covariate_dependent/polynomial/custom), spatial_effect (none/abs_diff/squared_diff/custom), `alpha_importance`, and custom callable fields.
- **`SamplerConfig`** (`models/config.py`): MCMC settings (draws, tune, chains, target_accept, nuts_sampler).

## Data Flow

1. Raw site coordinates + environmental predictors + condensed pairwise dissimilarities `y`
2. **Frequentist path**: `GDM.fit(X, y)` → `GDMPreprocessor.fit(X)` computes I-spline meshes → `GDMPreprocessor.transform(X)` → NNLS → `coef_`, `predictor_importance_`, `explained_`
3. **Bayesian path**: `spGDMM.fit(X, y)` → `build_model(X, y)` → `GDMPreprocessor.fit()` computes I-spline meshes and bases → PyMC model → MCMC → `InferenceData`
4. Prediction: `GDM.predict(X)` or `spGDMM._data_setter()` delegates to `GDMPreprocessor.transform()` using the stored meshes
