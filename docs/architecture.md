# Architecture

**gdmbayes** is a Python package for modeling ecological dissimilarities using I-spline basis functions with spatial and environmental predictors. It provides both a frequentist GDM estimator and a full Bayesian backend via PyMC/nutpie, and is the only Python GDM implementation.

## Class Descriptions

- **`GDM`** (`models/gdm.py`): Frequentist sklearn estimator implementing the R GDM algorithm. `fit(X, y)` applies the cloglog link to `y`, runs NNLS on the I-spline feature matrix, and stores coefficients. `predict(X)` returns pairwise dissimilarities via `1 - exp(-X_GDM @ coef_)`.
- **`GDMPreprocessor`** (`preprocessor.py`): sklearn-style transformer. Init params: `deg`, `knots`, `mesh_choice`, `distance_measure`, `extrapolation`, `custom_dist_mesh`, `custom_predictor_mesh`. Owns all data-transformation logic: I-spline mesh construction, geographic distance computation, and pairwise feature matrix assembly. Fitted state is saved/loaded via `to_xarray()` / `from_xarray()`.
- **`spGDMM`** (`models/spgdmm.py`): Bayesian estimator. Delegates preprocessing to `self.preprocessor` (a `GDMPreprocessor`). `fit()` calls `preprocessor.fit()`, builds a PyMC model, runs MCMC, and returns an inference-data `xarray.DataTree`. Includes `gdm_transform()`, `ispline_extract()`, and `summarise_sampling()` methods.
- **`plotting`** (`plotting/`): One module per plot — `isplines.py` (`plot_isplines`), `importance.py` (`plot_predictor_importance`), `fit.py` (`plot_obs_vs_pred`, `plot_link_curve`), `ppc.py` (`plot_ppc`), `scoring.py` (`crps_boxplot`).
- **`maps`** (`maps.py`): `rgb_biological_space()` for PCA-based RGB community-composition maps (works directly from a saved inference-data `DataTree`, no live model object needed).
- **`GDMPreprocessor.site_pairs`** (`preprocessor.py`): Static method that converts a site-index subset to condensed pair indices into a `y` vector, using the same upper-triangle ordering the preprocessor uses internally to assemble the pairwise feature matrix. Used in all CV loops to slice train/test pairs from the full pairwise dissimilarity vector.

## Configuration

- **`ModelConfig`** (`models/config.py`): Bayesian model structure — variance (homogeneous/covariate_dependent/polynomial/custom), spatial_effect (none/abs_diff/squared_diff/custom), `alpha_importance`, and custom callable fields.
- **`SamplerConfig`** (`models/config.py`): MCMC settings (draws, tune, chains, target_accept, nuts_sampler).

## Data Flow

1. Raw site coordinates + environmental predictors + condensed pairwise dissimilarities `y`
2. **Frequentist path**: `GDM.fit(X, y)` → `GDMPreprocessor.fit(X)` computes I-spline meshes → `GDMPreprocessor.transform(X)` → NNLS → `coef_`, `predictor_importance_`, `explained_`
3. **Bayesian path**: `spGDMM.fit(X, y)` → `build_model(X, y)` → `GDMPreprocessor.fit()` computes I-spline meshes and bases → PyMC model → MCMC → an inference-data `xarray.DataTree`
4. Prediction: `GDM.predict(X)` or `spGDMM._data_setter()` delegates to `GDMPreprocessor.transform()` using the stored meshes
