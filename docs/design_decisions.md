# Design Decisions

## Preprocessor Separation from Bayesian Estimator

`GDMPreprocessor` is passed as a constructor arg to `spGDMM` (not as a pipeline step) because `build_model()` needs scalar metadata from it (`location_values_train_`, `length_scale_`, dimension counts) that cannot travel through a standard `Pipeline`. This follows the same idiom as `GaussianProcessRegressor(kernel=RBF())`.

## State Serialization via xarray

Transformation state (spline knots, column indices, spatial metadata) is saved into `idata` via `GDMPreprocessor.to_xarray()` (called by `spGDMM._save_input_params()`) and reconstructed via `GDMPreprocessor.from_xarray()`.

## ModelConfig Deprecation Handling

`ModelConfig` now contains only Bayesian model-structure fields. Legacy preprocessing keys passed via `model_config` trigger a `DeprecationWarning`.

## Masked-Holdout Cross-Validation

White et al. (2024) fits on ALL sites with held-out pairs masked as NA, so the GP samples `psi` at test-site locations. We replicate this by splitting the likelihood: `pm.Censored` for observed train pairs + `pm.Normal` for held-out pairs (free latent RVs). Use `spGDMM.fit(X, y, holdout_mask=mask)` and `extract_holdout_predictions()` to retrieve posterior samples. The `holdout_pairs(n_sites, test_sites)` utility returns indices where EITHER site is a test site (matching White et al.'s masking strategy).

## Concurrent File Locking

Full-data `.nc` files in the Panama example use `fcntl.flock` to avoid HDF5 race conditions when SLURM array jobs write concurrently.

## Distance Calculations

Distance calculations (`distances/_general.py`, `distances/_ocean.py`) are modular and support Euclidean, geodesic, and grid-based ocean path distances.
