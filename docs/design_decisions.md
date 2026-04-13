# Design Decisions

## Preprocessor Separation from Bayesian Estimator

`GDMPreprocessor` is passed as a constructor arg to `spGDMM` (not as a pipeline step) because `build_model()` needs scalar metadata from it (`location_values_train_`, `length_scale_`, dimension counts) that cannot travel through a standard `Pipeline`. This follows the same idiom as `GaussianProcessRegressor(kernel=RBF())`. All preprocessing hyperparameters (`deg`, `knots`, `mesh_choice`, `distance_measure`, `extrapolation`, etc.) are direct `__init__` params on `GDMPreprocessor`, following the standard sklearn estimator pattern.

## State Serialization via xarray

Transformation state (spline knots, column indices, spatial metadata) is saved into `idata` via `GDMPreprocessor.to_xarray()` (called by `spGDMM._save_input_params()`) and reconstructed via `GDMPreprocessor.from_xarray()`. All hyperparameters are stored as dataset attrs and round-trip correctly (custom mesh arrays are not serialized since the fitted meshes capture the result).

## Cross-Validation — Standard sklearn fit/predict with GP Conditional

### Approach

CV uses standard sklearn semantics: `fit(X_train, y_train)` / `predict(X_test)`.  Caller subsets sites and pairs before calling `fit`.

For spatial models with a GP random effect, predicting at test-site locations requires **GP kriging** — `psi_pred` at test sites is not known from fitting on training sites alone.  This is handled transparently inside `predict()` via `_predict_gp_conditional()`:

1. `gp.conditional("psi_pred", pred_coords)` is registered in the fitted model context (PyMC knows how to sample from `p(psi_pred | psi_train, posterior_hyperparams)`).
2. `pm.sample_posterior_predictive(idata, var_names=["psi_pred"])` draws `psi_pred` for each posterior sample.
3. The full linear predictor and variance are assembled in NumPy using `psi_pred` samples and `idata.posterior` draws of `beta_0`, `beta`, `sigma2` / `beta_sigma`.
4. `log_y_pred ~ Normal(mu, sigma)` is sampled in NumPy; the result is exponentiated (and clipped at 1) inside `predict_posterior()` and returned on the dissimilarity scale as a `(n_chains, n_draws, n_pred_pairs)` array.

This avoids PyTensor graph shape conflicts that arise from mixing a batched GP conditional variable with existing model RVs inside `pm.sample_posterior_predictive`.

### Which pairs are evaluated

`site_pairs(n_sites, test_sites)` returns the `n_test*(n_test-1)/2` pairs where **both** endpoints are test sites.  These are the pairs passed to the CV metric computation.  `holdout_pairs` remains exported (returns pairs where **either** endpoint is a test site) but is no longer used by the main CV loop.

### GP conditional routing

`_data_setter` routes to `_predict_gp_conditional` when `spatial_effect != "none"` and `len(X_pred) != len(X_train)`.  For same-size prediction (e.g. full-data posterior predictive after `fit`) or no spatial effect, the standard `pm.set_data` path is used.

### Usage

```python
from gdmbayes import spGDMM, site_pairs
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_sites, test_sites in kf.split(np.arange(n_sites)):
    train_pair_idx = site_pairs(n_sites, train_sites)
    test_pair_idx  = site_pairs(n_sites, test_sites)

    model = make_spgdmm()
    model.fit(X.iloc[train_sites].reset_index(drop=True), y[train_pair_idx])
    y_post = model.predict_posterior(
        X.iloc[test_sites].reset_index(drop=True), combined=True, extend_idata=False
    )
    y_samples = y_post.values   # dissimilarity scale, already in (0, 1]
    y_pred_mean = y_samples.mean(axis=-1)
    # compute rmse, mae, crps against y[test_pair_idx]
```

## Concurrent File Locking

Full-data `.nc` files in the Panama example use `fcntl.flock` to avoid HDF5 race conditions when SLURM array jobs write concurrently.

## Distance Calculations

Pairwise geographic distances are computed internally by `GDMPreprocessor.pw_distance()` using `scipy.spatial.distance.pdist` (euclidean) or `geopy.distance.geodesic` (geodesic).
