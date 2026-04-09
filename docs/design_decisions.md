# Design Decisions

## Preprocessor Separation from Bayesian Estimator

`GDMPreprocessor` is passed as a constructor arg to `spGDMM` (not as a pipeline step) because `build_model()` needs scalar metadata from it (`location_values_train_`, `length_scale_`, dimension counts) that cannot travel through a standard `Pipeline`. This follows the same idiom as `GaussianProcessRegressor(kernel=RBF())`.

## State Serialization via xarray

Transformation state (spline knots, column indices, spatial metadata) is saved into `idata` via `GDMPreprocessor.to_xarray()` (called by `spGDMM._save_input_params()`) and reconstructed via `GDMPreprocessor.from_xarray()`.

## ModelConfig Deprecation Handling

`ModelConfig` now contains only Bayesian model-structure fields. Legacy preprocessing keys passed via `model_config` trigger a `DeprecationWarning`.

## Masked-Holdout Cross-Validation

### Motivation

Standard train/test splitting drops test sites entirely and fits the model on training sites only. For spatial models with a GP random effect, this is problematic: the GP has never seen the test-site locations, so the spatial contribution `psi` at those locations is undefined. Masked-holdout CV solves this by fitting on **all** sites while masking held-out pairs as latent variables. The GP can then sample `psi` at every site (including test sites), and the held-out dissimilarities are predicted from the full model without having contributed to the likelihood.

### Which pairs are held out

A pair `(i, j)` is held out if **either** endpoint belongs to the test-site fold — not just test–test pairs. This prevents any information about test-site environments from leaking through the likelihood. The utility `holdout_pairs(n_sites, test_sites)` returns these indices; `site_pairs(n_sites, site_idx)` returns pairs where **both** endpoints are in a subset (used by the frequentist GDM which drops test sites entirely).

### Likelihood split

The model is built on all `n*(n-1)/2` pairs but the likelihood is split into two terms:

- **Observed pairs** (`holdout_mask == False`): modelled with `pm.Censored(pm.Normal.dist(mu, sigma), upper=0)`, with the observed log-dissimilarity as data. The upper censoring at 0 (i.e. dissimilarity capped at 1 on the natural scale) handles the `y == 1` boundary.
- **Held-out pairs** (`holdout_mask == True`): modelled as `pm.Normal("log_y_holdout", mu, sigma)` with no observed data. These are free latent random variables — the sampler draws from the posterior predictive distribution conditioned on the training-pair likelihood only.

Both terms share the same linear predictor `mu` (including the GP spatial effect) and variance model `sigma`, so the held-out predictions incorporate all model components.

### Initialisation

BFGS initial values (`_compute_initvals`) are computed from observed pairs only — the holdout mask filters `X_GDM`, `log_y`, pair indices, and `X_sigma` before optimisation so that held-out dissimilarities do not influence starting values.

### Extracting predictions

After sampling, `extract_holdout_predictions()` retrieves the posterior samples of `log_y_holdout` from `idata.posterior`, transforms them back to the dissimilarity scale via `Z = min(1, exp(log_V))`, and returns per-pair posterior means and full sample matrices. CV metrics (RMSE, MAE, CRPS) are computed against the true held-out dissimilarities.

### Usage

```python
from gdmbayes import spGDMM, holdout_pairs
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_sites, test_sites in kf.split(np.arange(n_sites)):
    hold_idx = holdout_pairs(n_sites, test_sites)
    mask = np.zeros(n_pairs, dtype=bool)
    mask[hold_idx] = True

    model = make_spgdmm()
    model.fit(X, y, holdout_mask=mask)
    result = model.extract_holdout_predictions()
    # result["y_pred_mean"], result["y_pred_samples"], result["hold_idx"]
```

## Concurrent File Locking

Full-data `.nc` files in the Panama example use `fcntl.flock` to avoid HDF5 race conditions when SLURM array jobs write concurrently.

## Distance Calculations

Distance calculations (`distances/_general.py`, `distances/_ocean.py`) are modular and support Euclidean, geodesic, and grid-based ocean path distances.
