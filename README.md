# gdmbayes: Generalised Dissimilarity Modelling in Python

A Python package for modelling ecological dissimilarities using spatial and
environmental predictors with I-spline basis functions. Provides both a
frequentist GDM estimator (sklearn-compatible) and a full Bayesian backend via
PyMC/nutpie, and is the only Python GDM implementation.

## Features

- **Frequentist GDM**: sklearn-compatible `GDM` class that implements the R GDM algorithm (NNLS on I-spline features)
- **Bayesian GDM**: `spGDMM` / `GDMModel` for full posterior inference via PyMC
- **I-spline Transformations**: Flexible non-linear modelling of predictor effects
- **Spatial Effects**: Gaussian process-based spatial random effects
- **Flexible Distance Calculation**: Euclidean, geodesic, and custom grid-based distances
- **Scikit-learn Compatible**: `fit()` / `predict()` API with `clone()`, `get_params()`, `set_params()`
- **GDM Compatible**: Input/output format matching the R GDM package

## Quick Start

### Frequentist GDM

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from gdmbayes import GDM

# Site-level data: xc, yc, time_idx, then environmental predictors
X = pd.DataFrame({
    "xc": [10.5, 11.2, 10.8, 12.0],
    "yc": [60.1, 60.5, 59.9, 61.0],
    "time_idx": [0, 0, 0, 0],
    "temperature": [12.5, 13.1, 12.0, 14.2],
    "salinity": [32.5, 33.0, 32.3, 33.5],
})

biomass = np.random.exponential(1, (4, 10))
y = pdist(biomass, "braycurtis")

model = GDM(geo=True)
model.fit(X, y)

print(f"Deviance explained: {model.explained_:.4f}")
print(f"Predictor importance: {model.predictor_importance_}")

# Pairwise dissimilarity predictions
preds = model.predict(X)   # values in [0, 1)
```

### Bayesian spGDMM

```python
from gdmbayes import spGDMM, ModelConfig, PreprocessorConfig, SamplerConfig

model = spGDMM(
    preprocessor=PreprocessorConfig(deg=3, knots=2),
    model_config=ModelConfig(variance="homogeneous", spatial_effect="none"),
    sampler_config=SamplerConfig(draws=1000, tune=1000, chains=4),
)

idata = model.fit(X, y, random_seed=42)

# Point predictions (posterior mean)
predictions = model.predict(X)
```

### Bayesian GDMModel (R-compatible output)

```python
from gdmbayes import GDMModel

model = GDMModel(geo=True, splines=3)
result = model.fit(X, y, dataname="my_survey")

print(f"Deviance explained: {result.explained:.1f}%")
print(f"Predictors: {result.predictors}")
print(f"Coefficients: {result.coefficients}")
```

## Installation

### From PyPI (when published)

```bash
pip install gdmbayes
```

### Development Installation

```bash
cd spgdmm
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. All runtime and dev dependencies are declared in `pyproject.toml`.

## Configuration

### Model Configuration

`ModelConfig` controls the Bayesian model structure (variance and spatial effect).
Preprocessing settings live separately in `PreprocessorConfig`.

```python
from gdmbayes import ModelConfig

config = ModelConfig(
    # Variance structure: "homogeneous", "covariate_dependent", "polynomial", or callable
    variance="homogeneous",

    # Spatial random effect: "none", "abs_diff", "squared_diff", or callable
    spatial_effect="none",

    # Estimate per-predictor importance weights
    alpha_importance=True,
)
```

### Preprocessor Configuration

```python
from gdmbayes import PreprocessorConfig

preprocessor_config = PreprocessorConfig(
    deg=3,                           # Degree of I-spline basis
    knots=2,                         # Number of internal knots
    mesh_choice="percentile",        # Knot placement: "percentile", "even", "custom"
    distance_measure="euclidean",    # "euclidean", "geodesic", "ocean_distance"
    custom_predictor_mesh=None,      # Optional custom mesh for predictors
    custom_dist_mesh=None,           # Optional custom mesh for distances
)
```

### Sampler Configuration

```python
from gdmbayes import SamplerConfig

sampler_config = SamplerConfig(
    draws=1000,              # Number of posterior samples
    tune=1000,               # Burn-in iterations
    chains=4,                # Number of MCMC chains
    target_accept=0.95,      # Target acceptance rate
    nuts_sampler="nutpie",   # NUTS sampler implementation
    random_seed=42,          # Random seed for reproducibility
)

model = spGDMM(sampler_config=sampler_config)
```

## Variance and Spatial Effect Options

| String value | Description |
|---|---|
| `"homogeneous"` | Constant variance across all predictions |
| `"covariate_dependent"` | Variance depends on pairwise distance |
| `"polynomial"` | Polynomial variance as function of mean |
| `"none"` | No spatial random effects (spatial_effect only) |
| `"abs_diff"` | Absolute difference in GP latent values |
| `"squared_diff"` | Squared difference in GP latent values |

## Custom Variance and Spatial Functions

Both `variance` and `spatial_effect` in `ModelConfig` accept a callable in
addition to the built-in string options. Callables are executed inside a PyMC
model context, so you can define new priors directly.

### Custom variance

Signature: `fn(mu, X_sigma) -> sigma2`

- `mu`: PyTensor vector — the linear predictor for each site pair.
- `X_sigma`: `np.ndarray` of shape `(n_pairs, k)` or `None` — auxiliary covariates (currently pairwise geographic distance).
- Returns a PyTensor scalar or vector representing `sigma²`.

```python
import pymc as pm
from gdmbayes import ModelConfig

def my_variance(mu, X_sigma):
    beta_s = pm.HalfNormal("beta_s", sigma=1)
    return beta_s * pm.math.exp(mu)

model_config = ModelConfig(variance=my_variance)
```

### Custom spatial effect

Signature: `fn(psi, row_ind, col_ind) -> effect`

- `psi`: PyTensor vector of length `n_sites` — GP latent values at each training site.
- `row_ind`, `col_ind`: integer arrays — upper-triangle pair indices from `np.triu_indices(n_sites, k=1)`.
- Returns a PyTensor vector of length `n_pairs` to add to `mu`.

```python
import pymc as pm
from gdmbayes import ModelConfig

def my_spatial(psi, row_ind, col_ind):
    return pm.math.tanh(psi[row_ind] - psi[col_ind])

model_config = ModelConfig(spatial_effect=my_spatial)
```

### Registries

The built-in functions are stored in `VARIANCE_FUNCTIONS` and `SPATIAL_FUNCTIONS`:

```python
from gdmbayes import VARIANCE_FUNCTIONS, SPATIAL_FUNCTIONS

print(list(VARIANCE_FUNCTIONS))   # ['homogeneous', 'covariate_dependent', 'polynomial']
print(list(SPATIAL_FUNCTIONS))    # ['abs_diff', 'squared_diff']
```

## Plotting and Diagnostics

```python
from gdmbayes import plot_isplines, crps_boxplot, summarise_sampling, plot_ppc

# Check sampling diagnostics
diagnostics = summarise_sampling(idata)

# Plot I-spline effects (one figure per predictor)
figs = plot_isplines(model)

# Posterior predictive check
plot_ppc(idata, y_test)

# CRPS skill score vs climatological null
crps_boxplot(y_test, predictions, y_train)
```

## Distance Calculation

```python
from gdmbayes import (
    DistanceCalculator,
    compute_distance_matrix,
    euclidean_distance,
    geodesic_distance,
)
import numpy as np

# Euclidean (straight-line) distance
coords = [[0, 0], [3, 0], [0, 4]]
distances = euclidean_distance(np.array(coords))

# Geodesic (great-circle) distance for lat/lon coordinates
locations = [[60.0, 10.0], [60.1, 10.1]]
distances = geodesic_distance(np.array(locations))

# Custom distance calculator
calc = DistanceCalculator(metric="euclidean")
distances = calc.compute(np.array(locations))
```

## API Reference

### Core Classes

- `GDM`: Frequentist sklearn-compatible GDM estimator
- `spGDMM`: Bayesian GDM estimator (PyMC)
- `GDMModel`: Bayesian GDM wrapper returning `GDMResult` (R-compatible)
- `GDMResult`: Output dataclass matching R GDM output attributes
- `ModelConfig`: Dataclass for Bayesian model structure configuration
- `PreprocessorConfig`: Dataclass for I-spline and distance settings
- `SamplerConfig`: Dataclass for MCMC sampler settings

### Main Methods

- `GDM.fit(X, y)`: Fit frequentist model; returns `self`
- `GDM.predict(X)`: Pairwise dissimilarity predictions in `[0, 1)`
- `spGDMM.fit(X, y)`: Fit Bayesian model; returns `self`
- `spGDMM.predict(X)`: Posterior-mean dissimilarity predictions in `(0, 1]`
- `spGDMM.save(fname)` / `spGDMM.load(fname)`: Serialise/restore model
- `GDMModel.fit(X, y)`: Fit Bayesian model; returns `GDMResult`
- `GDMModel.predict(X)`: Predicted pairwise dissimilarities in `[0, 1)`

### Plotting Functions

- `plot_isplines(model)`: I-spline effect curves per predictor (matches R `gdm::plot.gdm`)
- `plot_predictor_importance(model)`: Posterior importance bar chart (Bayesian analogue of R `gdm::gdm.varImp`)
- `plot_obs_vs_pred(model, X, y)`: Observed vs. predicted dissimilarity scatter
- `plot_link_curve(model, X, y)`: Fitted link curve ``y = exp(η)`` with observed points overlaid
- `plot_ppc(idata, y_obs)`: Posterior predictive check
- `crps_boxplot(y_test, y_pred, y_train)`: CRPS skill vs climatological null
- `summarise_sampling(idata)`: ESS, R-hat and divergence diagnostics

## Development

### Running Tests

```bash
pytest src/gdmbayes/tests/ -v
pytest src/gdmbayes/tests/ --cov=src/gdmbayes
```

### Running Examples

```bash
python examples/basic_usage.py
python examples/model_variants.py
```

### Code Quality

```bash
# Lint and auto-fix
ruff check src/ --fix

# Type check
mypy src/gdmbayes/
```

## Citation

```bibtex
@software{gdmbayes2025,
  title = {gdmbayes: Bayesian and Frequentist Generalised Dissimilarity Modelling in Python},
  author = {Horsley, Harold},
  year = {2025},
  url = {https://github.com/harryhorsley9/spgdmm}
}
```

## License

Apache 2.0
