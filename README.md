# spGDMM: Spatial Generalized Dissimilarity Mixed Model

A Bayesian Python package for modeling ecological dissimilarities using spatial and environmental predictors with I-spline basis functions.

**This package is being refactored to be more general and similar to the R GDM package.**
The API is still evolving, and some features may change in future versions.

## Features

- **9 Model Variants**: Different combinations of variance structures and spatial random effects
- **Modern Python API**: Type-safe enums and dataclasses for configuration
- **I-spline Transformations**: Flexible non-linear modeling of predictor effects
- **Spatial Effects**: Gaussian process-based spatial random effects
- **Flexible Distance Calculation**: Euclidean, geodesic, and custom grid-based distances
- **Bayesian Inference**: Built with PyMC for full posterior inference
- **Scikit-learn Compatible**: Easy-to-use `fit()`/`predict()` API
- **GDM Compatible**: Input/output format matching R GDM package

## Quick Start (GDM Compatible)

```python
from spgdmm import format_site_pair, BioFormat, gdm
import pandas as pd

# 1. Prepare biological data (long format)
bio_data = pd.DataFrame({
    "site": ["A", "A", "B", "B"],
    "species": ["sp1", "sp2", "sp1", "sp2"],
    "xCoord": [0.0, 0.0, 1.0, 1.0],
    "yCoord": [0.0, 0.0, 1.0, 1.0],
    "abundance": [10, 5, 8, 3]
})

# 2. Prepare environmental data
pred_data = pd.DataFrame({
    "temp": {"A": 10.0, "B": 15.0},
    "precip": {"A": 500.0, "B": 400.0}
})

# 3. Create site-pair table (matches R's formatsitepair)
site_pair = format_site_pair(
    bio_data=bio_data,
    bio_format=BioFormat.FORMAT2,
    pred_data=pred_data,
    x_column="xCoord",
    y_column="yCoord",
    species_column="species",
    site_column="site",
    abund_column="abundance"
)

# 4. Fit GDM model (matches R's gdm function)
result = gdm(site_pair, geo=True, splines=3)

# 5. Access results (same as R GDM)
print(f"Deviance explained: {result.explained:.1f}%")
print(f"Predictors: {result.predictors}")
```

## GDM Compatibility

**NEW**: spGDMM now includes a GDM-compatible interface that matches the R GDM package's input/output format:

- `format_site_pair()` - Equivalent to R's `formatsitepair()`
- `gdm()` - Equivalent to R's `gdm()`
- `GDMModel` - Class-based interface with same API
- `GDMResult` - Output object with same attributes as R GDM

See [docs/GDM_COMPATIBILITY.md](docs/GDM_COMPATIBILITY.md) for full details.

## Installation

### From PyPI (when published)

```bash
pip install spgdmm
```

### Development Installation with Mamba

Clone the repository and set up the development environment:

```bash
cd spgdmm-package

# Create and activate the mamba environment
mamba env create -f environment.yml
mamba activate spgdmm

# Or use the convenience script
bash setup_env.sh
```

The environment includes:
- Python 3.10
- PyMC, ArviZ, nutpie (Bayesian modeling)
- NumPy, Pandas, SciPy, Xarray (Scientific computing)
- scikit-learn, scikit-image (Machine learning)
- dms-variants (I-splines)
- All other required dependencies

## Quick Start

```python
from spgdmm import spGDMM, ModelVariant
import pandas as pd
from scipy.spatial.distance import pdist

# Prepare your data
# X: DataFrame with columns [xc, yc, time_idx, predictor1, predictor2, ...]
# y: Pre-computed Bray-Curtis dissimilarities
X = pd.DataFrame({
    "xc": [10.5, 11.2, 10.8, ...],
    "yc": [60.1, 60.5, 59.9, ...],
    "time_idx": [0, 0, 0, ...],
    "temperature": [12.5, 13.1, 12.0, ...],
    "salinity": [32.5, 33.0, 32.3, ...],
})
biomass = [...]  # Your species abundance matrix
y = pdist(biomass, "braycurtis")

# Fit the model
model = spGDMM.from_variant(ModelVariant.MODEL1, deg=3, knots=2)
idata = model.fit(X, y, random_seed=42)

# Make predictions
X_pred = pd.DataFrame({
    "xc": [10.7, 11.0],
    "yc": [60.2, 60.4],
    "time_idx": [0, 0],
    "temperature": [12.8, 13.0],
    "salinity": [32.7, 32.9],
})
predictions = model.predict_posterior(X_pred)
```

## Model Variants

| Model | Variance Structure | Spatial Effects | Description |
|-------|-------------------|-----------------|-------------|
| MODEL1 | Homogeneous | None | Baseline model |
| MODEL2 | Covariate-Dependent | None | Heteroscedastic |
| MODEL3 | Polynomial | None | Non-linear variance |
| MODEL4 | Homogeneous | Abs Diff | Basic spatial |
| MODEL5 | Covariate-Dependent | Abs Diff | Spatial + heteroscedastic |
| MODEL6 | Polynomial | Abs Diff | Complex spatial |
| MODEL7 | Homogeneous | Squared Diff | Alternative spatial |
| MODEL8 | Covariate-Dependent | Squared Diff | Alternative spatial + heteroscedastic |
| MODEL9 | Polynomial | Squared Diff | Most complex spatial |

Using pre-configured variants:

```python
from spgdmm import spGDMM, ModelVariant

# Create model with absolute difference spatial effects
model = spGDMM.from_variant(ModelVariant.MODEL4)
```

Or build custom configurations:

```python
from spgdmm import spGDMM, ModelConfig, VarianceType, SpatialEffectType

config = ModelConfig(
    deg=4,
    knots=3,
    variance_type=VarianceType.HOMOGENEOUS,
    spatial_effect_type=SpatialEffectType.ABS_DIFF,
    alpha_importance=True,
)
model = spGDMM(config=config)
```

## Configuration

### Model Configuration

```python
from spgdmm import ModelConfig

config = ModelConfig(
    # I-spline settings
    deg=3,                           # Degree of I-spline
    knots=2,                           # Internal knots
    mesh_choice="percentile",          # Mesh placement: "percentile", "even", "custom"

    # Distance measure
    distance_measure="euclidean",      # "euclidean", "geodesic", "ocean_distance"

    # Predictor settings
    alpha_importance=True,            # Estimate feature importance weights
    custom_predictor_mesh=None,        # Optional custom mesh for predictors
    custom_dist_mesh=None,             # Optional custom mesh for distances

    # Variance & spatial
    variance_type="homogeneous",       # "homogeneous", "covariate_dependent", "polynomial"
    spatial_effect_type="none",        # "none", "abs_diff", "squared_diff"
    length_scale=None,                 # GP length scale (auto-computed if None)

    # Other settings
    diss_metric="braycurtis",
    time_predictor=None,
    connected_pairs_only=False,
)
```

### Sampler Configuration

```python
from spgdmm import SamplerConfig

sampler_config = SamplerConfig(
    draws=1000,              # Number of posterior samples
    tune=1000,               # Burn-in iterations
    chains=4,                # Number of MCMC chains
    target_accept=0.95,      # Target acceptance rate
    nuts_sampler="nutpie",   # NUTS sampler implementation
    random_seed=42,          # Random seed for reproducibility
)

model = spGDMM(sampler_config=sampler_config.to_dict())
```

## Plotting and Diagnostics

```python
from spgdmm import plot_isplines, plot_crps_comparison, summarise_sampling, plot_ppc

# Check sampling diagnostics
diagnostics = summarise_sampling(idata)

# Plot I-spline effects
plot_isplines(model)

# Posterior predictive check
plot_ppc(idata, y_test)

# CRPS comparison
plot_crps_comparison(y_test, predictions, y_train)
```

## Distance Calculation

The package provides flexible distance calculation utilities:

```python
from spgdmm import (
    DistanceCalculator,
    compute_distance_matrix,
    euclidean_distance,
    geodesic_distance,
)

# Euclidean (straight-line) distance
coords = [[0, 0], [3, 0], [0, 4]]
distances = euclidean_distance(np.array(coords))

# Geodesic (great-circle) distance for lat/lon coordinates
locations = [[60.0, 10.0], [60.1, 10.1]]
distances = geodesic_distance(np.array(locations))

# Custom distance calculator
calc = DistanceCalculator(metric="euclidean")
distances = calc.compute(locations)
```

## API Reference

### Core Classes

- `spGDMM`: Main model class
- `ModelVariant`: Enum of pre-configured model variants
- `ModelConfig`: Dataclass for model configuration
- `SamplerConfig`: Dataclass for sampler configuration
- `VarianceType`: Enum for variance structure types
- `SpatialEffectType`: Enum for spatial effect types

### Main Methods

- `spGDMM.fit(X, y)`: Fit the model
- `spGDMM.predict(X)`: Point predictions
- `spGDMM.predict_posterior(X)`: Full posterior predictive samples
- `spGDMM.save(fname)`: Save model to file
- `spGDMM.load(fname)`: Load model from file

### Plotting Functions

- `plot_isplines(model)`: Plot I-spline effect curves
- `plot_crps_comparison(y_test, y_pred, y_train)`: Compare model vs null baseline
- `summarise_sampling(idata)`: ESS and R-hat diagnostics
- `plot_ppc(idata, y_obs)`: Posterior predictive check

## Development

### Running Tests

```bash
mamba activate spgdmm
pytest tests/
```

### Running Examples

```bash
mamba activate spgdmm
python examples/basic_usage.py
python examples/model_variants.py
python examples/gdm_compatible_example.py  # GDM-compatible interface
```

### Code Formatting

```bash
# Install dev tools
mamba install -c conda-forge ruff mypy pytest

# Format code
ruff check src/ --fix

# Type check
mypy src/spgdmm/
```

## Citation

```bibtex
@software{spgdmm2025,
  title = {spGDMM: Spatial Generalized Dissimilarity Mixed Model},
  author = {Horsley, Harold},
  year = {2025},
  url = {https://github.com/harryhorsley9/spgdmm}
}
```

## License

Apache 2.0