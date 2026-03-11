# GDM Compatibility Interface

This document describes the GDM-compatible interface in spGDMM, which provides inputs and outputs in the same form as the R GDM package ([CRAN](https://cran.r-project.org/package=gdm)).

**Note**: spGDMM is being refactored to be more general and similar to R GDM. Some features and APIs may change.

## Overview

The spGDMM package now includes functions that match the R GDM package's workflow:

1. **`format_site_pair()`** - Equivalent to R's `formatsitepair()` function
2. **`gdm()`** - Equivalent to R's `gdm()` function
3. **`GDMModel`** - Class-based interface with `fit()` and `predict()` methods
4. **`GDMResult`** - Output object with same attributes as R GDM model

## Distance Calculation

spGDMM provides flexible distance calculation utilities that don't require ocean-specific features:

```python
from spgdmm import (
    DistanceCalculator,
    compute_distance_matrix,
    euclidean_distance,
    geodesic_distance,
)

# Euclidean distance (straight-line)
distances = euclidean_distance(locations)

# Geodesic distance (great-circle for lat/lon)
distances = geodesic_distance(lat_lon_locations)

# Custom distance with DistanceCalculator
calc = DistanceCalculator(metric="euclidean")
distances = calc.compute(locations)

# Create distance matrix from locations
dist_matrix = compute_distance_matrix(locations, metric="geodesic")
```

## Quick Start

```python
from spgdmm import format_site_pair, BioFormat, gdm

# 1. Prepare biological data (long format)
import pandas as pd
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

# 5. Access results
print(f"Deviance explained: {result.explained:.1f}%")
print(f"Predictors: {result.predictors}")
print(f"Coefficients: {result.coefficients}")
```

## Input Format

### Site-Pair Table (from `format_site_pair()`)

The site-pair table format matches the output of R's `formatsitepair()`:

| Column | Description |
|--------|-------------|
| 0 | Biological distance/dissimilarity (response variable) |
| 1 | Weight for model fitting |
| 2-3 | s1.xCoord, s1.yCoord (first site coordinates) |
| 4-5 | s2.xCoord, s2.yCoord (second site coordinates) |
| 6+ | s1.{predictor}, s2.{predictor} for each environmental predictor |

### Biological Data Formats

The `format_site_pair()` function supports multiple biological data formats:

| Format | Description |
|--------|-------------|
| FORMAT1 (1) | Site x Species abundance matrix |
| FORMAT2 (2) | Long format with site, species, coordinate, abundance columns |
| FORMAT3 (3) | Dissimilarity matrix format |
| FORMAT4 (4) | Already in site-pair format (no transformation) |

## Output Format

The `GDMResult` object contains the same attributes as the R GDM model object:

| Attribute | Type | Description |
|-----------|------|-------------|
| `dataname` | str | Name of the data table used |
| `geo` | bool | Whether geographic distance was included |
| `gdmdeviance` | float | Deviance of fitted model |
| `nulldeviance` | float | Null model deviance |
| `explained` | float | Percentage of null deviance explained |
| `intercept` | float | Fitted intercept term |
| `predictors` | list[str] | Predictor names ordered by turnover |
| `coefficients` | dict | Spline coefficients per predictor |
| `knots` | dict | Knot positions per predictor |
| `splines` | list[int] | I-spline basis function counts |
| `creationdate` | str | Date/time of model creation |
| `observed` | np.ndarray | Observed response per site pair |
| `predicted` | np.ndarray | Predicted response per site pair |
| `ecological` | np.ndarray | Linear predictor (before link) |
| `idata` | xr.InferenceData | Full Bayesian inference (spGDMM extension) |

## R vs Python Comparison

| R GDM | Python spGDMM | Description |
|-------|---------------|-------------|
| `formatsitepair()` | `format_site_pair()` | Create site-pair table |
| `gdm()` | `gdm()` | Fit GDM model |
| `gdmMod$explained` | `result.explained` | Deviance explained |
| `gdmMod$(coefficients)` | `result.coefficients` | Coefficients |
| `gdmMod$predictors` | `result.predictors` | Predictor names |
| `gdm.predict()` | `model.predict()` | Make predictions |

## API Reference

### `format_site_pair()`

```python
format_site_pair(
    bio_data: pd.DataFrame,
    bio_format: Union[int, str, BioFormat] = 2,
    dist: str = "bray",
    is_abundance: bool = True,
    site_column: str = "site",
    x_column: str = "xCoord",
    y_column: str = "yCoord",
    species_column: str = "species",
    abund_column: str = "abundance",
    spp_filter: int = 0,
    pred_data: pd.DataFrame = None,
    dist_preds: dict = None,
    weight_type: str = "equal",
    custom_weights: pd.DataFrame = None,
    sample_sites: float = None,
    verbose: bool = False
) -> pd.DataFrame
```

### `gdm()` Function

```python
gdm(
    data: pd.DataFrame,
    geo: bool = False,
    splines: Union[int, List[int]] = None,
    knots: Union[List[float], Dict[str, np.ndarray]] = None,
    **kwargs
) -> GDMResult
```

### `GDMModel` Class

```python
class GDMModel:
    def __init__(
        self,
        splines: Union[int, List[int]] = None,
        knots: Union[List[float], Dict] = None,
        geo: bool = False,
        model_config: ModelConfig = None,
        sampler_config: SamplerConfig = None,
        **kwargs
    )

    def fit(self, data: pd.DataFrame, dataname: str = None, **kwargs) -> GDMResult
    def predict(self, new_data: pd.DataFrame, **kwargs) -> GDMResult
```

## Extended Bayesian Features

While the GDM-compatible interface provides R-like simplicity, spGDMM also exposes full Bayesian inference:

```python
# Access full InferenceData
idata = result.idata

# Posterior samples
posterior = idata.posterior
beta_0_samples = posterior["beta_0"].values

# Posterior predictive
if "posterior_predictive" in idata:
    post_pred = idata.posterior_predictive

# Diagnostic plots
import arviz as az
az.plot_trace(idata, var_names=["beta_0"])
az.plot_posterior(idata, var_names=["beta_0"])
```

## Distance Calculation Examples

See the examples directory for complete examples:

```bash
python examples/gdm_compatible_example.py  # GDM-compatible interface
```

### Custom Distance Calculator

For grid-based distance calculations (e.g., accounting for land barriers):

```python
import numpy as np
from spgdmm import DistanceCalculator

# Create a custom distance function
def custom_distance(locations, cost_grid):
    """Custom distance that accounts for a cost surface."""
    # Implementation would use grid routing
    pass

# Use with DistanceCalculator
calc = DistanceCalculator(metric="custom")
# Add your custom implementation as needed
```

## Migration from Original spGDMM

The original spGDMM interface remains available:

```python
# Original interface (still works)
from spgdmm import spGDMM, ModelVariant
model = spGDMM.from_variant(ModelVariant.MODEL1, deg=3, knots=2)
idata = model.fit(X, y)

# New GDM-compatible interface
from spgdmm import format_site_pair, gdm
site_pair = format_site_pair(bio_data, pred_data=pred_data)
result = gdm(site_pair)
```

Both interfaces share the same underlying implementation and can be used interchangeably.

## References

- R GDM Package: https://cran.r-project.org/package=gdm
- GDM Documentation: https://www.rdocumentation.org/packages/gdm/
- Ferrier et al. (2007) - "Using generalized dissimilarity modelling to analyse and predict patterns of beta diversity"