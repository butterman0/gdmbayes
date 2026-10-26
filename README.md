# gdmbayes

[![CI](https://github.com/butterman0/gdmbayes/actions/workflows/ci.yml/badge.svg)](https://github.com/butterman0/gdmbayes/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gdmbayes?cacheSeconds=3600)](https://pypi.org/project/gdmbayes/)
[![Python](https://img.shields.io/pypi/pyversions/gdmbayes?cacheSeconds=3600)](https://pypi.org/project/gdmbayes/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/butterman0/gdmbayes/blob/main/LICENSE)
[![Docs](https://readthedocs.org/projects/gdmbayes/badge/?version=latest)](https://gdmbayes.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23568886.svg)](https://doi.org/10.5281/zenodo.23568886)

**Generalised Dissimilarity Modelling in Python — frequentist and Bayesian.**

Generalised Dissimilarity Modelling (GDM; [Ferrier et al. 2007](#references)) relates pairwise
compositional dissimilarity between sites to environmental and geographic distance
using monotone I-spline basis functions. gdmbayes provides:

- **`GDM`** — a frequentist estimator implementing the Ferrier et al. (2007) algorithm
  (I-splines + non-negative least squares) as used in the R `gdm` package, which it
  reproduces numerically, with a scikit-learn `fit` / `predict` interface.
- **`spGDMM`** — a fully Bayesian GDM in PyMC (NUTS via nutpie): posterior
  uncertainty on every coefficient and prediction, three variance structures,
  Gaussian-process spatial random effects, and prediction for unseen sites.
  Implements the spatial Bayesian GDM of [White et al. (2024)](#references).
- **`GDMPreprocessor`** — the shared scikit-learn transformer: I-spline meshes and
  Euclidean, geodesic or custom pairwise distances.
- Plotting for I-spline response curves, predictor importance, posterior predictive
  checks, CRPS skill and RGB "biological space" maps.

For a demonstration of gdmbayes on a real problem — predicting plankton beta-diversity
from ocean-model output — see [Horsley et al. (2026)](#references).

## Installation

```bash
pip install gdmbayes
```

Requires Python ≥ 3.12. Installing from PyPI pulls in PyMC, nutpie and ArviZ.

## Quick start

Site-level data is a DataFrame whose first three columns are `xc`, `yc` (site
coordinates) and `time_idx`, followed by the environmental predictors. The response
`y` is the condensed vector of pairwise dissimilarities (e.g. Bray–Curtis) in
`scipy.spatial.distance.pdist` order.

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from gdmbayes import GDM

n = 30
rng = np.random.default_rng(0)
X = pd.DataFrame({
    "xc": rng.uniform(0, 1e5, n),          # metres (projected CRS)
    "yc": rng.uniform(0, 1e5, n),
    "time_idx": np.zeros(n, dtype=int),
    "temperature": rng.uniform(5, 20, n),
    "salinity": rng.uniform(30, 35, n),
})
y = pdist(rng.exponential(1, (n, 20)), "braycurtis")

model = GDM(geo=True).fit(X, y)
print(model.explained_)               # fraction of deviance explained
print(model.predictor_importance_)    # per-predictor sum of I-spline coefficients
preds = model.predict(X)              # pairwise dissimilarities in [0, 1)
```

Bayesian version of the same model:

```python
from gdmbayes import spGDMM, GDMPreprocessor, ModelConfig, SamplerConfig

model = spGDMM(
    preprocessor=GDMPreprocessor(deg=3, knots=2),
    model_config=ModelConfig(variance="homogeneous", spatial_effect="abs_diff"),
    sampler_config=SamplerConfig(draws=1000, tune=1000, chains=4, random_seed=42),
)
model.fit(X, y)

idata = model.idata_                                   # ArviZ-style xarray DataTree
diag = model.summarise_sampling()                      # R-hat / ESS table
preds = model.predict(X)                               # posterior-mean dissimilarity per pair
draws = model.predict_posterior(X, combined=True, extend_idata=False)  # full posterior draws
```

## Configuring the Bayesian model

`spGDMM` is the model of [White et al. (2024)](#references); see that paper for the full
specification and priors. In their notation, the (log-scale) dissimilarity between sites
$s_i$ and $s_j$ is

$$V(s_i, s_j) \sim \mathcal{N}\big(\mu(s_i, s_j) + \eta(s_i, s_j),\; \sigma^2(s_i, s_j)\big),$$

$$\mu(s_i, s_j) = \beta_0 + \beta_1\, h\big(\lVert s_i - s_j \rVert\big) + \sum_k \alpha_k \big\lvert f_k(X_k(s_i)) - f_k(X_k(s_j)) \big\rvert ,$$

where $h$ and $f_k$ are monotone I-spline warping functions of geographic distance and of
each environmental covariate, $\eta$ is a spatial random effect built from a Gaussian
process $\psi(s)$ over sites, and $\sigma^2$ is the variance. `ModelConfig` chooses the
forms of $\eta$ and $\sigma^2$; `GDMPreprocessor` builds $h$ and $f_k$; `SamplerConfig`
controls MCMC.

```python
ModelConfig(
    variance="homogeneous",      # "homogeneous" | "covariate_dependent" | "polynomial" | callable
    spatial_effect="none",       # "none" | "abs_diff" | "squared_diff" | callable
    alpha_importance=True,       # hierarchical per-predictor importance weights
)

GDMPreprocessor(
    deg=3,                       # I-spline degree
    knots=2,                     # interior knots (deg + knots basis functions per predictor)
    mesh_choice="percentile",    # knot placement: "percentile" | "even" | "custom"
    distance_measure="euclidean",# "euclidean" | "geodesic" | callable (see below)
    extrapolation="clip",        # outside the training mesh: "clip" | "error" | "nan"
)

SamplerConfig(draws=1000, tune=1000, chains=4, target_accept=0.95,
              nuts_sampler="nutpie", random_seed=42)
```

| `variance` | $\sigma^2(s_i, s_j)$ | [White et al. (2024)](#references) |
|---|---|---|
| `"homogeneous"` | $\sigma^2$ constant | homogeneous variance |
| `"covariate_dependent"` | $\log \sigma^2 = g\big(\lVert s_i - s_j \rVert\big)$, $g$ a cubic polynomial in geographic distance | heterogeneous variance in distance |
| `"polynomial"` | $\log \sigma^2 = g\big(\mu(s_i, s_j)\big)$, $g$ a cubic polynomial in the mean | heterogeneous variance in the mean |

| `spatial_effect` | $\eta(s_i, s_j)$ |
|---|---|
| `"none"` | $0$ |
| `"abs_diff"` | $\lvert \psi(s_i) - \psi(s_j) \rvert$ |
| `"squared_diff"` | $\big(\psi(s_i) - \psi(s_j)\big)^2$ |

with $\psi(s) \sim \mathcal{GP}$ with an exponential covariance. Models 1–9 of [White et al. (2024)](#references)
are the nine combinations of the three variance and three spatial-effect choices.

### Geographic distance

R `gdm` computes Euclidean distance directly on the input coordinates, in whatever
units they are given, so you project first. gdmbayes does the same by default but
**assumes projected coordinates in metres and converts to kilometres**, so that the
Gaussian-process length scale (`length_scale_`) is in km. Two further options:

```python
# Projected coordinates in metres; distances in km
GDMPreprocessor(distance_measure="euclidean")

# Longitude/latitude in degrees: great-circle distance in km
GDMPreprocessor(distance_measure="geodesic")

# Anything else (least-cost, along-coast, ...): a callable returning the condensed
# pairwise distance vector, in km, in scipy.spatial.distance.pdist order
from scipy.spatial.distance import pdist

def manhattan_km(coords):           # coords: (n_sites, 2) array of [xc, yc]
    return pdist(coords, "cityblock") / 1000.0

GDMPreprocessor(distance_measure=manhattan_km)
```

A saved model records a custom distance function by import path and re-imports it on
`spGDMM.load()`. That works for any function defined in an importable module; a lambda
or a function defined inside a notebook can't be re-imported, and `load()` then raises
an error asking you to pass it: `spGDMM.load("fit.nc", distance_measure=manhattan_km)`.

### Custom variance and spatial-effect functions

Callables run inside the PyMC model context, so you can declare new priors directly:

```python
import pymc as pm

def my_variance(mu, X_sigma):               # mu: linear predictor per pair
    beta_s = pm.HalfNormal("beta_s", sigma=1)
    return beta_s * pm.math.exp(mu)

def my_spatial(psi, row_ind, col_ind):      # psi: GP value per site
    return pm.math.tanh(psi[row_ind] - psi[col_ind])

ModelConfig(variance=my_variance, spatial_effect=my_spatial)
```

The built-ins live in the `VARIANCE_FUNCTIONS` and `SPATIAL_FUNCTIONS` registries.

## Plotting and diagnostics

Every function below is used, with output, in
[`examples/southwest_running_example.ipynb`](https://github.com/butterman0/gdmbayes/blob/main/examples/southwest_running_example.ipynb).

```python
from gdmbayes import (plot_isplines, plot_predictor_importance, plot_obs_vs_pred,
                      plot_link_curve, plot_ppc, crps_boxplot, rgb_biological_space)

figs = plot_isplines(model)                       # one figure per predictor (cf. R gdm::plot.gdm)
fig, ax = plot_predictor_importance(model)        # posterior importance with HDI bars
fig, ax = plot_obs_vs_pred(model, X_test, y_test)
fig, ax = plot_ppc(model.idata_, y_train)         # posterior predictive check

draws = model.predict_posterior(X_test, combined=True, extend_idata=False)
fig, axes = crps_boxplot(y_test, draws, y_train)  # CRPS skill vs. climatological null
```

`spGDMM.save("fit.nc")` / `spGDMM.load("fit.nc")` round-trip a fitted model through
NetCDF, including the fitted preprocessor.

## Examples

| Notebook | What it shows |
|---|---|
| [`examples/basic_usage.ipynb`](https://github.com/butterman0/gdmbayes/blob/main/examples/basic_usage.ipynb) | `GDM` and `spGDMM` on a small synthetic dataset, diagnostics, prediction for new sites (~1 min) |
| [`examples/model_variants.ipynb`](https://github.com/butterman0/gdmbayes/blob/main/examples/model_variants.ipynb) | Configuring variance structures and spatial effects, custom callables, serialisation (no sampling) |
| [`examples/southwest_running_example.ipynb`](https://github.com/butterman0/gdmbayes/blob/main/examples/southwest_running_example.ipynb) | Full analysis of the SW Australia dataset shipped in `examples/data/`: fit, cross-validation, response curves, importance, PPC, CRPS skill and biological-space map |

## Documentation

API reference and developer notes: **https://gdmbayes.readthedocs.io**

## Citation

If you use gdmbayes, please cite the package (a `CITATION.cff` is included — GitHub's
"Cite this repository" button exports it) together with the method papers: [Ferrier
et al. (2007)](#references) for GDM and [White et al. (2024)](#references) for the spatial Bayesian formulation.

```bibtex
@software{gdmbayes,
  title   = {gdmbayes: Bayesian and Frequentist Generalised Dissimilarity Modelling in Python},
  author  = {Horsley, Harold},
  year    = {2026},
  version = {0.1.0},
  license = {MIT},
  doi     = {10.5281/zenodo.23568886},
  url     = {https://github.com/butterman0/gdmbayes}
}
```

## References

- Ferrier, S., Manion, G., Elith, J. & Richardson, K. (2007). Using generalized
  dissimilarity modelling to analyse and predict patterns of beta diversity in
  regional biodiversity assessment. *Diversity and Distributions*, 13, 252–264.
  https://doi.org/10.1111/j.1472-4642.2007.00341.x
- White, P. A., Frye, H. A., Slingsby, J. A., Silander, J. A. & Gelfand, A. E. (2024).
  Spatially explicit generalized dissimilarity modelling: a Bayesian approach to
  modelling species turnover with spatial random effects. *Methods in Ecology and
  Evolution*, 15, 214–226. https://doi.org/10.1111/2041-210X.14259
- Mokany, K., Ware, C., Woolley, S. N. C., Ferrier, S. & Fitzpatrick, M. C. (2022).
  A working guide to harnessing generalized dissimilarity modelling for biodiversity
  analysis and conservation assessment. *Global Ecology and Biogeography*, 31, 802–821.
  https://doi.org/10.1111/geb.13459
- Fitzpatrick, M. C., Mokany, K., Manion, G., Nieto-Lugilde, D. & Ferrier, S. (2025).
  *gdm: Generalized Dissimilarity Modeling*. R package. https://CRAN.R-project.org/package=gdm
- Horsley, H., Ellingsen, I., White, P. A., O'Hara, R. B. & Alver, M. O. (2026). The role
  of temporal alignment and connectivity-informed spatial separation in plankton
  beta-diversity prediction. *Ecological Modelling*, 522, 111779.
  https://doi.org/10.1016/j.ecolmodel.2026.111779 — a demonstration of gdmbayes in use.

## Provenance and acknowledgements

`spGDMM` is a from-scratch Python/PyMC reimplementation of the spatial Bayesian GDM of
[White et al. (2024)](#references). Its model structure (I-spline link, LogNormal coefficient priors,
the three variance specifications and the two Gaussian-process spatial-effect forms)
follows their reference NIMBLE implementation,
https://github.com/philawhite/spGDMM-code (Zenodo: https://doi.org/10.5281/zenodo.10091442),
which was also used to validate this package; the Panama (BCI) and South Africa
benchmark datasets were obtained from that repository. The frequentist `GDM`
estimator implements the Ferrier et al. (2007) algorithm as used in the R `gdm` package
([Fitzpatrick et al. 2025](#references), GPL-3): I-spline features and iteratively
reweighted non-negative least squares with a complementary log-log link. It is written
in Python on NumPy/SciPy, with the `gdm` C++ source consulted as the specification so
that results agree to floating-point precision (see the regression test); no code is
copied from it. The Southwest Australia example data are derived from the `southwest`
dataset distributed with `gdm` — see `examples/data/southwest/README.md`. Thanks to Philip White for
making the reference code and data publicly available.

## Contributing and licence

Contributions are welcome — [CONTRIBUTING.md](https://github.com/butterman0/gdmbayes/blob/main/CONTRIBUTING.md)
has the development setup (`pip install -e ".[dev]"`), test and style instructions; see also the
[Code of Conduct](https://github.com/butterman0/gdmbayes/blob/main/CODE_OF_CONDUCT.md). Released under the MIT licence.
