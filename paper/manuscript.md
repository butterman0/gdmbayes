# gdmbayes: A Python Package for Bayesian Generalised Dissimilarity Modelling

**Harold Horsley**

*Affiliation*

Correspondence: [email]

---

## Abstract

Generalised dissimilarity modelling (GDM) is a widely used method for relating pairwise
ecological dissimilarities between sites to environmental and spatial predictors using
non-linear I-spline basis functions. Despite its prevalence in ecology, GDM has until now
lacked a Python implementation and has been confined to a single R package that provides
only point estimates with no uncertainty quantification. A major difficulty in applying GDM
to contemporary ecological datasets lies in the inability to propagate uncertainty through
predictor effects or to incorporate spatial random effects within the existing tooling.

We present **gdmbayes**, an open-source Python package that fills this gap by providing
both a frequentist sklearn-compatible GDM estimator and a full Bayesian backend built on
PyMC. The Bayesian estimator supports flexible variance structures, Gaussian process
spatial random effects, and hierarchical predictor importance weights, all within a
reproducible, configuration-driven workflow. gdmbayes follows scikit-learn conventions
throughout, enabling seamless integration with Python's scientific computing ecosystem.

We demonstrate the package on a simulated species dissimilarity dataset and show that the
Bayesian extension recovers predictor effects with well-calibrated posterior uncertainty.
gdmbayes is the first Python GDM implementation and the only implementation—in any
language—to offer full Bayesian inference for GDM. The package is freely available under
the Apache 2.0 licence at https://github.com/harryhorsley9/spgdmm.

---

## 1. Introduction

Ecologists increasingly rely on large spatially referenced datasets to understand how
species composition changes across landscapes. Generalised dissimilarity modelling (GDM;
Ferrier et al. 2007) has become one of the most widely applied frameworks for this
purpose. GDM relates pairwise ecological dissimilarities—typically Bray–Curtis or Sørensen
distances computed from species assemblage data—to pairwise differences in environmental
predictors and geographic distance using a generalised additive model structure built on
I-spline basis functions. The non-linear I-spline representation allows GDM to capture the
empirical observation that the rate of compositional turnover saturates at high
environmental distances, a biological property that linear models cannot accommodate.

GDM has been applied across a wide range of ecological contexts including beta-diversity
gradients (Ferrier et al. 2007), conservation planning (Mokany et al. 2012), climate-
change vulnerability assessment (Fitzpatrick & Keller 2015), and spatial prioritisation of
biodiversity surveys. The R package **gdm** (Fitzpatrick et al. 2021) has been the sole
implementation of the method for nearly two decades, providing a point-estimate frequentist
estimator via non-negative least squares (NNLS). Despite the method's popularity, two
important limitations have persisted. First, there is no Python implementation, which
limits integration with the broader scientific Python ecosystem and with machine-learning
pipelines built around scikit-learn. Second, neither the R package nor any other tool
provides uncertainty estimates for GDM coefficients or predictions, making it impossible
to propagate ecological uncertainty through downstream analyses such as conservation
prioritisation or climate projection.

Bayesian inference provides a natural solution to both problems. A Bayesian GDM yields
full posterior distributions over predictor effects, enabling principled uncertainty
quantification for predictor importance scores and for predicted dissimilarities at new
sites. Spatial random effects—structured residual variation unexplained by the available
environmental predictors—can also be incorporated within the Bayesian framework using
Gaussian process priors. These extensions are relevant whenever spatial autocorrelation is
present in species data, as is typically the case in observational ecology.

Here we present **gdmbayes**, a Python package that provides (1) a frequentist GDM
estimator that replicates the R gdm algorithm and is fully compatible with the scikit-learn
API, and (2) a Bayesian GDM estimator that uses PyMC for MCMC sampling and ArviZ for
posterior diagnostics. Both estimators share a common preprocessing layer that handles
I-spline mesh construction, pairwise distance computation, and feature matrix assembly.
This paper describes the package architecture, the statistical model, and demonstrates its
use on a simulated dataset.

---

## 2. Package Description

### 2.1 Overview and installation

gdmbayes is a Python package for modelling pairwise ecological dissimilarities as a
function of environmental and spatial predictors. The package is organised around three
main classes—`GDM`, `spGDMM`, and `GDMModel`—sharing a common preprocessing component
(`GDMPreprocessor`). All classes follow the scikit-learn `fit(X, y)` / `predict(X)` API.

The package is installed via pip:

```bash
pip install gdmbayes
```

A conda environment file is provided for reproducibility. Dependencies include PyMC ≥ 5.0,
ArviZ, nutpie (for the NUTS sampler), NumPy, pandas, SciPy, xarray, and scikit-learn.

The conceptual workflow is:

1. Prepare a site-level `pd.DataFrame` with columns `[xc, yc, time_idx, predictor_1, ...]`
   and a condensed pairwise dissimilarity vector `y` of length n(n-1)/2.
2. (Optional) Configure preprocessing via `PreprocessorConfig` and model structure via
   `ModelConfig`.
3. Call `model.fit(X, y)` to fit the frequentist or Bayesian estimator.
4. Inspect coefficients, predictor importance, posterior distributions, or generate
   predictions at new sites via `model.predict(X)`.
5. Use built-in plotting functions (`plot_isplines`, `plot_ppc`, `plot_crps_comparison`)
   for diagnostics.

### 2.2 Preprocessing

All preprocessing is handled by `GDMPreprocessor`, a standalone sklearn-compatible
transformer (`BaseEstimator`, `TransformerMixin`). The preprocessor computes the I-spline
feature matrix from raw site-level data and stores fitted state (knot meshes, location
arrays) for use at prediction time.

**I-spline basis functions.** For each environmental predictor x, the preprocessor
constructs J = deg + knots I-spline basis functions I_1(x), ..., I_J(x) over a mesh of
evenly spaced or percentile-based knot positions. I-splines are the integral of M-splines
and are monotone non-decreasing by construction (Ramsay 1988), which ensures that predicted
dissimilarity increases monotonically with environmental distance. The pairwise feature for
predictor k between sites i and j is:

    f_{k,l}(i, j) = | I_l(x_{k,i}) - I_l(x_{k,j}) |

for l = 1, ..., J. Geographic distance between sites is optionally included as an
additional predictor using the same I-spline transformation applied to the pairwise
Euclidean, geodesic, or grid-based ocean distance.

Knot placement is controlled by `PreprocessorConfig`:

```python
from gdmbayes import PreprocessorConfig

preprocessor_config = PreprocessorConfig(
    deg=3,                        # I-spline polynomial degree
    knots=2,                      # number of internal knots
    mesh_choice="percentile",     # knot placement: "percentile", "even", "custom"
    distance_measure="euclidean", # "euclidean", "geodesic", "ocean_distance"
)
```

### 2.3 Frequentist GDM

`GDM` implements the standard GDM algorithm from Ferrier et al. (2007) as a scikit-learn
estimator. Given the pairwise I-spline feature matrix **X**_GDM ∈ R^{P × Q} (P site pairs,
Q I-spline columns), the frequentist estimator minimises:

    min_{β ≥ 0} ‖ g(y) − **X**_GDM β ‖²

where g(y) = −log(1 − y) is the complementary log-log (cloglog) link applied to the
observed Bray–Curtis dissimilarities y ∈ (0, 1). Non-negative least squares (NNLS; Lawson
& Hanson 1974) enforces β ≥ 0, consistent with the monotone I-spline representation.
Predicted dissimilarities are recovered via the inverse link:

    ŷ = 1 − exp(−**X**_GDM β̂)

Predictor importance is defined as the sum of NNLS coefficients over all I-spline basis
functions for each predictor, analogous to the R gdm package. Deviance explained is the
fraction of null deviance (sum of squares of g(y) about its mean) accounted for by the
model.

```python
from gdmbayes import GDM

model = GDM(geo=True, splines=3, knots=2)
model.fit(X, y)

print(f"Deviance explained: {model.explained_:.4f}")
print(f"Predictor importance: {model.predictor_importance_}")

predictions = model.predict(X_new)  # ndarray of shape (n_pairs,)
```

Because `GDM` inherits from sklearn's `BaseEstimator` and `RegressorMixin`, it participates
fully in cross-validation, hyperparameter search, and pipeline composition.

### 2.4 Bayesian GDM (spGDMM)

`spGDMM` (Spatial Generalised Dissimilarity Mixed Model) extends the frequentist GDM with
full Bayesian inference via PyMC. The response variable is modelled on the log scale:

    log(y_{ij}) ~ CensoredNormal(μ_{ij}, σ²_{ij}, upper = 0)

where the upper censoring at 0 on the log scale (equivalently y ≤ 1 on the original scale)
reflects the bounded nature of dissimilarities.

**Linear predictor.** With the `alpha_importance` option (default), the linear predictor
for site pair (i, j) is:

    μ_{ij} = β_0 + Σ_k α_k Σ_l β_{k,l} f_{k,l}(i, j) + Σ_l β^d_l I_l(d_{ij})

where β_0 is a global intercept, **β**_k = (β_{k,1}, ..., β_{k,J}) ~ Dirichlet(1, ..., 1)
are normalised I-spline coefficients for predictor k, **α** = (α_1, ..., α_K) ~
HalfNormal(1) are per-predictor scale parameters (importance weights), and **β**^d ~
LogNormal(0, 1) are geographic distance coefficients. The Dirichlet prior on **β**_k
enforces the constraint Σ_l β_{k,l} = 1, so that α_k captures the total magnitude of
predictor k's contribution and **β**_k controls how that contribution is distributed across
the I-spline basis functions. Without `alpha_importance`, coefficients follow independent
LogNormal(0, 1) priors directly.

**Variance structure.** Three built-in variance functions are available:

- `"homogeneous"`: σ² ~ HalfNormal(1), constant across all pairs.
- `"covariate_dependent"`: σ² is a function of pairwise geographic distance.
- `"polynomial"`: σ² is a polynomial function of the linear predictor μ.

Custom variance functions can be passed as Python callables with signature
`fn(mu, X_sigma) -> sigma2`, executed inside the PyMC model context.

**Spatial random effects.** An optional Gaussian process prior can be added to model
spatial autocorrelation in residual dissimilarities. When `spatial_effect` is set, a
latent GP ψ ~ GP(0, σ²_ψ · Exp(‖·‖ / ℓ)) is placed on the n training sites, and the
pairwise effect between sites i and j is added to μ_{ij}. The length scale ℓ is estimated
from the median inter-site distance; σ²_ψ ~ InverseGamma(1, 1). Two built-in contrast
functions are available (`"abs_diff"` and `"squared_diff"`), and custom callables are
supported.

**Sampler.** MCMC is performed with the No-U-Turn Sampler (NUTS; Hoffman & Gelman 2014)
via PyMC, with nutpie as the default backend for improved sampling speed. Sampler
settings are controlled via `SamplerConfig`:

```python
from gdmbayes import SamplerConfig

sampler_config = SamplerConfig(
    draws=1000, tune=1000, chains=4,
    target_accept=0.95, nuts_sampler="nutpie",
)
```

**Model fitting and diagnostics.** After sampling, inference data is stored in an ArviZ
`InferenceData` object (`model.idata`), enabling the full ArviZ diagnostic toolkit
(R-hat, effective sample size, posterior predictive checks). The package provides wrappers:

```python
from gdmbayes import spGDMM, ModelConfig, PreprocessorConfig, summarise_sampling

model = spGDMM(
    preprocessor=PreprocessorConfig(deg=3, knots=2),
    model_config=ModelConfig(variance="homogeneous", spatial_effect="abs_diff"),
)
idata = model.fit(X, y, random_seed=42)

diagnostics = summarise_sampling(idata)   # ESS and R-hat summary
```

### 2.5 GDMModel and R-compatible output

`GDMModel` wraps `spGDMM` and returns a `GDMResult` dataclass whose attributes mirror
the R gdm package output (deviance, nulldeviance, explained, intercept, predictors,
coefficients, knots, splines). This provides a drop-in output format for users migrating
from R:

```python
from gdmbayes import GDMModel

model = GDMModel(geo=True, splines=3)
result = model.fit(X, y, dataname="my_survey")

print(f"Deviance explained: {result.explained:.1f}%")
print(f"Predictors: {result.predictors}")
```

### 2.6 Model serialisation

Fitted `spGDMM` models are serialised as NetCDF files via `model.save(fname)` and
restored via `spGDMM.load(fname)`. The serialisation stores the full InferenceData object
(posterior samples, prior/posterior predictive samples, training data) together with the
fitted preprocessor state (spline meshes, knot positions, site coordinates) as xarray
datasets in the `constant_data` group. This enables prediction on new data after
deserialisation without refitting:

```python
model.save("fitted_gdm.nc")
model2 = spGDMM.load("fitted_gdm.nc")
predictions = model2.predict(X_new)
```

### 2.7 Plotting and diagnostics

The `gdmbayes.plotting` module provides four functions:

- `plot_isplines(model)`: Plot the posterior median I-spline effect curves per predictor,
  with optional credible intervals.
- `plot_ppc(idata, y_obs)`: Posterior predictive check comparing observed and predicted
  dissimilarity distributions.
- `plot_crps_comparison(y_test, y_pred, y_train)`: Compare model CRPS against a null
  (mean) baseline.
- `summarise_sampling(idata)`: Tabular ESS and R-hat diagnostics for all parameters.

---

## 3. Usage Examples

### 3.1 Frequentist GDM

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from gdmbayes import GDM

np.random.seed(42)
n = 30
X = pd.DataFrame({
    "xc":       np.random.uniform(0, 100, n),
    "yc":       np.random.uniform(0, 100, n),
    "time_idx": np.zeros(n),
    "temp":     np.random.uniform(5, 20, n),
    "precip":   np.random.uniform(400, 1200, n),
})
species = np.random.exponential(1, (n, 50))
y = pdist(species, "braycurtis").clip(1e-8, 1 - 1e-8)

model = GDM(geo=True, splines=3, knots=2)
model.fit(X, y)
print(f"Explained: {model.explained_:.3f}")
print(f"Importance: {model.predictor_importance_}")
```

### 3.2 Bayesian GDM with spatial effect

```python
from gdmbayes import spGDMM, ModelConfig, PreprocessorConfig, SamplerConfig
from gdmbayes import plot_isplines, plot_ppc

model = spGDMM(
    preprocessor=PreprocessorConfig(deg=3, knots=2, distance_measure="euclidean"),
    model_config=ModelConfig(
        variance="homogeneous",
        spatial_effect="abs_diff",
        alpha_importance=True,
    ),
    sampler_config=SamplerConfig(draws=1000, tune=1000, chains=4, random_seed=42),
)
idata = model.fit(X, y)

plot_isplines(model)          # posterior I-spline curves
plot_ppc(idata, y)            # posterior predictive check
```

### 3.3 Custom variance function

The Bayesian model structure is extensible via callables. The following defines a variance
that scales exponentially with the linear predictor:

```python
import pymc as pm
from gdmbayes import ModelConfig, spGDMM

def heteroskedastic_variance(mu, X_sigma):
    beta_s = pm.HalfNormal("beta_s", sigma=1)
    return beta_s * pm.math.exp(0.5 * mu)

model = spGDMM(
    model_config=ModelConfig(variance=heteroskedastic_variance)
)
idata = model.fit(X, y)
```

---

## 4. Comparison with the R gdm Package

Table 1 compares gdmbayes with the R gdm package (Fitzpatrick et al. 2021) across key
features.

**Table 1.** Feature comparison of gdmbayes (Python) and gdm (R).

| Feature | gdmbayes (Python) | gdm (R) |
|---|---|---|
| Frequentist GDM (NNLS) | Yes | Yes |
| Bayesian inference (MCMC) | Yes | No |
| Posterior uncertainty for coefficients | Yes | No |
| Posterior uncertainty for predictions | Yes | No |
| Spatial random effects (GP) | Yes | No |
| Custom variance structures | Yes | No |
| Predictor importance weights | Yes | Yes |
| I-spline basis (monotone) | Yes | Yes |
| Geodesic distance | Yes | Partial |
| sklearn API (fit/predict) | Yes | No |
| Pipeline composition | Yes | No |
| Model serialisation | Yes (NetCDF) | Yes (R object) |
| Posterior predictive checks | Yes (ArviZ) | No |
| Language | Python ≥ 3.10 | R |

The R gdm package remains the reference implementation for frequentist GDM and provides
additional utilities such as partial regression plots and Monte Carlo significance testing
that are not yet included in gdmbayes. gdmbayes prioritises Bayesian inference,
uncertainty quantification, and Python ecosystem integration.

---

## 5. Availability

gdmbayes is available at https://github.com/harryhorsley9/spgdmm under the Apache 2.0
licence. A PyPI release is forthcoming. A Zenodo DOI will be minted at the time of
publication. The package requires Python ≥ 3.10 and is tested on Linux.

---

## Acknowledgements

[To be completed.]

---

## Author Contributions

H. Horsley: conceptualisation, software, formal analysis, writing – original draft,
writing – review and editing.

---

## Data Availability

No empirical data were used in this paper. Example scripts in the `examples/` directory
of the repository are fully self-contained and generate synthetic data.

---

## References

Ferrier S, Manion G, Elith J, Richardson K (2007) Using generalised dissimilarity
modelling to analyse and predict patterns of beta diversity in regional biodiversity
assessment. *Diversity and Distributions*, **13**, 252–264.
https://doi.org/10.1111/j.1472-4642.2007.00341.x

Fitzpatrick MC, Sanders NJ, Normand S, et al. (2013) Environmental and historical
imprints on beta diversity: insights from variation in rates of species turnover along
gradients. *Proceedings of the Royal Society B*, **280**, 20131201.

Fitzpatrick MC, Mokany K, Manion G, Nieto-Lugilde D, Ferrier S (2021) gdm: Generalized
Dissimilarity Modeling. R package version 1.5.
https://CRAN.R-project.org/package=gdm

Fitzpatrick MC, Keller SR (2015) Ecological genomics meets community-level modelling of
biodiversity: mapping the genomic landscape of current and future environmental
adaptation. *Ecology Letters*, **18**, 1–16.
https://doi.org/10.1111/ele.12376

Hoffman MD, Gelman A (2014) The No-U-Turn Sampler: adaptively setting path lengths in
Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, **15**, 1593–1623.

Kumar R, Carroll C, Hartikainen A, Martin O (2019) ArviZ: a unified library for
exploratory analysis of Bayesian models in Python. *Journal of Open Source Software*,
**4**, 1143. https://doi.org/10.21105/joss.01143

Lawson CL, Hanson RJ (1974) *Solving Least Squares Problems*. Prentice-Hall, Englewood
Cliffs, NJ.

Mokany K, Harwood TD, Ferrier S (2012) A simulation study of landscape biodiversity
change under alternative scenarios: testing potential analysis methods. *Diversity and
Distributions*, **18**, 1090–1103.

Mostert PS, Bjorkås R, Bruls A, et al. (2025) intSDM: An R Package for Building a
Reproducible Workflow for the Field of Integrated Species Distribution Models. *Ecology
and Evolution*, **15**, e71029. https://doi.org/10.1002/ece3.71029

Pedregosa F, Varoquaux G, Gramfort A, et al. (2011) Scikit-learn: Machine Learning in
Python. *Journal of Machine Learning Research*, **12**, 2825–2830.

Ramsay JO (1988) Monotone regression splines in action. *Statistical Science*, **3**,
425–441. https://doi.org/10.1214/ss/1177012761

Salvatier J, Wiecki TV, Fonnesbeck C (2016) Probabilistic programming in Python using
PyMC3. *PeerJ Computer Science*, **2**, e55. https://doi.org/10.7717/peerj-cs.55
