# Variance Model Sampling Issues in gdmbayes/spGDMM

**Date:** 2026-03-21
**Purpose:** Self-contained diagnostic document for handing off to another AI for diagnosis and fixing.

---

## Table of Contents

1. [Mathematical Model Specification](#1-mathematical-model-specification)
2. [What Works vs What Fails](#2-what-works-vs-what-fails)
3. [Root Cause Analysis — The Funnel](#3-root-cause-analysis--the-funnel)
4. [What's Been Tried](#4-whats-been-tried)
5. [Identified Code Issues](#5-identified-code-issues)
6. [Remaining Options to Try](#6-remaining-options-to-try)
7. [Key Files](#7-key-files)
8. [Diagnostic Output](#8-diagnostic-output)
9. [Constraints](#9-constraints)
10. [Source Code](#10-source-code)

---

## 1. Mathematical Model Specification

### Overview

spGDMM models pairwise ecological dissimilarities between sites using I-spline basis functions. The response is the **log** of pairwise Bray-Curtis dissimilarity, upper-censored at 0 (since dissimilarity is bounded [0, 1], log-dissimilarity is bounded (-inf, 0]).

### Full Probabilistic Model

**Data:**
- `y_ij` = pairwise Bray-Curtis dissimilarity between sites i and j, in [0, 1]
- `X` = site-level environmental predictors (n_sites x n_predictors)
- Site coordinates (xc, yc) for spatial distance computation

**Transformed data:**
- `log_y_ij = log(y_ij)` — the response, upper-censored at 0
- `X_GDM` = pairwise I-spline difference matrix. For each predictor, I-spline bases are computed at each site using the predictor values, then pairwise absolute differences of these bases form the columns. Geographic distance splines are appended.
- `X_sigma` = 4-column polynomial design matrix for variance model (see below)

**Model (no alpha_importance, matching White et al.):**

```
beta_0 ~ Normal(0, 10)                          # intercept
beta_k ~ LogNormal(0, 10)  for k = 1..p         # I-spline coefficients (p = n_cols in X_GDM)

mu_ij = beta_0 + X_GDM[ij, :] @ beta            # linear predictor

sigma2_ij = f(mu_ij, X_sigma_ij)                 # variance (see below)

log_y_ij ~ CensoredNormal(mu_ij, sqrt(sigma2_ij), upper=0)
```

The censored normal means: if `log_y_ij = 0` (i.e., `y_ij = 1.0`, complete dissimilarity), the observation contributes `log(1 - Phi((0 - mu) / sigma))` to the log-likelihood, where `Phi` is the standard normal CDF.

### Variance Models

#### Homogeneous (works)
```
sigma2 ~ InverseGamma(1, 1)
```
Single scalar variance shared across all pairs.

#### Covariate-Dependent (FAILS)
```
d_ij = pairwise geographic distance between sites i and j
d_z_ij = (d_ij - mean(d_train)) / std(d_train)     # standardized

X_sigma_ij = [1, d_z_ij, d_z_ij^2, d_z_ij^3]       # 4-column polynomial

beta_sigma_raw ~ Normal(0, 1)   shape=(4,)           # non-centered
beta_sigma = 10 * beta_sigma_raw                      # implied prior: Normal(0, 10)

sigma2_ij = exp(clip(X_sigma_ij @ beta_sigma, -20, 20))
```

#### Polynomial (works after non-centered fix)
```
beta_sigma_raw ~ Normal(0, 1)   shape=(4,)           # non-centered
beta_sigma = 10 * beta_sigma_raw                      # implied prior: Normal(0, 10)

poly_ij = beta_sigma[0] + beta_sigma[1]*mu_ij + beta_sigma[2]*mu_ij^2 + beta_sigma[3]*mu_ij^3

sigma2_ij = exp(clip(poly_ij, -20, 20))
```

### How X_sigma Is Constructed

In `_generate_and_preprocess_model_data()`:

1. Compute pairwise geographic distances for all training sites
2. Clip to the distance mesh range
3. Standardize: `d_z = (pw_dist - mean) / std`
4. Build 4-column matrix: `X_sigma = [ones, d_z, d_z^2, d_z^3]`
5. Store `d_mean` and `d_std` on the model (`self._d_mean`, `self._d_std`) for consistent standardization at prediction time

**Critical detail:** `X_sigma` is only created when `n_predictors > 0`. If there are no environmental predictors, `X_sigma = None` and all variance models silently fall back to `InverseGamma(1, 1)`.

### Key Difference Between Covariate-Dependent and Polynomial

| Aspect | Covariate-Dependent | Polynomial |
|--------|---------------------|------------|
| Input to exp() | `X_sigma @ beta_sigma` (geographic distance polynomial) | `beta_sigma[0] + beta_sigma[1]*mu + ...` (mu polynomial) |
| What varies | Raw standardized distances (can have large values) | mu (bounded by data and I-spline structure) |
| Initvals | zeros (neutral) | White et al.'s tuned `[-5, -20, 12, 2]` |

---

## 2. What Works vs What Fails

### Homogeneous: Works Perfectly
- 0.1-0.4% divergences
- Good ESS (hundreds to thousands)
- Sensible predictions (range 0.35-0.99)
- R-hat ~1.0

### Polynomial (non-centered): Now Works
- Full-data: **0.1% divergences** (3/2000), sensible predictions (0.69-0.99)
- CV fold: **19.8% divergences** (395/2000), still produces sensible predictions (0.78-0.89)
- **BUT:** beta posteriors are bimodal — medians near 0 but means ~10^11, indicating some chains found good modes and others exploded
- R-hat 1.5-2.9, ESS bulk 5-8 — needs more draws/chains but is fundamentally working
- Train RMSE: 0.1005, CV RMSE: 0.0922

### Covariate-Dependent (non-centered): FAILS Completely
- **27.4% divergences** (549/2000 full-data, 480/2000 CV)
- Beta coefficients explode to **10^20 - 10^24**
- All predictions = **1.0** (complete dissimilarity for every pair)
- ESS = **4** (minimum possible with 4 chains — each chain is a single point)
- R-hat up to **152,854,748** (chains in completely different regions of parameter space)
- beta_sigma medians: `[3.2, 0.5, 6.6, -0.5]` — not extreme, but beta is catastrophically unconstrained

---

## 3. Root Cause Analysis — The Funnel

### Neal's Funnel Geometry

The core issue is **Neal's funnel** — a pathological posterior geometry that arises when a scale parameter controls the width of other parameters:

1. When `beta_sigma` drifts to values that make `exp(X_sigma @ beta_sigma)` large → variance `sigma2` becomes huge
2. Huge variance → likelihood is nearly flat (any `mu` value is equally likely under very wide normal)
3. Flat likelihood → `beta` coefficients are unconstrained → beta explodes to 10^20+
4. Once beta is huge → `mu` is huge → predictions saturate at `1 - exp(-exp(mu))` ≈ 1.0
5. The sampler gets trapped: the narrow neck of the funnel (reasonable beta_sigma) has high curvature, while the wide body (large beta_sigma, any beta) is flat

### Why NIMBLE Handles It But NUTS Can't

White et al.'s original R/NIMBLE implementation uses **scalar-at-a-time Gibbs/slice sampling**:
- Each parameter is updated one at a time, conditional on all others
- Slice sampling can handle the varying curvature because it adapts step size per-parameter per-iteration
- The funnel geometry is less problematic because the sampler never needs to jointly navigate the narrow neck

**NUTS** (used by PyMC/nutpie) does **joint exploration**:
- All parameters are updated simultaneously using Hamiltonian dynamics
- A single step size and mass matrix must work for the entire parameter space
- The funnel's extreme curvature variation (tight at the neck, flat at the body) causes:
  - Step size too large → overshoots the neck → divergences
  - Step size too small → can't explore the body → gets stuck

### Why Non-Centered Helped Polynomial But Not Covariate-Dependent

**Polynomial:** `sigma2 = exp(beta_sigma[0] + beta_sigma[1]*mu + ...)`. Here `mu` is itself a function of `beta`, creating a **direct coupling** between `beta_sigma` and `beta` through `mu`. Non-centered parameterization partially breaks this coupling by making `beta_sigma_raw` independent of the prior scale in the sampling space.

**Covariate-dependent:** `sigma2 = exp(X_sigma @ beta_sigma)`. Here `X_sigma` is **fixed data** (not a function of other parameters), so the coupling between `beta_sigma` and `beta` is purely through the likelihood. The non-centered parameterization helps with the prior geometry but doesn't address the likelihood-mediated funnel. The key problem is that `X_sigma` contains raw standardized distances that can be large, meaning even moderate `beta_sigma` values can produce extreme `exp()` outputs.

---

## 4. What's Been Tried

### 1. Increased Tuning and Target Accept
- Tuning: 1000 → 4000 steps
- Target accept: 0.95 → 0.97
- **Result:** No improvement for covariate_dependent. More tuning doesn't help when the geometry is fundamentally pathological.

### 2. White et al. BFGS Initialization
- Implemented BFGS optimization on `sum((log_y - beta_0 - X @ exp(log_beta))^2)` for `beta_0` and `beta`
- Correctly passed to nutpie via `nuts_sampler_kwargs["init_mean"]` in unconstrained space (nutpie silently ignores standard `initvals`)
- **Result:** Helps homogeneous converge faster. Does not fix covariate_dependent — the sampler quickly leaves the initial values and enters the funnel.

### 3. Non-Centered Parameterization
- Changed from `beta_sigma ~ Normal(0, 10)` to `beta_sigma_raw ~ Normal(0, 1); beta_sigma = 10 * beta_sigma_raw`
- **Result:** Fixed polynomial (0.1% divergences full-data). Did NOT fix covariate_dependent (still 27% divergences, beta ~10^24).

### 4. Compound Step Sampling (Considered)
- Idea: Use Slice sampler for `beta_sigma` + NUTS for everything else
- **Result:** PyMC's CompoundStep with mixed samplers is extremely slow and poorly supported with nutpie. Not viable.

---

## 5. Identified Code Issues

### Issue 1: Covariate-Dependent Initvals Are Weak

In `_compute_initvals()`, beta_sigma initialization differs by variance type:
- **Polynomial:** Gets White et al.'s tuned `[-5, -20, 12, 2]` (divided by 10 for raw parameterization) — these are carefully chosen values that put the sampler near a good mode
- **Covariate-dependent:** Gets **zeros** (i.e., `sigma2 = exp(0) = 1` everywhere) — a neutral but potentially poor starting point

There is no BFGS-based initialization for `beta_sigma` in either case. The BFGS optimization only covers `beta_0` and `beta`.

### Issue 2: Clip Bounds [-20, 20] Create Gradient Death Zones

Both variance functions clip the linear predictor to `[-20, 20]` before `exp()`:
```python
pm.math.exp(pt.clip(pm.math.dot(X_sigma, beta_sigma), -20, 20))
```

When the linear predictor hits the clip boundary:
- Gradient with respect to `beta_sigma` → 0
- The sampler receives no signal about which direction to move
- Parameters can get stuck at the boundary

The covariate-dependent model is more susceptible because `X_sigma` contains raw standardized distances (potentially large values like `d_z^3`), so even moderate `beta_sigma` values can hit the clip. The polynomial model uses `mu` which is more bounded.

### Issue 3: X_sigma Only Created When n_predictors > 0

```python
X_sigma=np.column_stack([...]) if n_predictors > 0 else None,
```

If there are no environmental predictors, `X_sigma` is silently set to `None`, and the variance function falls back to `InverseGamma(1, 1)` without any warning. This is a silent behavior change that could confuse users who explicitly request `variance="covariate_dependent"`.

### Issue 4: "Covariate-Dependent" Name Is Misleading

The "covariate_dependent" variance model uses a polynomial of **standardized pairwise geographic distance**, not actual environmental covariates. The name suggests it uses environmental predictors. A more accurate name would be "distance_dependent" or "distance_polynomial".

### Issue 5: No BFGS Optimization for beta_sigma

`_compute_initvals()` only optimizes `beta_0` and `beta` via BFGS. `beta_sigma` gets either:
- Hardcoded White et al. values (polynomial): `[-5, -20, 12, 2]`
- Zeros (covariate_dependent)

A joint BFGS over all parameters (`beta_0`, `beta`, AND `beta_sigma`) would give better starting points, especially for covariate_dependent where the current zeros may be far from the posterior mode.

### Issue 6: Polynomial Variance Has Bimodal Posterior

The polynomial config 2 diagnostic output shows:
- Beta medians near 0 (e.g., 0.0002, 0.023) but means ~10^11
- This means some chains found good modes (small beta) and others exploded (huge beta)
- R-hat 1.5-2.9, ESS 5-8
- Needs more draws and/or chains to get reliable inference, but the model is fundamentally working

---

## 6. Remaining Options to Try

### 1. Joint BFGS Initialization Including beta_sigma
Extend `_compute_initvals()` to jointly optimize over `beta_0`, `beta`, AND `beta_sigma` for the covariate_dependent case. The objective would be:
```
minimize sum((log_y - mu)^2 / sigma2 + log(sigma2))
```
where `sigma2 = exp(X_sigma @ beta_sigma)`. This is the negative log-likelihood of a normal with heterogeneous variance.

### 2. Increase max_treedepth
Default `max_treedepth=10` limits each leapfrog trajectory to 2^10 = 1024 steps. If the posterior has long, curved ridges (as funnels do), increasing to 12-15 allows the sampler to follow them further. Trade-off: each sample takes longer.

### 3. DEMetropolisZ Ensemble Sampler
PyMC's `DEMetropolisZ` is a differential-evolution Metropolis sampler that:
- Doesn't require gradient computation (avoids the gradient death zone from clipping)
- Uses an ensemble of chains with adaptive proposals
- Has been shown to handle funnel geometries better than NUTS in some cases
- Trade-off: much slower convergence, needs many more samples

### 4. Sequential Warm-Start from Homogeneous Posteriors
1. Fit the homogeneous model first (works well)
2. Use the posterior `beta_0` and `beta` values as initialization for the covariate_dependent model
3. This puts the sampler near a good mode for the mean parameters, so only `beta_sigma` needs to be explored from scratch

### 5. Tighter or Soft Clipping
- **Tighter clip bounds:** `[-10, 10]` instead of `[-20, 20]` would constrain `sigma2` to `[exp(-10), exp(10)]` ≈ `[0.00005, 22026]`, still a wide range but preventing the most extreme gradient death
- **Soft clipping (tanh-based):** Replace `clip(x, -20, 20)` with `20 * tanh(x/20)`, which smoothly saturates and always has non-zero gradient

### 6. Stan/cmdstanpy Backend
Stan's NUTS implementation has more sophisticated adaptation:
- Windowed warmup with mass matrix estimation
- Better step size adaptation for pathological geometries
- PyMC supports `nuts_sampler="numpyro"` and `nuts_sampler="blackjax"` as alternatives; cmdstanpy would require more integration work

---

## 7. Key Files

| File | Description |
|------|-------------|
| `src/gdmbayes/models/_variance.py` | Variance functions (homogeneous, covariate_dependent, polynomial) |
| `src/gdmbayes/models/_spgdmm.py` | Model building, initvals, sampling, full spGDMM class |
| `src/gdmbayes/models/_config.py` | ModelConfig, SamplerConfig dataclasses |
| `examples/panama_test_diagnostics.py` | Diagnostic test script that produced the output in Section 8 |
| `benchmarks/white2024_table1.csv` | Ground-truth benchmarks from White et al. (2024) |
| `examples/panama_example.py` | Panama dataset example (Bayesian CV) |
| `src/gdmbayes/preprocessing/_preprocessor.py` | GDMPreprocessor (I-spline computation, pairwise features) |

---

## 8. Diagnostic Output

### Config 1: Covariate-Dependent (FAILS)

Job 24213489, task 1. Settings: `tune=4000, draws=500, chains=4, target_accept=0.97`, nutpie sampler with BFGS initvals, non-centered parameterization.

```
=== Panama TEST — config_idx=1  tune=4000  draws=500  nutpie+initvals ===
Start: Sat 21 Mar 2026 15:03:34 CET
X shape: (39, 5), y shape: (741,)
y range: [0.3447, 1.0000], y==1.0: 3

============================================================
CONFIG 1: none_covariate_dependent
tune=4000, draws=500, chains=4, target_accept=0.97
beta prior: LogNormal(mu=0, sigma=10)
============================================================

============================================================
FULL-DATA FIT
============================================================

Full-data fit time: 12s (0.2min)

--- FULL-DATA DIAGNOSTICS ---
Divergences: 549/2000 (27.4%)

beta posteriors (n=12):
  beta[0]: mean=4176500170112491323392.0000  median=2641819346609656823808.0000
  beta[1]: mean=4371161457775713189888.0000  median=4275698030107455651840.0000
  beta[2]: mean=5251776820275762429952.0000  median=5872844287642928414720.0000
  ...
  (all 12 beta coefficients are in the range 10^21 to 10^22)

beta_0: mean=0.1103  std=0.5402
beta_sigma: [ 3.23381156  0.52919309  6.5760716  -0.50336964]

R-hat range: [3.9956, 152854748.8331]
ESS bulk range: [4, 4]
ESS tail range: [4, 4]
--- END FULL-DATA DIAGNOSTICS ---

Full-data predictions: min=1.0000, max=1.0000, mean=1.0000
RMSE (train): 0.1948
MAE  (train): 0.1483
WARNING: Predictions look degenerate (mean > 0.95) — beta likely exploded!

============================================================
1-FOLD CV
============================================================
Train: 35 sites (595 pairs)
Test:  4 sites (6 pairs)

CV fold fit time: 7s (0.1min)

--- CV-FOLD DIAGNOSTICS ---
Divergences: 480/2000 (24.0%)

beta posteriors (n=12):
  (all beta coefficients again in range 10^21 to 10^22)

beta_0: mean=0.1103  std=0.5403
beta_sigma: [ 6.59033768 -0.96028528  5.56130522  0.61960893]

R-hat range: [8.9925, 152854748.8331]
ESS bulk range: [4, 4]
ESS tail range: [4, 35]
--- END CV-FOLD DIAGNOSTICS ---

CV predictions: min=1.0000, max=1.0000, mean=1.0000
RMSE (fold 1): 0.1215
MAE  (fold 1): 0.0894
CRPS (fold 1): 0.0894
WARNING: CV predictions look degenerate!

SUMMARY — config 1 (none_covariate_dependent)
Total time: 19s (0.3min)
Degenerate: YES — FAILED
```

**Key observations:**
- Fit time is only 12s — sampler is not exploring, just stuck
- ALL beta values are ~10^21 (complete explosion)
- beta_sigma values are moderate ([3.2, 0.5, 6.6, -0.5]) — the funnel is working: moderate beta_sigma → large enough variance → beta unconstrained
- ESS = 4 across the board (each chain is a single stuck point)
- R-hat up to 152 million (chains in completely different regions)
- Predictions all exactly 1.0

### Config 2: Polynomial (Works, With Caveats)

Job 24213489, task 2. Same settings.

```
=== Panama TEST — config_idx=2  tune=4000  draws=500  nutpie+initvals ===
Start: Sat 21 Mar 2026 15:03:34 CET
X shape: (39, 5), y shape: (741,)
y range: [0.3447, 1.0000], y==1.0: 3

============================================================
CONFIG 2: none_polynomial
tune=4000, draws=500, chains=4, target_accept=0.97
beta prior: LogNormal(mu=0, sigma=10)
============================================================

============================================================
FULL-DATA FIT
============================================================

Full-data fit time: 179s (3.0min)

--- FULL-DATA DIAGNOSTICS ---
Divergences: 3/2000 (0.1%)

beta posteriors (n=12):
  beta[0]: mean=430364635578.4594  median=0.0002
  beta[1]: mean=204959696619.2997  median=0.0230
  ...
  (means ~10^11 but medians near 0 — bimodal)

beta_0: mean=-9577428.6405  std=16588592.0676
beta_sigma: [ -4.1491504  -16.32809964 -12.13781451   4.59804053]

R-hat range: [1.5253, 1.5485]
ESS bulk range: [7, 8]
ESS tail range: [4, 1163]
--- END FULL-DATA DIAGNOSTICS ---

Full-data predictions: min=0.6924, max=0.9941, mean=0.8881
RMSE (train): 0.1005
MAE  (train): 0.0727

============================================================
1-FOLD CV
============================================================
Divergences: 395/2000 (19.8%)
(beta posteriors similar pattern — means ~10^11, medians small)

R-hat range: [1.6292, 2.9394]
ESS bulk range: [5, 7]

CV predictions: min=0.7807, max=0.8855, mean=0.8369
RMSE (fold 1): 0.0922
CV fold CRPS: 0.0444

SUMMARY — config 2 (none_polynomial)
Total time: 340s (5.7min)
Degenerate: No — looks good
```

**Key observations:**
- Fit time 179s (vs 12s for covariate_dependent) — sampler is actually exploring
- Only 0.1% divergences full-data (3/2000)
- Beta posteriors are bimodal: medians near 0 (good chains) but means ~10^11 (exploded chains)
- R-hat 1.5-2.9, ESS 5-8 — poor but not catastrophic
- Predictions are sensible: 0.69-0.99 range
- beta_sigma values: `[-4.1, -16.3, -12.1, 4.6]` — larger magnitude than covariate_dependent's, but the model is working because mu (the polynomial input) is well-bounded

---

## 9. Constraints

These constraints are non-negotiable for reproducing White et al. (2024) results:

1. **Priors MUST match White et al.:**
   - `LogNormal(mu=0, sigma=10)` for beta coefficients
   - `Normal(mu=0, sigma=10)` for beta_sigma coefficients
   - `InverseGamma(alpha=1, beta=1)` for sigma2 (homogeneous)
   - **Never change priors to work around sampler issues**

2. **Must use nutpie sampler** (via `pm.sample(nuts_sampler="nutpie")`)

3. **Must reproduce White et al. Table 1 results within ~5-10%** on all 4 datasets

4. **10-fold site-level CV** for all datasets

5. **`alpha_importance=False`** for White et al. comparison (White et al. uses LogNormal priors, no Dirichlet/alpha structure)

---

## 10. Source Code

### 10.1 Variance Functions (`src/gdmbayes/models/_variance.py`)

Complete file:

```python
"""Built-in variance functions for spGDMM.

Each function is called inside a PyMC model context and returns a PyTensor
expression for the variance ``sigma2``.  Pass the string name or any callable
with the same signature to :class:`~gdmbayes.ModelConfig`:

.. code-block:: python

    from gdmbayes import ModelConfig, variance_polynomial

    # Built-in by name
    cfg = ModelConfig(variance="polynomial")

    # Built-in by reference
    cfg = ModelConfig(variance=variance_polynomial)

    # Custom callable
    import pymc as pm
    def my_variance(mu, X_sigma):
        beta_s = pm.HalfNormal("beta_s", sigma=1)
        return beta_s * pm.math.exp(mu)

    cfg = ModelConfig(variance=my_variance)

**Signature**::

    fn(mu, X_sigma) -> sigma2

- ``mu``: PyTensor vector — the linear predictor (mean) for each site pair.
- ``X_sigma``: np.ndarray of shape ``(n_pairs, k)`` or ``None`` — auxiliary
  covariates for the variance model (currently pairwise geographic distance).
- Returns a PyTensor scalar or vector representing ``sigma2``.
"""

from typing import Callable, Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def variance_homogeneous(mu, X_sigma):
    """Constant (homoscedastic) variance.

    Places an InverseGamma(1, 1) prior on a single shared ``sigma2``.
    """
    return pm.InverseGamma("sigma2", alpha=1, beta=1)


def variance_covariate_dependent(mu, X_sigma):
    """Variance as an exponential linear function of ``X_sigma``.

    If ``X_sigma`` is provided, fits ``sigma2 = exp(X_sigma @ beta_sigma)``
    with Normal(0, 10) priors on ``beta_sigma``.  ``X_sigma`` may be a numpy
    array or a PyMC ``pm.Data`` variable (shape ``(n_pairs, k)``).  The number
    of columns ``k`` is read from the static shape when ``X_sigma`` is a
    symbolic tensor, falling back to 1 (the default: pairwise geographic distance).
    The linear predictor is clipped to [-20, 20] before exp() to prevent
    overflow during nutpie initialization.
    Falls back to :func:`variance_homogeneous` when ``X_sigma`` is None.

    Uses a non-centered parameterization (``beta_sigma_raw ~ Normal(0, 1)``,
    ``beta_sigma = 10 * beta_sigma_raw``) to mitigate Neal's funnel geometry
    that arises when ``beta_sigma`` and ``beta`` interact through the likelihood.
    The implied prior on ``beta_sigma`` is unchanged: Normal(0, 10).
    """
    if X_sigma is not None:
        # Support both numpy arrays (shape known directly) and pm.Data/symbolic
        # tensors (static shape may be unknown, so evaluate to get the concrete shape).
        if isinstance(X_sigma, np.ndarray):
            n_cols = X_sigma.shape[1]
        else:
            n_cols = int(X_sigma.shape.eval()[1])
        # Non-centered parameterization: sample in unit-scale space to avoid
        # Neal's funnel between beta_sigma and beta.
        beta_sigma_raw = pm.Normal("beta_sigma_raw", mu=0, sigma=1, shape=n_cols)
        beta_sigma = pm.Deterministic("beta_sigma", 10 * beta_sigma_raw)
        return pm.math.exp(pt.clip(pm.math.dot(X_sigma, beta_sigma), -20, 20))
    return pm.InverseGamma("sigma2", alpha=1, beta=1)


def variance_polynomial(mu, X_sigma):
    """Variance as a cubic polynomial function of the mean ``mu``.

    Fits ``sigma2 = exp(b0 + b1*mu + b2*mu^2 + b3*mu^3)`` with
    Normal(0, 10) priors on all four coefficients.  The polynomial is clipped
    to [-20, 20] before exp() to prevent overflow during nutpie initialization.

    Uses a non-centered parameterization (``beta_sigma_raw ~ Normal(0, 1)``,
    ``beta_sigma = 10 * beta_sigma_raw``) to mitigate Neal's funnel geometry.
    The implied prior on ``beta_sigma`` is unchanged: Normal(0, 10).
    """
    # Non-centered parameterization: sample in unit-scale space.
    beta_sigma_raw = pm.Normal("beta_sigma_raw", mu=0, sigma=1, shape=4)
    beta_sigma = pm.Deterministic("beta_sigma", 10 * beta_sigma_raw)
    poly = (
        beta_sigma[0] + beta_sigma[1] * mu +
        beta_sigma[2] * mu ** 2 +
        beta_sigma[3] * mu ** 3
    )
    return pm.math.exp(pt.clip(poly, -20, 20))


VARIANCE_FUNCTIONS: Dict[str, Callable] = {
    "homogeneous": variance_homogeneous,
    "covariate_dependent": variance_covariate_dependent,
    "polynomial": variance_polynomial,
}
```

### 10.2 `_compute_initvals()` (`src/gdmbayes/models/_spgdmm.py`, lines 520-578)

```python
def _compute_initvals(self):
    """Compute initial values matching White et al. (2024) NIMBLE code.

    Runs BFGS on ``sum((log_y - beta_0 - X @ exp(log_beta))^2)`` to get
    reasonable starting values for ``beta_0`` and ``beta``.  For
    ``beta_sigma``, uses White et al.'s hardcoded ``[-5, -20, 12, 2]``
    (polynomial) or ``[0]`` (covariate_dependent).
    """
    from scipy.optimize import minimize

    X_GDM = self.X_transformed
    log_y = self.y_transformed
    p = X_GDM.shape[1]

    # OLS on log_y ~ X_GDM for starting guess
    from numpy.linalg import lstsq
    A = np.column_stack([np.ones(len(log_y)), X_GDM])
    ols_coefs, _, _, _ = lstsq(A, log_y, rcond=None)
    x0_beta0 = 0.3
    x0_log_beta = np.array([
        np.log(c) if c > 0 else -10.0 for c in ols_coefs[1:]
    ])
    x0 = np.concatenate([[x0_beta0], x0_log_beta])

    def obj(par):
        b0 = par[0]
        log_b = par[1:]
        pred = b0 + X_GDM @ np.exp(log_b)
        return np.sum((log_y - pred) ** 2)

    res = minimize(obj, x0, method="BFGS")
    beta_0_init = float(res.x[0])
    log_beta_init = res.x[1:]

    initvals = {
        "beta_0": beta_0_init,
        "beta": np.exp(log_beta_init),
    }

    var_names = [v.name for v in self.model.free_RVs]
    if "beta_sigma_raw" in var_names:
        # Non-centered parameterization: beta_sigma = 10 * beta_sigma_raw,
        # so beta_sigma_raw = beta_sigma_init / 10.
        beta_sigma_raw_var = self.model["beta_sigma_raw"]
        n_sigma = beta_sigma_raw_var.type.shape[0] or 1
        if n_sigma == 4:
            # Polynomial variance: White et al. uses [-5, -20, 12, 2]
            initvals["beta_sigma_raw"] = np.array([-5.0, -20.0, 12.0, 2.0]) / 10.0
        else:
            # Covariate-dependent: start at 0 (sigma2 = exp(0) = 1)
            initvals["beta_sigma_raw"] = np.zeros(n_sigma)

    if "sigma2" in var_names:
        initvals["sigma2"] = 1.0

    if "sig2_psi" in var_names:
        initvals["sig2_psi"] = 1.0

    return initvals
```

### 10.3 `_sample_model()` (`src/gdmbayes/models/_spgdmm.py`, lines 580-628)

```python
def _sample_model(self, **kwargs) -> az.InferenceData:
    """Sample from the PyMC model.

    Uses White et al. (2024) BFGS-based initial values for ``beta_0``
    and ``beta``, and their hardcoded init for ``beta_sigma``.

    For nutpie, initvals are passed via ``nuts_sampler_kwargs["init_mean"]``
    in the unconstrained (transformed) parameter space, since nutpie silently
    ignores the standard ``initvals`` argument.
    """
    if self.model is None:
        raise RuntimeError(
            "The model hasn't been built yet, call .build_model() first or .fit() instead."
        )
    with self.model:
        sampler_args = {**self.sampler_config, **kwargs}

        # Compute White et al. initial values (constrained space)
        initvals = self._compute_initvals()

        is_nutpie = sampler_args.get("nuts_sampler", "pymc") == "nutpie"

        if is_nutpie:
            # nutpie ignores `initvals` — pass via nuts_sampler_kwargs["init_mean"]
            # in the unconstrained (transformed) parameter space.
            unconstrained = {}
            for rv in self.model.free_RVs:
                if rv.name not in initvals:
                    continue
                value_var = self.model.rvs_to_values[rv]
                transform = self.model.rvs_to_transforms.get(rv, None)
                val = np.asarray(initvals[rv.name], dtype=np.float64)
                if transform is not None:
                    unconstrained[value_var.name] = transform.forward(val).eval()
                else:
                    unconstrained[value_var.name] = val

            sampler_args.setdefault("nuts_sampler_kwargs", {})
            sampler_args["nuts_sampler_kwargs"]["init_mean"] = unconstrained
        else:
            sampler_args["initvals"] = initvals

        idata = pm.sample(**sampler_args)

        idata.extend(pm.sample_prior_predictive(), join="right")
        idata.extend(pm.sample_posterior_predictive(idata), join="right")

    idata = self._set_idata_attrs(idata)
    return idata
```

### 10.4 `build_model()` (`src/gdmbayes/models/_spgdmm.py`, lines 288-401)

```python
def build_model(
    self,
    X: pd.DataFrame | np.ndarray,
    log_y: pd.Series | np.ndarray,
    **kwargs,
) -> None:
    """Build the PyMC model."""
    if self.metadata is None:
        self._generate_and_preprocess_model_data(X, log_y)

    X_values = self.X_transformed
    log_y_values = self.y_transformed

    cfg = self.preprocessor._get_config()
    n_spline_bases = cfg.deg + cfg.knots

    self.model_coords = {
        "obs_pair": np.arange(X_values.shape[0]),
        "predictor": self.metadata.column_names,
        "site_train": np.arange(self.metadata.no_sites_train),
        "feature": self.metadata.predictors,
        "basis_function": np.arange(1, n_spline_bases + 1),
    }

    with pm.Model(coords=self.model_coords) as model:
        X_data = pm.Data("X_data", X_values, dims=("obs_pair", "predictor"))
        log_y_data = pm.Data("log_y_data", log_y_values, dims=("obs_pair",))

        beta_0 = pm.Normal("beta_0", mu=0, sigma=10)

        if self._config.alpha_importance:
            J = n_spline_bases
            F = len(self.metadata.predictors)

            if F > 0:
                beta = pm.Dirichlet(
                    "beta", a=np.ones(J),
                    shape=(F, J),
                    dims=("feature", "basis_function")
                )

                n_cols_env = self.metadata.no_cols_env
                n_cols_dist = self.metadata.no_cols_dist
                X_env = X_data[:, :n_cols_env]
                X_reshaped = X_env.reshape((-1, F, J))
                warped = (X_reshaped * beta[None, :, :]).sum(axis=2)

                alpha = pm.HalfNormal("alpha", sigma=2, shape=F, dims=("feature",))
                mu = beta_0 + pm.math.dot(warped, alpha)

                if n_cols_dist > 0:
                    dist_cols = X_data[:, n_cols_env:]
                    beta_dist = pm.LogNormal("beta_dist", mu=0, sigma=10, shape=n_cols_dist)
                    mu = mu + pm.math.dot(dist_cols, beta_dist)
            else:
                mu = beta_0
        else:
            # White et al. comparison mode: flat LogNormal priors on all coefficients
            beta = pm.LogNormal("beta", mu=0, sigma=10, shape=self.metadata.no_cols)
            mu = beta_0 + pm.math.dot(X_data, beta)

        # --- Variance model ---
        variance_fn = (
            self._config.variance
            if callable(self._config.variance)
            else VARIANCE_FUNCTIONS[self._config.variance]
        )
        if self.metadata.X_sigma is not None:
            X_sigma_data = pm.Data("X_sigma_data", self.metadata.X_sigma)
        else:
            X_sigma_data = None
        sigma2 = variance_fn(mu, X_sigma_data)

        # --- Spatial random effect ---
        if self._config.spatial_effect != "none":
            sig2_psi = pm.InverseGamma("sig2_psi", alpha=1, beta=1)
            location_values = self.training_metadata.location_values_train
            length_scale = self.training_metadata.length_scale

            cov = sig2_psi * pm.gp.cov.Exponential(2, ls=length_scale)
            gp = pm.gp.Latent(cov_func=cov)
            psi = gp.prior("psi", X=location_values, dims=("site_train",))

            row_ind, col_ind = self._pair_indices
            row_indices = pm.Data("row_indices", row_ind.astype(np.int32))
            col_indices = pm.Data("col_indices", col_ind.astype(np.int32))

            spatial_fn = (
                self._config.spatial_effect
                if callable(self._config.spatial_effect)
                else SPATIAL_FUNCTIONS[self._config.spatial_effect]
            )
            mu += spatial_fn(psi, row_indices, col_indices)

        # --- Likelihood ---
        pm.Censored(
            "log_y",
            pm.Normal.dist(mu=mu, sigma=pm.math.sqrt(sigma2)),
            lower=None, upper=0,
            observed=log_y_data,
        )

    self.model = model
```

### 10.5 `_generate_and_preprocess_model_data()` — X_sigma construction (`src/gdmbayes/models/_spgdmm.py`, lines 179-287)

```python
def _generate_and_preprocess_model_data(
    self, X: pd.DataFrame, y: pd.Series
) -> None:
    """Preprocess model data before fitting."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    y_array = np.asarray(y, dtype=float)
    log_y = np.log(np.maximum(y_array, np.finfo(float).eps))

    self.X = X
    self.y = log_y

    n_sites = X.shape[0]
    all_row_ind, all_col_ind = np.triu_indices(n_sites, k=1)

    # Fit the preprocessor only if it hasn't been fitted yet
    if not hasattr(self.preprocessor, "n_predictors_"):
        self.preprocessor.fit(X)
    prep = self.preprocessor

    cfg = prep._get_config()
    n_spline_bases = prep.n_spline_bases_
    n_predictors = prep.n_predictors_
    predictor_names_from_data = prep.predictor_names_

    # Build pairwise I-spline diffs
    I_spline_bases = prep.I_spline_bases_
    if I_spline_bases.shape[1] > 0:
        I_spline_bases_diffs = np.array([
            pdist(I_spline_bases[:, i].reshape(-1, 1), metric="euclidean")
            for i in range(I_spline_bases.shape[1])
        ]).T
    else:
        I_spline_bases_diffs = np.empty((n_sites * (n_sites - 1) // 2, 0))

    # Compute distance splines using pw_distance on training locations
    pw_distance = prep.pw_distance(prep.location_values_train_)
    dist_mesh = prep.dist_mesh_
    pw_dist_clipped = np.clip(pw_distance, dist_mesh[0], dist_mesh[-1])
    from dms_variants.ispline import Isplines
    dist_spline_bases = np.column_stack([
        Isplines(cfg.deg, dist_mesh, pw_dist_clipped).I(j)
        for j in range(1, n_spline_bases + 1)
    ])

    # Combine features
    X_GDM = np.column_stack([I_spline_bases_diffs, dist_spline_bases])

    X_GDM_fit = X_GDM
    log_y_fit = log_y
    pw_dist_fit = pw_dist_clipped
    self._pair_indices = (all_row_ind, all_col_ind)

    n_cols_env = I_spline_bases_diffs.shape[1] if I_spline_bases_diffs.size > 0 else 0
    n_cols_dist = dist_spline_bases.shape[1] if dist_spline_bases.size > 0 else 0

    # Standardise pairwise distances before polynomial expansion
    d_mean = float(pw_dist_fit.mean())
    d_std = float(pw_dist_fit.std()) + float(np.finfo(float).eps)
    d_z = (pw_dist_fit - d_mean) / d_std

    self.metadata = ModelMetadata(
        no_sites_train=n_sites,
        no_predictors=n_predictors,
        no_rows=X_GDM_fit.shape[0],
        no_cols=X_GDM_fit.shape[1],
        no_cols_env=n_cols_env,
        no_cols_dist=n_cols_dist,
        predictors=predictor_names_from_data if n_predictors > 0 else [],
        column_names=[f"x_{i}" for i in range(X_GDM_fit.shape[1])],
        # X_sigma: 4-column polynomial of standardized pairwise distance
        X_sigma=np.column_stack([
            np.ones_like(d_z),
            d_z,
            d_z ** 2,
            d_z ** 3,
        ]) if n_predictors > 0 else None,
        p_sigma=1 if n_predictors > 0 else 0,
        d_mean=d_mean,
        d_std=d_std,
    )

    self.training_metadata = TrainingMetadata(
        location_values_train=prep.location_values_train_,
        predictor_mesh=prep.predictor_mesh_,
        dist_mesh=prep.dist_mesh_,
        length_scale=prep.length_scale_,
        I_spline_bases=prep.I_spline_bases_,
    )

    self.n_features_in_ = n_predictors

    self.X_transformed = X_GDM_fit
    self.y_transformed = log_y_fit
```
