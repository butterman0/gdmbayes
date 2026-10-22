# Changelog

All notable changes to gdmbayes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-10-22

Initial public release.

- `GDM`: frequentist Generalised Dissimilarity Model (I-splines + non-negative
  least squares) with a scikit-learn `fit` / `predict` interface, matching the
  R `gdm` algorithm.
- `spGDMM`: Bayesian GDM in PyMC with NUTS (nutpie) sampling, three variance
  structures, Gaussian-process spatial random effects, posterior prediction for
  unseen sites via the GP conditional, CRPS scoring, and NetCDF save/load.
- `GDMPreprocessor`: scikit-learn transformer for I-spline meshes and
  Euclidean, geodesic or custom (callable) pairwise distances; custom callables
  are restored by import path when a saved model is loaded.
- Plotting: I-spline response curves, predictor importance, observed-vs-predicted,
  link curve, posterior predictive checks, CRPS skill boxplots, RGB
  biological-space maps.
- Three example notebooks, including a full worked analysis of the SW Australia
  dataset with cross-validated comparison against White et al. (2024).
- Requires Python ≥ 3.12, PyMC ≥ 6, ArviZ ≥ 1.1.
