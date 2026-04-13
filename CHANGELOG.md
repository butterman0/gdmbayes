# Changelog

All notable changes to gdmbayes are documented here.

## [Unreleased]

### Added
- `TestGDMModel` test class covering `GDMResult` fields, coefficient extraction,
  `predict()` range, and round-trip serialization via `to_dict` / `from_dict`.
- Three new plots that bring the package to parity-plus-Bayes with R
  `gdm::plot.gdm` and `gdm::gdm.varImp`:
  - `plot_obs_vs_pred(model, X, y)` — observed vs. posterior-mean
    predicted dissimilarity with 1:1 line and Pearson r. Analogue of R
    `gdm::plot.gdm` panel 1.
  - `plot_link_curve(model, X, y)` — fitted link curve ``y = exp(η)``
    with observed ``(η̂, y)`` overlaid and the censoring bound marked.
    Analogue of R `gdm::plot.gdm` panel 2 adapted to spGDMM's
    log-normal-with-right-censoring parameterisation.
  - `plot_predictor_importance(model)` — horizontal bar chart of
    ``sum_j β[d, k, j]`` per predictor with HDI error bars. Bayesian
    analogue of R `gdm::gdm.varImp`.

### Changed
- **Plotting module restructure.** `src/gdmbayes/plotting/plots.py` split into
  `isplines.py`, `ppc.py`, and `scoring.py` — one concept per file.
  `summarise_sampling` moved to new top-level `diagnostics.py` (it is not a
  plot). `rgb_from_biological_space` / `rgb_biological_space` moved to new
  top-level `maps.py`.
- `plot_isplines` now returns `list[(fig, ax)]` instead of calling
  `plt.show()` per figure, so plots compose in notebooks, CI, and headless
  runs. Default rendering matches R `gdm::plot.gdm` (summed curve + HDI
  band); pass `decompose=True` to overlay per-basis contributions.
- Renamed `plot_crps_comparison` → `crps_boxplot` and removed the broken
  `use_log` parameter (it logged `y_test`/`y_train` without logging
  `y_pred`, producing mixed-scale CRPS). Callers who want log-scale scores
  should pass logged inputs directly.

### Fixed
- `rgb_biological_space`: removed a redundant second `pca.transform` pass
  over the same rows — the PCA is now fit once on the valid samples and
  its outputs placed back into the grid.
- Beta coefficient indexing in `GDMModel.fit()` for `alpha_importance=True`:
  when `beta` is 2-D (Dirichlet prior shape `(F, J)`), coefficients are now
  extracted as `beta_median[i, :]` per predictor rather than a 1-D slice.

- README rewritten to reflect the `gdmbayes` package name and current API.
  All references to the old `spgdmm` import path, `format_site_pair`,
  `BioFormat`, and `ModelVariant` have been removed.
- `examples/basic_usage.py` rewritten to demonstrate `GDM` and `spGDMM`
  using the current API.
- `examples/model_variants.py` import fixed from `spgdmm` → `gdmbayes`.

---

## [1.0.0] — 2025

Initial public release as **gdmbayes** (formerly spGDMM).

### Added
- `GDM` — frequentist sklearn-compatible GDM estimator implementing the
  R GDM algorithm (I-splines + NNLS). Attributes: `coef_`,
  `predictor_importance_`, `explained_`, `knots_`, `null_deviance_`,
  `model_deviance_`.
- `spGDMM` — Bayesian GDM estimator built on PyMC. Full posterior inference
  via NUTS (nutpie or PyMC sampler). Sklearn-style `fit()` / `predict()` /
  `save()` / `load()`.
- `GDMModel` — Bayesian GDM wrapper returning `GDMResult` dataclass with
  R-compatible attributes: `gdmdeviance`, `nulldeviance`, `explained`,
  `intercept`, `predictors`, `coefficients`, `knots`, `splines`.
- `GDMPreprocessor` — standalone sklearn transformer owning all
  data-transformation logic: I-spline mesh construction, geographic distance
  computation, pairwise feature matrix assembly. Can be used independently or
  composed into pipelines.
- `PreprocessorConfig` — dataclass for I-spline and distance settings (`deg`,
  `knots`, `mesh_choice`, `distance_measure`, `extrapolation`,
  `custom_dist_mesh`, `custom_predictor_mesh`). Separated from `ModelConfig`
  in this release.
- `ModelConfig` — dataclass for Bayesian model structure only (`variance`,
  `spatial_effect`, `alpha_importance`, custom callables). Legacy preprocessing
  keys passed to `ModelConfig.from_dict()` trigger a `DeprecationWarning`.
- `SamplerConfig` — dataclass for MCMC settings.
- Built-in variance functions: `"homogeneous"`, `"covariate_dependent"`,
  `"polynomial"` (custom callable supported).
- Built-in spatial effect functions: `"abs_diff"`, `"squared_diff"` (custom
  callable supported).
- Distance utilities: `euclidean_distance`, `geodesic_distance`,
  `DistanceCalculator`, `compute_distance_matrix`.
- Plotting utilities: `plot_isplines`, `plot_crps_comparison`,
  `summarise_sampling`, `plot_ppc`, `rgb_from_biological_space`.
- Convenience functions: `gdm()`, `gdm_transform()`, `ispline_extract()`,
  `rgb_biological_space()`.
- `test_distances.py` — 21 tests for distance utilities.
- Full sklearn interface: `clone()`, `get_params()`, `set_params()`,
  `check_is_fitted()`, `__sklearn_is_fitted__`.
- Save/load round-trip via ArviZ `InferenceData` (NetCDF); preprocessor state
  serialised via `GDMPreprocessor.to_xarray()` / `from_xarray()`.

### Removed
- `format_site_pair()` and `BioFormat` — site-pair table construction removed;
  users should supply condensed pairwise `y` directly.
- `ModelVariant` enum and `spGDMM.from_variant()` — replaced by direct
  `ModelConfig(variance=..., spatial_effect=...)` construction.
- Preprocessing fields (`deg`, `knots`, `mesh_choice`, `distance_measure`,
  `extrapolation`) removed from `ModelConfig`; they now live exclusively in
  `PreprocessorConfig`.
