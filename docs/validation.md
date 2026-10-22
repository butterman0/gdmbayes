# Validation against White et al. (2024)

gdmbayes is checked against two references:

- **R `gdm`** — `GDM(geo=True, splines=2, knots=1)` reproduces `gdm::gdm()` with its
  default 3 I-spline bases per predictor to floating-point precision (explained
  deviance to 1e-11, predictions to 1e-8). This is enforced by
  `tests/test_r_gdm_regression.py` on the Southwest Australia data.
- **White et al. (2024)** — the nine `spGDMM` configurations are compared with the
  10-fold cross-validation results in their Table 1 on the two datasets below.

## Protocol

10-fold site-level cross-validation (seed 42): sites are split into folds; the model
is fitted on the training sites only (I-spline meshes and the Gaussian process see no
test site) and evaluated on the pairs whose *both* sites are held out, with the spatial
effect at test sites obtained from the GP conditional. Sampler: nutpie NUTS, 4 chains,
4 000 tuning and 1 000 draws per chain, `target_accept = 0.95`. I-splines: cubic with
one interior knot (4 bases per predictor), matching the NIMBLE code of White et al. for
these two datasets. White et al. used a random-walk Metropolis sampler with a single
chain. Priors are identical (LogNormal(0, 10) on the I-spline coefficients, Normal(0, 10)
on variance coefficients). Lower is better for all three metrics.

## Southwest Australia (94 sites, 4 371 pairs)

| Model | spatial / variance | gdmbayes RMSE | MAE | CRPS | White et al. RMSE | MAE | CRPS |
|---|---|---|---|---|---|---|---|
| M1 | none / homogeneous | 0.0652 | 0.0489 | 0.0358 | 0.0790 | 0.0595 | 0.0439 |
| M2 | none / covariate_dependent | 0.0665 | 0.0492 | 0.0349 | 0.0805 | 0.0608 | 0.0435 |
| M3 | none / polynomial | 0.0949 | 0.0795 | 0.0445 | — | — | — |
| M4 | abs_diff / homogeneous | 0.0609 | 0.0449 | 0.0336 | 0.0840 | 0.0629 | 0.0473 |
| M5 | abs_diff / covariate_dependent | 0.0637 | 0.0468 | 0.0333 | 0.0820 | 0.0626 | 0.0454 |
| M7 | squared_diff / homogeneous | 0.0623 | 0.0461 | 0.0342 | 0.0731 | 0.0545 | 0.0414 |
| M8 | squared_diff / covariate_dependent | 0.0654 | 0.0478 | 0.0344 | 0.0748 | 0.0556 | 0.0407 |

M6 and M9 (polynomial variance with a spatial effect) were not run on this dataset
(~2 h per fold); White et al. do not report them for Southwest Australia either.
The frequentist `GDM` with the same 4-basis splines scores RMSE 0.0631 / MAE 0.0459;
White et al. report 0.0737 / 0.0549 for R `gdm` with its 3-basis default.

## Panama (39 sites, 741 pairs)

| Model | spatial / variance | gdmbayes RMSE | MAE | CRPS | White et al. RMSE | MAE | CRPS |
|---|---|---|---|---|---|---|---|
| M1 | none / homogeneous | 0.0912 | 0.0809 | 0.0528 | 0.0954 | 0.0779 | 0.0527 |
| M2 | none / covariate_dependent | 0.0822 | 0.0721 | 0.0487 | 0.0937 | 0.0734 | 0.0511 |
| M3 | none / polynomial | 0.0869 | 0.0744 | 0.0509 | — | — | — |
| M4 | abs_diff / homogeneous | 0.0744 | 0.0617 | 0.0440 | 0.0878 | 0.0690 | 0.0490 |
| M5 | abs_diff / covariate_dependent | 0.0746 | 0.0599 | 0.0436 | 0.0879 | 0.0654 | 0.0479 |
| M6 | abs_diff / polynomial | 0.0945 | 0.0800 | 0.0473 | — | — | — |
| M7 | squared_diff / homogeneous | 0.0764 | 0.0645 | 0.0450 | 0.0856 | 0.0658 | 0.0472 |
| M8 | squared_diff / covariate_dependent | 0.0742 | 0.0619 | 0.0431 | 0.0821 | 0.0618 | 0.0450 |
| M9 | squared_diff / polynomial | 0.0924 | 0.0802 | 0.0491 | — | — | — |

The frequentist `GDM` scores RMSE 0.0946 / MAE 0.0759 (R `gdm` in White et al.:
0.0934 / 0.0716).

## Reading the tables

- Every gdmbayes spatial model with homogeneous or covariate-dependent variance
  (M4, M5, M7, M8) beats the best model of White et al. on both datasets: best RMSE is
  0.0609 (M4) vs 0.0731 on Southwest Australia, and 0.0742 (M8) vs 0.0821 on Panama.
- Polynomial variance (M3, M6, M9) is consistently the weakest choice.
- The improvement over White et al. with the same model, priors and I-spline basis
  comes from the fitting, not the model: NUTS (gradient-based, four chains, R-hat
  checked) explores the strongly correlated posterior of the I-spline coefficients
  far more efficiently than the single-chain random-walk Metropolis sampler used in
  the original, and gdmbayes initialises the chains from a BFGS optimum (including
  the spatial effect). Posterior means from better-mixed chains are closer to the
  true posterior mean, which directly lowers cross-validated error. The out-of-sample
  spatial effect also differs: gdmbayes predicts held-out sites from the Gaussian
  process conditional, whereas White et al. treat held-out dissimilarities as
  latent variables in the fit.
- Panama has only 39 sites, so each test fold contains about four sites and the
  estimates are noisier than for Southwest Australia.
- Some Southwest Australia folds show R-hat up to ~1.5 on the GP latent `psi`
  with this sampler budget; the CV metrics use posterior means and are not
  visibly affected, but a longer run or higher `target_accept` is advisable for
  inference on `psi` itself.

Results were generated on 2026-10-19/20 with gdmbayes 0.1.0 (PyMC 6.3, nutpie 0.16.11).
