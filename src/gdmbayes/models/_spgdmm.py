"""
Spatial Generalized Dissimilarity Mixed Model (spGDMM).

This module implements the spGDMM class for modeling pairwise ecological
dissimilarities as a function of environmental predictors and spatial distance.
"""

import hashlib
import json
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ..preprocessing._config import PreprocessorConfig
from ..preprocessing._preprocessor import GDMPreprocessor
from ._config import ModelConfig, SamplerConfig
from ._spatial import SPATIAL_FUNCTIONS
from ._variance import VARIANCE_FUNCTIONS, poly_fit, poly_predict


class spGDMM(BaseEstimator):
    """
    Spatial Generalized Dissimilarity Mixed Model.

    spGDMM models pairwise ecological dissimilarities as a function of
    environmental and spatial distance using I-spline basis functions.

    Parameters
    ----------
    preprocessor : GDMPreprocessor, PreprocessorConfig, or None, optional
        Data preprocessing configuration or a pre-built preprocessor.
        If None, a default ``GDMPreprocessor(PreprocessorConfig())`` is used.
        If a ``PreprocessorConfig`` is passed, it is wrapped in a ``GDMPreprocessor``.
    model_config : dict or ModelConfig, optional
        Model structure configuration (variance type, spatial effects, etc.).
        Defaults to ``ModelConfig()``.

        .. deprecated::
            Passing preprocessing fields (``deg``, ``knots``, ``mesh_choice``,
            ``distance_measure``, ``extrapolation``, etc.) via ``model_config``
            is deprecated. Pass them via ``preprocessor`` instead.
    sampler_config : dict or SamplerConfig, optional
        MCMC sampler configuration. Defaults to PyMC/nutpie defaults.

    Examples
    --------
    >>> from gdmbayes import spGDMM, ModelConfig, PreprocessorConfig
    >>> prep_cfg = PreprocessorConfig(deg=3, knots=2, distance_measure="euclidean")
    >>> model_cfg = ModelConfig(variance="homogeneous", spatial_effect="abs_diff")
    >>> model = spGDMM(preprocessor=prep_cfg, model_config=model_cfg)
    >>> idata = model.fit(X, y)
    """

    _model_type = "spGDMM"
    version = "1.0.0"

    def __init__(
        self,
        preprocessor: "GDMPreprocessor | PreprocessorConfig | None" = None,
        model_config: "dict | ModelConfig | None" = None,
        sampler_config: "dict | SamplerConfig | None" = None,
    ):
        # ------------------------------------------------------------------ #
        # Resolve preprocessor
        # ------------------------------------------------------------------ #
        if isinstance(preprocessor, GDMPreprocessor):
            self.preprocessor = preprocessor
        elif isinstance(preprocessor, PreprocessorConfig):
            self.preprocessor = GDMPreprocessor(config=preprocessor)
        elif preprocessor is None:
            self.preprocessor = GDMPreprocessor(config=PreprocessorConfig())
        else:
            raise TypeError(
                f"preprocessor must be a GDMPreprocessor, PreprocessorConfig, or None; "
                f"got {type(preprocessor)!r}"
            )

        if isinstance(model_config, dict):
            self._config = ModelConfig.from_dict(model_config)
            # Store the exact dict reference so sklearn.clone() identity check passes.
            self.model_config = model_config
        elif isinstance(model_config, ModelConfig):
            self._config = model_config
            self.model_config = self._config.to_dict()
        else:
            self._config = ModelConfig()
            self.model_config = self._config.to_dict()

        # ------------------------------------------------------------------ #
        # Sampler config
        # ------------------------------------------------------------------ #
        if isinstance(sampler_config, SamplerConfig):
            sampler_config = sampler_config.to_dict()
        elif sampler_config is None:
            sampler_config = self.get_default_sampler_config()

        self.sampler_config = sampler_config

        self.model = None
        self.idata: az.InferenceData | None = None
        self._config_dict = self._config.to_dict()
        # Raw inputs (set by _generate_and_preprocess_model_data)
        self.X: pd.DataFrame | None = None  # site-level input
        self.y: np.ndarray | None = None    # raw pairwise dissimilarities
        # Transformed for model (set by _generate_and_preprocess_model_data)
        self.X_GDM: np.ndarray | None = None  # pairwise GDM feature matrix
        self.log_y: np.ndarray | None = None        # log-transformed dissimilarities
        # Variance model state (covariate-dependent only)
        self._X_sigma: np.ndarray | None = None
        self._poly_alpha: np.ndarray | None = None
        self._poly_norm2: np.ndarray | None = None
        # Holdout mask for masked-holdout CV (White et al. 2024 strategy)
        self._holdout_mask: np.ndarray | None = None

    def __sklearn_is_fitted__(self) -> bool:
        """Return True only after a successful fit (idata with posterior exists)."""
        return self.idata is not None and "posterior" in self.idata

    @property
    def config(self) -> dict:
        """Get model configuration as dictionary (legacy interface)."""
        return self._config_dict

    def _generate_and_preprocess_model_data(
        self, X: pd.DataFrame, y: pd.Series, holdout_mask: np.ndarray | None = None,
    ) -> None:
        """
        Preprocess model data before fitting.

        Delegates to ``self.preprocessor.fit()`` to compute spline meshes
        (unless the preprocessor is already fitted), then builds the
        pairwise feature matrix and variance model state.

        Parameters
        ----------
        X : pd.DataFrame
            Site-level data with columns [xc, yc, time_idx, predictor1, ...]
        y : pd.Series
            Pairwise Bray-Curtis dissimilarities (length n_sites*(n_sites-1)//2).
        holdout_mask : np.ndarray of bool or None, optional
            Boolean mask (length n_pairs, True = held-out pair).  When set,
            the model is built on ALL sites but the likelihood is split:
            observed pairs get the Censored likelihood, held-out pairs become
            free Normal RVs.  This replicates the White et al. (2024) CV
            strategy where latent variables are sampled for masked pairs.
        """
        self._holdout_mask = holdout_mask
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        y_array = np.asarray(y, dtype=float)

        self.X = X
        self.y = y_array
        self.log_y = np.log(np.maximum(y_array, np.finfo(float).eps))

        # Fit the preprocessor only if it hasn't been fitted yet (e.g., on load path).
        try:
            check_is_fitted(self.preprocessor)
        except NotFittedError:
            self.preprocessor.fit(X)
        prep = self.preprocessor

        # Delegate pairwise I-spline diffs + distance splines to preprocessor.
        self.X_GDM = prep.transform(X)
        self.n_features_in_ = prep.n_predictors_

        # Build orthogonal polynomial basis for covariate-dependent variance.
        # Matches White et al.'s R code: X_sigma = cbind(1, poly(vec_distance, 3))
        # Only computed when variance="covariate_dependent".
        variance_type = (
            self._config.variance if isinstance(self._config.variance, str) else "custom"
        )
        if variance_type == "covariate_dependent":
            pw_distance = prep.pw_distance(prep.location_values_train_)
            pw_dist_fit = np.clip(pw_distance, prep.dist_mesh_[0], prep.dist_mesh_[-1])
            poly_cols, alpha, norm2 = poly_fit(pw_dist_fit, degree=3)
            self._poly_alpha = alpha
            self._poly_norm2 = norm2
            self._X_sigma = np.column_stack([np.ones(len(pw_dist_fit)), poly_cols])
        else:
            self._X_sigma = None
            self._poly_alpha = None
            self._poly_norm2 = None

    def _build_sigma_basis(self, pw_distance: np.ndarray) -> np.ndarray:
        """Build orthogonal polynomial basis for covariate-dependent variance.

        Applies the same orthogonal polynomial transform (matching R's
        ``poly()``) computed at training time to a new vector of pairwise
        distances, then prepends a column of ones (matching White et al.'s
        ``cbind(1, poly(vec_distance, 3))``).
        """
        dist_mesh = self.preprocessor.dist_mesh_
        pw_dist = np.clip(pw_distance, dist_mesh[0], dist_mesh[-1])
        poly_cols = poly_predict(pw_dist, self._poly_alpha, self._poly_norm2)
        return np.column_stack([np.ones(len(pw_dist)), poly_cols])

    def build_model(
        self,
        X: "pd.DataFrame | None" = None,
        y: "pd.Series | np.ndarray | None" = None,
        holdout_mask: "np.ndarray | None" = None,
    ) -> None:
        """Preprocess data (if provided) and build the PyMC model.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Site-level data. If provided together with ``y``, preprocessing
            is run first via ``_generate_and_preprocess_model_data``.
            If omitted, assumes preprocessing was already done (e.g. on
            the load path).
        y : array-like, optional
            Pairwise dissimilarities. Required when ``X`` is provided.
        holdout_mask : np.ndarray of bool, optional
            Boolean mask for masked-holdout CV (True = held-out pair).
        """
        if X is not None:
            if y is None:
                raise ValueError("y is required when X is provided.")
            y_array = np.asarray(y, dtype=float)
            self._generate_and_preprocess_model_data(X, y_array, holdout_mask=holdout_mask)
        elif self.X_GDM is None:
            raise ValueError("No preprocessed data. Pass X and y, or call fit().")

        X_values = self.X_GDM
        log_y_values = self.log_y

        prep = self.preprocessor
        n_spline_bases = prep.n_spline_bases_
        n_predictors = prep.n_predictors_
        predictor_names = prep.predictor_names_

        column_names = [
            f"{name}_s{j}" for name in predictor_names
            for j in range(1, n_spline_bases + 1)
        ] + [f"dist_s{j}" for j in range(1, n_spline_bases + 1)]

        n_sites_train = prep.location_values_train_.shape[0]

        self.model_coords = {
            "obs_pair": np.arange(X_values.shape[0]),
            "predictor": column_names,
            "site_train": np.arange(n_sites_train),
            "feature": predictor_names,
            "basis_function": np.arange(1, n_spline_bases + 1),
        }

        with pm.Model(coords=self.model_coords) as model:
            X_data = pm.Data("X_data", X_values, dims=("obs_pair", "predictor"))
            log_y_data = pm.Data("log_y_data", log_y_values, dims=("obs_pair",))

            beta_0 = pm.Normal("beta_0", mu=0, sigma=10)

            if self._config.alpha_importance:
                J = n_spline_bases
                F = n_predictors

                if F > 0:
                    beta = pm.Dirichlet(
                        "beta", a=np.ones(J),
                        shape=(F, J),
                        dims=("feature", "basis_function")
                    )

                    n_cols_env = n_predictors * n_spline_bases
                    n_cols_dist = n_spline_bases
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
                beta = pm.LogNormal(
                    "beta", mu=0, sigma=10,
                    shape=(n_predictors + 1) * n_spline_bases,
                )
                mu = beta_0 + pm.math.dot(X_data, beta)

            variance_fn = (
                self._config.variance
                if callable(self._config.variance)
                else VARIANCE_FUNCTIONS[self._config.variance]
            )
            if self._X_sigma is not None:
                X_sigma_data = pm.Data("X_sigma_data", self._X_sigma)
            else:
                X_sigma_data = None
            sigma2 = variance_fn(mu, X_sigma_data)

            if self._config.spatial_effect != "none":
                sig2_psi = pm.InverseGamma("sig2_psi", alpha=1, beta=1)
                location_values = self.preprocessor.location_values_train_
                length_scale = self.preprocessor.length_scale_

                # Match White et al.: cov(d) = sig2_psi * exp(-d / rho) where
                # rho = max(pw_distance) / 10 and d is in km.  PyMC's Exponential
                # kernel uses exp(-d / (2*ls)), so ls = rho / 2.  Coordinates must
                # be in the same units as pw_distance (km); for euclidean mode the
                # raw coordinates are in metres, so divide by 1000.
                gp_coords = location_values / 1000.0
                cov = sig2_psi * pm.gp.cov.Exponential(2, ls=length_scale / 2)
                gp = pm.gp.Latent(cov_func=cov)
                psi = gp.prior("psi", X=gp_coords, dims=("site_train",))

                # Use pm.Data so indices can be updated via pm.set_data during prediction
                # when X_pred has a different number of sites than X_train.
                row_ind, col_ind = np.triu_indices(n_sites_train, k=1)
                row_indices = pm.Data("row_indices", row_ind.astype(np.int32))
                col_indices = pm.Data("col_indices", col_ind.astype(np.int32))

                spatial_fn = (
                    self._config.spatial_effect
                    if callable(self._config.spatial_effect)
                    else SPATIAL_FUNCTIONS[self._config.spatial_effect]
                )
                mu += spatial_fn(psi, row_indices, col_indices)

            if self._holdout_mask is not None:
                obs_idx = np.where(~self._holdout_mask)[0]
                hold_idx = np.where(self._holdout_mask)[0]

                if sigma2.ndim > 0:
                    sigma_obs = pm.math.sqrt(sigma2[obs_idx])
                    sigma_hold = pm.math.sqrt(sigma2[hold_idx])
                else:
                    sigma_obs = sigma_hold = pm.math.sqrt(sigma2)

                pm.Censored(
                    "log_y",
                    pm.Normal.dist(mu=mu[obs_idx], sigma=sigma_obs),
                    lower=None, upper=0,
                    observed=log_y_values[obs_idx],
                )
                pm.Normal(
                    "log_y_holdout",
                    mu=mu[hold_idx], sigma=sigma_hold,
                    shape=len(hold_idx),
                )
            else:
                pm.Censored(
                    "log_y",
                    pm.Normal.dist(mu=mu, sigma=pm.math.sqrt(sigma2)),
                    lower=None, upper=0,
                    observed=log_y_data,
                )

        self.model = model

    def _data_setter(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> None:
        """Update mutable data containers in the existing model for prediction.

        For spatial models, also updates ``row_indices`` and ``col_indices`` to
        match the prediction sites so the spatial effect term has the correct shape.
        """
        if self.idata is None or "posterior" not in self.idata:
            raise ValueError("Model must be fitted before predicting.")
        X_transformed = self.preprocessor.transform(X)
        n_pred_pairs = X_transformed.shape[0]

        if y is None:
            log_y = np.zeros(n_pred_pairs)
        else:
            log_y = np.log(np.maximum(np.asarray(y, dtype=float), np.finfo(float).eps))

        set_data_dict: dict = {"X_data": X_transformed, "log_y_data": log_y}

        # For covariate_dependent variance: update X_sigma_data with pairwise distances
        # for the prediction sites so its shape matches X_data.
        if self._X_sigma is not None:
            pred_locations = (
                X.iloc[:, :2].values if isinstance(X, pd.DataFrame) else X[:, :2]
            )
            pw_dist_pred = self.preprocessor.pw_distance(pred_locations)
            set_data_dict["X_sigma_data"] = self._build_sigma_basis(pw_dist_pred)

        # For spatial models: recompute pair indices for the prediction sites so the
        # spatial term shape matches X_data.  n_pred_sites is recovered from n_pred_pairs
        # via the quadratic formula: n_pairs = n*(n-1)/2.
        if self._config.spatial_effect != "none":
            n_pred_sites = round((1 + (1 + 8 * n_pred_pairs) ** 0.5) / 2)
            pred_row_ind, pred_col_ind = np.triu_indices(n_pred_sites, k=1)
            set_data_dict["row_indices"] = pred_row_ind.astype(np.int32)
            set_data_dict["col_indices"] = pred_col_ind.astype(np.int32)

        with self.model:
            pm.set_data(
                set_data_dict,
                coords={"obs_pair": np.arange(n_pred_pairs)},
            )

    def _save_input_params(self, idata: az.InferenceData) -> None:
        """Persist transformation state (meshes, coordinates) in idata.constant_data.

        Delegates to ``self.preprocessor.to_xarray()`` so that models can be
        serialized and reloaded for prediction without refitting.
        """
        ds = self.preprocessor.to_xarray()
        idata.attrs["predictor_names"] = json.dumps(self.preprocessor.predictor_names_)
        if hasattr(idata, "constant_data"):
            # pm.sample already created constant_data for pm.Data variables — merge in place.
            merged = idata.constant_data.merge(ds)
            idata.constant_data = merged
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="The group constant_data is not defined",
                )
                idata.add_groups(constant_data=ds)

    # ------------------------------------------------------------------ #
    # Lifecycle methods (inlined from ModelBuilder, bugs fixed)
    # ------------------------------------------------------------------ #

    def _compute_initvals(self):
        """Compute initial values via multi-stage BFGS optimization.

        **Stage 1** (mean model, no spatial): BFGS on
        ``sum((log_y - beta_0 - X @ exp(log_beta))^2)`` for ``beta_0`` and
        ``beta``, assuming constant variance and no spatial effect.

        **Stage 1b** (mean model + spatial): If a spatial effect is configured,
        re-optimises jointly over ``[beta_0, log_beta, psi]`` including the
        spatial term.  This matches White et al.'s initialisation strategy and
        is critical for NUTS convergence on spatial models.

        **Stage 2** (variance model): Profile NLL optimization for ``beta_sigma``,
        holding the stage-1 mean parameters fixed.  Minimises the Gaussian
        negative log-likelihood with heterogeneous variance::

            NLL = 0.5 * sum(r^2 / sigma2 + log(sigma2))

        where ``r = log_y - mu_fixed`` and ``sigma2 = exp(linear_predictor)``.

        Both stages only apply when ``alpha_importance=False`` (flat LogNormal
        beta).  With ``alpha_importance=True``, beta is Dirichlet and the
        OLS/BFGS init is not applicable.
        """
        from scipy.optimize import minimize

        # When using holdout CV, compute initvals from observed pairs only
        if self._holdout_mask is not None:
            obs = ~self._holdout_mask
            X_GDM = self.X_GDM[obs]
            log_y = self.log_y[obs]
        else:
            X_GDM = self.X_GDM
            log_y = self.log_y
        p = X_GDM.shape[1]

        # Pair indices (for spatial init), filtered to observed pairs if holdout
        n_train = self.preprocessor.location_values_train_.shape[0]
        row_ind, col_ind = np.triu_indices(n_train, k=1)
        if self._holdout_mask is not None:
            row_ind = row_ind[~self._holdout_mask]
            col_ind = col_ind[~self._holdout_mask]

        # Resolve the spatial function once (works on both numpy and pytensor
        # since the registered functions use only generic ops like abs() and **).
        spatial_effect = self._config.spatial_effect
        if spatial_effect != "none":
            spatial_fn = (
                spatial_effect if callable(spatial_effect)
                else SPATIAL_FUNCTIONS[spatial_effect]
            )

        initvals = {}

        # Stage 1: mean model (beta_0, beta) via squared-error BFGS.
        # Only applies for flat LogNormal beta (alpha_importance=False).
        if not self._config.alpha_importance:
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
            initvals["beta_0"] = float(res.x[0])
            initvals["beta"] = np.maximum(np.exp(res.x[1:]), np.finfo(float).tiny)

            # Stage 1b: joint optimisation including psi (spatial effect).
            # White et al. initialises psi by optimising
            #   sum((log_y - b0 - X @ exp(log_b) - spatial(psi))^2)
            # jointly over [beta_0, log_beta, psi].
            if spatial_effect != "none":
                ns = n_train

                x0_psi = np.random.default_rng(42).normal(0, 0.1, size=ns)
                x0_joint = np.concatenate([res.x, x0_psi])

                def obj_spatial(par):
                    b0 = par[0]
                    log_b = par[1:p + 1]
                    psi = par[p + 1:]
                    pred = b0 + X_GDM @ np.exp(log_b)
                    pred = pred + spatial_fn(psi, row_ind, col_ind)
                    return np.sum((log_y - pred) ** 2)

                res_sp = minimize(obj_spatial, x0_joint, method="BFGS")
                initvals["beta_0"] = float(res_sp.x[0])
                initvals["beta"] = np.maximum(
                    np.exp(res_sp.x[1:p + 1]), np.finfo(float).tiny
                )
                initvals["psi"] = res_sp.x[p + 1:]

        # Stage 2: variance model (beta_sigma) via profile NLL.
        var_names = [v.name for v in self.model.free_RVs]
        if "beta_sigma_raw" in var_names and not self._config.alpha_importance:
            beta_sigma_raw_var = self.model["beta_sigma_raw"]
            n_sigma = beta_sigma_raw_var.type.shape[0] or 4

            mu_fixed = initvals["beta_0"] + X_GDM @ initvals["beta"]
            # Include spatial effect in mu_fixed if psi was initialized
            if "psi" in initvals:
                mu_fixed = mu_fixed + spatial_fn(
                    initvals["psi"], row_ind, col_ind
                )
            residuals = log_y - mu_fixed

            variance_type = (
                self._config.variance
                if isinstance(self._config.variance, str)
                else "custom"
            )

            def _profile_nll(beta_sigma, linear_pred_fn):
                """Gaussian NLL with heterogeneous variance."""
                lp = np.clip(linear_pred_fn(beta_sigma), -20, 20)
                return 0.5 * np.sum(residuals ** 2 * np.exp(-lp) + lp)

            beta_sigma_init = None

            if variance_type == "covariate_dependent" and self._X_sigma is not None:
                X_sig = self._X_sigma
                if self._holdout_mask is not None:
                    X_sig = X_sig[~self._holdout_mask]
                res_sigma = minimize(
                    _profile_nll, np.zeros(n_sigma), args=(lambda bs: X_sig @ bs,),
                    method="BFGS",
                )
                if res_sigma.success:
                    beta_sigma_init = res_sigma.x

            elif variance_type == "polynomial":
                def poly_linear(bs):
                    return bs[0] + bs[1] * mu_fixed + bs[2] * mu_fixed ** 2 + bs[3] * mu_fixed ** 3

                # Try from zeros first; fall back to White et al. values if it fails.
                res_sigma = minimize(
                    _profile_nll, np.zeros(n_sigma), args=(poly_linear,),
                    method="BFGS",
                )
                if res_sigma.success:
                    beta_sigma_init = res_sigma.x
                else:
                    res_sigma = minimize(
                        _profile_nll, np.array([-5.0, -20.0, 12.0, 2.0]),
                        args=(poly_linear,), method="BFGS",
                    )
                    if res_sigma.success:
                        beta_sigma_init = res_sigma.x

            # Non-centered: beta_sigma = 10 * beta_sigma_raw
            if beta_sigma_init is not None:
                initvals["beta_sigma_raw"] = beta_sigma_init / 10.0
            else:
                initvals["beta_sigma_raw"] = np.zeros(n_sigma)

        elif "beta_sigma_raw" in var_names:
            # alpha_importance=True or custom variance: use zeros.
            beta_sigma_raw_var = self.model["beta_sigma_raw"]
            n_sigma = beta_sigma_raw_var.type.shape[0] or 4
            initvals["beta_sigma_raw"] = np.zeros(n_sigma)

        if "sigma2" in var_names:
            initvals["sigma2"] = 1.0

        if "sig2_psi" in var_names:
            initvals["sig2_psi"] = 1.0

        return initvals

    def _sample_model(self, **kwargs) -> az.InferenceData:
        """Sample from the PyMC model.

        Uses BFGS-based initial values for ``beta_0``, ``beta``, and
        ``beta_sigma``.  Initial values are injected via
        ``model.set_initval()`` so that both PyMC's internal sampler and
        external samplers (nutpie) pick them up during model compilation.
        """
        if self.model is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first or .fit() instead."
            )
        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}

            # Compute initial values (constrained space) and inject into model
            initvals = self._compute_initvals()

            for rv in self.model.free_RVs:
                if rv.name in initvals:
                    self.model.set_initval(rv, initvals[rv.name])

            if sampler_args.get("nuts_sampler", "pymc") != "nutpie":
                sampler_args["initvals"] = initvals
            idata = pm.sample(**sampler_args)

            # Skip prior/posterior predictive when using holdout CV
            # (unnecessary for CV metrics and posterior predictive on
            # the Censored term is slow)
            if self._holdout_mask is None:
                idata.extend(pm.sample_prior_predictive(), join="right")
                idata.extend(pm.sample_posterior_predictive(idata), join="right")

        idata = self._set_idata_attrs(idata)
        return idata

    def _set_idata_attrs(self, idata: az.InferenceData | None = None) -> az.InferenceData:
        """Set metadata attributes on an InferenceData object."""
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        idata.attrs["id"] = self.id
        idata.attrs["model_type"] = self._model_type
        idata.attrs["version"] = self.version
        idata.attrs["sampler_config"] = json.dumps(self.sampler_config)
        idata.attrs["model_config"] = json.dumps(self._serializable_model_config)
        self._save_input_params(idata)
        return idata

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        holdout_mask: np.ndarray | None = None,
        progressbar: bool = True,
        random_seed=None,
        **kwargs,
    ) -> "spGDMM":
        """
        Fit the model using the provided data.

        Parameters
        ----------
        X : pd.DataFrame
            Site-level data with columns [xc, yc, time_idx, predictor1, ...]
        y : array-like
            Pairwise Bray-Curtis dissimilarities (length n_sites*(n_sites-1)//2).
        holdout_mask : np.ndarray of bool or None, optional
            Boolean mask (length n_pairs, True = held-out pair).  When set,
            the model is built on ALL sites but the likelihood is split into
            observed (Censored) and held-out (free Normal) terms.  Use
            ``extract_holdout_predictions()`` to retrieve posterior samples for
            the held-out pairs.
        progressbar : bool
            Whether to display a progress bar during sampling.
        random_seed : int or None
            Random seed for reproducibility.
        **kwargs
            Additional keyword arguments forwarded to the sampler.

        Returns
        -------
        self : spGDMM
            The fitted estimator. Access inference data via ``self.idata``.
        """
        sampler_updates: dict = {"progressbar": progressbar, **kwargs}
        if random_seed is not None:
            sampler_updates["random_seed"] = random_seed
        self.sampler_config = {**self.sampler_config, **sampler_updates}

        if y is None:
            y = np.zeros(X.shape[0])
        y_series = pd.Series(np.asarray(y, dtype=float), name=self.output_var)

        self.build_model(X, y_series.values, holdout_mask=holdout_mask)

        self.idata = self._sample_model()

        # Save only site-level X in fit_data. Pair-level y is recoverable from
        # idata.constant_data["log_y_data"]. Concatenating X (n_sites rows) with
        # y_series (n_pairs rows) misaligns pandas indices and inflates fit_data to
        # n_pairs rows, causing n_sites to be reconstructed as n_pairs on load().
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group fit_data is not defined in the InferenceData scheme",
            )
            self.idata.add_groups(fit_data=X.to_xarray())

        return self

    def save(self, fname: str) -> None:
        """Save inference data to a file.

        Parameters
        ----------
        fname : str
            Path to the output file.
        """
        if self.idata is not None and "posterior" in self.idata:
            self.idata.to_netcdf(str(fname))
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")

    @classmethod
    def load(cls, fname: str) -> "spGDMM":
        """Load a fitted model from a file.

        Parameters
        ----------
        fname : str
            Path to a saved InferenceData file.

        Returns
        -------
        spGDMM
        """
        filepath = Path(str(fname))
        idata = az.from_netcdf(filepath)

        model_config = ModelConfig.from_dict(json.loads(idata.attrs["model_config"]))
        sampler_config = SamplerConfig.from_dict(json.loads(idata.attrs["sampler_config"]))

        # Restore the fitted preprocessor from serialised state — no refitting needed.
        preprocessor = GDMPreprocessor.from_xarray(idata.constant_data)

        model = cls(
            preprocessor=preprocessor,
            model_config=model_config,
            sampler_config=sampler_config,
        )
        model.idata = idata

        # Populate metadata from stored training data using the already-fitted preprocessor.
        # fit_data contains site-level X; pair-level y is recovered from constant_data.
        fit_df = idata.fit_data.to_dataframe()
        if "log_y" in fit_df.columns:
            # Legacy format: X and y were concat'd with mismatched lengths. Recover X
            # by dropping rows that are NaN (rows beyond n_sites), and y from constant_data.
            X = fit_df.drop(columns=["log_y"]).dropna()
        else:
            X = fit_df
        # y is stored as log_y_data (log-transformed) in constant_data; exp to recover.
        log_y_stored = idata.constant_data["log_y_data"].values
        y_raw = np.exp(log_y_stored)
        model.build_model(X, y_raw)

        if model.id != idata.attrs["id"]:
            raise ValueError(
                f"The file '{fname}' does not contain an inference data of the same model "
                f"or configuration as '{cls._model_type}'"
            )

        return model

    def predict(
        self,
        X_pred: pd.DataFrame,
        extend_idata: bool = True,
        predictions: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict on new data, returning the posterior mean of the output variable.

        Parameters
        ----------
        X_pred : pd.DataFrame
            Site-level data.
        extend_idata : bool
            Whether to add predictions to the InferenceData object.
        predictions : bool
            Whether to use the predictions group for posterior predictive sampling.

        Returns
        -------
        np.ndarray
        """
        posterior_predictive_samples = self.sample_posterior_predictive(
            X_pred, extend_idata, combined=False, predictions=predictions, **kwargs
        )
        if self.output_var not in posterior_predictive_samples:
            raise KeyError(
                f"Output variable {self.output_var} not found in posterior predictive samples."
            )
        return posterior_predictive_samples[self.output_var].mean(
            dim=["chain", "draw"], keep_attrs=True
        ).data

    def predict_posterior(
        self,
        X_pred: pd.DataFrame,
        y_pred_obs=None,
        extend_idata: bool = True,
        combined: bool = True,
        predictions: bool = True,
        **kwargs,
    ) -> xr.DataArray:
        """
        Generate posterior predictive samples on new data.

        Parameters
        ----------
        X_pred : pd.DataFrame
            Site-level data.
        extend_idata : bool
            Whether to add predictions to InferenceData.
        combined : bool
            Combine chain and draw dims into sample.
        predictions : bool
            Whether to use the predictions group.

        Returns
        -------
        xr.DataArray
        """
        posterior_predictive_samples = self.sample_posterior_predictive(
            X_pred, extend_idata, combined,
            predictions=predictions, **kwargs
        )
        if self.output_var not in posterior_predictive_samples:
            raise KeyError(
                f"Output variable {self.output_var} not found in posterior predictive samples."
            )
        return posterior_predictive_samples[self.output_var]

    def predict_proba(
        self,
        X_pred: pd.DataFrame,
        extend_idata: bool = True,
        combined: bool = False,
        **kwargs,
    ) -> xr.DataArray:
        """Alias for ``predict_posterior``, for sklearn probabilistic estimator compatibility."""
        return self.predict_posterior(X_pred, extend_idata=extend_idata, combined=combined, **kwargs)

    def sample_posterior_predictive(
        self, X_pred, extend_idata, combined, predictions=True, **kwargs
    ):
        """Sample from the model's posterior predictive distribution."""
        self._data_setter(X_pred)
        with self.model:
            post_pred = pm.sample_posterior_predictive(
                self.idata, predictions=predictions, **kwargs
            )
            if extend_idata:
                self.idata.extend(post_pred, join="right")
        group_name = "predictions" if predictions else "posterior_predictive"
        return az.extract(post_pred, group_name, combined=combined)

    def sample_prior_predictive(
        self,
        X_pred,
        y_pred=None,
        samples: int | None = None,
        extend_idata: bool = False,
        combined: bool = True,
        **kwargs,
    ):
        """Sample from the model's prior predictive distribution."""
        if y_pred is None:
            y_pred = pd.Series(np.zeros(len(X_pred)), name=self.output_var)
        if samples is None:
            samples = self.sampler_config.get("draws", 500)

        if self.model is None:
            self.build_model(X_pred, y_pred)
        else:
            self._data_setter(X_pred, y_pred)
        with self.model:
            prior_pred: az.InferenceData = pm.sample_prior_predictive(samples, **kwargs)
            self._set_idata_attrs(prior_pred)
            if extend_idata:
                if self.idata is not None:
                    self.idata.extend(prior_pred, join="right")
                else:
                    self.idata = prior_pred
        return az.extract(prior_pred, "prior_predictive", combined=combined)

    def extract_holdout_predictions(self) -> dict:
        """Extract posterior predictions for held-out pairs.

        Returns a dict with keys:

        - ``hold_idx``: indices into the full pairwise vector for held-out pairs
        - ``y_pred_mean``: posterior mean of dissimilarity (on [0, 1] scale)
        - ``y_pred_samples``: full posterior samples, shape (n_hold, n_samples)

        The transform ``Z = min(1, exp(log_V))`` matches White et al. (2024).
        """
        if self._holdout_mask is None:
            raise RuntimeError("No holdout mask was set during fit().")
        if self.idata is None or "posterior" not in self.idata:
            raise RuntimeError("Model must be fitted before extracting predictions.")

        log_y_hold = self.idata.posterior["log_y_holdout"]
        # (chain, draw, n_hold) → (n_hold, n_samples)
        samples = log_y_hold.stack(sample=("chain", "draw")).values
        y_samples = np.minimum(1.0, np.exp(samples))
        return {
            "hold_idx": np.where(self._holdout_mask)[0],
            "y_pred_mean": y_samples.mean(axis=1),
            "y_pred_samples": y_samples,
        }

    @property
    def id(self) -> str:
        """Generate a unique hash value for the model based on config and version."""
        hasher = hashlib.sha256()
        hasher.update(str(self._config.to_dict().values()).encode())
        hasher.update(self.version.encode())
        hasher.update(self._model_type.encode())
        return hasher.hexdigest()[:16]

    @property
    def output_var(self) -> str:
        """Return the output variable name."""
        return "log_y"

    @staticmethod
    def get_default_model_config() -> dict:
        """Return default model configuration."""
        return ModelConfig().to_dict()

    @staticmethod
    def get_default_sampler_config() -> dict:
        """Return default sampler configuration."""
        return {
            "draws": 1000,
            "tune": 1000,
            "chains": 4,
            "target_accept": 0.95,
            "nuts_sampler": "nutpie",
            "progressbar": True,
            "random_seed": None,
        }

    @property
    def _serializable_model_config(self) -> dict:
        """Return serializable model config."""
        return self._config.to_dict()


__all__ = ["spGDMM"]
