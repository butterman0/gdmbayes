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

from ..preprocessing.config import PreprocessorConfig
from ..preprocessing.preprocessor import GDMPreprocessor
from .config import ModelConfig, SamplerConfig
from .spatial import SPATIAL_FUNCTIONS
from .variance import VARIANCE_FUNCTIONS, poly_fit, poly_predict


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

        # ------------------------------------------------------------------ #
        # Model config
        # ------------------------------------------------------------------ #
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

        self.model_ = None
        self.idata_: az.InferenceData | None = None

        # Raw inputs (set by _generate_and_preprocess_model_data)
        self.X_: pd.DataFrame | None = None  # site-level input
        self.y_: np.ndarray | None = None    # raw pairwise dissimilarities
        # Transformed for model (set by _generate_and_preprocess_model_data)
        self.X_GDM_: np.ndarray | None = None  # pairwise GDM feature matrix
        self.log_y_: np.ndarray | None = None        # log-transformed dissimilarities
        # Variance model state (covariate-dependent only)
        self._X_sigma: np.ndarray | None = None
        self._poly_alpha: np.ndarray | None = None
        self._poly_norm2: np.ndarray | None = None
        # GP object — stored after build_model() for use in predict() via gp.conditional()
        self.gp_: pm.gp.Latent | None = None

    @property
    def _variance_type(self) -> str:
        """Return the variance type as a string ('custom' for callable)."""
        v = self._config.variance
        return v if isinstance(v, str) else "custom"

    @property
    def _spatial_type(self) -> str:
        """Return the spatial effect type as a string ('custom' for callable)."""
        s = self._config.spatial_effect
        return s if isinstance(s, str) else "custom"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        progressbar: bool = True,
        random_seed=None,
        **kwargs,
    ) -> "spGDMM":
        """
        Fit the model on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training site-level data with columns [xc, yc, time_idx, predictor1, ...].
            Pass only training sites; for cross-validation, subset with
            ``X.iloc[train_sites].reset_index(drop=True)``.
        y : array-like
            Pairwise Bray-Curtis dissimilarities for training-site pairs
            (length n_train*(n_train-1)//2).  Use ``site_pairs(n_sites, train_sites)``
            to extract the correct subset from the full pairwise vector.
        progressbar : bool
            Whether to display a progress bar during sampling.
        random_seed : int or None
            Random seed for reproducibility.
        **kwargs
            Additional keyword arguments forwarded to the sampler.

        Returns
        -------
        self : spGDMM
            The fitted estimator. Access inference data via ``self.idata_``.
            Call ``predict(X_test)`` for out-of-sample predictions; the GP
            spatial effect is conditioned on the training posterior via kriging.
        """
        if y is None:
            raise ValueError(
                "y is required for fit(). Pass pairwise dissimilarities "
                "(length n_train*(n_train-1)//2)."
            )

        sampler_args: dict = {**self.sampler_config, "progressbar": progressbar, **kwargs}
        if random_seed is not None:
            sampler_args["random_seed"] = random_seed

        y_series = pd.Series(np.asarray(y, dtype=float), name=self.output_var)

        self.build_model(X, y_series.values)

        self.idata_ = self._sample_model(sampler_args=sampler_args)

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
            self.idata_.add_groups(fit_data=X.to_xarray())

        return self

    def _generate_and_preprocess_model_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Preprocess model data before fitting.

        Fits spline meshes from ``X`` (training sites only), then builds the
        pairwise feature matrix and variance model state.  For out-of-sample
        prediction at new sites, the GP is conditioned on the training posterior
        via ``gp.conditional()`` inside ``predict()``.

        Parameters
        ----------
        X : pd.DataFrame
            Training site-level data with columns [xc, yc, time_idx, predictor1, ...].
        y : pd.Series
            Pairwise Bray-Curtis dissimilarities (length n_sites*(n_sites-1)//2)
            for training-site pairs only.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        y_array = np.asarray(y, dtype=float)

        self.X_ = X
        self.y_ = y_array
        self.log_y_ = np.log(np.maximum(y_array, np.finfo(float).eps))
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)

        # Fit the preprocessor on training sites only (skips refitting on load path).
        try:
            check_is_fitted(self.preprocessor)
        except NotFittedError:
            self.preprocessor.fit(X)
        prep = self.preprocessor

        # Pairwise I-spline diffs + distance splines for train-train pairs.
        self.X_GDM_ = prep.transform(X)
        self.n_features_in_ = prep.n_predictors_

        # Build orthogonal polynomial basis for covariate-dependent variance.
        # Matches White et al.'s R code: X_sigma = cbind(1, poly(vec_distance, 3))
        # Fitted from training-pair distances only; applied to test pairs via
        # _build_sigma_basis() in _predict_gp_conditional().
        if self._variance_type == "covariate_dependent":
            pw_dist = prep.pw_distance(prep.location_values_train_)
            pw_dist_clipped = np.clip(pw_dist, prep.dist_mesh_[0], prep.dist_mesh_[-1])
            _, alpha, norm2 = poly_fit(pw_dist_clipped, degree=3)
            self._poly_alpha = alpha
            self._poly_norm2 = norm2
            poly_cols = poly_predict(pw_dist_clipped, alpha, norm2)
            self._X_sigma = np.column_stack([np.ones(len(pw_dist)), poly_cols])
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

        X_GDM = self.X_GDM_
        log_y = self.log_y_
        p = X_GDM.shape[1]

        n_train = self.preprocessor.location_values_train_.shape[0]
        row_ind, col_ind = np.triu_indices(n_train, k=1)

        # Resolve the spatial function once (works on both numpy and pytensor
        # since the registered functions use only generic ops like abs() and **).
        if self._spatial_type != "none":
            spatial_fn = (
                self._config.spatial_effect if self._spatial_type == "custom"
                else SPATIAL_FUNCTIONS[self._config.spatial_effect]
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
            if self._spatial_type != "none":
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
        var_names = [v.name for v in self.model_.free_RVs]
        if "beta_sigma_raw" in var_names and not self._config.alpha_importance:
            beta_sigma_raw_var = self.model_["beta_sigma_raw"]
            n_sigma = beta_sigma_raw_var.type.shape[0] or 4

            mu_fixed = initvals["beta_0"] + X_GDM @ initvals["beta"]
            # Include spatial effect in mu_fixed if psi was initialized
            if "psi" in initvals:
                mu_fixed = mu_fixed + spatial_fn(
                    initvals["psi"], row_ind, col_ind
                )
            residuals = log_y - mu_fixed

            def _profile_nll(beta_sigma, linear_pred_fn):
                """Gaussian NLL with heterogeneous variance."""
                lp = np.clip(linear_pred_fn(beta_sigma), -20, 20)
                return 0.5 * np.sum(residuals ** 2 * np.exp(-lp) + lp)

            beta_sigma_init = None

            if self._variance_type == "covariate_dependent" and self._X_sigma is not None:
                X_sig = self._X_sigma
                res_sigma = minimize(
                    _profile_nll, np.zeros(n_sigma), args=(lambda bs: X_sig @ bs,),
                    method="BFGS",
                )
                if res_sigma.success:
                    beta_sigma_init = res_sigma.x

            elif self._variance_type == "polynomial":
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
            beta_sigma_raw_var = self.model_["beta_sigma_raw"]
            n_sigma = beta_sigma_raw_var.type.shape[0] or 4
            initvals["beta_sigma_raw"] = np.zeros(n_sigma)

        if "sigma2" in var_names:
            initvals["sigma2"] = 1.0

        if "sig2_psi" in var_names:
            initvals["sig2_psi"] = 1.0

        return initvals

    def _sample_model(self, sampler_args: "dict | None" = None, **kwargs) -> az.InferenceData:
        """Sample from the PyMC model.

        Uses BFGS-based initial values for ``beta_0``, ``beta``, and
        ``beta_sigma``.  Initial values are injected via
        ``model.set_initval()`` so that both PyMC's internal sampler and
        external samplers (nutpie) pick them up during model compilation.
        """
        if self.model_ is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first or .fit() instead."
            )
        with self.model_:
            if sampler_args is None:
                sampler_args = dict(self.sampler_config)
            sampler_args.update(kwargs)

            # Compute initial values (constrained space) and inject into model
            initvals = self._compute_initvals()

            for rv in self.model_.free_RVs:
                if rv.name in initvals:
                    self.model_.set_initval(rv, initvals[rv.name])

            if sampler_args.get("nuts_sampler", "pymc") != "nutpie":
                sampler_args["initvals"] = initvals
            idata = pm.sample(**sampler_args)
            idata.extend(pm.sample_prior_predictive(), join="right")
            idata.extend(pm.sample_posterior_predictive(idata), join="right")

        idata = self._set_idata_attrs(idata)
        return idata

    def _set_idata_attrs(self, idata: az.InferenceData | None = None) -> az.InferenceData:
        """Set metadata attributes on an InferenceData object."""
        if idata is None:
            idata = self.idata_
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        idata.attrs["id"] = self.id
        idata.attrs["model_type"] = self._model_type
        idata.attrs["version"] = self.version
        idata.attrs["sampler_config"] = json.dumps(self.sampler_config)
        idata.attrs["model_config"] = json.dumps(self._serializable_model_config)
        self._save_input_params(idata)
        return idata

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
        return self.predict_posterior(
            X_pred, extend_idata=extend_idata, combined=False,
            predictions=predictions, **kwargs
        ).mean(dim=["chain", "draw"], keep_attrs=True).data

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
        """Sample from the model's posterior predictive distribution.

        When predicting at new sites (n_pred != n_train) with a spatial effect,
        ``_data_setter`` triggers GP conditional (kriging) and stores the result
        in ``self._gp_pred_result``.  This method consumes that result and returns
        a Dataset with the standard ``output_var`` key so that ``predict()`` and
        ``predict_posterior()`` work transparently.
        """
        self._data_setter(X_pred)

        # GP conditional path — result was computed inside _data_setter as a numpy array.
        if hasattr(self, "_gp_pred_result"):
            log_y_arr = self._gp_pred_result   # (n_chains, n_draws, n_pred_pairs)
            del self._gp_pred_result
            raw = xr.DataArray(log_y_arr, dims=["chain", "draw", "obs_dim_0"])
            if combined:
                raw = raw.stack(sample=("chain", "draw"))
            return xr.Dataset({self.output_var: raw})

        # Standard path — pm.set_data was used; sample from existing model.
        with self.model_:
            post_pred = pm.sample_posterior_predictive(
                self.idata_, predictions=predictions, **kwargs
            )
            if extend_idata:
                self.idata_.extend(post_pred, join="right")
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

        if self.model_ is None:
            self.build_model(X_pred, y_pred)
        else:
            self._data_setter(X_pred, y_pred)
        with self.model_:
            prior_pred: az.InferenceData = pm.sample_prior_predictive(samples, **kwargs)
            self._set_idata_attrs(prior_pred)
            if extend_idata:
                if self.idata_ is not None:
                    self.idata_.extend(prior_pred, join="right")
                else:
                    self.idata_ = prior_pred
        return az.extract(prior_pred, "prior_predictive", combined=combined)

    def _data_setter(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> None:
        """Update mutable data containers in the existing model for prediction.

        When predicting at new sites (different count from training), and a spatial
        effect is active, delegates to ``_predict_gp_conditional()`` which uses
        ``gp.conditional()`` (GP kriging) to obtain ``psi`` at new locations.
        Result is stored in ``self._gp_pred_result`` for
        ``sample_posterior_predictive()`` to consume.

        For same-count prediction (e.g. full-data posterior predictive), updates
        ``pm.Data`` containers in place via ``pm.set_data()``.
        """
        if self.idata_ is None or "posterior" not in self.idata_:
            raise ValueError("Model must be fitted before predicting.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_pred_sites = len(X)
        n_train_sites = self.preprocessor.location_values_train_.shape[0]

        # Route to GP conditional for new-site prediction when a spatial effect is active.
        if self._spatial_type != "none" and n_pred_sites != n_train_sites:
            self._predict_gp_conditional(X)
            return

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
            pred_locations = X.iloc[:, :2].values
            pw_dist_pred = self.preprocessor.pw_distance(pred_locations)
            set_data_dict["X_sigma_data"] = self._build_sigma_basis(pw_dist_pred)

        # For spatial models: recompute pair indices for the prediction sites so the
        # spatial term shape matches X_data.  n_pred_sites is recovered from n_pred_pairs
        # via the quadratic formula: n_pairs = n*(n-1)/2.
        if self._spatial_type != "none":
            pred_row_ind, pred_col_ind = np.triu_indices(n_pred_sites, k=1)
            set_data_dict["row_indices"] = pred_row_ind.astype(np.int32)
            set_data_dict["col_indices"] = pred_col_ind.astype(np.int32)

        with self.model_:
            pm.set_data(
                set_data_dict,
                coords={"obs_pair": np.arange(n_pred_pairs)},
            )

    def _predict_gp_conditional(self, X_pred: pd.DataFrame) -> None:
        """Build and sample GP conditional predictions for new (test) sites.

        Called by ``_data_setter()`` when the number of prediction sites differs
        from the number of training sites and a spatial effect is active.

        Strategy: sample ``psi_pred`` from the GP conditional via PyMC (so that
        the conditioning on training-site posterior is handled correctly), then
        assemble the full linear predictor and draw posterior predictive samples
        in NumPy.  This avoids pytensor batch-dimension shape conflicts that arise
        when mixing a batched GP conditional with other model RVs inside
        ``pm.sample_posterior_predictive``.

        Result is stored in ``self._gp_pred_result`` (ndarray, shape
        ``(n_chains, n_draws, n_pred_pairs)``) for ``sample_posterior_predictive()``
        to consume.
        """
        prep = self.preprocessor
        n_pred = len(X_pred)
        n_pred_pairs = n_pred * (n_pred - 1) // 2
        row_p, col_p = np.triu_indices(n_pred, k=1)

        # Feature matrix for pred-pred pairs (uses already-fitted meshes).
        X_pred_features = prep.transform(X_pred)   # (n_pred_pairs, J)

        # GP coordinates in km (matching training convention).
        pred_coords = X_pred.iloc[:, :2].values / 1000.0

        # Variance basis for covariate_dependent.
        X_sigma_pred = None
        if self._variance_type == "covariate_dependent" and self._poly_alpha is not None:
            pw_dist_pred = prep.pw_distance(X_pred.iloc[:, :2].values)
            X_sigma_pred = self._build_sigma_basis(pw_dist_pred)   # (n_pred_pairs, 4)

        # Unique suffix — prevents name clashes across repeated predict() calls.
        self._pred_call_count = getattr(self, "_pred_call_count", 0) + 1
        sfx = self._pred_call_count
        psi_name = f"psi_pred_{sfx}"

        # --- Step 1: sample psi_pred via PyMC GP conditional ---
        # For each posterior sample (psi_train_s, length_scale_s, sig2_psi_s), PyMC
        # draws psi_pred_s from the conditional MvNormal.  This correctly propagates
        # all GP hyperparameter uncertainty to the prediction locations.
        with self.model_:
            self.gp_.conditional(psi_name, pred_coords)
            psi_idata = pm.sample_posterior_predictive(
                self.idata_, var_names=[psi_name], progressbar=False
            )

        # psi_samples: (chains, draws, n_pred_sites)
        psi_samples = psi_idata.posterior_predictive[psi_name].values
        n_chains, n_draws = psi_samples.shape[:2]
        n_samples = n_chains * n_draws
        psi_flat = psi_samples.reshape(n_samples, n_pred)   # (n_samples, n_pred_sites)

        # --- Step 2: extract posterior parameter samples from idata ---
        post = self.idata_.posterior
        beta_0 = post["beta_0"].values.reshape(n_samples)           # (n_samples,)
        beta   = post["beta"].values.reshape(n_samples, -1)         # (n_samples, J)

        # --- Step 3: linear predictor (vectorised NumPy) ---
        # env + intercept: (n_samples, n_pred_pairs)
        mu = beta_0[:, None] + (beta @ X_pred_features.T)

        # spatial contribution
        if self._spatial_type == "squared_diff":
            diff = psi_flat[:, row_p] - psi_flat[:, col_p]   # (n_samples, n_pred_pairs)
            mu = mu + diff ** 2
        elif self._spatial_type == "abs_diff":
            diff = psi_flat[:, row_p] - psi_flat[:, col_p]
            mu = mu + np.abs(diff)
        elif self._spatial_type == "custom":
            # Custom callable: apply per sample (rare path)
            spatial_contrib = np.stack([
                np.asarray(self._config.spatial_effect(psi_flat[s], row_p, col_p))
                for s in range(n_samples)
            ])
            mu = mu + spatial_contrib

        # --- Step 4: variance ---
        if self._variance_type == "homogeneous":
            sigma2 = post["sigma2"].values.reshape(n_samples)[:, None]     # (n_samples, 1)
        elif self._variance_type == "covariate_dependent" and X_sigma_pred is not None:
            bs = post["beta_sigma"].values.reshape(n_samples, -1)           # (n_samples, 4)
            log_s2 = np.clip(X_sigma_pred @ bs.T, -20, 20).T               # (n_samples, n_pred_pairs)
            sigma2 = np.exp(log_s2)
        elif self._variance_type == "polynomial":
            bs = post["beta_sigma"].values.reshape(n_samples, -1)           # (n_samples, 4)
            log_s2 = np.clip(
                bs[:, 0:1] + bs[:, 1:2] * mu + bs[:, 2:3] * mu ** 2 + bs[:, 3:4] * mu ** 3,
                -20, 20,
            )
            sigma2 = np.exp(log_s2)                                         # (n_samples, n_pred_pairs)
        else:
            sigma2 = post["sigma2"].values.reshape(n_samples)[:, None]

        # --- Step 5: draw posterior predictive samples ---
        seed = self.sampler_config.get("random_seed", None) if hasattr(self, "sampler_config") else None
        rng = np.random.default_rng(seed)
        log_y_samples = rng.normal(mu, np.sqrt(sigma2))   # (n_samples, n_pred_pairs)

        # Store as (chains, draws, pairs) for consistent handling in sample_posterior_predictive.
        self._gp_pred_result = log_y_samples.reshape(n_chains, n_draws, n_pred_pairs)

    def build_model(
        self,
        X: "pd.DataFrame | None" = None,
        y: "pd.Series | np.ndarray | None" = None,
    ) -> None:
        """Preprocess data (if provided) and build the PyMC model.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Training site-level data. If provided together with ``y``, preprocessing
            is run first via ``_generate_and_preprocess_model_data``.
            If omitted, assumes preprocessing was already done (e.g. on
            the load path).
        y : array-like, optional
            Pairwise dissimilarities (training pairs only). Required when ``X`` is provided.
        """
        self.gp_ = None  # reset; set below if spatial effect is active
        if X is not None:
            if y is None:
                raise ValueError("y is required when X is provided.")
            y_array = np.asarray(y, dtype=float)
            self._generate_and_preprocess_model_data(X, y_array)
        elif self.X_GDM_ is None:
            raise ValueError("No preprocessed data. Pass X and y, or call fit().")

        X_values = self.X_GDM_
        log_y_values = self.log_y_

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

            if self._spatial_type != "none":
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
                self.gp_ = gp  # stored for predict() via gp.conditional()

                # Use pm.Data so indices can be updated via pm.set_data when
                # predict() is called on the same number of sites as training.
                row_ind, col_ind = np.triu_indices(n_sites_train, k=1)
                row_indices = pm.Data("row_indices", row_ind.astype(np.int32))
                col_indices = pm.Data("col_indices", col_ind.astype(np.int32))

                spatial_fn = (
                    self._config.spatial_effect if self._spatial_type == "custom"
                    else SPATIAL_FUNCTIONS[self._config.spatial_effect]
                )
                mu += spatial_fn(psi, row_indices, col_indices)

            pm.Censored(
                "log_y",
                pm.Normal.dist(mu=mu, sigma=pm.math.sqrt(sigma2)),
                lower=None, upper=0,
                observed=log_y_data,
            )

        self.model_ = model

    def save(self, fname: str) -> None:
        """Save inference data to a file.

        Parameters
        ----------
        fname : str
            Path to the output file.
        """
        if self.idata_ is not None and "posterior" in self.idata_:
            self.idata_.to_netcdf(str(fname))
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
        model.idata_ = idata

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

    def gdm_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform site-level predictors to the biological importance scale.

        Equivalent to R's ``gdm.transform(model, data)``. Returns per-site
        I-spline basis values with columns named ``{predictor}_{basis}``.

        Parameters
        ----------
        X : pd.DataFrame
            Site-level data with columns [xc, yc, time_idx, predictor1, ...]
            matching the structure used during training.

        Returns
        -------
        pd.DataFrame
            I-spline-transformed predictor values per site.
        """
        check_is_fitted(self, "idata_")
        I_spline_bases = self.preprocessor.transform(X, biological_space=True)
        n_predictors = self.preprocessor.n_predictors_
        n_basis = self.preprocessor.n_spline_bases_
        predictor_names = self.preprocessor.predictor_names_ or [
            f"pred_{i}" for i in range(n_predictors)
        ]
        columns = [
            f"{pred}_{j}"
            for pred in predictor_names
            for j in range(1, n_basis + 1)
        ]
        return pd.DataFrame(I_spline_bases, index=X.index, columns=columns)

    def ispline_extract(self) -> dict:
        """Extract I-spline knot positions and posterior median coefficients.

        Equivalent to R's ``isplineExtract(model)``.

        Returns
        -------
        dict
            Maps each predictor name to ``{"x": knot_positions, "y": median_coefficients}``.
            Geographic distance is returned under the key ``"distance"`` when present.
        """
        check_is_fitted(self, "idata_")
        predictor_mesh = self.preprocessor.predictor_mesh_
        dist_mesh = self.preprocessor.dist_mesh_
        predictor_names = self.preprocessor.predictor_names_

        result = {}
        if "beta" in self.idata_.posterior:
            beta_median = self.idata_.posterior.beta.median(dim=["chain", "draw"])
            for i, pred in enumerate(predictor_names):
                result[pred] = {
                    "x": predictor_mesh[i],
                    "y": beta_median.sel(feature=pred).values,
                }

        if "beta_dist" in self.idata_.posterior:
            beta_dist_median = self.idata_.posterior.beta_dist.median(dim=["chain", "draw"])
            result["distance"] = {
                "x": dist_mesh,
                "y": beta_dist_median.values,
            }

        return result

    @property
    def id(self) -> str:
        """Generate a unique hash value for the model based on config and version."""
        hasher = hashlib.sha256()
        hasher.update(json.dumps(self._config.to_dict(), sort_keys=True).encode())
        hasher.update(self.version.encode())
        hasher.update(self._model_type.encode())
        return hasher.hexdigest()[:16]

    @property
    def output_var(self) -> str:
        """Return the output variable name."""
        return "log_y"

    @property
    def _serializable_model_config(self) -> dict:
        """Return serializable model config."""
        return self._config.to_dict()

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

    def __sklearn_is_fitted__(self) -> bool:
        """Return True only after a successful fit (idata with posterior exists)."""
        return self.idata_ is not None and "posterior" in self.idata_

    def _more_tags(self):
        return {"no_validation": True, "non_deterministic": True}


__all__ = ["spGDMM"]
