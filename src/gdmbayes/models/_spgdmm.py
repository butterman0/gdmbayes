"""
Spatial Generalized Dissimilarity Mixed Model (spGDMM).

This module implements the spGDMM class for modeling pairwise ecological
dissimilarities as a function of environmental predictors and spatial distance.
"""

import hashlib
import json
import typing as t
import warnings
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ..preprocessing._config import PreprocessorConfig
from ..preprocessing._preprocessor import GDMPreprocessor
from ._config import ModelConfig, SamplerConfig
from ._variance import VARIANCE_FUNCTIONS
from ._spatial import SPATIAL_FUNCTIONS


@dataclass
class ModelMetadata:
    """Column counts, predictor names, and variance-model data for a fitted spGDMM."""

    no_sites_train: int
    no_predictors: int
    no_rows: int
    no_cols: int
    no_cols_env: int
    no_cols_dist: int
    predictors: list        # semantic predictor names
    column_names: list      # feature-matrix column names
    X_sigma: np.ndarray | None   # pairwise distance column for variance model
    p_sigma: int            # 1 if X_sigma is used, 0 otherwise


@dataclass
class TrainingMetadata:
    """Spline meshes and spatial state needed to transform new data consistently."""

    location_values_train: np.ndarray  # (n_sites, 2) site coordinates
    predictor_mesh: np.ndarray         # (n_predictors, n_knot_points)
    dist_mesh: np.ndarray              # (n_knot_points,)
    length_scale: float                # GP spatial length scale
    I_spline_bases: np.ndarray         # (n_sites, n_predictors * n_bases)


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
        # Resolve model_config — detect legacy preprocessing keys and warn
        # ------------------------------------------------------------------ #
        _PREPROCESSING_KEYS = {
            "deg", "knots", "mesh_choice", "distance_measure",
            "custom_dist_mesh", "custom_predictor_mesh", "extrapolation",
            "diss_metric", "time_predictor", "connected_pairs_only",
            "updated_predictor_mesh", "time_varying", "connectivity_percentile",
            "length_scale",
        }

        if isinstance(model_config, dict):
            legacy_keys = _PREPROCESSING_KEYS.intersection(model_config)
            if legacy_keys and preprocessor is None:
                warnings.warn(
                    f"Passing preprocessing key(s) {sorted(legacy_keys)!r} via model_config "
                    f"is deprecated. Use PreprocessorConfig / GDMPreprocessor instead. "
                    f"The keys will be auto-forwarded to the preprocessor this time.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                prep_kwargs = {k: model_config[k] for k in legacy_keys}
                self.preprocessor = GDMPreprocessor(
                    config=PreprocessorConfig.from_dict(prep_kwargs)
                )
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
        self.metadata: ModelMetadata | None = None
        self.training_metadata: TrainingMetadata | None = None
        # Populated after each _transform_for_prediction call with clipping/NaN stats.
        self.prediction_metadata: dict | None = None

    def __sklearn_is_fitted__(self) -> bool:
        """Return True only after a successful fit (idata with posterior exists)."""
        return self.idata is not None and "posterior" in self.idata

    @property
    def config(self) -> dict:
        """Get model configuration as dictionary (legacy interface)."""
        return self._config_dict

    def _generate_and_preprocess_model_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """
        Preprocess model data before fitting.

        Delegates to ``self.preprocessor.fit()`` to compute spline meshes
        (unless the preprocessor is already fitted), then populates
        ``TrainingMetadata`` and ``ModelMetadata``.

        Parameters
        ----------
        X : pd.DataFrame
            Site-level data with columns [xc, yc, time_idx, predictor1, ...]
        y : pd.Series
            Pairwise Bray-Curtis dissimilarities (length n_sites*(n_sites-1)//2).
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        y_array = np.asarray(y, dtype=float)
        log_y = np.log(np.maximum(y_array, np.finfo(float).eps))

        self.X = X
        self.y = log_y

        n_sites = X.shape[0]
        all_row_ind, all_col_ind = np.triu_indices(n_sites, k=1)

        # Fit the preprocessor only if it hasn't been fitted yet (e.g., on load path).
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

        self.metadata = ModelMetadata(
            no_sites_train=n_sites,
            no_predictors=n_predictors,
            no_rows=X_GDM_fit.shape[0],
            no_cols=X_GDM_fit.shape[1],
            no_cols_env=n_cols_env,
            no_cols_dist=n_cols_dist,
            predictors=predictor_names_from_data if n_predictors > 0 else [],
            column_names=[f"x_{i}" for i in range(X_GDM_fit.shape[1])],
            X_sigma=pw_dist_fit.reshape(-1, 1) if n_predictors > 0 else None,
            p_sigma=1 if n_predictors > 0 else 0,
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

    def build_model(
        self,
        X: pd.DataFrame | np.ndarray,
        log_y: pd.Series | np.ndarray,
        **kwargs,
    ) -> None:
        """
        Build the PyMC model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The input data (raw site-level features). If metadata is already populated
            (e.g. after calling _generate_and_preprocess_model_data), the pre-computed
            X_transformed / y_transformed are used instead.
        log_y : pd.Series or np.ndarray
            The response variable (raw dissimilarities or log-dissimilarities).
        **kwargs : keyword arguments
            Additional arguments (ignored; accepted for API compatibility).
        """
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

                    alpha = pm.HalfNormal("alpha", sigma=1, shape=F, dims=("feature",))
                    mu = beta_0 + pm.math.dot(warped, alpha)

                    if n_cols_dist > 0:
                        dist_cols = X_data[:, n_cols_env:]
                        beta_dist = pm.LogNormal("beta_dist", mu=0, sigma=1, shape=n_cols_dist)
                        mu = mu + pm.math.dot(dist_cols, beta_dist)
                else:
                    mu = beta_0
            else:
                beta = pm.LogNormal("beta", mu=0, sigma=1, shape=self.metadata.no_cols)
                mu = beta_0 + pm.math.dot(X_data, beta)

            variance_fn = (
                self._config.variance
                if callable(self._config.variance)
                else VARIANCE_FUNCTIONS[self._config.variance]
            )
            sigma2 = variance_fn(mu, self.metadata.X_sigma)

            if self._config.spatial_effect != "none":
                sig2_psi = pm.InverseGamma("sig2_psi", alpha=1, beta=1)
                location_values = self.training_metadata.location_values_train
                length_scale = self.training_metadata.length_scale / 2

                cov = sig2_psi * pm.gp.cov.Exponential(2, ls=length_scale)
                gp = pm.gp.Latent(cov_func=cov)
                psi = gp.prior("psi", X=location_values, dims=("site_train",))

                row_ind, col_ind = self._pair_indices
                # Use pm.Data so indices can be updated via pm.set_data during prediction
                # when X_pred has a different number of sites than X_train.
                row_indices = pm.Data("row_indices", row_ind.astype(np.int32))
                col_indices = pm.Data("col_indices", col_ind.astype(np.int32))

                spatial_fn = (
                    self._config.spatial_effect
                    if callable(self._config.spatial_effect)
                    else SPATIAL_FUNCTIONS[self._config.spatial_effect]
                )
                mu += spatial_fn(psi, row_indices, col_indices)

            pm.Censored(
                "log_y",
                pm.Normal.dist(mu=mu, sigma=pm.math.sqrt(sigma2)),
                lower=None, upper=0,
                observed=log_y_data,
            )

        self.model = model

    def _transform_for_prediction(
        self, X_pred: pd.DataFrame | np.ndarray, biological_space: bool = False
    ) -> np.ndarray:
        """
        Transform site-level predictors into GDM feature space.

        Delegates to ``self.preprocessor.transform()``.

        Parameters
        ----------
        X_pred : pd.DataFrame or np.ndarray
            Site-level data with columns [xc, yc, time_idx, predictor1, ...]
        biological_space : bool, default=False
            If True, return per-site I-spline bases (n_sites × n_features×n_basis).
            If False, return pairwise GDM feature matrix with distance splines appended.

        Returns
        -------
        np.ndarray
            Transformed feature matrix.
        """
        if not isinstance(X_pred, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be Pandas DataFrame or numpy array")

        if self.idata is None or "posterior" not in self.idata:
            raise ValueError("Model must be fitted before transforming data.")

        X_values = X_pred.iloc[:, 3:].values if isinstance(X_pred, pd.DataFrame) else X_pred[:, 3:]
        if self.n_features_in_ is not None and X_values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_pred has {X_values.shape[1]} environmental predictor(s) but the model "
                f"was trained with {self.n_features_in_}."
            )

        result = self.preprocessor.transform(X_pred, biological_space=biological_space)
        self.prediction_metadata = getattr(
            self.preprocessor, "_last_prediction_metadata", None
        )
        return result

    def _data_setter(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> None:
        """Update mutable data containers in the existing model for prediction.

        For spatial models, also updates ``row_indices`` and ``col_indices`` to
        match the prediction sites so the spatial effect term has the correct shape.
        """
        X_transformed = self._transform_for_prediction(X)
        n_pred_pairs = X_transformed.shape[0]

        if y is None:
            log_y = np.zeros(n_pred_pairs)
        else:
            log_y = np.log(np.maximum(np.asarray(y, dtype=float), np.finfo(float).eps))

        set_data_dict: dict = {"X_data": X_transformed, "log_y_data": log_y}

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
        idata.attrs["predictor_names"] = json.dumps(self.metadata.predictors)
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

    def _sample_model(self, **kwargs) -> az.InferenceData:
        """Sample from the PyMC model."""
        if self.model is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first or .fit() instead."
            )
        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}
            idata = pm.sample(**sampler_args)
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
        self.sampler_config = {
            **self.sampler_config,
            "progressbar": progressbar,
            "random_seed": random_seed,
            **kwargs,
        }

        if y is None:
            y = np.zeros(X.shape[0])
        y_series = pd.Series(np.asarray(y, dtype=float), name=self.output_var)

        self._generate_and_preprocess_model_data(X, y_series.values)
        self.build_model(self.X, self.y)

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
        model._generate_and_preprocess_model_data(X, y_raw)
        model.build_model(model.X, model.y)

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
