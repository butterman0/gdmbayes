"""
Spatial Generalized Dissimilarity Mixed Model (spGDMM).

This module implements the spGDMM class for modeling pairwise ecological
dissimilarities as a function of environmental predictors and spatial distance.
"""

import json
import typing as t
import warnings
from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from scipy.spatial.distance import pdist

from ..core.base import ModelBuilder
from ..core.config import PreprocessorConfig
from ..preprocessing.preprocessor import GDMPreprocessor
from .variants import ModelConfig, SamplerConfig, VarianceType, SpatialEffectType


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


if t.TYPE_CHECKING:
    from ..distances.ocean import ocean_path_distance_pdist


class spGDMM(ModelBuilder):
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
    >>> from spgdmm import spGDMM, ModelConfig, PreprocessorConfig, GDMPreprocessor
    >>> from spgdmm import VarianceType, SpatialEffectType
    >>> prep_cfg = PreprocessorConfig(deg=3, knots=2, distance_measure="euclidean")
    >>> model_cfg = ModelConfig(
    ...     variance_type=VarianceType.HOMOGENEOUS,
    ...     spatial_effect_type=SpatialEffectType.ABS_DIFF,
    ... )
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
                # Auto-forward to preprocessor
                prep_kwargs = {k: model_config[k] for k in legacy_keys}
                self.preprocessor = GDMPreprocessor(
                    config=PreprocessorConfig.from_dict(prep_kwargs)
                )
            self._config = ModelConfig.from_dict(model_config)
        elif isinstance(model_config, ModelConfig):
            self._config = model_config
        else:
            self._config = ModelConfig()

        # ------------------------------------------------------------------ #
        # Sampler config
        # ------------------------------------------------------------------ #
        if isinstance(sampler_config, SamplerConfig):
            sampler_config = sampler_config.to_dict()

        self._config_dict = self._config.to_dict()
        self.metadata: ModelMetadata | None = None
        self.training_metadata: TrainingMetadata | None = None
        # Populated after each _transform_for_prediction call with clipping/NaN stats.
        self.prediction_metadata: dict | None = None
        self.n_features_in_: int | None = None

        super().__init__(model_config=self._config.to_dict(), sampler_config=sampler_config)

    @property
    def config(self) -> dict:
        """Get model configuration as dictionary (legacy interface)."""
        return self._config_dict

    def _generate_and_preprocess_model_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """
        Preprocess model data before fitting.

        Delegates to ``self.preprocessor.fit()`` to compute spline meshes,
        then populates ``TrainingMetadata`` and ``ModelMetadata``.

        Parameters
        ----------
        X : pd.DataFrame
            Site-level data with columns [xc, yc, time_idx, predictor1, ...]
        y : pd.Series
            Pairwise Bray-Curtis dissimilarities
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Handle y as pre-computed condensed dissimilarities
        y_array = np.asarray(y, dtype=float)
        log_y = np.log(np.maximum(y_array, np.finfo(float).eps))

        self.X = X
        self.y = log_y

        n_sites = X.shape[0]
        self._pair_indices = np.triu_indices(n_sites, k=1)

        # Fit the preprocessor — computes meshes and per-site I-spline bases
        self.preprocessor.fit(X)
        prep = self.preprocessor

        # Retrieve preprocessor config for spline counting
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

        n_cols_env = I_spline_bases_diffs.shape[1] if I_spline_bases_diffs.size > 0 else 0
        n_cols_dist = dist_spline_bases.shape[1] if dist_spline_bases.size > 0 else 0

        self.metadata = ModelMetadata(
            no_sites_train=n_sites,
            no_predictors=n_predictors,
            no_rows=X_GDM.shape[0],
            no_cols=X_GDM.shape[1],
            no_cols_env=n_cols_env,
            no_cols_dist=n_cols_dist,
            predictors=predictor_names_from_data if n_predictors > 0 else [],
            column_names=[f"x_{i}" for i in range(X_GDM.shape[1])],
            X_sigma=pw_dist_clipped.reshape(-1, 1) if n_predictors > 0 else None,
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

        self.X_transformed = X_GDM
        self.y_transformed = log_y

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
        # When called by load(), metadata is None — run preprocessing first.
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

        # Model construction
        with pm.Model(coords=self.model_coords) as model:
            # Define mutable data containers
            X_data = pm.Data("X_data", X_values, dims=("obs_pair", "predictor"))
            log_y_data = pm.Data("log_y_data", log_y_values, dims=("obs_pair",))

            # Define priors
            beta_0 = pm.Normal("beta_0", mu=0, sigma=10)

            if self._config.alpha_importance:
                J = n_spline_bases
                F = len(self.metadata.predictors)

                if F > 0:
                    # Dirichlet prior over I-spline weights
                    beta = pm.Dirichlet(
                        "beta", a=np.ones(J),
                        shape=(F, J),
                        dims=("feature", "basis_function")
                    )

                    # I-spline dot product
                    n_cols_env = self.metadata.no_cols_env
                    n_cols_dist = self.metadata.no_cols_dist
                    X_env = X_data[:, :n_cols_env]
                    X_reshaped = X_env.reshape((-1, F, J))
                    warped = (X_reshaped * beta[None, :, :]).sum(axis=2)

                    # Alpha importance weights
                    alpha = pm.HalfNormal("alpha", sigma=1, shape=F, dims=("feature",))
                    mu = beta_0 + pm.math.dot(warped, alpha)

                    # Add distance spline coefficients
                    if n_cols_dist > 0:
                        dist_cols = X_data[:, n_cols_env:]
                        beta_dist = pm.LogNormal("beta_dist", mu=0, sigma=1, shape=n_cols_dist)
                        mu = mu + pm.math.dot(dist_cols, beta_dist)
                else:
                    # No predictors, just distance
                    mu = beta_0
            else:
                beta = pm.LogNormal("beta", mu=0, sigma=1, shape=self.metadata.no_cols)
                mu = beta_0 + pm.math.dot(X_data, beta)

            # Variance structure
            if self._config.variance_type == VarianceType.HOMOGENEOUS:
                sigma2 = pm.InverseGamma("sigma2", alpha=1, beta=1)
            elif self._config.variance_type == VarianceType.COVARIATE_DEPENDENT:
                if self.metadata.X_sigma is not None and self.metadata.X_sigma.shape[1] > 0:
                    X_sigma = self.metadata.X_sigma
                    beta_sigma = pm.Normal("beta_sigma", mu=0, sigma=5, shape=X_sigma.shape[1])
                    sigma2 = pm.math.exp(pm.math.dot(X_sigma, beta_sigma))
                else:
                    sigma2 = pm.InverseGamma("sigma2", alpha=1, beta=1)
            elif self._config.variance_type == VarianceType.POLYNOMIAL:
                beta_sigma = pm.Normal("beta_sigma", mu=0, sigma=5, shape=4)
                sigma2 = pm.math.exp(
                    beta_sigma[0] + beta_sigma[1] * mu +
                    beta_sigma[2] * mu ** 2 +
                    beta_sigma[3] * mu ** 3
                )
            elif self._config.variance_type == VarianceType.CUSTOM:
                if self._config.custom_variance_fn is None:
                    raise ValueError(
                        "variance_type=CUSTOM requires custom_variance_fn to be set in ModelConfig."
                    )
                X_sigma = self.metadata.X_sigma
                sigma2 = self._config.custom_variance_fn(mu, X_sigma)
            else:
                sigma2 = pm.InverseGamma("sigma2", alpha=1, beta=1)

            # Spatial random effects
            if self._config.spatial_effect_type != SpatialEffectType.NONE:
                sig2_psi = pm.InverseGamma("sig2_psi", alpha=1, beta=1)
                location_values = self.training_metadata.location_values_train
                length_scale = self.training_metadata.length_scale / 2

                cov = sig2_psi * pm.gp.cov.Exponential(2, ls=length_scale)
                gp = pm.gp.Latent(cov_func=cov)
                psi = gp.prior("psi", X=location_values, dims=("site_train",))

                row_ind, col_ind = self._pair_indices

                if self._config.spatial_effect_type == SpatialEffectType.ABS_DIFF:
                    mu += pm.math.abs(psi[row_ind] - psi[col_ind])
                elif self._config.spatial_effect_type == SpatialEffectType.SQUARED_DIFF:
                    mu += pm.math.abs(psi[row_ind] - psi[col_ind]) ** 2
                elif self._config.spatial_effect_type == SpatialEffectType.CUSTOM:
                    if self._config.custom_spatial_effect_fn is None:
                        raise ValueError(
                            "spatial_effect_type=CUSTOM requires custom_spatial_effect_fn "
                            "to be set in ModelConfig."
                        )
                    mu += self._config.custom_spatial_effect_fn(psi, row_ind, col_ind)

            # Observed data with censored likelihood
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

        # Validate predictor count against training
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

    def rgb_biological_space(
        self, X_pred: pd.DataFrame, metric: str = "median"
    ) -> xr.DataArray:
        """
        Compute RGB biological-space map via PCA on I-spline transformed predictors.

        Equivalent to R's gdm.transform() + manual PCA + RGB colour assignment.

        Parameters
        ----------
        X_pred : pd.DataFrame
            Site-level data. Index must be a MultiIndex with levels (yc, xc)
            so that grid_cell can be unstacked into a spatial grid.
        metric : str, default="median"
            Posterior summary: "median" or "mean".

        Returns
        -------
        xr.DataArray
            Dims (time, xc, yc, rgb) with RGB values normalised to [0, 1].
        """
        from spgdmm.plotting.plots import rgb_from_biological_space

        beta = (
            self.idata.posterior.beta.mean(dim=["chain", "draw"])
            if metric == "mean"
            else self.idata.posterior.beta.median(dim=["chain", "draw"])
        )
        X_splined = self._transform_for_prediction(X_pred, biological_space=True).reshape(
            1, -1, beta.sizes["feature"], beta.sizes["basis_function"]
        )
        transformed = (
            xr.DataArray(
                X_splined,
                dims=("time", "grid_cell", "feature", "basis_function"),
                coords={
                    "time": [0],
                    "grid_cell": X_pred.index,
                    "feature": beta["feature"].values,
                    "basis_function": beta["basis_function"].values,
                },
            )
            * beta
        ).sum(dim="basis_function", skipna=False)
        return rgb_from_biological_space(transformed)

    def _data_setter(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> None:
        """Update mutable data containers in the existing model for prediction."""
        X_transformed = self._transform_for_prediction(X)

        if y is None:
            log_y = np.zeros(X_transformed.shape[0])
        else:
            log_y = np.log(np.maximum(np.asarray(y, dtype=float), np.finfo(float).eps))

        with self.model:
            pm.set_data(
                {"X_data": X_transformed, "log_y_data": log_y},
                coords={"obs_pair": np.arange(X_transformed.shape[0])},
            )

    def _save_input_params(self, idata: az.InferenceData) -> None:
        """Persist transformation state (meshes, coordinates) in idata.constant_data.

        Delegates to ``self.preprocessor.to_xarray()`` so that models can be
        serialized and reloaded for prediction without refitting.
        """
        ds = self.preprocessor.to_xarray()
        idata.attrs["predictor_names"] = json.dumps(self.metadata.predictors)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The group constant_data is not defined",
            )
            idata.add_groups(constant_data=ds)

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
