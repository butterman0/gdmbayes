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
from geopy.distance import geodesic
from scipy.spatial.distance import pdist
from dms_variants.ispline import Isplines

from ..core.base import ModelBuilder
from .variants import ModelConfig, VarianceType, SpatialEffectType


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
    config : ModelConfig, optional
        Model configuration. Defaults to ``ModelConfig()`` (homogeneous variance,
        no spatial effects).
    sampler_config : dict, optional
        MCMC sampler configuration.

    Examples
    --------
    >>> from spgdmm import spGDMM, ModelConfig, VarianceType, SpatialEffectType
    >>> config = ModelConfig(
    ...     deg=3, knots=2,
    ...     variance_type=VarianceType.HOMOGENEOUS,
    ...     spatial_effect_type=SpatialEffectType.ABS_DIFF,
    ... )
    >>> model = spGDMM(config=config)
    >>> idata = model.fit(X, y)
    """

    _model_type = "spGDMM"
    version = "1.0.0"

    def __init__(
        self,
        model_config: dict | ModelConfig | None = None,
        sampler_config: dict | None = None,
        config: ModelConfig | None = None,  # backward-compat alias
    ):
        # model_config takes precedence; config is a legacy alias
        effective = model_config if model_config is not None else config
        if isinstance(effective, ModelConfig):
            self._config = effective
        elif isinstance(effective, dict):
            self._config = ModelConfig.from_dict(effective)
        else:
            self._config = ModelConfig()

        self._config_dict = self._config.to_dict()
        self.metadata = None
        self.training_metadata = None
        self.prediction_metadata = None

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

        Transforms predictors using I-spline basis functions and computes
        pairwise dissimilarities.

        Parameters
        ----------
        X : pd.DataFrame
            Site-level data with columns [xc, yc, time_idx, predictor1, ...]
        y : pd.Series
            Pairwise Bray-Curtis dissimilarities
        """
        # Ensure X is a DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Extract components from X
        location_values = X.iloc[:, :2].values if X.shape[1] >= 2 else np.zeros((X.shape[0], 2))
        time_idxs = X.iloc[:, 2].values if X.shape[1] >= 3 else np.zeros(X.shape[0])

        # Environmental predictors
        if X.shape[1] >= 4:
            X_values = X.iloc[:, 3:].values
        else:
            X_values = np.empty((X.shape[0], 0))

        # Handle y as pre-computed condensed dissimilarities
        # Add small epsilon to avoid log(0) - zeros are handled by censored likelihood
        y_array = np.asarray(y, dtype=float)
        log_y = np.log(np.maximum(y_array, np.finfo(float).eps))

        # Store location values
        self.X = X
        self.y = log_y

        # Create metadata
        n_sites = X.shape[0]

        # Define predictor mesh for I-splines
        if X_values.shape[1] > 0:
            mesh_choice = self._config.mesh_choice
            deg = self._config.deg
            knots = self._config.knots

            # Ensure at least deg + 1 knot points for I-splines
            # Even with knots=0, we need deg + 1 = 4 points for deg=3
            n_knot_points = max(knots + 2, deg + 1)

            if mesh_choice == "percentile":
                predictor_mesh = np.percentile(
                    X_values, np.linspace(0, 100, n_knot_points), axis=0
                ).T
            elif mesh_choice == "even":
                min_vals = X_values.min(axis=0)
                max_vals = X_values.max(axis=0)
                predictor_mesh = np.linspace(
                    min_vals, max_vals, n_knot_points, axis=0
                ).T
            else:  # custom
                predictor_mesh = self._config.custom_predictor_mesh
                if predictor_mesh is None:
                    raise ValueError("custom_predictor_mesh must be provided for mesh_choice='custom'")

            # Ensure mesh values are unique and sorted
            # For predictors with non-unique values, expand to a small range
            # Note: predictor_mesh has shape (n_predictors, n_knot_points), so we access rows with [i]
            for i in range(predictor_mesh.shape[0]):
                unique_vals = np.unique(predictor_mesh[i])
                if len(unique_vals) < n_knot_points:
                    # Values are not unique, use min/max with small perturbation
                    min_val = np.min(X_values[:, i])
                    max_val = np.max(X_values[:, i])
                    if min_val == max_val:
                        # All values are the same, create a small range
                        range_val = np.finfo(float).eps * 1000
                        predictor_mesh[i] = np.linspace(
                            min_val - range_val, max_val + range_val, n_knot_points
                        )
                    else:
                        predictor_mesh[i] = np.linspace(min_val, max_val, n_knot_points)
        else:
            predictor_mesh = np.empty((2, 0))

        # Compute pairwise distance (simplified for basic model)
        distance_measure = self._config.distance_measure
        if distance_measure == "euclidean":
            pw_distance = pdist(location_values, metric="euclidean") / 1000.0
        elif distance_measure == "geodesic":
            pw_distance = pdist(location_values, lambda u, v: geodesic(u, v).kilometers)
        else:
            # Default to euclidean
            pw_distance = pdist(location_values, metric="euclidean") / 1000.0

        # Create distance mesh
        if self._config.custom_dist_mesh is not None:
            dist_mesh = self._config.custom_dist_mesh
        else:
            # Ensure at least deg + 1 knot points for I-splines
            dist_n_knot_points = max(self._config.knots + 2, self._config.deg + 1)
            dist_mesh = np.percentile(pw_distance, np.linspace(0, 100, dist_n_knot_points))

            # Ensure mesh values are unique and sorted
            unique_vals = np.unique(dist_mesh)
            if len(unique_vals) < dist_n_knot_points:
                min_val = np.min(pw_distance)
                max_val = np.max(pw_distance)
                if min_val == max_val:
                    # All distances are the same, create a small range
                    range_val = np.finfo(float).eps * 1000
                    dist_mesh = np.linspace(
                        min_val - range_val, max_val + range_val, dist_n_knot_points
                    )
                else:
                    dist_mesh = np.linspace(min_val, max_val, dist_n_knot_points)

        # Transform environmental predictors with I-splines
        if X_values.shape[1] > 0:
            n_predictors = X_values.shape[1]
            n_spline_bases = self._config.deg + self._config.knots

            Expanded_mesh = predictor_mesh
            X_expanded = X_values

            # Clip values to mesh bounds
            X_clipped = np.clip(
                X_expanded,
                Expanded_mesh[:, 0],
                Expanded_mesh[:, -1],
            )

            # Compute I-spline bases for each predictor
            I_spline_bases = np.column_stack([
                Isplines(self._config.deg, Expanded_mesh[i], X_clipped[:, i]).I(j)
                for i in range(n_predictors)
                for j in range(1, n_spline_bases + 1)
            ])
        else:
            I_spline_bases = np.empty((n_sites, 0))
            n_predictors = 0
            n_spline_bases = self._config.deg + self._config.knots

        # Compute pairwise differences of I-spline bases
        if I_spline_bases.shape[1] > 0:
            I_spline_bases_diffs = np.array([
                pdist(I_spline_bases[:, i].reshape(-1, 1), metric='euclidean')
                for i in range(I_spline_bases.shape[1])
            ]).T
        else:
            I_spline_bases_diffs = np.empty((n_sites * (n_sites - 1) // 2, 0))

        # Compute distance-spline bases
        pw_dist_clipped = np.clip(pw_distance, dist_mesh[0], dist_mesh[-1])
        dist_spline_bases = np.column_stack([
            Isplines(self._config.deg, dist_mesh, pw_dist_clipped).I(j)
            for j in range(1, n_spline_bases + 1)
        ])

        # Combine features
        X_GDM = np.column_stack([I_spline_bases_diffs, dist_spline_bases])

        # Store metadata
        # no_cols_env = number of environmental columns (F * J)
        # no_cols_total = total columns (F * J + J for distance)
        n_cols_env = I_spline_bases_diffs.shape[1] if I_spline_bases_diffs.size > 0 else 0
        n_cols_dist = dist_spline_bases.shape[1] if dist_spline_bases.size > 0 else 0

        self.metadata = {
            "no_sites_train": n_sites,
            "no_sites": n_sites,
            "no_predictors": n_predictors,
            "no_rows": X_GDM.shape[0],
            "no_cols": X_GDM.shape[1],
            "no_cols_env": n_cols_env,
            "no_cols_dist": n_cols_dist,
            "predictors": [f"pred_{i}" for i in range(n_predictors)] if n_predictors > 0 else [],
            "column_names": [f"x_{i}" for i in range(X_GDM.shape[1])],
            "X_sigma": pw_dist_clipped.reshape(-1, 1) if n_predictors > 0 else None,
            "p_sigma": 1 if n_predictors > 0 else 0,
        }

        self.training_metadata = {
            "location_values_train": location_values,
            "predictor_mesh": predictor_mesh,
            "dist_mesh": dist_mesh,
            "length_scale": np.median(pw_distance) if len(pw_distance) > 0 else 100,
            "I_spline_bases": I_spline_bases,
        }

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

        self.model_coords = {
            "obs_pair": np.arange(X_values.shape[0]),
            "predictor": self.metadata["column_names"],
            "site_train": np.arange(self.metadata["no_sites_train"]),
            "feature": self.metadata["predictors"],
            "basis_function": np.arange(1, self._config.deg + self._config.knots + 1),
        }

        # Model construction
        with pm.Model(coords=self.model_coords) as model:
            # Define mutable data containers
            X_data = pm.Data("X_data", X_values, dims=("obs_pair", "predictor"))
            log_y_data = pm.Data("log_y_data", log_y_values, dims=("obs_pair",))

            # Define priors
            beta_0 = pm.Normal("beta_0", mu=0, sigma=10)

            if self._config.alpha_importance:
                J = self._config.deg + self._config.knots
                F = len(self.metadata["predictors"])

                if F > 0:
                    # Dirichlet prior over I-spline weights
                    beta = pm.Dirichlet(
                        "beta", a=np.ones(J),
                        shape=(F, J),
                        dims=("feature", "basis_function")
                    )

                    # I-spline dot product
                    # X_data contains [env_diffs, dist_splines], env_diffs has F*J columns
                    n_cols_env = self.metadata.get("no_cols_env", F * J)
                    n_cols_dist = self.metadata.get("no_cols_dist", 0)
                    X_env = X_data[:, :n_cols_env]
                    X_reshaped = X_env.reshape((-1, F, J))
                    warped = (X_reshaped * beta[None, :, :]).sum(axis=2)

                    # Alpha importance weights
                    alpha = pm.HalfNormal("alpha", sigma=1, shape=F, dims=("feature",))
                    mu = beta_0 + pm.math.dot(warped, alpha)

                    # Add distance spline coefficients
                    # Distance splines are the remaining columns
                    if n_cols_dist > 0:
                        dist_cols = X_data[:, n_cols_env:]
                        beta_dist = pm.LogNormal("beta_dist", mu=0, sigma=1, shape=n_cols_dist)
                        mu = mu + pm.math.dot(dist_cols, beta_dist)
                else:
                    # No predictors, just distance
                    mu = beta_0
            else:
                beta = pm.LogNormal("beta", mu=0, sigma=1, shape=self.metadata["no_cols"])
                mu = beta_0 + pm.math.dot(X_data, beta)

            # Variance structure
            if self._config.variance_type == VarianceType.HOMOGENEOUS:
                sigma2 = pm.InverseGamma("sigma2", alpha=1, beta=1)
            elif self._config.variance_type == VarianceType.COVARIATE_DEPENDENT:
                if self.metadata["X_sigma"] is not None and self.metadata["X_sigma"].shape[1] > 0:
                    X_sigma = self.metadata["X_sigma"]
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
                X_sigma = self.metadata.get("X_sigma")
                sigma2 = self._config.custom_variance_fn(mu, X_sigma)
            else:
                sigma2 = pm.InverseGamma("sigma2", alpha=1, beta=1)

            # Spatial random effects
            if self._config.spatial_effect_type != SpatialEffectType.NONE:
                sig2_psi = pm.InverseGamma("sig2_psi", alpha=1, beta=1)
                location_values = self.training_metadata["location_values_train"]
                length_scale = self.training_metadata["length_scale"] / 2

                cov = sig2_psi * pm.gp.cov.Exponential(2, ls=length_scale)
                gp = pm.gp.Latent(cov_func=cov)
                psi = gp.prior("psi", X=location_values, dims=("site_train",))

                row_ind, col_ind = np.triu_indices(location_values.shape[0], k=1)

                if self._config.spatial_effect_type == SpatialEffectType.ABS_DIFF:
                    mu += pm.math.abs(psi[row_ind] - psi[col_ind])
                elif self._config.spatial_effect_type == SpatialEffectType.SQUARED_DIFF:
                    mu += pm.math.abs(psi[row_ind] - psi[col_ind]) ** 2
                elif self._config.spatial_effect_type == SpatialEffectType.CUSTOM:
                    if self._config.custom_spatial_effect_fn is None:
                        raise ValueError(
                            "spatial_effect_type=CUSTOM requires custom_spatial_effect_fn to be set in ModelConfig."
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

        location_values = X_pred.iloc[:, :2].values if isinstance(X_pred, pd.DataFrame) else X_pred[:, :2]
        X_values = X_pred.iloc[:, 3:].values if isinstance(X_pred, pd.DataFrame) else X_pred[:, 3:]

        # predictor_mesh shape: (n_predictors, n_knot_points)
        predictor_mesh = self.training_metadata["predictor_mesh"]

        X_values_clipped = np.clip(
            X_values,
            predictor_mesh[:, 0],
            predictor_mesh[:, -1],
        )

        n_clipped = np.sum(X_values_clipped != X_values)
        if n_clipped > 0:
            warnings.warn(f"{n_clipped} env values were clipped to predictor_mesh bounds.")

        NaN_mask = np.isnan(X_values_clipped)
        X_clipped_nonan = X_values_clipped[~NaN_mask.any(axis=1)]

        n_spline_bases = self._config.deg + self._config.knots
        I_spline_bases = np.column_stack([
            Isplines(self._config.deg, predictor_mesh[i], X_clipped_nonan[:, i]).I(j)
            for i in range(X_clipped_nonan.shape[1])
            for j in range(1, n_spline_bases + 1)
        ])

        I_spline_bases_full = np.full((X_values_clipped.shape[0], I_spline_bases.shape[1]), np.nan)
        I_spline_bases_full[~NaN_mask.any(axis=1)] = I_spline_bases

        if biological_space:
            return I_spline_bases_full

        I_spline_bases_diffs = np.array([
            pdist(I_spline_bases_full[:, i].reshape(-1, 1), metric="euclidean")
            for i in range(I_spline_bases_full.shape[1])
        ]).T

        pw_distance = self.pw_distance(location_values)
        pw_distance_clipped = np.clip(
            pw_distance,
            self.training_metadata["dist_mesh"][0],
            self.training_metadata["dist_mesh"][-1],
        )

        dist_predictors = np.column_stack([
            Isplines(self._config.deg, self.training_metadata["dist_mesh"], pw_distance_clipped).I(j)
            for j in range(1, n_spline_bases + 1)
        ])

        return np.column_stack([I_spline_bases_diffs, dist_predictors])

    def _predict_biological_space(
        self, X_pred: pd.DataFrame, metric: str = "median"
    ) -> xr.DataArray:
        """
        Apply posterior beta weights to I-spline transformed predictors.

        Parameters
        ----------
        X_pred : pd.DataFrame
            Site-level data. Index must be a MultiIndex with (yc, xc) levels
            for downstream unstacking in rgb_biological_space.
        metric : str, default="median"
            Posterior summary metric: "median" or "mean".

        Returns
        -------
        xr.DataArray
            Dims (time, grid_cell, feature).
        """
        if metric == "mean":
            beta_posterior_summary = self.idata.posterior.beta.mean(dim=["chain", "draw"])
        else:
            beta_posterior_summary = self.idata.posterior.beta.median(dim=["chain", "draw"])

        # In this package, beta covers only environmental predictors (no "distance" feature)
        X_pred_splined = self._transform_for_prediction(X_pred, biological_space=True)
        X_pred_splined = X_pred_splined.reshape(
            1, -1,
            beta_posterior_summary.sizes["feature"],
            beta_posterior_summary.sizes["basis_function"],
        )
        X_pred_splined_da = xr.DataArray(
            X_pred_splined,
            dims=("time", "grid_cell", "feature", "basis_function"),
            coords={
                "time": [0],
                "grid_cell": X_pred.index,
                "feature": beta_posterior_summary["feature"].values,
                "basis_function": beta_posterior_summary["basis_function"].values,
            },
        )

        out_da = (X_pred_splined_da * beta_posterior_summary).sum(dim="basis_function", skipna=False)
        return out_da

    def rgb_biological_space(
        self, X_pred: pd.DataFrame, metric: str = "median", add_idata: bool = False
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
        add_idata : bool, default=False
            Reserved for future use (not used currently).

        Returns
        -------
        xr.DataArray
            Dims (time, xc, yc, rgb) with RGB values normalised to [0, 1].
        """
        from sklearn.decomposition import PCA

        transformed_features = self._predict_biological_space(X_pred, metric=metric)

        tf = transformed_features.unstack("grid_cell")
        valid = ~tf.isnull().any(dim="feature")
        X_all = tf.where(valid).stack(sample=("time", "yc", "xc")).dropna(dim="sample")
        if X_all.sizes["sample"] == 0:
            raise ValueError("No fully observed rows available for PCA.")
        X_mat = X_all.transpose("sample", "feature").values

        pca = PCA(n_components=3).fit(X_mat)
        PC_ref = pca.transform(X_mat)
        pc_min, pc_max = PC_ref.min(axis=0), PC_ref.max(axis=0)
        pc_rng = np.where(pc_max > pc_min, pc_max - pc_min, 1.0)

        X_full = tf.transpose("time", "yc", "xc", "feature").stack(sample=("time", "yc", "xc"))
        X_valid = X_full.sel(sample=X_all["sample"]).transpose("sample", "feature").values
        PC_curr = pca.transform(X_valid)

        pcs_full = xr.DataArray(
            np.full((tf.sizes["time"], tf.sizes["yc"], tf.sizes["xc"], 3), np.nan, dtype=float),
            dims=("time", "yc", "xc", "pc"),
            coords={"time": tf["time"], "yc": tf["yc"], "xc": tf["xc"], "pc": range(3)},
        ).stack(sample=("time", "yc", "xc"))

        pcs_full.loc[dict(sample=X_all["sample"])] = xr.DataArray(
            PC_curr, dims=("sample", "pc"), coords={"sample": X_all["sample"], "pc": pcs_full["pc"]}
        )
        pcs_full = pcs_full.unstack("sample")

        pcs_norm = (
            pcs_full - xr.DataArray(pc_min, dims="pc", coords={"pc": pcs_full["pc"]})
        ) / xr.DataArray(pc_rng, dims="pc", coords={"pc": pcs_full["pc"]})
        pcs_norm = xr.where(
            xr.DataArray(pc_max == pc_min, dims="pc", coords={"pc": pcs_full["pc"]}),
            0.5,
            pcs_norm,
        )

        rgb_da = pcs_norm.assign_coords(pc=["R", "G", "B"]).rename(pc="rgb")
        return rgb_da.transpose("time", "xc", "yc", "rgb")

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

    def pw_distance(
        self, location_values: np.ndarray, distance_measure: str = "euclidean"
    ) -> np.ndarray:
        """Compute pairwise geographical distance."""
        if distance_measure == "geodesic":
            return pdist(location_values, lambda u, v: geodesic(u, v).kilometers)
        elif distance_measure == "euclidean":
            return pdist(location_values, metric="euclidean") / 1000.0
        else:
            raise ValueError(f"Unknown distance measure: {distance_measure}")


__all__ = ["spGDMM"]