"""GDMPreprocessor: sklearn-style transformer for GDM data preprocessing.

This module extracts all data transformation logic from spGDMM into a
standalone, composable transformer that follows the sklearn estimator API.
"""

import json
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from geopy.distance import geodesic
from scipy.spatial.distance import pdist
from dms_variants.ispline import Isplines
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._config import PreprocessorConfig


class GDMPreprocessor(BaseEstimator, TransformerMixin):
    """sklearn-style transformer that owns all GDM data preprocessing logic.

    Computes I-spline basis functions for environmental predictors and geographic
    distances, and produces the pairwise feature matrix consumed by spGDMM.

    Parameters
    ----------
    config : PreprocessorConfig or None, default None
        Preprocessing configuration. If None, defaults to ``PreprocessorConfig()``.

    Attributes (set after fit)
    --------------------------
    predictor_mesh_ : np.ndarray of shape (n_predictors, n_knot_points)
        Knot mesh for environmental predictor I-splines.
    dist_mesh_ : np.ndarray of shape (n_dist_knot_points,)
        Knot mesh for geographic distance I-splines.
    location_values_train_ : np.ndarray of shape (n_sites, 2)
        Site coordinates from training data.
    I_spline_bases_ : np.ndarray of shape (n_sites, n_predictors * n_spline_bases)
        Per-site I-spline basis values computed during training.
    length_scale_ : float
        Median pairwise distance used as GP spatial length scale.
    n_predictors_ : int
        Number of environmental predictors.
    n_spline_bases_ : int
        Number of I-spline basis functions per predictor (deg + knots).
    predictor_names_ : list[str]
        Names of environmental predictors from training data.
    """

    def __init__(self, config: "PreprocessorConfig | None" = None):
        self.config = config

    def _get_config(self) -> PreprocessorConfig:
        """Return the resolved config (default if None)."""
        if self.config is None:
            return PreprocessorConfig()
        return self.config

    def fit(self, X: pd.DataFrame | np.ndarray, y=None) -> "GDMPreprocessor":
        """Fit the preprocessor by computing spline meshes from training data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Site-level data with columns [xc, yc, time_idx, predictor1, ...].
        y : ignored
            Not used; present for sklearn API compatibility.

        Returns
        -------
        self
        """
        cfg = self._get_config()

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Extract components
        location_values = X.iloc[:, :2].values if X.shape[1] >= 2 else np.zeros((X.shape[0], 2))

        if X.shape[1] >= 4:
            pred_cols = X.iloc[:, 3:]
            X_values = pred_cols.values
            predictor_names = list(pred_cols.columns)
        else:
            X_values = np.empty((X.shape[0], 0))
            predictor_names = []

        n_sites = X.shape[0]
        n_spline_bases = cfg.deg + cfg.knots

        # Build predictor mesh
        if X_values.shape[1] > 0:
            n_predictors = X_values.shape[1]
            n_knot_points = max(cfg.knots + 2, cfg.deg + 1)

            if cfg.mesh_choice == "percentile":
                predictor_mesh = np.percentile(
                    X_values, np.linspace(0, 100, n_knot_points), axis=0
                ).T
            elif cfg.mesh_choice == "even":
                min_vals = X_values.min(axis=0)
                max_vals = X_values.max(axis=0)
                predictor_mesh = np.linspace(min_vals, max_vals, n_knot_points, axis=0).T
            else:  # custom
                predictor_mesh = cfg.custom_predictor_mesh
                if predictor_mesh is None:
                    raise ValueError(
                        "custom_predictor_mesh must be provided for mesh_choice='custom'"
                    )

            # Ensure mesh values are unique; expand if necessary
            for i in range(predictor_mesh.shape[0]):
                unique_vals = np.unique(predictor_mesh[i])
                if len(unique_vals) < n_knot_points:
                    min_val = np.min(X_values[:, i])
                    max_val = np.max(X_values[:, i])
                    if min_val == max_val:
                        range_val = np.finfo(float).eps * 1000
                        predictor_mesh[i] = np.linspace(
                            min_val - range_val, max_val + range_val, n_knot_points
                        )
                    else:
                        predictor_mesh[i] = np.linspace(min_val, max_val, n_knot_points)

            if not predictor_names or len(predictor_names) != n_predictors:
                predictor_names = [f"pred_{i}" for i in range(n_predictors)]
        else:
            predictor_mesh = np.empty((0, 0))
            n_predictors = 0

        # Compute pairwise distance
        pw_distance = self.pw_distance(location_values)

        # Build distance mesh
        if cfg.custom_dist_mesh is not None:
            dist_mesh = cfg.custom_dist_mesh
        else:
            dist_n_knot_points = max(cfg.knots + 2, cfg.deg + 1)
            dist_mesh = np.percentile(pw_distance, np.linspace(0, 100, dist_n_knot_points))

            unique_vals = np.unique(dist_mesh)
            if len(unique_vals) < dist_n_knot_points:
                min_val = np.min(pw_distance)
                max_val = np.max(pw_distance)
                if min_val == max_val:
                    range_val = np.finfo(float).eps * 1000
                    dist_mesh = np.linspace(
                        min_val - range_val, max_val + range_val, dist_n_knot_points
                    )
                else:
                    dist_mesh = np.linspace(min_val, max_val, dist_n_knot_points)

        # Compute per-site I-spline bases
        if X_values.shape[1] > 0:
            X_clipped = np.clip(X_values, predictor_mesh[:, 0], predictor_mesh[:, -1])
            I_spline_bases = np.column_stack([
                Isplines(cfg.deg, predictor_mesh[i], X_clipped[:, i]).I(j)
                for i in range(n_predictors)
                for j in range(1, n_spline_bases + 1)
            ])
        else:
            I_spline_bases = np.empty((n_sites, 0))

        length_scale = float(np.median(pw_distance)) if len(pw_distance) > 0 else 100.0

        # Store fitted state
        self.predictor_mesh_ = predictor_mesh
        self.dist_mesh_ = dist_mesh
        self.location_values_train_ = location_values
        self.I_spline_bases_ = I_spline_bases
        self.length_scale_ = length_scale
        self.n_predictors_ = n_predictors
        self.n_spline_bases_ = n_spline_bases
        self.predictor_names_ = predictor_names

        return self

    def transform(
        self, X: pd.DataFrame | np.ndarray, biological_space: bool = False
    ) -> np.ndarray:
        """Transform site-level predictors into the GDM feature space.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Site-level data with columns [xc, yc, time_idx, predictor1, ...].
        biological_space : bool, default False
            If True, return per-site I-spline bases (n_sites × n_features×n_basis).
            If False, return pairwise GDM feature matrix with distance splines appended.

        Returns
        -------
        np.ndarray
            Transformed feature matrix.
        """
        check_is_fitted(self)
        cfg = self._get_config()

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError("X must be a Pandas DataFrame or numpy array")

        location_values = X.iloc[:, :2].values if isinstance(X, pd.DataFrame) else X[:, :2]
        X_values = X.iloc[:, 3:].values if isinstance(X, pd.DataFrame) else X[:, 3:]

        predictor_mesh = self.predictor_mesh_
        mode = cfg.extrapolation

        # Out-of-range handling for environmental predictors
        NaN_mask_raw = np.isnan(X_values)
        valid_mask = ~NaN_mask_raw.any(axis=1)
        X_valid = X_values[valid_mask]

        if X_values.shape[1] > 0 and predictor_mesh.shape[0] > 0:
            lo, hi = predictor_mesh[:, 0], predictor_mesh[:, -1]
            out_of_range_env = (X_valid < lo) | (X_valid > hi)
            n_clipped_env = int(out_of_range_env.sum())
        else:
            out_of_range_env = np.zeros((X_valid.shape[0], X_values.shape[1]), dtype=bool)
            n_clipped_env = 0

        if n_clipped_env > 0:
            if mode == "error":
                raise ValueError(
                    f"{n_clipped_env} env values are outside predictor_mesh bounds."
                )
            elif mode == "clip":
                warnings.warn(f"{n_clipped_env} env values were clipped to predictor_mesh bounds.")

        if X_values.shape[1] > 0 and predictor_mesh.shape[0] > 0:
            lo, hi = predictor_mesh[:, 0], predictor_mesh[:, -1]
            if mode == "nan" and n_clipped_env > 0:
                X_valid_processed = X_valid.copy().astype(float)
                X_valid_processed[out_of_range_env.any(axis=1)] = np.nan
            else:
                X_valid_processed = np.clip(X_valid, lo, hi)
        else:
            X_valid_processed = X_valid.copy()

        X_values_clipped = np.where(NaN_mask_raw, np.nan, 0.0)
        X_values_clipped[valid_mask] = X_valid_processed
        X_values_clipped[~valid_mask] = np.nan

        NaN_mask = np.isnan(X_values_clipped)
        X_clipped_nonan = X_values_clipped[~NaN_mask.any(axis=1)]
        n_nan_rows = int((~valid_mask).sum())

        cfg_obj = self._get_config()
        n_spline_bases = self.n_spline_bases_

        if X_clipped_nonan.shape[1] > 0 and predictor_mesh.shape[0] > 0:
            I_spline_bases = np.column_stack([
                Isplines(cfg_obj.deg, predictor_mesh[i], X_clipped_nonan[:, i]).I(j)
                for i in range(X_clipped_nonan.shape[1])
                for j in range(1, n_spline_bases + 1)
            ])
            I_spline_bases_full = np.full(
                (X_values_clipped.shape[0], I_spline_bases.shape[1]), np.nan
            )
            I_spline_bases_full[~NaN_mask.any(axis=1)] = I_spline_bases
        else:
            I_spline_bases_full = np.empty((X_values_clipped.shape[0], 0))

        if biological_space:
            self._last_prediction_metadata = {
                "n_sites_pred": X_values.shape[0],
                "n_clipped_env": n_clipped_env,
                "n_nan_rows": n_nan_rows,
            }
            return I_spline_bases_full

        if I_spline_bases_full.shape[1] > 0:
            I_spline_bases_diffs = np.array([
                pdist(I_spline_bases_full[:, i].reshape(-1, 1), metric="euclidean")
                for i in range(I_spline_bases_full.shape[1])
            ]).T
        else:
            n_sites = X_values.shape[0]
            I_spline_bases_diffs = np.empty((n_sites * (n_sites - 1) // 2, 0))

        pw_distance = self.pw_distance(location_values)
        dist_mesh = self.dist_mesh_
        mode = cfg.extrapolation
        out_of_range_dist = (pw_distance < dist_mesh[0]) | (pw_distance > dist_mesh[-1])
        n_clipped_dist = int(out_of_range_dist.sum())

        if n_clipped_dist > 0:
            if mode == "error":
                raise ValueError(
                    f"{n_clipped_dist} pairwise distances are outside dist_mesh bounds."
                )
            elif mode == "clip":
                warnings.warn(f"{n_clipped_dist} pairwise distances were clipped to dist_mesh bounds.")

        if mode == "nan" and n_clipped_dist > 0:
            pw_distance_processed = pw_distance.astype(float).copy()
            pw_distance_processed[out_of_range_dist] = np.nan
            valid_dist = ~out_of_range_dist
            dist_predictors = np.full((len(pw_distance), n_spline_bases), np.nan)
            if valid_dist.any():
                dist_predictors[valid_dist] = np.column_stack([
                    Isplines(
                        cfg_obj.deg, dist_mesh, pw_distance_processed[valid_dist]
                    ).I(j)
                    for j in range(1, n_spline_bases + 1)
                ])
        else:
            pw_distance_processed = np.clip(pw_distance, dist_mesh[0], dist_mesh[-1])
            dist_predictors = np.column_stack([
                Isplines(cfg_obj.deg, dist_mesh, pw_distance_processed).I(j)
                for j in range(1, n_spline_bases + 1)
            ])

        self._last_prediction_metadata = {
            "n_sites_pred": X_values.shape[0],
            "n_pairs_pred": len(pw_distance),
            "n_clipped_env": n_clipped_env,
            "n_clipped_dist": n_clipped_dist,
            "n_nan_rows": n_nan_rows,
        }

        return np.column_stack([I_spline_bases_diffs, dist_predictors])

    def pw_distance(self, location_values: np.ndarray) -> np.ndarray:
        """Compute pairwise geographic distance.

        Parameters
        ----------
        location_values : np.ndarray of shape (n_sites, 2)
            Site coordinates.

        Returns
        -------
        np.ndarray
            Condensed pairwise distance vector.
        """
        cfg = self._get_config()
        distance_measure = cfg.distance_measure

        if distance_measure == "geodesic":
            # location_values columns are [xc, yc] = [lon, lat]; geopy geodesic expects (lat, lon)
            return pdist(location_values, lambda u, v: geodesic((u[1], u[0]), (v[1], v[0])).kilometers)
        elif distance_measure == "euclidean":
            return pdist(location_values, metric="euclidean") / 1000.0
        else:
            # Default to euclidean
            return pdist(location_values, metric="euclidean") / 1000.0

    def to_xarray(self) -> xr.Dataset:
        """Serialize fitted state to an xarray Dataset.

        Returns
        -------
        xr.Dataset
            Dataset containing all fitted transformation parameters, using the
            same variable names as spGDMM._save_input_params for compatibility.
        """
        check_is_fitted(self)
        ds = xr.Dataset(
            {
                "predictor_mesh": xr.DataArray(
                    self.predictor_mesh_, dims=("feature", "mesh_knot")
                ),
                "dist_mesh": xr.DataArray(self.dist_mesh_, dims=("dist_knot",)),
                "location_values_train": xr.DataArray(
                    self.location_values_train_, dims=("site_train", "coord")
                ),
                "I_spline_bases_train": xr.DataArray(
                    self.I_spline_bases_, dims=("site_train", "spline_col")
                ),
                "length_scale": xr.DataArray(self.length_scale_),
            }
        )
        ds.attrs["predictor_names"] = json.dumps(self.predictor_names_)
        cfg = self._get_config()
        ds.attrs["deg"] = cfg.deg
        ds.attrs["knots"] = cfg.knots
        return ds

    @classmethod
    def from_xarray(
        cls, ds: xr.Dataset, config: "PreprocessorConfig | None" = None
    ) -> "GDMPreprocessor":
        """Reconstruct a fitted GDMPreprocessor from a saved xarray Dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset produced by ``to_xarray()`` or loaded from idata.constant_data.
        config : PreprocessorConfig or None
            Configuration to attach to the reconstructed preprocessor. If None,
            a default PreprocessorConfig is used (only affects future transform calls).

        Returns
        -------
        GDMPreprocessor
        """
        if config is None and "deg" in ds.attrs and "knots" in ds.attrs:
            config = PreprocessorConfig(deg=int(ds.attrs["deg"]), knots=int(ds.attrs["knots"]))
        obj = cls(config=config)
        obj.predictor_mesh_ = ds["predictor_mesh"].values
        obj.dist_mesh_ = ds["dist_mesh"].values
        obj.location_values_train_ = ds["location_values_train"].values
        obj.I_spline_bases_ = ds["I_spline_bases_train"].values
        obj.length_scale_ = float(ds["length_scale"].values)

        predictor_names_raw = ds.attrs.get("predictor_names", "[]")
        obj.predictor_names_ = json.loads(predictor_names_raw)

        obj.n_predictors_ = obj.predictor_mesh_.shape[0]
        cfg = obj._get_config()
        obj.n_spline_bases_ = cfg.deg + cfg.knots

        return obj


__all__ = ["GDMPreprocessor"]
