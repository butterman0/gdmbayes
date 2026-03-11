"""GDMModel: GDM-compatible interface for spGDMM.

This module provides a GDMModel class that interfaces with spGDMM using
the same input/output formats as the R GDM package.

The GDMModel accepts site-pair formatted data (from format_site_pair)
and returns results in R GDM-compatible format.
"""

import json
import typing as t
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import arviz as az

from .spgdmm import spGDMM
from .variants import ModelConfig, SamplerConfig, VarianceType
from ..core.config import PreprocessorConfig
from ..preprocessing.preprocessor import GDMPreprocessor

# Type alias for InferenceData
InferenceData = az.InferenceData


@dataclass
class GDMResult:
    """
    Result object matching R GDM package output format.

    Attributes
    ----------
    dataname : str
        Name of the data table used for fitting.
    geo : bool
        Whether geographic distance was included as a predictor.
    gdmdeviance : float
        Deviance of the fitted GDM model.
    nulldeviance : float
        Deviance of the null model.
    explained : float
        Percentage of null deviance explained by the fitted model.
    intercept : float
        Fitted intercept term value.
    predictors : list[str]
        Names of predictors used, ordered by turnover magnitude.
    coefficients : dict
        Dictionary of spline coefficients for each predictor.
    knots : dict
        Vector of knots for each predictor.
    splines : list[int]
        Number of I-spline basis functions for each predictor.
    creationdate : str
        Date and time of model creation.
    observed : np.ndarray
        Observed response for each site pair.
    predicted : np.ndarray
        Predicted response for each site pair (after link function).
    ecological : np.ndarray
        Linear predictor values before link function application.
    idata : InferenceData
        Full PyMC inference data object (spGDMM extension).
    """

    dataname: str
    geo: bool
    gdmdeviance: float
    nulldeviance: float
    explained: float
    intercept: float
    predictors: list[str]
    coefficients: dict
    knots: dict
    splines: list[int]
    creationdate: str
    observed: np.ndarray
    predicted: np.ndarray
    ecological: np.ndarray
    idata: InferenceData | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "dataname": self.dataname,
            "geo": self.geo,
            "gdmdeviance": self.gdmdeviance,
            "nulldeviance": self.nulldeviance,
            "explained": self.explained,
            "intercept": self.intercept,
            "predictors": self.predictors,
            "coefficients": {k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in self.coefficients.items()},
            "knots": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.knots.items()},
            "splines": self.splines,
            "creationdate": self.creationdate,
            "observed": self.observed.tolist() if isinstance(self.observed, np.ndarray) else self.observed,
            "predicted": self.predicted.tolist() if isinstance(self.predicted, np.ndarray) else self.predicted,
            "ecological": self.ecological.tolist() if isinstance(self.ecological, np.ndarray) else self.ecological,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GDMResult":
        """Create from dictionary."""
        return cls(
            dataname=data["dataname"],
            geo=data["geo"],
            gdmdeviance=data["gdmdeviance"],
            nulldeviance=data["nulldeviance"],
            explained=data["explained"],
            intercept=data["intercept"],
            predictors=data["predictors"],
            coefficients={k: np.array(v) if isinstance(v, list) else v
                         for k, v in data["coefficients"].items()},
            knots={k: np.array(v) if isinstance(v, list) else v
                   for k, v in data["knots"].items()},
            splines=data["splines"],
            creationdate=data["creationdate"],
            observed=np.array(data["observed"]) if isinstance(data["observed"], list) else data["observed"],
            predicted=np.array(data["predicted"]) if isinstance(data["predicted"], list) else data["predicted"],
            ecological=np.array(data["ecological"]) if isinstance(data["ecological"], list) else data["ecological"],
            idata=None,
        )


class GDMModel:
    """
    GDM-compatible interface for spGDMM.

    This class provides a fitting and prediction interface that matches
    the R GDM package's gdm function for input/output format compatibility.

    Parameters
    ----------
    splines : list[int] or int, optional
        Number of I-spline basis functions per predictor. If an int,
        applies to all predictors. If None, defaults to 3 for all predictors.
    knots : list[float] or dict, optional
        Knot positions in predictor variable units. If provided with splines=None,
        length must equal predictors multiplied by knot count (default 3).
        When both splines and knots specified, knot vector length matches
        sum of splines vector values. Can be a dict mapping predictor names
        to knot vectors.
    geo : bool, default=False
        Whether to include geographic distance as a predictor term.
    s_sites : str or pd.DataFrame, optional
        Site data. If str, path to file. If DataFrame, site-level data.
    model_config : ModelConfig, optional
        spGDMM model configuration object.
    sampler_config : SamplerConfig, optional
        MCMC sampler configuration object.
    **kwargs
        Additional model configuration parameters.

    Examples
    --------
    >>> import pandas as pd
    >>> from spgdmm import format_site_pair, GDMModel
    >>>
    >>> # Create site-pair data
    >>> site_pair = format_site_pair(bio_data, bio_format=2, pred_data=pred_data)
    >>>
    >>> # Fit GDM model (with geographic distance)
    >>> gdm = GDMModel(geo=True, splines=3)
    >>> result = gdm.fit(site_pair, dataname="my_data")
    >>>
    >>> # Access results in R GDM format
    >>> print(f"Deviance explained: {result.explained:.2f}%")
    >>> print(f"Intercept: {result.intercept:.3f}")
    >>> print(f"Predictors: {result.predictors}")
    >>>
    >>> # Make predictions
    >>> predictions = gdm.predict(new_site_pair)
    """

    def __init__(
        self,
        splines: list[int] | int | None = None,
        knots: list[float] | dict[str, np.ndarray] | None = None,
        geo: bool = False,
        s_sites: str | pd.DataFrame | None = None,
        model_config: ModelConfig | None = None,
        sampler_config: SamplerConfig | None = None,
        **kwargs,
    ):
        self.splines = splines
        self.knots = knots
        self.geo = geo
        self.s_sites = s_sites

        # Get or create model config
        if model_config is not None:
            self.model_config = model_config
        else:
            self.model_config = ModelConfig()

        # Build preprocessor config from splines/knots kwargs
        if isinstance(splines, int):
            knots_count = splines - 3 if splines >= 3 else 0
        elif isinstance(splines, list):
            knots_count = max([s - 3 if s >= 3 else 0 for s in splines]) if splines else 2
        else:
            knots_count = 2

        # Allow passing preprocessor kwargs via **kwargs (distance_measure, mesh_choice, etc.)
        prep_kwarg_keys = {
            "deg", "mesh_choice", "distance_measure",
            "custom_dist_mesh", "custom_predictor_mesh", "extrapolation",
        }
        prep_kwargs = {k: v for k, v in kwargs.items() if k in prep_kwarg_keys}
        remaining_kwargs = {k: v for k, v in kwargs.items() if k not in prep_kwarg_keys}

        prep_config = PreprocessorConfig(
            deg=prep_kwargs.pop("deg", 3),
            knots=knots_count,
            **prep_kwargs,
        )
        self._preprocessor = GDMPreprocessor(config=prep_config)

        self.sampler_config = sampler_config if sampler_config else SamplerConfig()

        self._spgdmm = spGDMM(
            preprocessor=self._preprocessor,
            model_config=self.model_config,
            sampler_config=self.sampler_config.to_dict(),
        )
        self._fit_data: pd.DataFrame | None = None
        self._fit_result: GDMResult | None = None
        self._site_pair_columns: dict | None = None
        self._predictor_names: list[str] | None = None

    def fit(
        self,
        data: pd.DataFrame,
        dataname: str | None = None,
        **kwargs,
    ) -> GDMResult:
        """
        Fit the GDM model using site-pair formatted data.

        Parameters
        ----------
        data : pd.DataFrame
            Site-pair dataframe in GDM-compatible format (from format_site_pair):
            - Column 0: Bio distance (response)
            - Column 1: Weight
            - Columns 2-3: s1.xCoord, s1.yCoord
            - Columns 4-5: s2.xCoord, s2.yCoord
            - Remaining: s1.{pred}, s2.{pred} for each predictor
        dataname : str, optional
            Name for the dataset. If None, derived from data name.
        **kwargs
            Additional arguments passed to spGDMM.fit()

        Returns
        -------
        GDMResult
            Result object matching R GDM output format.

        Examples
        --------
        >>> site_pair = format_site_pair(bio_data, pred_data=pred_data)
        >>> result = gdm.fit(site_pair, dataname="my_study")
        """
        # Store raw data
        self._fit_data = data.copy()
        if dataname is None:
            dataname = "site_pair_data"
        self._dataname = dataname

        # Parse site-pair columns
        self._parse_site_pair_columns(data)

        # Extract response variable (biological distance)
        bio_distance = data.iloc[:, 0].values

        # Extract coordinates
        s1_x = data["s1.xCoord"].values
        s1_y = data["s1.yCoord"].values
        s2_x = data["s2.xCoord"].values
        s2_y = data["s2.yCoord"].values

        # Create predictor pairs for spGDMM
        X_pairs_df = self._create_predictor_dataframe(data)

        # Fit the underlying spGDMM model
        idata = self._spgdmm.fit(X_pairs_df, bio_distance, **kwargs)

        # Calculate deviances
        predicted_log = self._spgdmm.predict(X_pairs_df)
        predicted = np.exp(predicted_log)

        # Observed vs predicted for deviance calculation
        observed = bio_distance

        # Calculate null deviance (mean model)
        mean_obs = np.mean(observed[observed > 0])
        null_dev = -2 * np.sum(
            np.log(1e-10 + observed) * observed - (1e-10 + observed)
        )

        # Calculate GDM deviance using censored model
        # (log likelihood with censoring at 0)
        mu = predicted_log
        # Extract posterior sigma samples
        if "sigma2" in idata.posterior:
            sigma_samples = np.sqrt(idata.posterior["sigma2"].values)
            sigma = np.mean(sigma_samples)
        else:
            sigma = 1.0

        # Log likelihood for censored normal
        # For y > 0: log(N(y|mu, sigma))
        # For y = 0: log(Phi(0|mu, sigma))
        from scipy.stats import norm

        ll = np.where(
            observed > 0,
            norm.logpdf(observed, loc=mu, scale=sigma),
            norm.logcdf(0, loc=mu, scale=sigma)
        )
        gdm_dev = -2 * np.sum(ll)

        # Explained deviance
        explained = max(0, (null_dev - gdm_dev) / null_dev * 100) if null_dev > 0 else 0

        # Get intercept and coefficients
        intercept = float(np.mean(idata.posterior["beta_0"].values))

        coefficients = {}
        variance_type = self.model_config.variance_type

        if "alpha" in idata.posterior and variance_type != VarianceType.HOMOGENEOUS:
            # Importance-weighted coefficients
            alpha_samples = idata.posterior["alpha"].values
            alpha = np.mean(alpha_samples, axis=(0, 1))

            if alpha_importance := self.model_config.alpha_importance:
                # Get warped predictor values
                # Coefficients are alpha * Dirichlet-averaged beta
                # For simplicity, return alpha as importance weights
                for i, pred in enumerate(self._predictor_names):
                    coefficients[pred] = alpha[i] if i < len(alpha) else 1.0
            else:
                for i, pred in enumerate(self._predictor_names):
                    # Note: Full coefficient extraction would require
                    # storing the beta posterior and evaluating splines
                    coefficients[pred] = 1.0  # Placeholder

        # Getting knots from training metadata
        knots_dict = {}
        if hasattr(self._spgdmm, 'training_metadata') and self._spgdmm.training_metadata:
            pred_mesh = self._spgdmm.training_metadata.get("predictor_mesh", None)
            if pred_mesh is not None and len(pred_mesh) > 0:
                for i, pred in enumerate(self._predictor_names):
                    if i < pred_mesh.shape[0]:
                        knots_dict[pred] = pred_mesh[i]

        # Get knots for distance if geo=True
        if self.geo and hasattr(self._spgdmm, 'training_metadata'):
            dist_mesh = self._spgdmm.training_metadata.get("dist_mesh", None)
            if dist_mesh is not None:
                knots_dict["geographic"] = dist_mesh

        # Determine spline counts
        if isinstance(self.splines, list):
            spline_counts = self.splines
        elif isinstance(self.splines, int):
            n_preds = len(self._predictor_names)
            if self.geo:
                n_preds += 1
            spline_counts = [self.splines] * n_preds
        else:
            n_preds = len(self._predictor_names)
            if self.geo:
                n_preds += 1
            spline_counts = [3] * n_preds

        # Rank predictors by coefficient magnitude
        pred_importance = sorted(
            coefficients.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        predictors = [p[0] for p in pred_importance]

        # Transform predicted to ecological distance (linear predictor)
        ecological = predicted_log

        # Create result object
        result = GDMResult(
            dataname=dataname,
            geo=self.geo,
            gdmdeviance=float(gdm_dev),
            nulldeviance=float(null_dev),
            explained=float(explained),
            intercept=float(intercept),
            predictors=predictors,
            coefficients=coefficients,
            knots=knots_dict,
            splines=spline_counts,
            creationdate=datetime.now().isoformat(),
            observed=observed,
            predicted=predicted,
            ecological=ecological,
            idata=idata,
        )

        self._fit_result = result
        return result

    def _parse_site_pair_columns(self, data: pd.DataFrame) -> None:
        """Parse column names from site-pair format."""
        # Expected columns: 0 bio_distance, 1 weight, 2 s1.x, 3 s1.y, 4 s2.x, 5 s2.y
        # Then: s1.{pred1}, s2.{pred1}, s1.{pred2}, s2.{pred2}, ...
        columns = data.columns.tolist()

        # Check coordinate columns exist
        if not all(col in columns for col in ["s1.xCoord", "s1.yCoord", "s2.xCoord", "s2.yCoord"]):
            raise ValueError(
                "Data must be in site-pair format with s1.xCoord, s1.yCoord, "
                "s2.xCoord, s2.yCoord columns. Use format_site_pair() to create."
            )

        # Find predictor columns (those starting with s1. or s2. that aren't coordinates)
        coord_cols = {"s1.xCoord", "s1.yCoord", "s2.xCoord", "s2.yCoord"}
        s1_cols = [c for c in columns if c.startswith("s1.") and c not in coord_cols]
        s2_cols = [c for c in columns if c.startswith("s2.") and c not in coord_cols]

        # Extract predictor names (remove "s1." prefix)
        predictors = [c.replace("s1.", "") for c in s1_cols]

        # Store column info
        self._site_pair_columns = {
            "bio_distance": columns[0],
            "weight": columns[1],
            "s1_x": "s1.xCoord",
            "s1_y": "s1.yCoord",
            "s2_x": "s2.xCoord",
            "s2_y": "s2.yCoord",
        }
        self._predictor_names = predictors

    def _create_predictor_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create predictor matrix from site-pair data for spGDMM.

        This converts the site-pair format (s1.pred, s2.pred) to
        absolute differences for each predictor.
        """
        n_pairs = len(data)
        features = []

        # Coordinate differences (geographic distance)
        s1_x = data["s1.xCoord"].values
        s1_y = data["s1.yCoord"].values
        s2_x = data["s2.xCoord"].values
        s2_y = data["s2.yCoord"].values

        # Euclidean distance between coordinates
        geo_dist = np.sqrt((s1_x - s2_x)**2 + (s1_y - s2_y)**2)

        if self.geo:
            features.append(geo_dist)

        # Add environmental predictor differences
        for pred in self._predictor_names:
            s1_pred = data[f"s1.{pred}"].values
            s2_pred = data[f"s2.{pred}"].values
            # Use absolute difference
            features.append(np.abs(s1_pred - s2_pred))

        # Also add the individual predictor values if not using geo
        if not self.geo:
            for pred in self._predictor_names:
                features.append(data[f"s1.{pred}"].values)
                features.append(data[f"s2.{pred}"].values)

        return np.column_stack(features)

    def _create_predictor_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create site-level predictor DataFrame from site-pair data for spGDMM.

        spGDMM expects site-level data (one row per site), not site-pair data.
        This method extracts unique sites and their environmental predictor values,
        then spGDMM will compute pairwise distances internally.

        Returns a DataFrame with columns xc, yc, time_idx, predictor1, ...
        matching the format expected by spGDMM.
        """
        # Extract unique sites from site-pair data
        # Site-pair data has s1.{pred} and s2.{pred} columns for each predictor
        # We need to extract unique site IDs and their predictor values

        # Get unique sites from s1 and s2 site identifiers (if available)
        # or use the coordinate pairs as site identifiers
        s1_x = data["s1.xCoord"].values
        s1_y = data["s1.yCoord"].values
        s2_x = data["s2.xCoord"].values
        s2_y = data["s2.yCoord"].values

        # Create site identifiers from coordinates
        s1_coords = list(zip(s1_x, s1_y))
        s2_coords = list(zip(s2_x, s2_y))

        # Get unique site coordinates
        all_coords = list(set(s1_coords + s2_coords))

        # Create site-level data - each unique site gets one row
        n_sites = len(all_coords)
        site_coords = np.array(all_coords)

        # For environmental predictors, we need to extract site-level values
        # This is tricky because site-pair data doesn't directly store site-level values
        # We'll use the s1 values as the site values (since s1 represents the "first" site in each pair)
        # This assumes that the site-pair data is consistent (same site always has same predictor values)

        # Get predictor values for unique sites
        site_pred_values = {}
        for pred in self._predictor_names:
            # Get unique (coord, pred_value) pairs for s1
            s1_pred = data[f"s1.{pred}"].values
            coord_pred_pairs = list(zip(s1_coords, s1_pred))
            unique_coord_pred = list(set(coord_pred_pairs))

            # For each unique coordinate, get the predictor value
            site_pred_values[pred] = {}
            for coord, pred_val in unique_coord_pred:
                site_pred_values[pred][coord] = pred_val

        # Build site-level environmental data
        env_data = []
        for coord in all_coords:
            row = []
            for pred in self._predictor_names:
                row.append(site_pred_values[pred].get(coord, 0))
            env_data.append(row)

        env_array = np.array(env_data) if env_data else np.empty((n_sites, 0))

        # spGDMM expects columns: xc, yc, time_idx, predictor1, ...
        df_dict = {
            "xCoord": site_coords[:, 0],
            "yCoord": site_coords[:, 1],
            "time_idx": np.zeros(n_sites),
        }

        # Add environmental predictors
        for i, pred in enumerate(self._predictor_names):
            df_dict[pred] = env_array[:, i]

        X_df = pd.DataFrame(df_dict)

        return X_df

    def predict(
        self,
        new_data: pd.DataFrame,
        **kwargs,
    ) -> GDMResult:
        """
        Make predictions on new site-pair data.

        Parameters
        ----------
        new_data : pd.DataFrame
            Site-pair dataframe in same format as training data.
        **kwargs
            Additional arguments passed to spGDMM.predict()

        Returns
        -------
        GDMResult
            Prediction result object with predicted values.
        """
        if self._fit_result is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # Create predictor dataframe
        X_pred = self._create_predictor_dataframe(new_data)

        # Make predictions
        pred_log = self._spgdmm.predict(X_pred, **kwargs)
        pred = np.exp(pred_log)

        # Create result object for predictions
        result = GDMResult(
            dataname=self._dataname + "_prediction",
            geo=self.geo,
            gdmdeviance=0.0,
            nulldeviance=0.0,
            explained=0.0,
            intercept=self._fit_result.intercept,
            predictors=self._fit_result.predictors,
            coefficients=self._fit_result.coefficients,
            knots=self._fit_result.knots,
            splines=self._fit_result.splines,
            creationdate=datetime.now().isoformat(),
            observed=new_data.iloc[:, 0].values,  # Observed if present
            predicted=pred,
            ecological=pred_log,
            idata=None,
        )

        return result

    @property
    def coefficients(self) -> dict:
        """Get model coefficients."""
        if self._fit_result is None:
            raise ValueError("Model must be fitted first.")
        return self._fit_result.coefficients

    @property
    def explained(self) -> float:
        """Get percent of deviance explained."""
        if self._fit_result is None:
            raise ValueError("Model must be fitted first.")
        return self._fit_result.explained


def gdm(
    data: pd.DataFrame,
    geo: bool = False,
    splines: list[int] | int | None = None,
    knots: list[float] | dict[str, np.ndarray] | None = None,
    **kwargs,
) -> GDMResult:
    """
    GDM-compatible fitting function (matches R GDM's gdm() function).

    This function provides a direct interface matching the R GDM package's
    gdm function signature for compatibility.

    Parameters
    ----------
    data : pd.DataFrame
        Site-pair dataframe created by format_site_pair().
    geo : bool, default=False
        Whether to include geographic distance as a predictor.
    splines : list[int] or int, optional
        Number of I-spline basis functions per predictor. Default is 3.
    knots : list[float] or dict, optional
        Custom knot positions per predictor.
    **kwargs
        Additional model configuration parameters.

    Returns
    -------
    GDMResult
        Fitted model result in R GDM-compatible format.

    Examples
    --------
    >>> from spgdmm import format_site_pair, gdm
    >>>
    >>> site_pair = format_site_pair(bio_data, bio_format=2, pred_data=pred_data)
    >>> result = gdm(site_pair, geo=True, splines=3)
    >>>
    >>> print(f"Explained deviance: {result.explained:.1f}%")
    """
    model = GDMModel(splines=splines, knots=knots, geo=geo, **kwargs)
    return model.fit(data)


def gdm_transform(model, newdata: pd.DataFrame) -> pd.DataFrame:
    """
    Transform site-level predictors to the biological importance scale.

    R equivalent: gdm.transform(model, data).

    Parameters
    ----------
    model : spGDMM or GDMModel
        Fitted model.
    newdata : pd.DataFrame
        Site-level data with columns [xc, yc, time_idx, predictor1, ...]
        matching the structure used during training.

    Returns
    -------
    pd.DataFrame
        I-spline-transformed predictor values per site, with columns named
        ``{predictor}_{basis}`` (e.g. ``pred_0_1``, ``pred_0_2``, ...).
    """
    spgdmm_model = model._spgdmm if isinstance(model, GDMModel) else model

    I_spline_bases = spgdmm_model._transform_for_prediction(newdata, biological_space=True)

    predictor_mesh = spgdmm_model.training_metadata["predictor_mesh"]
    n_predictors = predictor_mesh.shape[0]
    n_basis = spgdmm_model._config.deg + spgdmm_model._config.knots
    predictor_names = spgdmm_model.metadata.get(
        "predictors", [f"pred_{i}" for i in range(n_predictors)]
    )

    columns = [
        f"{pred}_{j}"
        for pred in predictor_names
        for j in range(1, n_basis + 1)
    ]

    return pd.DataFrame(I_spline_bases, index=newdata.index, columns=columns)


def ispline_extract(model) -> dict:
    """
    Extract I-spline knot positions and posterior median coefficients.

    R equivalent: isplineExtract(model).

    Parameters
    ----------
    model : spGDMM or GDMModel
        Fitted model with posterior inference data.

    Returns
    -------
    dict
        Maps each predictor name to ``{"x": knot_positions, "y": median_coefficients}``.
        Environmental predictors use names from training metadata; distance is
        returned under the key ``"distance"`` when present.
    """
    spgdmm_model = model._spgdmm if isinstance(model, GDMModel) else model

    predictor_mesh = spgdmm_model.training_metadata["predictor_mesh"]
    dist_mesh = spgdmm_model.training_metadata["dist_mesh"]
    predictor_names = spgdmm_model.metadata.get("predictors", [])

    result = {}

    if "beta" in spgdmm_model.idata.posterior:
        beta_median = spgdmm_model.idata.posterior.beta.median(dim=["chain", "draw"])
        for i, pred in enumerate(predictor_names):
            result[pred] = {
                "x": predictor_mesh[i],
                "y": beta_median.sel(feature=pred).values,
            }

    if "beta_dist" in spgdmm_model.idata.posterior:
        beta_dist_median = spgdmm_model.idata.posterior.beta_dist.median(dim=["chain", "draw"])
        result["distance"] = {
            "x": dist_mesh,
            "y": beta_dist_median.values,
        }

    return result


__all__ = ["GDMModel", "GDMResult", "gdm", "gdm_transform", "ispline_extract"]