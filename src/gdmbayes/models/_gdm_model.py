"""GDMModel: Bayesian GDM wrapper returning GDMResult.

Provides a GDMModel class that wraps spGDMM with a sklearn-style (X, y) fit
interface and returns results in R GDM-compatible format.
"""

import json
import typing as t
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import arviz as az

from sklearn.base import BaseEstimator

from ._spgdmm import spGDMM
from ._config import ModelConfig, SamplerConfig
from ..preprocessing._config import PreprocessorConfig
from ..preprocessing._preprocessor import GDMPreprocessor

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


class GDMModel(BaseEstimator):
    """
    Bayesian GDM wrapper with sklearn-style (X, y) interface.

    Wraps spGDMM and returns GDMResult in an R GDM-compatible format.

    Parameters
    ----------
    splines : list[int] or int, optional
        Number of I-spline basis functions per predictor. If an int,
        applies to all predictors. If None, defaults to 3 for all predictors.
    knots : list[float] or dict, optional
        Knot positions in predictor variable units. Can be a dict mapping
        predictor names to knot vectors.
    geo : bool, default=False
        Whether to include geographic distance as a predictor term.
    model_config : ModelConfig, optional
        spGDMM model configuration object.
    sampler_config : SamplerConfig, optional
        MCMC sampler configuration object.
    **kwargs
        Additional preprocessor configuration parameters (distance_measure,
        mesh_choice, deg, extrapolation, etc.).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from scipy.spatial.distance import pdist
    >>> from gdmbayes import GDMModel
    >>>
    >>> # Site-level data: xc, yc, time_idx, environmental predictors
    >>> X = pd.DataFrame({...})
    >>> y = pdist(biomass, 'braycurtis')
    >>>
    >>> model = GDMModel(geo=True, splines=3)
    >>> result = model.fit(X, y, dataname="my_data")
    >>> print(f"Deviance explained: {result.explained:.2f}%")
    """

    def __init__(
        self,
        splines: list[int] | int | None = None,
        knots: list[float] | dict[str, np.ndarray] | None = None,
        geo: bool = False,
        model_config: ModelConfig | None = None,
        sampler_config: SamplerConfig | None = None,
        **kwargs,
    ):
        self.splines = splines
        self.knots = knots
        self.geo = geo

        if model_config is not None:
            self.model_config = model_config
        else:
            self.model_config = ModelConfig()

        if isinstance(splines, int):
            knots_count = splines - 3 if splines >= 3 else 0
        elif isinstance(splines, list):
            knots_count = max([s - 3 if s >= 3 else 0 for s in splines]) if splines else 2
        else:
            knots_count = 2

        prep_kwarg_keys = {
            "deg", "mesh_choice", "distance_measure",
            "custom_dist_mesh", "custom_predictor_mesh", "extrapolation",
        }
        prep_kwargs = {k: v for k, v in kwargs.items() if k in prep_kwarg_keys}

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
        self._fit_result: GDMResult | None = None
        self._predictor_names: list[str] | None = None

    def __sklearn_is_fitted__(self):
        return hasattr(self, "_fit_result") and self._fit_result is not None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        dataname: str | None = None,
        **kwargs,
    ) -> GDMResult:
        """
        Fit the Bayesian GDM model.

        Parameters
        ----------
        X : pd.DataFrame
            Site-level data with columns [xc, yc, time_idx, predictor1, ...].
        y : np.ndarray of shape (n_pairs,)
            Condensed pairwise dissimilarities.
        dataname : str, optional
            Name for the dataset.
        **kwargs
            Additional arguments passed to spGDMM.fit().

        Returns
        -------
        GDMResult
            Result object matching R GDM output format.
        """
        y = np.asarray(y, dtype=float)
        if dataname is None:
            dataname = "data"
        self._dataname = dataname

        # Predictor names: columns after xc, yc, time_idx
        self._predictor_names = list(X.columns[3:])

        # Fit the underlying spGDMM model
        self._spgdmm.fit(X, y, **kwargs)
        idata = self._spgdmm.idata_

        # Posterior mean predictions (on log scale)
        predicted_log = self._spgdmm.predict(X)
        predicted = np.exp(predicted_log)

        observed = y

        # Null/model deviance: sum of squared residuals on the *response* scale.
        # Note: GDM (frequentist) computes deviance on the cloglog link scale; these
        # values are therefore not directly comparable across the two classes.
        null_dev = float(np.sum((observed - np.mean(observed)) ** 2))
        gdm_dev = float(np.sum((observed - predicted) ** 2))

        explained = max(0.0, (null_dev - gdm_dev) / null_dev * 100) if null_dev > 0 else 0.0

        # Intercept
        intercept = float(np.mean(idata.posterior["beta_0"].values))

        # Coefficients and knots from preprocessor
        coefficients = {}
        knots_dict = {}
        prep = self._spgdmm.preprocessor
        if hasattr(prep, "predictor_mesh_"):
            pred_mesh = prep.predictor_mesh_
            if pred_mesh is not None and pred_mesh.shape[0] > 0:
                n_bases = prep.n_spline_bases_
                beta_median = (
                    idata.posterior["beta"].median(dim=["chain", "draw"]).values
                    if "beta" in idata.posterior
                    else None
                )
                for i, pred in enumerate(self._predictor_names):
                    if i < pred_mesh.shape[0]:
                        knots_dict[pred] = pred_mesh[i]
                        if beta_median is not None:
                            if beta_median.ndim == 2:
                                coefficients[pred] = beta_median[i, :].tolist()
                            else:
                                start, end = i * n_bases, (i + 1) * n_bases
                                coefficients[pred] = beta_median[start:end].tolist()
                        else:
                            coefficients[pred] = []
            dist_mesh = prep.dist_mesh_
            if self.geo and dist_mesh is not None:
                knots_dict["geographic"] = dist_mesh

        # Spline counts
        if isinstance(self.splines, list):
            spline_counts = self.splines
        elif isinstance(self.splines, int):
            n_preds = len(self._predictor_names) + (1 if self.geo else 0)
            spline_counts = [self.splines] * n_preds
        else:
            n_preds = len(self._predictor_names) + (1 if self.geo else 0)
            spline_counts = [3] * n_preds

        predictors = list(coefficients.keys())

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
            ecological=predicted_log,
            idata=idata,
        )

        self._fit_result = result
        return result

    def predict(
        self,
        X: pd.DataFrame,
        **kwargs,
    ) -> np.ndarray:
        """
        Make predictions on new site-level data.

        Parameters
        ----------
        X : pd.DataFrame
            Site-level data with same column structure as training data.
        **kwargs
            Additional arguments passed to spGDMM.predict().

        Returns
        -------
        np.ndarray
            Predicted pairwise dissimilarities.
        """
        if self._fit_result is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        pred_log = self._spgdmm.predict(X, **kwargs)
        return np.exp(pred_log)

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
    X: pd.DataFrame,
    y: np.ndarray,
    geo: bool = False,
    splines: list[int] | int | None = None,
    knots: list[float] | dict[str, np.ndarray] | None = None,
    **kwargs,
) -> GDMResult:
    """
    Bayesian GDM fitting convenience function.

    Parameters
    ----------
    X : pd.DataFrame
        Site-level data with columns [xc, yc, time_idx, predictor1, ...].
    y : np.ndarray of shape (n_pairs,)
        Condensed pairwise dissimilarities.
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
    """
    model = GDMModel(splines=splines, knots=knots, geo=geo, **kwargs)
    return model.fit(X, y)


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

    I_spline_bases = spgdmm_model.preprocessor.transform(newdata, biological_space=True)

    prep = spgdmm_model.preprocessor
    n_predictors = prep.n_predictors_
    n_basis = prep.n_spline_bases_
    predictor_names = spgdmm_model.preprocessor.predictor_names_ or [f"pred_{i}" for i in range(n_predictors)]

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

    predictor_mesh = spgdmm_model.preprocessor.predictor_mesh_
    dist_mesh = spgdmm_model.preprocessor.dist_mesh_
    predictor_names = spgdmm_model.preprocessor.predictor_names_

    result = {}

    if "beta" in spgdmm_model.idata_.posterior:
        beta_median = spgdmm_model.idata_.posterior.beta.median(dim=["chain", "draw"])
        for i, pred in enumerate(predictor_names):
            result[pred] = {
                "x": predictor_mesh[i],
                "y": beta_median.sel(feature=pred).values,
            }

    if "beta_dist" in spgdmm_model.idata_.posterior:
        beta_dist_median = spgdmm_model.idata_.posterior.beta_dist.median(dim=["chain", "draw"])
        result["distance"] = {
            "x": dist_mesh,
            "y": beta_dist_median.values,
        }

    return result


def rgb_biological_space(model, X_pred: pd.DataFrame, metric: str = "median") -> xr.DataArray:
    """
    Compute RGB biological-space map via PCA on I-spline transformed predictors.

    R equivalent: gdm.transform() + PCA + RGB colour assignment.

    Parameters
    ----------
    model : spGDMM or GDMModel
        Fitted model.
    X_pred : pd.DataFrame
        Site-level data. Index must be MultiIndex (yc, xc).
    metric : str, default="median"
        Posterior summary: "mean" or "median".

    Returns
    -------
    xr.DataArray
        Dims (time, xc, yc, rgb) with rgb in {R, G, B}, values in [0, 1].
    """
    from gdmbayes.plotting._plots import rgb_biological_space as _rgb
    spgdmm_model = model._spgdmm if isinstance(model, GDMModel) else model
    return _rgb(spgdmm_model.idata_, X_pred, metric=metric)


__all__ = ["GDMModel", "GDMResult", "gdm", "gdm_transform", "ispline_extract", "rgb_biological_space"]
