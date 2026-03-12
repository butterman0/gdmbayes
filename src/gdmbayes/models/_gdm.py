"""GDM: Frequentist Generalised Dissimilarity Model."""

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ..preprocessing._config import PreprocessorConfig
from ..preprocessing._preprocessor import GDMPreprocessor


class GDM(BaseEstimator, RegressorMixin):
    """Frequentist Generalised Dissimilarity Model (sklearn-compatible).

    Implements the GDM algorithm from Ferrier et al. (2007) using I-spline
    basis functions and non-negative least squares. Follows the R gdm package
    algorithm while providing a full sklearn-compatible interface.

    Parameters
    ----------
    splines : int, default 3
        Degree of the I-spline polynomial basis (PreprocessorConfig.deg).
        Total basis functions per predictor = splines + knots.
    knots : int, default 4
        Number of mesh knot intervals for I-spline construction
        (PreprocessorConfig.knots).
    geo : bool, default False
        Whether to include geographic distance as a predictor.
    preprocessor_config : PreprocessorConfig or None, default None
        Override all preprocessor settings directly. When provided, takes
        precedence over splines and knots.

    Attributes (set after fit)
    --------------------------
    coef_ : ndarray of shape (n_gdm_features,)
        NNLS coefficients per I-spline column.
    predictor_importance_ : dict[str, float]
        Sum of coef_ per predictor (analogous to R gdm "importance").
    null_deviance_ : float
        SS of g about its mean (null model residual SS on link scale).
    model_deviance_ : float
        Residual SS after fitting (on link scale).
    explained_ : float
        Fraction of null deviance explained (1 - model_deviance_ / null_deviance_).
    knots_ : dict[str, ndarray]
        Fitted knot positions per predictor.
    preprocessor_ : GDMPreprocessor
        Fitted preprocessor.
    n_features_in_ : int
        Number of site-level input columns (sklearn convention).
    feature_names_in_ : ndarray
        Column names of X from fit (sklearn convention).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from scipy.spatial.distance import pdist
    >>> from gdmbayes import GDM
    >>> np.random.seed(42)
    >>> n = 20
    >>> X = pd.DataFrame({
    ...     'xc': np.random.uniform(0, 100, n), 'yc': np.random.uniform(0, 100, n),
    ...     'time_idx': np.zeros(n), 'temp': np.random.uniform(5, 20, n)
    ... })
    >>> y = pdist(np.random.exponential(1, (n, 10)), 'braycurtis').clip(1e-8, 1 - 1e-8)
    >>> m = GDM().fit(X, y)
    >>> print(m.predictor_importance_, m.explained_)
    """

    def __init__(self, splines=3, knots=4, geo=False, preprocessor_config=None):
        self.splines = splines
        self.knots = knots
        self.geo = geo
        self.preprocessor_config = preprocessor_config

    def _build_preprocessor(self) -> GDMPreprocessor:
        if self.preprocessor_config is not None:
            cfg = self.preprocessor_config
        else:
            cfg = PreprocessorConfig(deg=self.splines, knots=self.knots)
        return GDMPreprocessor(config=cfg)

    def _get_X_gdm(self, X: pd.DataFrame) -> np.ndarray:
        """Transform site-level X to pairwise GDM feature matrix."""
        X_full = self.preprocessor_.transform(X)
        n_env = self.preprocessor_.n_predictors_ * self.preprocessor_.n_spline_bases_
        if self.geo:
            return X_full  # env columns + dist columns
        return X_full[:, :n_env]  # env columns only

    def __sklearn_is_fitted__(self):
        return hasattr(self, "coef_")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GDM":
        """Fit the GDM.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_sites, n_features)
            Site-level data with columns [xc, yc, time_idx, predictor1, ...].
        y : ndarray of shape (n_pairs,)
            Condensed pairwise dissimilarities in (0, 1). Length must equal
            n_sites * (n_sites - 1) / 2.

        Returns
        -------
        self
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        y = np.asarray(y, dtype=float)

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)

        # Fit preprocessor
        self.preprocessor_ = self._build_preprocessor()
        self.preprocessor_.fit(X)

        # Build GDM feature matrix
        X_gdm = self._get_X_gdm(X)

        # Apply link: g = -log(1 - y)  (inverse cloglog)
        y_clipped = np.clip(y, 1e-10, 1 - 1e-10)
        g = -np.log1p(-y_clipped)

        # Non-negative least squares
        coef, _ = nnls(X_gdm, g)
        self.coef_ = coef

        # Deviances (SS on link scale)
        g_hat = X_gdm @ coef
        g_mean = np.mean(g)
        self.null_deviance_ = float(np.sum((g - g_mean) ** 2))
        self.model_deviance_ = float(np.sum((g - g_hat) ** 2))
        if self.null_deviance_ > 0:
            self.explained_ = float(1.0 - self.model_deviance_ / self.null_deviance_)
        else:
            self.explained_ = 0.0

        # Predictor importance: sum of coef per predictor
        n_bases = self.preprocessor_.n_spline_bases_
        pred_names = list(self.preprocessor_.predictor_names_)
        importance = {}
        for i, name in enumerate(pred_names):
            importance[name] = float(np.sum(coef[i * n_bases:(i + 1) * n_bases]))
        if self.geo:
            n_env = len(pred_names) * n_bases
            importance["geo"] = float(np.sum(coef[n_env:]))
        self.predictor_importance_ = importance

        # Knot positions per predictor
        self.knots_ = {
            name: self.preprocessor_.predictor_mesh_[i]
            for i, name in enumerate(pred_names)
        }
        if self.geo:
            self.knots_["geo"] = self.preprocessor_.dist_mesh_

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict pairwise dissimilarities for new site data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_sites, n_features)
            Site-level data with same column structure as training X.

        Returns
        -------
        ndarray of shape (n_pairs,)
            Predicted pairwise dissimilarities in [0, 1).
        """
        check_is_fitted(self)
        X_gdm = self._get_X_gdm(X)
        g_hat = X_gdm @ self.coef_
        return 1.0 - np.exp(-g_hat)

    def gdm_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Map sites to biological importance space (partial ecological distances).

        Returns per-site I-spline basis values weighted by NNLS coefficients.
        Geographic distance is excluded (it is pairwise, not site-level).

        Parameters
        ----------
        X : pd.DataFrame of shape (n_sites, n_features)
            Site-level data.

        Returns
        -------
        ndarray of shape (n_sites, n_env_features)
            Weighted I-spline basis values per site.
        """
        check_is_fitted(self)
        I_bases = self.preprocessor_.transform(X, biological_space=True)
        n_env = self.preprocessor_.n_predictors_ * self.preprocessor_.n_spline_bases_
        env_coef = self.coef_[:n_env]
        return I_bases * env_coef[np.newaxis, :]

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Fraction of null deviance explained on given data.

        Parameters
        ----------
        X : pd.DataFrame
            Site-level data.
        y : ndarray of shape (n_pairs,)
            Observed pairwise dissimilarities.

        Returns
        -------
        float
            1 - SS_res / SS_tot on the link scale.
        """
        check_is_fitted(self)
        y = np.asarray(y, dtype=float)
        y_clipped = np.clip(y, 1e-10, 1 - 1e-10)
        g = -np.log1p(-y_clipped)
        X_gdm = self._get_X_gdm(X)
        g_hat = X_gdm @ self.coef_
        ss_res = float(np.sum((g - g_hat) ** 2))
        ss_tot = float(np.sum((g - np.mean(g)) ** 2))
        return (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


__all__ = ["GDM"]
