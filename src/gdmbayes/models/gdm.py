"""GDM: Frequentist Generalised Dissimilarity Model."""

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ..preprocessor import GDMPreprocessor


class GDM(BaseEstimator, RegressorMixin):
    """Frequentist Generalised Dissimilarity Model (sklearn-compatible).

    Implements the GDM algorithm from Ferrier et al. (2007) using I-spline
    basis functions and non-negative least squares. Follows the R gdm package
    algorithm while providing a full sklearn-compatible interface.

    Parameters
    ----------
    splines : int, default 3
        Degree of the I-spline polynomial basis.
        Total basis functions per predictor = splines + knots.
    knots : int, default 4
        Number of mesh knot intervals for I-spline construction.
    geo : bool, default True
        Whether to include geographic distance as a predictor.
        Default matches R gdm.

    Attributes (set after fit)
    --------------------------
    coef_ : ndarray of shape (1 + n_gdm_features,)
        NNLS coefficients: coef_[0] is the intercept (alpha_0 ≥ 0),
        remaining entries are per I-spline column, matching R gdm layout.
    intercept_ : float
        The fitted intercept coefficient (= coef_[0]).
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

    def __init__(self, splines=3, knots=4, geo=True):
        self.splines = splines
        self.knots = knots
        self.geo = geo

    def _build_preprocessor(self) -> GDMPreprocessor:
        return GDMPreprocessor(deg=self.splines, knots=self.knots)

    def _get_X_gdm(self, X: pd.DataFrame) -> np.ndarray:
        """Transform site-level X to pairwise GDM feature matrix with intercept.

        Returns a matrix whose first column is all-ones (the intercept), matching
        R gdm's Gdmlib.cpp design matrix construction (lines 695-699).
        """
        X_full = self.preprocessor_.transform(X)
        n_env = self.preprocessor_.n_predictors_ * self.preprocessor_.n_spline_bases_
        X_feat = X_full if self.geo else X_full[:, :n_env]
        return np.column_stack([np.ones(X_feat.shape[0]), X_feat])

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

        # Fit preprocessor on all sites
        self.preprocessor_ = self._build_preprocessor()
        self.preprocessor_.fit(X)

        # Build GDM feature matrix
        X_gdm_fit = self._get_X_gdm(X)
        y_fit = y

        # Iteratively reweighted NNLS matching the R gdm C algorithm.
        # Ref: Gdmlib.cpp / NNLS_Double.cpp in gdm CRAN source.
        #
        # Unit weights (w=1) for all pairs.  Jeffreys-smoothed initial estimate:
        #   eta = -log(1 - (y + 0.5) / 2)
        # avoids eta -> inf when y == 1 and gives a proper GLM starting point.
        # Each iteration fits weighted NNLS with IRLS working response and weights
        # derived from the cloglog link and binomial variance V(mu) = mu(1-mu).
        w = np.ones(len(y_fit))
        eta = -np.log1p(-((y_fit * w + 0.5) / (w + 1.0)))

        coef = np.zeros(X_gdm_fit.shape[1])
        eps = 1e-8
        max_iter = 100
        prev_dev = np.inf
        for _ in range(max_iter):
            mu = 1.0 - np.exp(-eta)
            mu = np.clip(mu, eps, 1 - eps)
            irls_w = np.sqrt(w * (1.0 - mu) / mu)      # sqrt of IRLS weight
            z = eta + (y_fit - mu) / (1.0 - mu)         # working response
            # Weighted NNLS: min ||diag(irls_w) @ (X @ coef - z)||^2
            coef, _ = nnls(X_gdm_fit * irls_w[:, None], z * irls_w)
            eta = X_gdm_fit @ coef
            # Binomial deviance
            mu2 = 1.0 - np.exp(-eta)
            mu2 = np.clip(mu2, eps, 1 - eps)
            dev = -2.0 * np.sum(
                y_fit * np.log(mu2 / (y_fit + eps))
                + (1 - y_fit) * np.log((1 - mu2) / (1 - y_fit + eps))
            )
            if np.isfinite(prev_dev) and abs(prev_dev - dev) / (abs(prev_dev) + eps) < eps:
                break
            prev_dev = dev
        self.coef_ = coef

        # Deviances using the R gdm CalcGDMDevianceDouble formula:
        # D = 2 * sum(y*log(y/mu) + (1-y)*log((1-y)/(1-mu)))
        # Null model: mu_null is the intercept-only IRLS solution = mean(y).
        def _gdm_deviance(y_vals, mu_vals):
            eps_ = 1e-8
            mu_v = np.clip(mu_vals, eps_, 1 - eps_)
            t1 = np.where(y_vals == 0, 0.0, y_vals * np.log(np.clip(y_vals, eps_, 1) / mu_v))
            t2 = np.where(y_vals == 1, 0.0,
                          (1 - y_vals) * np.log(np.clip(1 - y_vals, eps_, 1) / (1 - mu_v)))
            return 2.0 * np.sum(t1 + t2)

        mu_fit = np.clip(1.0 - np.exp(-(X_gdm_fit @ coef)), 1e-8, 1 - 1e-8)
        # Null model intercept-only IRLS (converges to mean of cloglog-predicted y)
        mu_null_val = np.clip(np.mean(y_fit), 1e-8, 1 - 1e-8)
        mu_null = np.full_like(y_fit, mu_null_val)
        null_dev = _gdm_deviance(y_fit, mu_null)
        model_dev = _gdm_deviance(y_fit, mu_fit)
        self.null_deviance_ = float(null_dev)
        self.model_deviance_ = float(model_dev)
        if null_dev > 0:
            self.explained_ = float(1.0 - model_dev / null_dev)
        else:
            self.explained_ = 0.0

        # Predictor importance: sum of coef per predictor (coef[0] is intercept)
        self.intercept_ = float(coef[0])
        n_bases = self.preprocessor_.n_spline_bases_
        pred_names = list(self.preprocessor_.predictor_names_)
        importance = {}
        for i, name in enumerate(pred_names):
            importance[name] = float(np.sum(coef[1 + i * n_bases : 1 + (i + 1) * n_bases]))
        if self.geo:
            n_env = len(pred_names) * n_bases
            importance["geo"] = float(np.sum(coef[1 + n_env:]))
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
        env_coef = self.coef_[1:1 + n_env]  # skip intercept at coef_[0]
        return I_bases * env_coef[np.newaxis, :]

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        """Fraction of null deviance explained on given data.

        Matches R gdm's ``explained`` metric: 1 - model_deviance / null_deviance.

        Parameters
        ----------
        X : pd.DataFrame
            Site-level data.
        y : ndarray of shape (n_pairs,)
            Observed pairwise dissimilarities.

        Returns
        -------
        float
            1 - model_deviance / null_deviance (binomial deviance ratio).
        """
        check_is_fitted(self)
        y = np.asarray(y, dtype=float)
        eps = 1e-8
        mu_pred = np.clip(1.0 - np.exp(-(self._get_X_gdm(X) @ self.coef_)), eps, 1 - eps)
        mu_null = np.clip(np.full_like(y, np.mean(y)), eps, 1 - eps)
        y_safe = np.clip(y, eps, 1 - eps)

        def _deviance(y_vals, mu_vals):
            t1 = np.where(y_vals < eps, 0.0, y_vals * np.log(y_vals / mu_vals))
            t2 = np.where(y_vals > 1 - eps, 0.0,
                          (1 - y_vals) * np.log((1 - y_vals) / (1 - mu_vals)))
            return 2.0 * np.sum(t1 + t2)

        null_dev = _deviance(y_safe, mu_null)
        model_dev = _deviance(y_safe, mu_pred)
        return float(1.0 - model_dev / null_dev) if null_dev > 0 else 0.0


__all__ = ["GDM"]
