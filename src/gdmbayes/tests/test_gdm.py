"""Tests for the frequentist GDM estimator."""

import numpy as np
from sklearn.base import clone

from gdmbayes import GDM, GDMPreprocessor


class TestGDM:
    """Frequentist GDM: sklearn plumbing + fit/predict/score/transform."""

    def test_fit_returns_self(self, site_data):
        X, y = site_data
        m = GDM()
        assert m.fit(X, y) is m

    def test_fit_sets_coef(self, fitted_gdm):
        assert fitted_gdm.coef_.shape[0] > 0
        assert np.all(fitted_gdm.coef_ >= 0)  # NNLS guarantees non-negative

    def test_fit_sets_predictor_importance(self, fitted_gdm):
        assert "temp" in fitted_gdm.predictor_importance_
        assert "depth" in fitted_gdm.predictor_importance_
        assert all(v >= 0 for v in fitted_gdm.predictor_importance_.values())

    def test_fit_sets_explained(self, fitted_gdm):
        assert np.isfinite(fitted_gdm.explained_)
        assert fitted_gdm.explained_ <= 1.0

    def test_fit_sets_knots(self, fitted_gdm):
        assert "temp" in fitted_gdm.knots_
        assert "depth" in fitted_gdm.knots_
        assert "geo" in fitted_gdm.knots_

    def test_predict_shape_and_range(self, fitted_gdm, site_data):
        X, _ = site_data
        preds = fitted_gdm.predict(X)
        n = len(X)
        assert preds.shape == (n * (n - 1) // 2,)
        assert np.all(preds >= 0)
        assert np.all(preds < 1)

    def test_geo_false_excludes_distance(self, site_data):
        X, y = site_data
        m = GDM(geo=False).fit(X, y)
        assert "geo" not in m.predictor_importance_

    def test_geo_true_includes_distance(self, fitted_gdm):
        assert "geo" in fitted_gdm.predictor_importance_

    def test_sklearn_feature_metadata(self, fitted_gdm, site_data):
        X, _ = site_data
        assert fitted_gdm.n_features_in_ == X.shape[1]
        np.testing.assert_array_equal(fitted_gdm.feature_names_in_, np.array(X.columns))

    def test_sklearn_clone(self):
        cloned = clone(GDM(splines=4, geo=True))
        assert cloned.splines == 4
        assert cloned.geo is True
        assert not hasattr(cloned, "coef_")

    def test_sklearn_get_params(self):
        params = GDM(splines=4, knots=3, geo=True).get_params()
        assert params == {"splines": 4, "knots": 3, "geo": True}

    def test_gdm_transform_shape(self, fitted_gdm, site_data):
        X, _ = site_data
        T = fitted_gdm.gdm_transform(X)
        n_env = fitted_gdm.preprocessor_.n_predictors_ * fitted_gdm.preprocessor_.n_spline_bases_
        assert T.shape == (len(X), n_env)

    def test_score_in_range(self, fitted_gdm, site_data):
        X, y = site_data
        assert fitted_gdm.score(X, y) <= 1.0

    def test_splines_knots_override(self, site_data):
        X, y = site_data
        m = GDM(splines=2, knots=3).fit(X, y)
        assert m.preprocessor_.n_spline_bases_ == 2 + 3

    def test_deviances_nonnegative(self, fitted_gdm):
        assert fitted_gdm.null_deviance_ >= 0
        assert fitted_gdm.model_deviance_ >= 0

    def test_site_level_cv_fit_predict(self, site_data):
        """Fit on a training-site subset and predict on held-out test sites."""
        X, y = site_data
        n_sites = len(X)
        train_sites = np.arange(15)
        test_sites = np.arange(15, n_sites)

        train_pair_idx = GDMPreprocessor.site_pairs(n_sites, train_sites)
        test_pair_idx = GDMPreprocessor.site_pairs(n_sites, test_sites)

        X_train = X.iloc[train_sites].reset_index(drop=True)
        y_train = y[train_pair_idx]
        X_test = X.iloc[test_sites].reset_index(drop=True)

        preds = GDM().fit(X_train, y_train).predict(X_test)
        expected = len(test_sites) * (len(test_sites) - 1) // 2
        assert preds.shape == (expected,)
        assert len(test_pair_idx) == expected
