"""Tests for GDMPreprocessor."""

import numpy as np
import pandas as pd
import pytest

from gdmbayes import GDMPreprocessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_X():
    """Small site-level DataFrame (20 sites, 2 predictors)."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "xc": np.random.uniform(0, 100, n),
        "yc": np.random.uniform(0, 100, n),
        "time_idx": np.zeros(n, dtype=int),
        "temp": np.random.uniform(5, 20, n),
        "depth": np.random.uniform(0, 200, n),
    })


# ---------------------------------------------------------------------------
# TestGDMPreprocessor
# ---------------------------------------------------------------------------

class TestGDMPreprocessor:
    """Tests for GDMPreprocessor class."""

    def test_defaults(self):
        prep = GDMPreprocessor()
        assert prep.deg == 3
        assert prep.knots == 2
        assert prep.mesh_choice == "percentile"
        assert prep.distance_measure == "euclidean"
        assert prep.extrapolation == "clip"
        assert prep.custom_dist_mesh is None
        assert prep.custom_predictor_mesh is None

    def test_fit_sets_attributes(self, sample_X):
        prep = GDMPreprocessor()
        prep.fit(sample_X)
        assert hasattr(prep, "predictor_mesh_")
        assert hasattr(prep, "dist_mesh_")
        assert hasattr(prep, "location_values_train_")
        assert hasattr(prep, "I_spline_bases_")
        assert hasattr(prep, "length_scale_")
        assert hasattr(prep, "n_predictors_")
        assert hasattr(prep, "n_spline_bases_")
        assert hasattr(prep, "predictor_names_")

    def test_fit_predictor_count(self, sample_X):
        prep = GDMPreprocessor()
        prep.fit(sample_X)
        assert prep.n_predictors_ == 2  # temp + depth
        assert prep.predictor_names_ == ["temp", "depth"]

    def test_fit_mesh_shapes(self, sample_X):
        prep = GDMPreprocessor(deg=3, knots=2)
        prep.fit(sample_X)
        n_knot_points = max(prep.knots + 2, prep.deg + 1)
        assert prep.predictor_mesh_.shape == (2, n_knot_points)
        assert prep.dist_mesh_.shape == (n_knot_points,)

    def test_fit_ispline_bases_shape(self, sample_X):
        prep = GDMPreprocessor(deg=3, knots=2)
        prep.fit(sample_X)
        n_spline_bases = prep.deg + prep.knots
        n_sites = sample_X.shape[0]
        assert prep.I_spline_bases_.shape == (n_sites, 2 * n_spline_bases)

    def test_transform_pairwise_shape(self, sample_X):
        prep = GDMPreprocessor()
        prep.fit(sample_X)
        X_out = prep.transform(sample_X)
        n_sites = sample_X.shape[0]
        n_pairs = n_sites * (n_sites - 1) // 2
        assert X_out.shape[0] == n_pairs

    def test_transform_biological_space(self, sample_X):
        prep = GDMPreprocessor(deg=3, knots=2)
        prep.fit(sample_X)
        X_bio = prep.transform(sample_X, biological_space=True)
        n_spline_bases = prep.deg + prep.knots
        assert X_bio.shape == (sample_X.shape[0], 2 * n_spline_bases)

    def test_fit_transform(self, sample_X):
        prep = GDMPreprocessor()
        X_ft = prep.fit_transform(sample_X)
        assert X_ft.shape[0] == sample_X.shape[0] * (sample_X.shape[0] - 1) // 2

    def test_transform_before_fit_raises(self, sample_X):
        prep = GDMPreprocessor()
        with pytest.raises(Exception):
            prep.transform(sample_X)

    def test_to_xarray_and_from_xarray(self, sample_X):
        prep = GDMPreprocessor(deg=3, knots=2)
        prep.fit(sample_X)

        ds = prep.to_xarray()
        assert "predictor_mesh" in ds
        assert "dist_mesh" in ds
        assert "location_values_train" in ds
        assert "I_spline_bases_train" in ds
        assert "length_scale" in ds
        # All config fields should be in attrs
        assert ds.attrs["deg"] == 3
        assert ds.attrs["knots"] == 2
        assert ds.attrs["mesh_choice"] == "percentile"
        assert ds.attrs["distance_measure"] == "euclidean"
        assert ds.attrs["extrapolation"] == "clip"

        restored = GDMPreprocessor.from_xarray(ds)
        np.testing.assert_array_equal(restored.predictor_mesh_, prep.predictor_mesh_)
        np.testing.assert_array_equal(restored.dist_mesh_, prep.dist_mesh_)
        np.testing.assert_array_equal(restored.location_values_train_, prep.location_values_train_)
        np.testing.assert_array_equal(restored.I_spline_bases_, prep.I_spline_bases_)
        assert restored.length_scale_ == prep.length_scale_
        assert restored.predictor_names_ == prep.predictor_names_
        assert restored.deg == prep.deg
        assert restored.knots == prep.knots

    def test_from_xarray_transform_consistency(self, sample_X):
        """Restored preprocessor must produce identical transforms."""
        prep = GDMPreprocessor(deg=3, knots=2)
        prep.fit(sample_X)

        ds = prep.to_xarray()
        restored = GDMPreprocessor.from_xarray(ds)

        X_orig = prep.transform(sample_X)
        X_rest = restored.transform(sample_X)
        np.testing.assert_array_almost_equal(X_orig, X_rest)

    def test_pw_distance_euclidean(self, sample_X):
        prep = GDMPreprocessor(distance_measure="euclidean")
        locs = sample_X.iloc[:, :2].values
        dists = prep.pw_distance(locs)
        assert dists.shape == (sample_X.shape[0] * (sample_X.shape[0] - 1) // 2,)
        assert np.all(dists >= 0)

    def test_sklearn_clone(self, sample_X):
        """GDMPreprocessor should be cloneable by sklearn.clone."""
        from sklearn.base import clone
        prep = GDMPreprocessor(deg=4, knots=1)
        cloned = clone(prep)
        assert cloned.deg == 4
        assert cloned.knots == 1
