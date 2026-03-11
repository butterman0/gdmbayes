"""Tests for GDMPreprocessor and PreprocessorConfig."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from spgdmm import PreprocessorConfig, GDMPreprocessor


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
# TestPreprocessorConfig
# ---------------------------------------------------------------------------

class TestPreprocessorConfig:
    """Tests for PreprocessorConfig dataclass."""

    def test_defaults(self):
        cfg = PreprocessorConfig()
        assert cfg.deg == 3
        assert cfg.knots == 2
        assert cfg.mesh_choice == "percentile"
        assert cfg.distance_measure == "euclidean"
        assert cfg.extrapolation == "clip"
        assert cfg.custom_dist_mesh is None
        assert cfg.custom_predictor_mesh is None

    def test_to_dict(self):
        cfg = PreprocessorConfig(deg=4, knots=1, distance_measure="geodesic")
        d = cfg.to_dict()
        assert d["deg"] == 4
        assert d["knots"] == 1
        assert d["distance_measure"] == "geodesic"
        assert d["mesh_choice"] == "percentile"
        assert d["extrapolation"] == "clip"
        # Arrays are not serialized
        assert "custom_dist_mesh" not in d
        assert "custom_predictor_mesh" not in d

    def test_from_dict_round_trip(self):
        original = PreprocessorConfig(deg=2, knots=3, mesh_choice="even", extrapolation="nan")
        restored = PreprocessorConfig.from_dict(original.to_dict())
        assert restored.deg == 2
        assert restored.knots == 3
        assert restored.mesh_choice == "even"
        assert restored.extrapolation == "nan"

    def test_from_dict_defaults_on_empty(self):
        cfg = PreprocessorConfig.from_dict({})
        assert cfg.deg == 3
        assert cfg.extrapolation == "clip"


# ---------------------------------------------------------------------------
# TestGDMPreprocessor
# ---------------------------------------------------------------------------

class TestGDMPreprocessor:
    """Tests for GDMPreprocessor class."""

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
        cfg = PreprocessorConfig(deg=3, knots=2)
        prep = GDMPreprocessor(config=cfg)
        prep.fit(sample_X)
        n_knot_points = max(cfg.knots + 2, cfg.deg + 1)
        assert prep.predictor_mesh_.shape == (2, n_knot_points)
        assert prep.dist_mesh_.shape == (n_knot_points,)

    def test_fit_ispline_bases_shape(self, sample_X):
        cfg = PreprocessorConfig(deg=3, knots=2)
        prep = GDMPreprocessor(config=cfg)
        prep.fit(sample_X)
        n_spline_bases = cfg.deg + cfg.knots
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
        cfg = PreprocessorConfig(deg=3, knots=2)
        prep = GDMPreprocessor(config=cfg)
        prep.fit(sample_X)
        X_bio = prep.transform(sample_X, biological_space=True)
        n_spline_bases = cfg.deg + cfg.knots
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
        cfg = PreprocessorConfig(deg=3, knots=2)
        prep = GDMPreprocessor(config=cfg)
        prep.fit(sample_X)

        ds = prep.to_xarray()
        assert "predictor_mesh" in ds
        assert "dist_mesh" in ds
        assert "location_values_train" in ds
        assert "I_spline_bases_train" in ds
        assert "length_scale" in ds

        restored = GDMPreprocessor.from_xarray(ds, config=cfg)
        np.testing.assert_array_equal(restored.predictor_mesh_, prep.predictor_mesh_)
        np.testing.assert_array_equal(restored.dist_mesh_, prep.dist_mesh_)
        np.testing.assert_array_equal(restored.location_values_train_, prep.location_values_train_)
        np.testing.assert_array_equal(restored.I_spline_bases_, prep.I_spline_bases_)
        assert restored.length_scale_ == prep.length_scale_
        assert restored.predictor_names_ == prep.predictor_names_

    def test_from_xarray_transform_consistency(self, sample_X):
        """Restored preprocessor must produce identical transforms."""
        cfg = PreprocessorConfig(deg=3, knots=2)
        prep = GDMPreprocessor(config=cfg)
        prep.fit(sample_X)

        ds = prep.to_xarray()
        restored = GDMPreprocessor.from_xarray(ds, config=cfg)

        X_orig = prep.transform(sample_X)
        X_rest = restored.transform(sample_X)
        np.testing.assert_array_almost_equal(X_orig, X_rest)

    def test_pw_distance_euclidean(self, sample_X):
        prep = GDMPreprocessor(config=PreprocessorConfig(distance_measure="euclidean"))
        locs = sample_X.iloc[:, :2].values
        dists = prep._pw_distance(locs)
        assert dists.shape == (sample_X.shape[0] * (sample_X.shape[0] - 1) // 2,)
        assert np.all(dists >= 0)

    def test_sklearn_clone(self, sample_X):
        """GDMPreprocessor should be cloneable by sklearn.clone."""
        from sklearn.base import clone
        prep = GDMPreprocessor(config=PreprocessorConfig(deg=4, knots=1))
        cloned = clone(prep)
        assert cloned.config.deg == 4
        assert cloned.config.knots == 1
