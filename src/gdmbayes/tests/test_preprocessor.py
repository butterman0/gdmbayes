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
    return pd.DataFrame(
        {
            "xc": np.random.uniform(0, 100, n),
            "yc": np.random.uniform(0, 100, n),
            "time_idx": np.zeros(n, dtype=int),
            "temp": np.random.uniform(5, 20, n),
            "depth": np.random.uniform(0, 200, n),
        }
    )


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
        # Config fields are stored as scalar data variables (survive netcdf roundtrip)
        assert int(ds["preprocessor_deg"]) == 3
        assert int(ds["preprocessor_knots"]) == 2
        assert str(ds["preprocessor_mesh_choice"].values) == "percentile"
        assert str(ds["preprocessor_distance_measure"].values) == "euclidean"
        assert str(ds["preprocessor_extrapolation"].values) == "clip"

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

    def test_site_pairs_count(self, sample_X):
        """site_pairs returns n_subset*(n_subset-1)//2 indices for a site subset."""
        n_sites = len(sample_X)
        idx = GDMPreprocessor.site_pairs(n_sites, np.arange(13))
        assert len(idx) == 13 * 12 // 2


# ---------------------------------------------------------------------------
# Extrapolation modes on transform()
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_prep_with_X():
    """Fitted preprocessor + a known in-range X for extrapolation tests."""
    np.random.seed(0)
    n_sites = 15
    X = pd.DataFrame(
        {
            "xc": np.linspace(0, 100, n_sites),
            "yc": np.linspace(0, 100, n_sites),
            "time_idx": np.zeros(n_sites, dtype=int),
            "temp": np.linspace(5, 20, n_sites),
        }
    )
    prep = GDMPreprocessor().fit(X)
    return prep, X


def _oob_X(X_train):
    """Copy X with one site's temp pushed well above training max."""
    X_pred = X_train.copy()
    X_pred.iloc[0, 3] = X_train.iloc[:, 3].max() + 50
    return X_pred


class TestExtrapolation:
    """`GDMPreprocessor.transform` extrapolation modes: clip / error / nan."""

    def test_default_extrapolation_is_clip(self):
        assert GDMPreprocessor().extrapolation == "clip"

    def test_clip_mode_warns_and_returns_finite(self, fitted_prep_with_X):
        prep, X = fitted_prep_with_X
        with pytest.warns(UserWarning, match="clipped"):
            result = prep.transform(_oob_X(X))
        assert isinstance(result, np.ndarray)
        assert not np.isnan(result).any()

    def test_error_mode_raises_on_oob(self, fitted_prep_with_X):
        prep, X = fitted_prep_with_X
        prep.extrapolation = "error"
        with pytest.raises(ValueError, match="outside predictor_mesh bounds"):
            prep.transform(_oob_X(X))

    def test_error_mode_silent_in_range(self, fitted_prep_with_X):
        prep, X = fitted_prep_with_X
        prep.extrapolation = "error"
        prep.transform(X)  # must not raise

    def test_nan_mode_propagates_and_leaves_inrange_finite(self, fitted_prep_with_X):
        prep, X = fitted_prep_with_X
        prep.extrapolation = "nan"
        oob_result = prep.transform(_oob_X(X))
        in_result = prep.transform(X)
        assert np.isnan(oob_result).any()
        assert not np.isnan(in_result).any()


def manhattan_km(coords):
    """Module-level custom distance used by the round-trip tests."""
    from scipy.spatial.distance import pdist

    return pdist(coords, "cityblock") / 1000.0


class TestCustomDistanceRoundTrip:
    """Callable distance measures are restored by import path, or must be re-supplied."""

    def test_module_level_callable_is_restored_automatically(self, sample_X):
        prep = GDMPreprocessor(distance_measure=manhattan_km).fit(sample_X)
        ds = prep.to_xarray()
        assert str(ds["preprocessor_distance_measure"].values).endswith(":manhattan_km")
        restored = GDMPreprocessor.from_xarray(ds)
        assert restored.distance_measure is manhattan_km
        np.testing.assert_allclose(
            restored.pw_distance(prep.location_values_train_),
            prep.pw_distance(prep.location_values_train_),
        )

    def test_lambda_requires_explicit_override(self, sample_X):
        fn = lambda coords: manhattan_km(coords)  # noqa: E731
        prep = GDMPreprocessor(distance_measure=fn).fit(sample_X)
        ds = prep.to_xarray()
        assert str(ds["preprocessor_distance_measure"].values) == "custom"
        with pytest.raises(ValueError, match="distance_measure=my_function"):
            GDMPreprocessor.from_xarray(ds)
        restored = GDMPreprocessor.from_xarray(ds, distance_measure=fn)
        assert restored.distance_measure is fn

    def test_unimportable_path_gives_clear_error(self, sample_X):
        prep = GDMPreprocessor().fit(sample_X)
        ds = prep.to_xarray()
        ds["preprocessor_distance_measure"] = "no_such_module_xyz:fn"
        with pytest.raises(ValueError, match="could not be imported"):
            GDMPreprocessor.from_xarray(ds)

    def test_builtin_names_round_trip(self, sample_X):
        X_lonlat = sample_X.copy()
        X_lonlat["xc"] = X_lonlat["xc"] / 100 * 20  # keep geodesic coordinates in range
        X_lonlat["yc"] = X_lonlat["yc"] / 100 * 60
        for name in ("euclidean", "geodesic"):
            ds = GDMPreprocessor(distance_measure=name).fit(X_lonlat).to_xarray()
            assert GDMPreprocessor.from_xarray(ds).distance_measure == name
