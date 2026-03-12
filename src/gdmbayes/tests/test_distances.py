"""Tests for distance calculations."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from gdmbayes.distances._general import (
    DistanceCalculator,
    compute_distance_matrix,
    euclidean_distance,
    geodesic_distance,
    manhattan_distance,
)


# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------

class TestEuclideanDistance:
    def test_basic_3point(self):
        locs = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        dists = euclidean_distance(locs)
        # 0-1: 3, 0-2: 4, 1-2: 5
        np.testing.assert_allclose(dists, [3.0, 4.0, 5.0])

    def test_condensed_length(self):
        n = 5
        locs = np.random.RandomState(0).rand(n, 2)
        dists = euclidean_distance(locs)
        assert len(dists) == n * (n - 1) // 2

    def test_matches_scipy(self):
        locs = np.random.RandomState(1).rand(8, 2)
        np.testing.assert_allclose(euclidean_distance(locs), pdist(locs, metric="euclidean"))

    def test_identical_points_zero(self):
        locs = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        dists = euclidean_distance(locs)
        assert dists[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# geodesic_distance
# ---------------------------------------------------------------------------

class TestGeodesicDistance:
    def test_oslo_bergen(self):
        # Oslo (~59.91°N, 10.75°E) to Bergen (~60.39°N, 5.32°E)
        # Great-circle (haversine) ≈ 305 km; road distance is ~480 km
        oslo = [59.91, 10.75]
        bergen = [60.39, 5.32]
        dists = geodesic_distance(np.array([oslo, bergen]))
        assert len(dists) == 1
        assert 280_000 < dists[0] < 340_000  # meters

    def test_same_point_zero(self):
        locs = np.array([[10.0, 20.0], [10.0, 20.0]])
        dists = geodesic_distance(locs)
        assert dists[0] == pytest.approx(0.0, abs=1.0)

    def test_condensed_length(self):
        n = 4
        locs = np.random.RandomState(2).rand(n, 2) * 10 + 50
        dists = geodesic_distance(locs)
        assert len(dists) == n * (n - 1) // 2

    def test_symmetry(self):
        a = np.array([59.91, 10.75])
        b = np.array([60.39, 5.32])
        d_ab = geodesic_distance(np.array([a, b]))[0]
        d_ba = geodesic_distance(np.array([b, a]))[0]
        assert d_ab == pytest.approx(d_ba, rel=1e-10)


# ---------------------------------------------------------------------------
# manhattan_distance
# ---------------------------------------------------------------------------

class TestManhattanDistance:
    def test_basic(self):
        locs = np.array([[0.0, 0.0], [2.0, 3.0]])
        dists = manhattan_distance(locs)
        assert dists[0] == pytest.approx(5.0)

    def test_matches_scipy(self):
        locs = np.random.RandomState(3).rand(6, 2)
        np.testing.assert_allclose(manhattan_distance(locs), pdist(locs, metric="cityblock"))


# ---------------------------------------------------------------------------
# DistanceCalculator
# ---------------------------------------------------------------------------

class TestDistanceCalculator:
    def test_euclidean_default(self):
        locs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        calc = DistanceCalculator(metric="euclidean")
        dists = calc.compute(locs)
        np.testing.assert_allclose(dists, [1.0, 1.0, np.sqrt(2)])

    def test_geodesic_dispatch(self):
        locs = np.array([[59.91, 10.75], [60.39, 5.32], [55.0, 12.0]])
        calc = DistanceCalculator(metric="geodesic")
        dists = calc.compute(locs, coord_type="geodesic")
        assert len(dists) == 3
        assert all(d >= 0 for d in dists)

    def test_pdist_alias(self):
        locs = np.random.RandomState(4).rand(5, 2)
        calc = DistanceCalculator(metric="euclidean")
        np.testing.assert_allclose(calc.pdist(locs), calc.compute(locs))

    def test_squareform_shape(self):
        n = 4
        locs = np.random.RandomState(5).rand(n, 2)
        calc = DistanceCalculator(metric="euclidean")
        sq = calc.squareform(locs)
        assert sq.shape == (n, n)
        np.testing.assert_allclose(sq, sq.T)
        np.testing.assert_allclose(np.diag(sq), 0.0)

    def test_unknown_metric_raises(self):
        calc = DistanceCalculator(metric="unknown")
        locs = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="Unknown metric"):
            calc.compute(locs)


# ---------------------------------------------------------------------------
# compute_distance_matrix
# ---------------------------------------------------------------------------

class TestComputeDistanceMatrix:
    def test_euclidean_square(self):
        locs = [[0, 0], [1, 0], [0, 1]]
        mat = compute_distance_matrix(locs)
        assert mat.shape == (3, 3)
        np.testing.assert_allclose(np.diag(mat), 0.0)
        assert mat[0, 1] == pytest.approx(1.0)

    def test_geodesic(self):
        locs = [[59.91, 10.75], [60.39, 5.32]]
        mat = compute_distance_matrix(locs, metric="geodesic", coord_type="geodesic")
        assert mat.shape == (2, 2)
        assert mat[0, 0] == pytest.approx(0.0, abs=1.0)
        assert mat[0, 1] == pytest.approx(mat[1, 0], rel=1e-10)

    def test_accepts_list_input(self):
        locs = [[0, 0], [3, 4]]
        mat = compute_distance_matrix(locs)
        assert mat[0, 1] == pytest.approx(5.0)
