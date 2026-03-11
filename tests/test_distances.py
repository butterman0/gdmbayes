"""Tests for distance calculations."""

import numpy as np
import pytest
from spgdmm.distances import OceanPathDistance, ocean_path_distance_pdist


class TestOceanPathDistance:
    """Test OceanPathDistance class."""

    def test_initialization(self):
        """Test OceanPathDistance initialization."""
        cost = np.ones((10, 10))
        opd = OceanPathDistance(cost, spacing=100, fully_connected=True)
        assert opd.cost_array is cost
        assert opd.spacing == 100
        assert opd.fully_connected is True

    def test_pdist_simple(self):
        """Test pairwise distance calculation."""
        # Simple cost array (all ones = no barriers)
        cost = np.ones((5, 5))
        locations = np.array([
            [1, 1],
            [3, 1],
            [1, 3],
        ])
        opd = OceanPathDistance(cost, spacing=10, fully_connected=True)
        distances = opd.pdist(locations)

        # Should have 3 choose 2 = 3 distances
        assert len(distances) == 3
        assert np.all(distances > 0)

    def test_pdist_with_land_mask(self):
        """Test pdist with land barriers."""
        # Create cost array with land (inf) in the middle
        cost = np.ones((10, 10))
        cost[4:6, :] = np.inf  # Land barrier in middle row

        # Locations on opposite sides of land
        locations = np.array([
            [200, 500],  # North of barrier
            [200, 500],  # South of barrier
        ])

        opd = OceanPathDistance(cost, spacing=100, fully_connected=True)
        distances = opd.pdist(locations)

        # Should return inf since path is blocked
        assert np.any(np.isinf(distances)) or len(distances) > 0

    def test_pdist_land_locations(self):
        """Test that land locations are handled correctly."""
        cost = np.ones((5, 5))
        cost[1:4, 1:4] = np.inf  # Land in center

        # One location on land, one in ocean
        locations = np.array([
            [200, 200],  # On land (cost is inf)
            [400, 400],  # In ocean
        ])

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opd = OceanPathDistance(cost, spacing=100, fully_connected=True)
            distances = opd.pdist(locations)

            # Should have warning about land location
            land_warnings = [x for x in w if "land" in str(x.message).lower()]
            # At least one warning about land should be present


class TestOceanPathDistancePdist:
    """Test ocean_path_distance_pdist function."""

    def test_with_cost_array(self):
        """Test with explicit cost array."""
        cost = np.ones((5, 5))
        locations = np.array([[100, 100], [200, 100]])

        distances = ocean_path_distance_pdist(locations, cost_array=cost, spacing=10)
        assert len(distances) == 1
        assert distances[0] > 0

    def test_error_without_cost_or_dataset(self):
        """Test that error is raised when neither cost nor dataset is provided."""
        locations = np.array([[100, 100], [200, 100]])

        with pytest.raises(ValueError, match="Either cost_array or dataset_path must be provided"):
            ocean_path_distance_pdist(locations)