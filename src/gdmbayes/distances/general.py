"""General distance calculation utilities.

This module provides flexible distance calculation utilities that support
multiple distance metrics, similar to R GDM's distance calculation approach.
"""

import warnings
from typing import Optional, Union

import numpy as np
from scipy.spatial.distance import pdist, squareform


class DistanceCalculator:
    """
    Flexible distance calculator for GDM modeling.

    This class provides a unified interface for computing various types of
    distances between locations, supporting:
    - Euclidean distance (straight-line)
    - Geodesic distance (great-circle, for lat/lon coordinates)
    - Manhattan distance (city-block)
    - Custom distance functions via grid-based routing

    The design follows R GDM's approach of accepting distance matrices or
    computing distances from coordinates.

    Parameters
    ----------
    metric : str, default="euclidean"
        Distance metric to use. Options:
        - "euclidean": Straight-line distance
        - "geodesic": Great-circle distance (for lat/lon coordinates)
        - "manhattan": City-block distance
        - "custom": Use a custom function
    spacing : float, default=1.0
        Grid spacing for custom distance calculations
    cost_array : np.ndarray, optional
        Cost surface for custom distance routing
    """

    def __init__(
        self,
        metric: str = "euclidean",
        spacing: float = 1.0,
        cost_array: Optional[np.ndarray] = None,
    ):
        self.metric = metric
        self.spacing = spacing
        self.cost_array = cost_array

    def compute(
        self,
        locations: np.ndarray,
        coord_type: str = "euclidean",
    ) -> np.ndarray:
        """
        Compute pairwise distances between locations.

        Parameters
        ----------
        locations : np.ndarray, shape (n, 2)
            Location coordinates. For geodesic distance, use [lat, lon] in degrees.
        coord_type : str, default="euclidean"
            Type of coordinates:
            - "euclidean": Cartesian coordinates (x, y)
            - "geodesic": Geographic coordinates (lat, lon) in degrees

        Returns
        -------
        distances : np.ndarray
            Condensed pairwise distances (upper triangular, as in scipy.pdist)
        """
        if coord_type == "geodesic" and self.metric == "geodesic":
            return geodesic_distance(locations)
        elif self.metric == "euclidean":
            return euclidean_distance(locations)
        elif self.metric == "manhattan":
            return manhattan_distance(locations)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def pdist(self, locations: np.ndarray) -> np.ndarray:
        """Compute condensed pairwise distances (alias for compute)."""
        return self.compute(locations)

    def squareform(self, locations: np.ndarray) -> np.ndarray:
        """Compute square distance matrix."""
        return squareform(self.pdist(locations))


def compute_distance_matrix(
    locations: Union[np.ndarray, list],
    metric: str = "euclidean",
    coord_type: str = "euclidean",
) -> np.ndarray:
    """
    Compute distance matrix from locations.

    This is a convenience function similar to R's dist() or scipy.spatial.distance.pdist,
    designed to match GDM's expected input format.

    Parameters
    ----------
    locations : np.ndarray or list
        Location coordinates, shape (n, 2)
    metric : str, default="euclidean"
        Distance metric. Options: "euclidean", "geodesic", "manhattan"
    coord_type : str, default="euclidean"
        Type of coordinates: "euclidean" or "geodesic"

    Returns
    -------
    distance_matrix : np.ndarray
        Square distance matrix (n, n)

    Examples
    --------
    >>> locations = [[0, 0], [1, 0], [0, 1]]
    >>> dist_mat = compute_distance_matrix(locations)
    >>> dist_mat.shape
    (3, 3)

    >>> # Geographic coordinates (degrees)
    >>> locations = [[60.0, 10.0], [60.1, 10.1], [59.9, 10.2]]
    >>> dist_mat = compute_distance_matrix(locations, coord_type="geodesic")
    """
    locations = np.asarray(locations)

    if coord_type == "geodesic" and metric == "geodesic":
        distances = geodesic_distance(locations)
    else:
        distances = pdist(locations, metric=metric)

    return squareform(distances)


def euclidean_distance(locations: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean (straight-line) distances between locations.

    Parameters
    ----------
    locations : np.ndarray, shape (n, 2)
        Location coordinates

    Returns
    -------
    distances : np.ndarray
        Condensed pairwise distances

    Examples
    --------
    >>> locs = np.array([[0, 0], [3, 0], [0, 4]])
    >>> dists = euclidean_distance(locs)
    >>> # Distances: 0-1: 3, 0-2: 4, 1-2: 5
    """
    return pdist(locations, metric="euclidean")


def geodesic_distance(locations: np.ndarray) -> np.ndarray:
    """
    Compute geodesic (great-circle) distances between geographic coordinates.

    This uses the haversine formula to compute shortest distances on a sphere.

    Parameters
    ----------
    locations : np.ndarray, shape (n, 2)
        Geographic coordinates as [latitude, longitude] in degrees

    Returns
    -------
    distances : np.ndarray
        Condensed pairwise distances in meters

    Examples
    --------
    >>> # Oslo and Bergen, Norway
    >>> oslo = [59.91, 10.75]
    >>> bergen = [60.39, 5.32]
    >>> dists = geodesic_distance([oslo, bergen])
    >>> # Result is approximately 480 km (480000 meters)
    """
    # Haversine formula implementation
    R = 6371000  # Earth's radius in meters

    locations = np.asarray(locations)
    n = locations.shape[0]

    # Convert to radians
    lat_rad = np.radians(locations[:, 0])
    lon_rad = np.radians(locations[:, 1])

    distances = np.zeros(n * (n - 1) // 2)

    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dlat = lat_rad[j] - lat_rad[i]
            dlon = lon_rad[j] - lon_rad[i]

            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat_rad[i]) * np.cos(lat_rad[j]) * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            distances[k] = R * c
            k += 1

    return distances


def manhattan_distance(locations: np.ndarray) -> np.ndarray:
    """
    Compute Manhattan (city-block) distances between locations.

    Parameters
    ----------
    locations : np.ndarray, shape (n, 2)
        Location coordinates

    Returns
    -------
    distances : np.ndarray
        Condensed pairwise distances
    """
    return pdist(locations, metric="cityblock")


def minkowski_distance(locations: np.ndarray, p: float = 2) -> np.ndarray:
    """
    Compute Minkowski distances between locations.

    Parameters
    ----------
    locations : np.ndarray, shape (n, 2)
        Location coordinates
    p : float, default=2
        Order of the Minkowski distance (p=2 is Euclidean, p=1 is Manhattan)

    Returns
    -------
    distances : np.ndarray
        Condensed pairwise distances
    """
    return pdist(locations, metric="minkowski", p=p)


# Backward compatibility - Ocean path distance (deprecated)
class OceanPathDistance:
    """Deprecated: Ocean path distance calculator.

    This class is deprecated. For grid-based custom distance calculations,
    use DistanceCalculator with a custom metric function.
    """

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "OceanPathDistance is deprecated. For custom distance calculations, "
            "use DistanceCalculator or implement a custom distance function.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.args = args
        self.kwargs = kwargs

    def pdist(self, locations):
        """Compute distances - requires grid-based routing implementation."""
        raise NotImplementedError(
            "OceanPathDistance is deprecated. "
            "Implement custom distance using DistanceCalculator."
        )


def ocean_path_distance_pdist(*args, **kwargs):
    """Deprecated: Ocean path distance calculation function."""
    import warnings

    warnings.warn(
        "ocean_path_distance_pdist is deprecated. "
        "Use DistanceCalculator for custom distance calculations.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError(
        "Ocean path distance is deprecated. "
        "Use DistanceCalculator or implement custom distance function."
    )
