"""Ocean path distance calculations.

This module provides utilities for computing shortest ocean-path distances
between locations, accounting for land barriers using graph routing.
"""

import warnings
from typing import Optional

import numpy as np
import xarray as xr
from skimage.graph import route_through_array


class OceanPathDistance:
    """
    Calculate shortest ocean-path distances between points.

    Computes condensed pairwise shortest ocean-path distances accounting
    for land barriers using graph routing through ocean cells.

    Parameters
    ----------
    cost_array : np.ndarray
        Cost surface where 1 = ocean, np.inf = land
    spacing : float, default=800
        Grid spacing in meters
    fully_connected : bool, default=True
        Allow 8-way movement
    """

    def __init__(
        self,
        cost_array: np.ndarray,
        spacing: float = 800,
        fully_connected: bool = True,
    ):
        self.cost_array = cost_array
        self.spacing = spacing
        self.fully_connected = fully_connected

    def pdist(self, locations: np.ndarray) -> np.ndarray:
        """
        Compute condensed pairwise ocean-path distances.

        Parameters
        ----------
        locations : np.ndarray, shape (n, 2)
            Grid indices (row, col) of ocean points

        Returns
        -------
        distances : np.ndarray
            Condensed array of pairwise distances in meters
        """
        n = locations.shape[0]
        n_dists = n * (n - 1) // 2
        distances = np.full(n_dists, np.inf)

        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                start = np.round(
                    [locations[i, 1] / self.spacing - 1, locations[i, 0] / self.spacing - 1]
                ).astype(int)  # (col, row) for (x, y)
                end = np.round(
                    [locations[j, 1] / self.spacing - 1, locations[j, 0] / self.spacing - 1]
                ).astype(int)  # (col, row) for (x, y)

                if self.cost_array[tuple(start)] == np.inf:
                    warnings.warn(
                        f"Start point {start} is on land, skipping path to {end}.",
                        UserWarning,
                        stacklevel=2,
                    )
                    k += 1
                    continue
                if self.cost_array[tuple(end)] == np.inf:
                    warnings.warn(
                        f"End point {end} is on land, skipping path from {start}.",
                        UserWarning,
                        stacklevel=2,
                    )
                    k += 1
                    continue

                try:
                    _, cost = route_through_array(
                        self.cost_array,
                        start=tuple(start),
                        end=tuple(end),
                        fully_connected=self.fully_connected,
                    )
                    distances[k] = cost * self.spacing  # convert to meters
                except Exception:
                    warnings.warn(
                        f"Failed to compute path from {start} to {end}. "
                        "It may be in an enclosed area or unreachable.",
                        UserWarning,
                        stacklevel=2,
                    )
                    distances[k] = np.inf
                k += 1

        return distances


def ocean_path_distance_pdist(
    locations: np.ndarray,
    cost_array: Optional[np.ndarray] = None,
    dataset_path: Optional[str] = None,
    spacing: float = 800,
    fully_connected: bool = True,
) -> np.ndarray:
    """
    Convenience function for ocean path distance calculations.

    Computes condensed pairwise shortest ocean-path distances (like scipy's pdist).

    Parameters
    ----------
    locations : np.ndarray, shape (n, 2)
        Grid indices (row, col) of ocean points
    cost_array : np.ndarray, optional
        Cost surface where 1 = ocean, np.inf = land. If None, will be loaded from dataset_path.
    dataset_path : str, optional
        Path to NetCDF file with geospatial data for creating cost array.
        Used only if cost_array is None. Default is None (cost_array must be provided).
    spacing : float, default=800
        Grid spacing in metres
    fully_connected : bool, default=True
        Whether to allow 8-way movement

    Returns
    -------
    distances : 1D condensed array of pairwise distances (in metres)

    Examples
    --------
    >>> # Using a custom cost array
    >>> cost = np.where(ocean_mask, 1, np.inf)
    >>> distances = ocean_path_distance_pdist(locations, cost)

    >>> # Using a NetCDF dataset
    >>> distances = ocean_path_distance_pdist(locations, dataset_path="ocean.nc")
    """
    if cost_array is None:
        if dataset_path is None:
            raise ValueError(
                "Either cost_array or dataset_path must be provided. "
                "dataset_path is required when cost_array is None."
            )
        try:
            with xr.open_dataset(dataset_path) as ds:
                # Use the first variable that looks like it has spatial data
                # Preference order: vorticity, temperature, salinity, etc.
                var_names = ["vorticity", "temp", "temperature", "salinity", "u", "v"]
                for var_name in var_names:
                    if var_name in ds.data_vars:
                        is_ocean_mask = ds[var_name].isel(time=0).notnull()
                        break
                else:
                    # Fallback: use the first 2D variable
                    for var in ds.data_vars.values():
                        if len(var.dims) >= 2:
                            is_ocean_mask = var.isel(time=0).notnull()
                            break
                    else:
                        raise ValueError("No suitable variable found in dataset for ocean mask.")

                cost_array = np.where(is_ocean_mask, 1, np.inf)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {dataset_path}: {e}") from e

    calculator = OceanPathDistance(cost_array, spacing, fully_connected)
    return calculator.pdist(locations)


__all__ = [
    "OceanPathDistance",
    "ocean_path_distance_pdist",
]