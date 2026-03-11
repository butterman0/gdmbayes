"""Distance calculation utilities for Generalized Dissimilarity Modeling.

This module provides flexible distance calculation utilities supporting
multiple distance metrics for GDM modeling, similar to the R GDM package.
"""

# Distance calculator classes
from .general import (
    DistanceCalculator,
    compute_distance_matrix,
)

# Standard distance functions
from .general import (
    euclidean_distance,
    geodesic_distance,
    manhattan_distance,
    minkowski_distance,
)

# Backward compatibility - deprecated
from .general import (
    OceanPathDistance as _DeprecatedOceanPathDistance,
    ocean_path_distance_pdist as _deprecated_ocean_path_distance_pdist,
)

__all__ = [
    # Distance calculator classes
    "DistanceCalculator",
    "compute_distance_matrix",
    # Standard distance functions
    "euclidean_distance",
    "geodesic_distance",
    "manhattan_distance",
    "minkowski_distance",
    # Backward compatibility (deprecated)
    "OceanPathDistance",
    "ocean_path_distance_pdist",
]


# Deprecated aliases for backward compatibility
class OceanPathDistance(_DeprecatedOceanPathDistance):
    """Deprecated: Use DistanceCalculator instead."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "OceanPathDistance is deprecated. Use DistanceCalculator for "
            "custom distance calculations.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def ocean_path_distance_pdist(*args, **kwargs):
    """Deprecated: Use DistanceCalculator.compute() instead."""
    import warnings

    warnings.warn(
        "ocean_path_distance_pdist is deprecated. Use DistanceCalculator "
        "or compute_distance_matrix() for distance calculations.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError(
        "Ocean path distance is deprecated. "
        "Use DistanceCalculator or implement custom distance function."
    )
