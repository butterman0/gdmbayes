"""
spGDMM: Spatial Generalized Dissimilarity Mixed Model.

A Python package for modeling ecological dissimilarities using spatial
and environmental predictors with Bayesian inference, compatible with
the R GDM package input/output format.
"""

from . import _version  # noqa: F401

__version__ = _version.__version__

# Core model
from .models._spgdmm import spGDMM

# GDM-compatible interface
from .models._gdm_model import GDMModel, GDMResult, gdm, gdm_transform, ispline_extract, rgb_biological_space

# Model configuration
from .models._config import ModelConfig, SamplerConfig

# Built-in variance functions
from .models._variance import (
    variance_homogeneous,
    variance_covariate_dependent,
    variance_polynomial,
    VARIANCE_FUNCTIONS,
)

# Built-in spatial effect functions
from .models._spatial import (
    spatial_abs_diff,
    spatial_squared_diff,
    SPATIAL_FUNCTIONS,
)

# Distance utilities (general, not ocean-specific)
from .distances import (
    DistanceCalculator,
    compute_distance_matrix,
    euclidean_distance,
    geodesic_distance,
)

# Backward compatibility (deprecated)
from .distances import (
    OceanPathDistance as _OceanPathDistance,
    ocean_path_distance_pdist as _ocean_path_distance_pdist,
)

# Plotting
from .plotting._plots import (
    plot_isplines,
    plot_crps_comparison,
    summarise_sampling,
    plot_ppc,
    rgb_from_biological_space,
)

# Core utilities
from .core._base import ModelBuilder
from .core._config import PreprocessorConfig

# Preprocessing
from .preprocessing._preprocessor import GDMPreprocessor

# Utilities
from .utils._format_site_pair import format_site_pair, BioFormat

__all__ = [
    # Version
    "__version__",
    # Core model
    "spGDMM",
    # GDM-compatible interface
    "GDMModel",
    "GDMResult",
    "gdm",
    "gdm_transform",
    "ispline_extract",
    "rgb_biological_space",
    # Model configuration
    "ModelConfig",
    "SamplerConfig",
    # Variance functions
    "variance_homogeneous",
    "variance_covariate_dependent",
    "variance_polynomial",
    "VARIANCE_FUNCTIONS",
    # Spatial effect functions
    "spatial_abs_diff",
    "spatial_squared_diff",
    "SPATIAL_FUNCTIONS",
    # Preprocessor
    "PreprocessorConfig",
    "GDMPreprocessor",
    # Distance utilities
    "DistanceCalculator",
    "compute_distance_matrix",
    "euclidean_distance",
    "geodesic_distance",
    # Plotting
    "plot_isplines",
    "plot_crps_comparison",
    "summarise_sampling",
    "plot_ppc",
    "rgb_from_biological_space",
    # Core
    "ModelBuilder",
    # Utilities
    "format_site_pair",
    "BioFormat",
]
