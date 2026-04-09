"""
gdmbayes: Bayesian and Frequentist Generalised Dissimilarity Modelling.

A Python package for modelling ecological dissimilarities using spatial
and environmental predictors with I-spline basis functions.
"""

from . import _version  # noqa: F401

__version__ = _version.__version__

# Core Bayesian model
from .models._spgdmm import spGDMM

# Frequentist GDM
from .models._gdm import GDM

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
    rgb_biological_space,
)

# Preprocessing
from .preprocessing._config import PreprocessorConfig
from .preprocessing._preprocessor import GDMPreprocessor

# Utilities
from .utils import site_pairs, holdout_pairs

__all__ = [
    # Version
    "__version__",
    # Core Bayesian model
    "spGDMM",
    # Frequentist GDM
    "GDM",
    # Plotting (re-exported)
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
    # Utilities
    "site_pairs",
    "holdout_pairs",
    # Plotting
    "plot_isplines",
    "plot_crps_comparison",
    "summarise_sampling",
    "plot_ppc",
    "rgb_from_biological_space",
]
