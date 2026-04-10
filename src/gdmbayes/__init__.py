"""
gdmbayes: Bayesian and Frequentist Generalised Dissimilarity Modelling.

A Python package for modelling ecological dissimilarities using spatial
and environmental predictors with I-spline basis functions.
"""

from . import version  # noqa: F401

__version__ = version.__version__

# Core Bayesian model
# Model configuration
from .models.config import ModelConfig, SamplerConfig

# Frequentist GDM
from .models.gdm import GDM

# Built-in spatial effect functions
from .models.spatial import (
    SPATIAL_FUNCTIONS,
    spatial_abs_diff,
    spatial_squared_diff,
)
from .models.spgdmm import spGDMM

# Built-in variance functions
from .models.variance import (
    VARIANCE_FUNCTIONS,
    variance_covariate_dependent,
    variance_homogeneous,
    variance_polynomial,
)

# Plotting
from .plotting.plots import (
    plot_crps_comparison,
    plot_isplines,
    plot_ppc,
    rgb_biological_space,
    rgb_from_biological_space,
    summarise_sampling,
)

# Preprocessing
from .preprocessor import GDMPreprocessor

# Utilities
from .utils import site_pairs

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
    "GDMPreprocessor",
    # Utilities
    "site_pairs",
    # Plotting
    "plot_isplines",
    "plot_crps_comparison",
    "summarise_sampling",
    "plot_ppc",
    "rgb_from_biological_space",
]
