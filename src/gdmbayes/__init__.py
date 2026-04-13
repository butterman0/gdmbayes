"""
gdmbayes: Bayesian and Frequentist Generalised Dissimilarity Modelling.

A Python package for modelling ecological dissimilarities using spatial
and environmental predictors with I-spline basis functions.
"""

from . import version  # noqa: F401

__version__ = version.__version__

# Diagnostics
from .diagnostics import summarise_sampling

# Biological-space RGB maps
from .maps import rgb_biological_space, rgb_from_biological_space

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
from .plotting import crps_boxplot, plot_isplines, plot_ppc

# Preprocessing
from .preprocessor import GDMPreprocessor

# Utilities
from .utils import site_pairs

__all__ = [
    "__version__",
    # Core models
    "spGDMM",
    "GDM",
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
    # Diagnostics
    "summarise_sampling",
    # Plotting
    "plot_isplines",
    "plot_ppc",
    "crps_boxplot",
    # Maps
    "rgb_from_biological_space",
    "rgb_biological_space",
]
