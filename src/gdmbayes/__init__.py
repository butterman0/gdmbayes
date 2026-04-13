"""
gdmbayes: Bayesian and Frequentist Generalised Dissimilarity Modelling.

A Python package for modelling ecological dissimilarities using spatial
and environmental predictors with I-spline basis functions.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("gdmbayes")
except PackageNotFoundError:  # pragma: no cover — editable install without metadata
    __version__ = "0.0.0+unknown"

# Biological-space RGB maps
from .maps import rgb_biological_space

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
from .plotting import (
    crps_boxplot,
    plot_isplines,
    plot_link_curve,
    plot_obs_vs_pred,
    plot_ppc,
    plot_predictor_importance,
)

# Preprocessing
from .preprocessor import GDMPreprocessor

# Cross-validation helpers
from .cv import site_pairs

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
    # Cross-validation helpers
    "site_pairs",
    # Plotting
    "plot_isplines",
    "plot_predictor_importance",
    "plot_obs_vs_pred",
    "plot_link_curve",
    "plot_ppc",
    "crps_boxplot",
    # Maps
    "rgb_biological_space",
]
