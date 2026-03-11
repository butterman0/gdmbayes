"""
spGDMM: Spatial Generalized Dissimilarity Mixed Model.

A Python package for modeling ecological dissimilarities using spatial
and environmental predictors with Bayesian inference, compatible with
the R GDM package input/output format.
"""

from . import _version  # noqa: F401

__version__ = _version.__version__

# Core model
from .models.spgdmm import spGDMM

# GDM-compatible interface
from .models.gdm_model import GDMModel, GDMResult, gdm, gdm_transform, ispline_extract

# Model configuration
from .models.variants import (
    ModelConfig,
    VarianceType,
    SpatialEffectType,
    SamplerConfig,
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
from .plotting.plots import (
    plot_isplines,
    plot_crps_comparison,
    summarise_sampling,
    plot_ppc,
)

# Core utilities
from .core.base import ModelBuilder
from .core.config import PreprocessorConfig

# Preprocessing
from .preprocessing.preprocessor import GDMPreprocessor

# Utilities
from .utils.format_site_pair import format_site_pair, BioFormat

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
    # Model configuration
    "ModelConfig",
    "VarianceType",
    "SpatialEffectType",
    "SamplerConfig",
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
    # Core
    "ModelBuilder",
    # Utilities
    "format_site_pair",
    "BioFormat",
]
