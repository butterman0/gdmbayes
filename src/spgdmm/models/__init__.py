"""Model implementations for spGDMM."""

from ._spgdmm import spGDMM
from ._config import ModelConfig, SamplerConfig
from ._variance import variance_homogeneous, variance_covariate_dependent, variance_polynomial, VARIANCE_FUNCTIONS
from ._spatial import spatial_abs_diff, spatial_squared_diff, SPATIAL_FUNCTIONS
from ._gdm_model import GDMModel, GDMResult, gdm

__all__ = [
    "spGDMM",
    "ModelConfig",
    "SamplerConfig",
    "variance_homogeneous",
    "variance_covariate_dependent",
    "variance_polynomial",
    "VARIANCE_FUNCTIONS",
    "spatial_abs_diff",
    "spatial_squared_diff",
    "SPATIAL_FUNCTIONS",
    "GDMModel",
    "GDMResult",
    "gdm",
]
