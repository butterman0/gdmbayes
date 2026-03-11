"""Configuration management for spGDMM models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import numpy as np


class VarianceType(str, Enum):
    """Variance structure types for spGDMM models."""

    HOMOGENEOUS = "homogeneous"
    COVARIATE_DEPENDENT = "covariate_dependent"
    POLYNOMIAL = "polynomial"


class SpatialEffectType(str, Enum):
    """Spatial random effect types for spGDMM models."""

    NONE = "none"
    ABS_DIFF = "abs_diff"
    SQUARED_DIFF = "squared_diff"


class ModelVariant(str, Enum):
    """Pre-configured model variants for spGDMM.

    Each variant combines a variance structure with a spatial effect type.
    """

    MODEL1 = "model1"  # Homogeneous variance, no spatial effects
    MODEL2 = "model2"  # Covariate-dependent variance, no spatial effects
    MODEL3 = "model3"  # Polynomial variance, no spatial effects
    MODEL4 = "model4"  # Homogeneous variance, abs diff spatial effects
    MODEL5 = "model5"  # Covariate-dependent variance, abs diff spatial effects
    MODEL6 = "model6"  # Polynomial variance, abs diff spatial effects
    MODEL7 = "model7"  # Homogeneous variance, squared diff spatial effects
    MODEL8 = "model8"  # Covariate-dependent variance, squared diff spatial effects
    MODEL9 = "model9"  # Polynomial variance, squared diff spatial effects

    @property
    def variance_type(self) -> VarianceType:
        """Get the variance type for this variant."""
        variant_map = {
            ModelVariant.MODEL1: VarianceType.HOMOGENEOUS,
            ModelVariant.MODEL2: VarianceType.COVARIATE_DEPENDENT,
            ModelVariant.MODEL3: VarianceType.POLYNOMIAL,
            ModelVariant.MODEL4: VarianceType.HOMOGENEOUS,
            ModelVariant.MODEL5: VarianceType.COVARIATE_DEPENDENT,
            ModelVariant.MODEL6: VarianceType.POLYNOMIAL,
            ModelVariant.MODEL7: VarianceType.HOMOGENEOUS,
            ModelVariant.MODEL8: VarianceType.COVARIATE_DEPENDENT,
            ModelVariant.MODEL9: VarianceType.POLYNOMIAL,
        }
        return variant_map[self]

    @property
    def spatial_effect_type(self) -> SpatialEffectType:
        """Get the spatial effect type for this variant."""
        variant_map = {
            ModelVariant.MODEL1: SpatialEffectType.NONE,
            ModelVariant.MODEL2: SpatialEffectType.NONE,
            ModelVariant.MODEL3: SpatialEffectType.NONE,
            ModelVariant.MODEL4: SpatialEffectType.ABS_DIFF,
            ModelVariant.MODEL5: SpatialEffectType.ABS_DIFF,
            ModelVariant.MODEL6: SpatialEffectType.ABS_DIFF,
            ModelVariant.MODEL7: SpatialEffectType.SQUARED_DIFF,
            ModelVariant.MODEL8: SpatialEffectType.SQUARED_DIFF,
            ModelVariant.MODEL9: SpatialEffectType.SQUARED_DIFF,
        }
        return variant_map[self]


@dataclass(frozen=True)
class SamplerConfig:
    """MCMC sampler configuration for spGDMM models."""

    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95
    nuts_sampler: str = "nutpie"
    progressbar: bool = True
    random_seed: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "draws": self.draws,
            "tune": self.tune,
            "chains": self.chains,
            "target_accept": self.target_accept,
            "nuts_sampler": self.nuts_sampler,
            "progressbar": self.progressbar,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SamplerConfig":
        """Create from dictionary."""
        # Filter to only valid fields
        valid_fields = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


@dataclass
class ModelConfig:
    """Configuration for spGDMM model.

    This dataclass encapsulates all configuration parameters for the spGDMM model,
    including I-spline settings, distance measures, predictor settings, variance
    structure, and spatial effects.
    """

    # I-spline settings
    deg: int = 3
    knots: int = 2
    mesh_choice: Literal["percentile", "even", "custom"] = "percentile"

    # Distance settings
    distance_measure: str = "euclidean"
    custom_dist_mesh: Optional[np.ndarray] = None

    # Predictor settings
    alpha_importance: bool = True
    custom_predictor_mesh: Optional[np.ndarray] = None

    # Variance structure
    variance_type: VarianceType = VarianceType.HOMOGENEOUS

    # Spatial effects
    spatial_effect_type: SpatialEffectType = SpatialEffectType.NONE
    length_scale: Optional[float] = None

    # Additional settings
    diss_metric: str = "braycurtis"
    time_predictor: Optional[str] = None
    connected_pairs_only: bool = False
    updated_predictor_mesh: bool = True
    time_varying: bool = True
    connectivity_percentile: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary suitable for serialization."""
        return {
            "deg": self.deg,
            "knots": self.knots,
            "mesh_choice": self.mesh_choice,
            "distance_measure": self.distance_measure,
            "custom_dist_mesh": self.custom_dist_mesh.tolist() if self.custom_dist_mesh is not None else None,
            "alpha_importance": self.alpha_importance,
            "custom_predictor_mesh": self.custom_predictor_mesh.tolist() if self.custom_predictor_mesh is not None else None,
            "variance_type": self.variance_type.value if isinstance(self.variance_type, str) else self.variance_type,
            "spatial_effect_type": self.spatial_effect_type.value if isinstance(self.spatial_effect_type, str) else self.spatial_effect_type,
            "length_scale": self.length_scale,
            "diss_metric": self.diss_metric,
            "time_predictor": self.time_predictor,
            "connected_pairs_only": self.connected_pairs_only,
            "updated_predictor_mesh": self.updated_predictor_mesh,
            "time_varying": self.time_varying,
            "connectivity_percentile": self.connectivity_percentile,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create from dictionary."""
        # Convert variance_type and spatial_effect_type to enums if they're strings
        variance_type = config_dict.get("variance_type", VarianceType.HOMOGENEOUS)
        if isinstance(variance_type, str):
            variance_type = VarianceType(variance_type)

        spatial_effect_type = config_dict.get("spatial_effect_type", SpatialEffectType.NONE)
        if isinstance(spatial_effect_type, str):
            spatial_effect_type = SpatialEffectType(spatial_effect_type)

        return cls(
            deg=config_dict.get("deg", 3),
            knots=config_dict.get("knots", 2),
            mesh_choice=config_dict.get("mesh_choice", "percentile"),
            distance_measure=config_dict.get("distance_measure", "euclidean"),
            custom_dist_mesh=np.array(config_dict["custom_dist_mesh"]) if config_dict.get("custom_dist_mesh") is not None else None,
            alpha_importance=config_dict.get("alpha_importance", True),
            custom_predictor_mesh=np.array(config_dict["custom_predictor_mesh"]) if config_dict.get("custom_predictor_mesh") is not None else None,
            variance_type=variance_type,
            spatial_effect_type=spatial_effect_type,
            length_scale=config_dict.get("length_scale"),
            diss_metric=config_dict.get("diss_metric", "braycurtis"),
            time_predictor=config_dict.get("time_predictor"),
            connected_pairs_only=config_dict.get("connected_pairs_only", False),
            updated_predictor_mesh=config_dict.get("updated_predictor_mesh", True),
            time_varying=config_dict.get("time_varying", True),
            connectivity_percentile=config_dict.get("connectivity_percentile"),
        )

    @classmethod
    def from_variant(cls, variant: ModelVariant, **overrides) -> "ModelConfig":
        """Create a ModelConfig from a pre-configured ModelVariant.

        Parameters
        ----------
        variant : ModelVariant
            The model variant to use.
        **overrides : keyword arguments
            Additional configuration parameters to override the defaults.

        Returns
        -------
        ModelConfig
            A configuration object with the variant's settings.

        Examples
        --------
        >>> config = ModelConfig.from_variant(ModelVariant.MODEL1, deg=4, knots=3)
        """
        config = cls(
            variance_type=variant.variance_type,
            spatial_effect_type=variant.spatial_effect_type,
        )
        # Apply any overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                object.__setattr__(config, key, value)
        return config


__all__ = [
    "ModelVariant",
    "VarianceType",
    "SpatialEffectType",
    "ModelConfig",
    "SamplerConfig",
]