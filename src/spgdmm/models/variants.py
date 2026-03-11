"""Model variant definitions for spGDMM."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Optional

import numpy as np


class VarianceType(str, Enum):
    """Variance structure types for spGDMM models."""

    HOMOGENEOUS = "homogeneous"
    COVARIATE_DEPENDENT = "covariate_dependent"
    POLYNOMIAL = "polynomial"
    CUSTOM = "custom"


class SpatialEffectType(str, Enum):
    """Spatial random effect types for spGDMM models."""

    NONE = "none"
    ABS_DIFF = "abs_diff"
    SQUARED_DIFF = "squared_diff"
    CUSTOM = "custom"


@dataclass(frozen=False)
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
        valid_fields = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


@dataclass
class ModelConfig:
    """Configuration for spGDMM model.

    This dataclass encapsulates all configuration parameters for the spGDMM model,
    including I-spline settings, distance measures, predictor settings, variance
    structure, and spatial effects.

    For custom variance or spatial effects, set the corresponding type to CUSTOM and
    provide a callable:

    - ``custom_variance_fn(mu, X_sigma) -> sigma2``: receives the linear predictor
      ``mu`` (a PyTensor variable) and ``X_sigma`` (np.ndarray or None), and must
      return a PyTensor expression for the variance ``sigma2``.

    - ``custom_spatial_effect_fn(psi, row_ind, col_ind) -> effect``: receives the
      GP latent variable ``psi``, and the upper-triangle index arrays ``row_ind`` and
      ``col_ind``, and must return a PyTensor expression to add to ``mu``.

    Examples
    --------
    Custom variance:

    >>> import pymc as pm
    >>> def my_variance(mu, X_sigma):
    ...     beta_s = pm.HalfNormal("beta_s", sigma=1)
    ...     return beta_s * pm.math.exp(mu)
    >>> config = ModelConfig(
    ...     variance_type=VarianceType.CUSTOM,
    ...     custom_variance_fn=my_variance,
    ... )

    Custom spatial effect:

    >>> def my_spatial(psi, row_ind, col_ind):
    ...     return pm.math.tanh(psi[row_ind] - psi[col_ind])
    >>> config = ModelConfig(
    ...     spatial_effect_type=SpatialEffectType.CUSTOM,
    ...     custom_spatial_effect_fn=my_spatial,
    ... )
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
    custom_variance_fn: Optional[Callable] = None

    # Spatial effects
    spatial_effect_type: SpatialEffectType = SpatialEffectType.NONE
    custom_spatial_effect_fn: Optional[Callable] = None
    length_scale: Optional[float] = None

    # Additional settings
    diss_metric: str = "braycurtis"
    time_predictor: Optional[str] = None
    connected_pairs_only: bool = False
    updated_predictor_mesh: bool = True
    time_varying: bool = True
    connectivity_percentile: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary suitable for serialization.

        Note: ``custom_variance_fn`` and ``custom_spatial_effect_fn`` are not
        serialized and will be ``None`` when reloaded from a saved model.
        """
        return {
            "deg": self.deg,
            "knots": self.knots,
            "mesh_choice": self.mesh_choice,
            "distance_measure": self.distance_measure,
            "custom_dist_mesh": self.custom_dist_mesh.tolist() if self.custom_dist_mesh is not None else None,
            "alpha_importance": self.alpha_importance,
            "custom_predictor_mesh": self.custom_predictor_mesh.tolist() if self.custom_predictor_mesh is not None else None,
            "variance_type": self.variance_type.value if isinstance(self.variance_type, VarianceType) else self.variance_type,
            "spatial_effect_type": self.spatial_effect_type.value if isinstance(self.spatial_effect_type, SpatialEffectType) else self.spatial_effect_type,
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


__all__ = [
    "VarianceType",
    "SpatialEffectType",
    "ModelConfig",
    "SamplerConfig",
]
