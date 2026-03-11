"""Model variant definitions for spGDMM."""

import warnings
from dataclasses import dataclass
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


# Keys that used to live in ModelConfig but now belong in PreprocessorConfig.
_PREPROCESSING_KEYS = frozenset({
    "deg", "knots", "mesh_choice", "distance_measure",
    "custom_dist_mesh", "custom_predictor_mesh", "extrapolation",
    "diss_metric", "time_predictor", "connected_pairs_only",
    "updated_predictor_mesh", "time_varying", "connectivity_percentile",
    "length_scale",
})


@dataclass
class ModelConfig:
    """Configuration for spGDMM model (Bayesian estimator settings only).

    This dataclass encapsulates the model-structure parameters for spGDMM:
    variance structure and spatial random effects.  Data-preprocessing settings
    (spline degree, knots, distance measure, etc.) now live in
    :class:`~spgdmm.core.config.PreprocessorConfig`.

    For custom variance or spatial effects, set the corresponding type to CUSTOM
    and provide a callable:

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

    # Predictor importance weighting
    alpha_importance: bool = True

    # Variance structure
    variance_type: VarianceType = VarianceType.HOMOGENEOUS
    custom_variance_fn: Optional[Callable] = None

    # Spatial effects
    spatial_effect_type: SpatialEffectType = SpatialEffectType.NONE
    custom_spatial_effect_fn: Optional[Callable] = None

    def to_dict(self) -> dict:
        """Convert to dictionary suitable for serialization.

        Note: ``custom_variance_fn`` and ``custom_spatial_effect_fn`` are not
        serialized and will be ``None`` when reloaded from a saved model.
        """
        return {
            "alpha_importance": self.alpha_importance,
            "variance_type": (
                self.variance_type.value
                if isinstance(self.variance_type, VarianceType)
                else self.variance_type
            ),
            "spatial_effect_type": (
                self.spatial_effect_type.value
                if isinstance(self.spatial_effect_type, SpatialEffectType)
                else self.spatial_effect_type
            ),
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create from dictionary.

        Deprecated preprocessing keys (``deg``, ``knots``, ``mesh_choice``, etc.)
        are silently ignored with a deprecation warning so that configs serialized
        with older versions of the library can still be loaded.
        """
        removed = _PREPROCESSING_KEYS.intersection(config_dict)
        if removed:
            warnings.warn(
                f"ModelConfig.from_dict() received deprecated preprocessing key(s) "
                f"{sorted(removed)!r}. These settings now belong in PreprocessorConfig "
                f"and are ignored here.",
                DeprecationWarning,
                stacklevel=2,
            )

        variance_type = config_dict.get("variance_type", VarianceType.HOMOGENEOUS)
        if isinstance(variance_type, str):
            variance_type = VarianceType(variance_type)

        spatial_effect_type = config_dict.get("spatial_effect_type", SpatialEffectType.NONE)
        if isinstance(spatial_effect_type, str):
            spatial_effect_type = SpatialEffectType(spatial_effect_type)

        return cls(
            alpha_importance=config_dict.get("alpha_importance", True),
            variance_type=variance_type,
            spatial_effect_type=spatial_effect_type,
        )


__all__ = [
    "VarianceType",
    "SpatialEffectType",
    "ModelConfig",
    "SamplerConfig",
]
