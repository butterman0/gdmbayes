"""Configuration dataclasses for the spGDMM Bayesian estimator."""

from dataclasses import dataclass
from typing import Callable, Optional, Union


@dataclass
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


_VALID_VARIANCE = frozenset({"homogeneous", "covariate_dependent", "polynomial"})
_VALID_SPATIAL = frozenset({"none", "abs_diff", "squared_diff"})


@dataclass
class ModelConfig:
    """Configuration for spGDMM model (Bayesian estimator settings only).

    This dataclass encapsulates the model-structure parameters for spGDMM:
    variance structure and spatial random effects.  Data-preprocessing settings
    (spline degree, knots, distance measure, etc.) now live in
    :class:`~gdmbayes.core._config.PreprocessorConfig`.

    Parameters
    ----------
    alpha_importance : bool
        Whether to use alpha importance weighting for predictors (default True).
    variance : str or callable
        Variance structure. Built-in strings: ``"homogeneous"`` (default),
        ``"covariate_dependent"``, ``"polynomial"``.  Pass a callable for a
        custom structure: ``fn(mu, X_sigma) -> sigma2``, where ``mu`` is a
        PyTensor variable and ``X_sigma`` is an np.ndarray or None.
        See :mod:`gdmbayes.models._variance` for the built-in implementations.
    spatial_effect : str or callable
        Spatial random effect. Built-in strings: ``"none"`` (default),
        ``"abs_diff"``, ``"squared_diff"``.  Pass a callable for a custom
        effect: ``fn(psi, row_ind, col_ind) -> effect``, where ``psi`` is the
        GP latent variable and ``row_ind``/``col_ind`` are index arrays.
        See :mod:`gdmbayes.models._spatial` for the built-in implementations.

    Examples
    --------
    Built-in variance and spatial effect:

    >>> config = ModelConfig(variance="polynomial", spatial_effect="abs_diff")

    Custom variance callable:

    >>> import pymc as pm
    >>> def my_variance(mu, X_sigma):
    ...     beta_s = pm.HalfNormal("beta_s", sigma=1)
    ...     return beta_s * pm.math.exp(mu)
    >>> config = ModelConfig(variance=my_variance)

    Custom spatial effect callable:

    >>> def my_spatial(psi, row_ind, col_ind):
    ...     return pm.math.tanh(psi[row_ind] - psi[col_ind])
    >>> config = ModelConfig(spatial_effect=my_spatial)
    """

    alpha_importance: bool = True
    variance: Union[str, Callable] = "homogeneous"
    spatial_effect: Union[str, Callable] = "none"

    def __post_init__(self):
        if isinstance(self.variance, str) and self.variance not in _VALID_VARIANCE:
            raise ValueError(
                f"Unknown variance={self.variance!r}. "
                f"Valid strings: {sorted(_VALID_VARIANCE)}. "
                "Pass a callable for custom variance."
            )
        if isinstance(self.spatial_effect, str) and self.spatial_effect not in _VALID_SPATIAL:
            raise ValueError(
                f"Unknown spatial_effect={self.spatial_effect!r}. "
                f"Valid strings: {sorted(_VALID_SPATIAL)}. "
                "Pass a callable for custom spatial effect."
            )

    def to_dict(self) -> dict:
        """Convert to dictionary suitable for serialization.

        Note: callable ``variance`` and ``spatial_effect`` values are not
        serialized; they will be ``None`` when reloaded from a saved model.
        """
        return {
            "alpha_importance": self.alpha_importance,
            "variance": self.variance if isinstance(self.variance, str) else None,
            "spatial_effect": self.spatial_effect if isinstance(self.spatial_effect, str) else None,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create from dictionary."""
        variance = config_dict.get("variance", "homogeneous")
        spatial_effect = config_dict.get("spatial_effect", "none")

        return cls(
            alpha_importance=config_dict.get("alpha_importance", True),
            variance=variance,
            spatial_effect=spatial_effect,
        )


__all__ = [
    "ModelConfig",
    "SamplerConfig",
]
