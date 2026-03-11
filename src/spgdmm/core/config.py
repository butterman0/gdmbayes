"""Configuration management for spGDMM models."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class PreprocessorConfig:
    """Configuration for GDMPreprocessor (data transformation pipeline).

    Parameters
    ----------
    deg : int, default 3
        Degree of I-spline basis functions.
    knots : int, default 2
        Number of interior knots for I-splines.
    mesh_choice : {"percentile", "even", "custom"}, default "percentile"
        Method for computing predictor mesh knot positions.
    distance_measure : str, default "euclidean"
        Geographic distance metric: "euclidean" or "geodesic".
    custom_dist_mesh : np.ndarray or None, default None
        Custom knot mesh for distance I-splines (overrides computed mesh).
    custom_predictor_mesh : np.ndarray or None, default None
        Custom knot mesh for predictor I-splines (used when mesh_choice="custom").
    extrapolation : {"clip", "error", "nan"}, default "clip"
        Controls behaviour when prediction data fall outside training mesh bounds.

        * ``"clip"``  — clamp out-of-range values to the mesh boundary and warn.
        * ``"error"`` — raise ``ValueError`` on any out-of-range value.
        * ``"nan"``   — propagate NaN for affected sites or pairs.
    """

    deg: int = 3
    knots: int = 2
    mesh_choice: Literal["percentile", "even", "custom"] = "percentile"
    distance_measure: str = "euclidean"
    custom_dist_mesh: Optional[np.ndarray] = None
    custom_predictor_mesh: Optional[np.ndarray] = None
    extrapolation: Literal["clip", "error", "nan"] = "clip"

    def to_dict(self) -> dict:
        """Convert to dictionary (numpy arrays excluded)."""
        return {
            "deg": self.deg,
            "knots": self.knots,
            "mesh_choice": self.mesh_choice,
            "distance_measure": self.distance_measure,
            "extrapolation": self.extrapolation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessorConfig":
        """Create from dictionary."""
        return cls(
            deg=d.get("deg", 3),
            knots=d.get("knots", 2),
            mesh_choice=d.get("mesh_choice", "percentile"),
            distance_measure=d.get("distance_measure", "euclidean"),
            extrapolation=d.get("extrapolation", "clip"),
        )


__all__ = ["PreprocessorConfig"]
