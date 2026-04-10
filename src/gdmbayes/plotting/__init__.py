"""Plotting utilities for spGDMM models."""

from .plots import (
    plot_crps_comparison,
    plot_isplines,
    plot_ppc,
    rgb_biological_space,
    rgb_from_biological_space,
    summarise_sampling,
)

__all__ = [
    "plot_isplines",
    "plot_crps_comparison",
    "summarise_sampling",
    "plot_ppc",
    "rgb_from_biological_space",
    "rgb_biological_space",
]
