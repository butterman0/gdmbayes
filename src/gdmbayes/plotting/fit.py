"""Model-fit diagnostic plots (analogues of R ``gdm::plot.gdm`` panels 1–2)."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..models.spgdmm import spGDMM


def plot_obs_vs_pred(
    model: "spGDMM",
    X: pd.DataFrame,
    y,
    figsize: tuple[float, float] = (5, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter of observed vs. posterior-mean predicted dissimilarity.

    Direct analogue of R ``gdm::plot.gdm`` panel 1. A well-fitting model
    produces points close to the 1:1 line; systematic departures indicate
    bias at particular dissimilarity levels.

    Parameters
    ----------
    model : fitted spGDMM
    X : pd.DataFrame
        Site-level data for the pairs to score. Can be train or test.
    y : array-like, shape (n_pairs,)
        Observed pairwise dissimilarities on the original Bray–Curtis scale.
    figsize : tuple, default (5, 5)

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    y_obs = np.asarray(y, dtype=float)
    y_hat = np.asarray(model.predict(X), dtype=float)

    if y_obs.shape != y_hat.shape:
        raise ValueError(
            f"Shape mismatch: y has shape {y_obs.shape}, predict(X) returned {y_hat.shape}."
        )

    lo = float(min(y_obs.min(), y_hat.min()))
    hi = float(max(y_obs.max(), y_hat.max()))
    pad = 0.02 * max(hi - lo, 1e-6)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_hat, y_obs, s=10, alpha=0.4, color="steelblue", edgecolor="none")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1, linestyle="--")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted dissimilarity")
    ax.set_ylabel("Observed dissimilarity")
    ax.set_title("Observed vs. predicted")
    ax.spines[["top", "right"]].set_visible(False)

    r = float(np.corrcoef(y_hat, y_obs)[0, 1])
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {r:.2f}\nn = {len(y_obs)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="lightgray"),
    )

    fig.tight_layout()
    return fig, ax


def plot_link_curve(
    model: "spGDMM",
    X: pd.DataFrame,
    y,
    figsize: tuple[float, float] = (6, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Fitted log-scale linear predictor vs. observed dissimilarity.

    Analogue of R ``gdm::plot.gdm`` panel 2 adapted to spGDMM's
    parameterisation. The model assumes ``log y ~ Normal(η, σ²)`` censored at
    0, so the link from the linear predictor ``η`` to the dissimilarity is
    ``y = exp(η)``. This plot draws that exponential curve over the fitted
    range of ``η`` and overlays the observed ``(η̂, y_obs)`` pairs, where
    ``η̂`` is the posterior mean linear predictor for each site-pair.

    Points hugging the curve → well-calibrated. Points above the curve near
    ``η → 0`` → model under-predicts high-dissimilarity pairs and the
    censoring bound is binding.

    Parameters
    ----------
    model : fitted spGDMM
    X : pd.DataFrame
        Site-level data for the pairs to score.
    y : array-like, shape (n_pairs,)
        Observed pairwise dissimilarities on the original scale.
    figsize : tuple, default (6, 4)

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    y_obs = np.asarray(y, dtype=float)

    samples = model.predict_posterior(X, combined=True, extend_idata=False)
    pair_dim = next(d for d in samples.dims if d != "sample")
    y_samples = samples.transpose(pair_dim, "sample").values  # (n_pairs, n_samples)
    log_y_samples = np.log(np.clip(y_samples, 1e-12, None))
    eta_hat = log_y_samples.mean(axis=1)

    if y_obs.shape != eta_hat.shape:
        raise ValueError(
            f"Shape mismatch: y has shape {y_obs.shape}, predictions gave {eta_hat.shape}."
        )

    eta_min = float(min(eta_hat.min(), np.log(max(y_obs.min(), 1e-12))))
    eta_max = float(max(eta_hat.max(), 0.0))
    pad = 0.05 * max(eta_max - eta_min, 1e-6)
    eta_grid = np.linspace(eta_min - pad, min(0.0, eta_max + pad), 200)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(eta_grid, np.exp(eta_grid), color="black", lw=1.5, label="y = exp(η)")
    ax.scatter(
        eta_hat,
        y_obs,
        s=10,
        alpha=0.4,
        color="steelblue",
        edgecolor="none",
        label="Observed",
    )
    ax.axhline(1.0, color="gray", linestyle=":", lw=0.8, label="Censoring bound")
    ax.set_xlabel("Posterior-mean linear predictor  η̂  =  E[log y | X]")
    ax.set_ylabel("Observed dissimilarity")
    ax.set_title("Link function: y = exp(η)")
    ax.set_ylim(0, max(1.05, float(y_obs.max()) * 1.05))
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, ax


__all__ = ["plot_obs_vs_pred", "plot_link_curve"]
