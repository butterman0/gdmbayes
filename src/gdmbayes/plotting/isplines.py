"""I-spline effect curves for fitted spGDMM models."""

from typing import TYPE_CHECKING

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from dms_variants.ispline import Isplines

if TYPE_CHECKING:
    from ..models.spgdmm import spGDMM


def plot_isplines(
    model: "spGDMM",
    features: list[str] | None = None,
    hdi_prob: float = 0.9,
    decompose: bool = False,
    figsize: tuple[float, float] = (6.5, 4),
) -> list[tuple[plt.Figure, plt.Axes]]:
    """Plot I-spline effect curves for each environmental predictor.

    By default shows one figure per predictor with the summed spline
    (black line, HDI band), matching R ``gdm::plot.gdm``. Pass
    ``decompose=True`` to additionally draw each basis-function component.

    Parameters
    ----------
    model : fitted spGDMM
        A model on which ``.fit()`` has been called.
    features : list of str, optional
        Subset of predictor names to plot. Defaults to all environmental
        features present on ``beta.feature``.
    hdi_prob : float, default 0.9
        Width of the highest-density interval band.
    decompose : bool, default False
        If True, overlay individual basis-function contributions and report
        each basis weight in the legend.
    figsize : tuple, default (6.5, 4)
        Figure size passed to matplotlib.

    Returns
    -------
    list of (Figure, Axes)
        One entry per plotted predictor.
    """
    idata = model.idata_
    prep = model.preprocessor

    beta = idata.posterior.beta  # (chain, draw, feature, basis_function)
    n_bases = beta.sizes["basis_function"]
    all_feats = list(beta.coords["feature"].values)

    if features is not None:
        missing = [f for f in features if f not in all_feats]
        if missing:
            raise ValueError(
                f"Unknown feature(s) {missing}; available: {all_feats}"
            )
        feats = list(features)
    else:
        feats = all_feats

    beta_mean = beta.mean(dim=["chain", "draw"])
    hdi_vals = az.hdi(beta, hdi_prob=hdi_prob)
    beta_lo = hdi_vals.sel(hdi="lower").beta
    beta_hi = hdi_vals.sel(hdi="higher").beta

    has_alpha = "alpha" in idata.posterior
    if has_alpha:
        alpha_mean = idata.posterior.alpha.mean(dim=["chain", "draw"])

    predictor_mesh = prep.predictor_mesh_
    deg = prep.deg

    results: list[tuple[plt.Figure, plt.Axes]] = []

    for feat in feats:
        i = all_feats.index(feat)
        mesh_col = predictor_mesh[i]
        x = np.linspace(mesh_col[0], mesh_col[-1], 300)

        spline = Isplines(deg, mesh_col, x)
        basis = np.column_stack([spline.I(j + 1) for j in range(n_bases)])

        bm = beta_mean.sel(feature=feat).values
        blo = beta_lo.sel(feature=feat).values
        bhi = beta_hi.sel(feature=feat).values

        total = basis @ bm
        total_lo = basis @ blo
        total_hi = basis @ bhi

        fig, ax = plt.subplots(figsize=figsize)

        ax.fill_between(
            x,
            total_lo,
            total_hi,
            color="black",
            alpha=0.15,
            label=f"{int(hdi_prob * 100)}% HDI",
        )
        ax.plot(x, total, color="black", lw=2, label="Posterior mean")

        if decompose:
            colors = plt.cm.tab10(np.linspace(0, 0.9, n_bases))
            for j in range(n_bases):
                ax.plot(
                    x,
                    basis[:, j] * bm[j],
                    lw=1,
                    color=colors[j],
                    label=f"I{j + 1}  β={bm[j]:.2f}",
                )

        if has_alpha:
            a_val = float(alpha_mean.sel(feature=feat).values)
            ax.plot([], [], " ", label=f"α = {a_val:.2f}")

        for ki, kv in enumerate(mesh_col[1:-1]):
            ax.axvline(
                kv,
                color="gray",
                linestyle="--",
                lw=0.8,
                label="Knot" if ki == 0 else None,
            )

        ax.set_title(feat)
        ax.set_xlabel("Predictor value")
        ax.set_ylabel("Partial ecological distance")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, ncol=2 if decompose else 1)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        results.append((fig, ax))

    return results


__all__ = ["plot_isplines"]
