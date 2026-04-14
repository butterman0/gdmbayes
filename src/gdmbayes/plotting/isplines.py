"""I-spline effect curves for fitted spGDMM models."""

from typing import TYPE_CHECKING

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dms_variants.ispline import Isplines

if TYPE_CHECKING:
    from ..models.spgdmm import spGDMM


def _beta_as_feature_basis(idata, prep) -> xr.DataArray:
    """Return beta as a (chain, draw, feature, basis_function) DataArray.

    When ``alpha_importance=True`` the model already declares beta with these
    dims. When ``alpha_importance=False`` beta is a flat LogNormal vector of
    length ``(n_predictors + 1) * n_spline_bases`` laid out as
    ``[env_0_s1..env_0_sJ, env_1_s1..env_1_sJ, ..., dist_s1..dist_sJ]``; slice
    off the trailing distance block and reshape so downstream plotting code is
    uniform across both cases.
    """
    beta = idata.posterior.beta
    if "feature" in beta.dims:
        return beta

    n_bases = prep.n_spline_bases_
    n_feat = prep.n_predictors_
    feat_names = list(prep.predictor_names_)
    env_len = n_feat * n_bases
    env_flat = beta.isel({beta.dims[-1]: slice(0, env_len)})
    reshaped = env_flat.values.reshape(
        env_flat.sizes["chain"], env_flat.sizes["draw"], n_feat, n_bases
    )
    return xr.DataArray(
        reshaped,
        dims=("chain", "draw", "feature", "basis_function"),
        coords={
            "chain": env_flat.coords["chain"].values,
            "draw": env_flat.coords["draw"].values,
            "feature": feat_names,
            "basis_function": np.arange(1, n_bases + 1),
        },
        name="beta",
    )


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

    beta = _beta_as_feature_basis(idata, prep)
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
