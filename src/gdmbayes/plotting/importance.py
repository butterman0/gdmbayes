"""Posterior predictor importance bar chart.

Bayesian analogue of R ``gdm::gdm.varImp``. Since I-splines are constructed
to saturate at 1.0 at the right mesh endpoint, the maximum of the summed
spline for predictor *k* on draw *d* is simply ``sum_j beta[d, k, j]``.
We use that as the per-draw importance, then summarise with posterior mean
and HDI.
"""

from typing import TYPE_CHECKING

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..models.spgdmm import spGDMM


def plot_predictor_importance(
    model: "spGDMM",
    hdi_prob: float = 0.9,
    include_distance: bool = True,
    figsize: tuple[float, float] = (6, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Posterior predictor-importance bar chart for a fitted spGDMM.

    Importance for each predictor is the maximum height of its summed
    I-spline — equivalently, ``sum_j beta[d, k, j]`` since I-splines
    saturate at 1 at the right mesh endpoint. Bars show the posterior mean
    of that quantity, with HDI error bars.

    Predictors are sorted by posterior-mean importance (descending).

    Parameters
    ----------
    model : fitted spGDMM
    hdi_prob : float, default 0.9
        Width of the highest-density interval shown as error bars.
    include_distance : bool, default True
        If True and the model has a geographic-distance spline
        (``beta_dist`` in the posterior), append a ``"distance"`` bar.
    figsize : tuple, default (6, 4)

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    idata = model.idata_
    post = idata.posterior

    if "beta" not in post:
        raise ValueError("Posterior does not contain a 'beta' variable.")

    from .isplines import _beta_as_feature_basis

    beta = _beta_as_feature_basis(idata, model.preprocessor)
    importance = beta.sum(dim="basis_function")  # (chain, draw, feature)

    mean = importance.mean(dim=["chain", "draw"])
    hdi = az.hdi(importance, hdi_prob=hdi_prob)
    lo = hdi.sel(hdi="lower")["beta"] if "beta" in hdi else hdi.sel(hdi="lower")[list(hdi.data_vars)[0]]
    hi = hdi.sel(hdi="higher")["beta"] if "beta" in hdi else hdi.sel(hdi="higher")[list(hdi.data_vars)[0]]

    names = [str(f) for f in mean.coords["feature"].values]
    vals = mean.values
    errs_lo = vals - lo.values
    errs_hi = hi.values - vals

    if include_distance:
        dist_imp = None
        if "beta_dist" in post:
            dist_imp = post.beta_dist.sum(
                dim=[d for d in post.beta_dist.dims if d not in ("chain", "draw")]
            )
        elif "feature" not in post.beta.dims:
            # Flat LogNormal beta — trailing n_spline_bases entries are the distance basis.
            n_bases = model.preprocessor.n_spline_bases_
            flat = post.beta
            dist_block = flat.isel({flat.dims[-1]: slice(-n_bases, None)})
            dist_imp = dist_block.sum(dim=flat.dims[-1])
        if dist_imp is not None:
            dist_mean = float(dist_imp.mean(dim=["chain", "draw"]).values)
            dist_hdi_ds = az.hdi(dist_imp, hdi_prob=hdi_prob)
            dist_lo = float(dist_hdi_ds.sel(hdi="lower")[list(dist_hdi_ds.data_vars)[0]].values)
            dist_hi = float(dist_hdi_ds.sel(hdi="higher")[list(dist_hdi_ds.data_vars)[0]].values)

            names = names + ["distance"]
            vals = np.concatenate([vals, [dist_mean]])
            errs_lo = np.concatenate([errs_lo, [dist_mean - dist_lo]])
            errs_hi = np.concatenate([errs_hi, [dist_hi - dist_mean]])

    order = np.argsort(vals)[::-1]
    names_sorted = [names[i] for i in order]
    vals_sorted = vals[order]
    errs_sorted = np.stack([errs_lo[order], errs_hi[order]])

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names_sorted))
    ax.barh(
        y_pos,
        vals_sorted,
        xerr=errs_sorted,
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        error_kw=dict(ecolor="black", capsize=3, lw=1),
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    ax.invert_yaxis()
    ax.set_xlabel(f"Posterior importance  (mean ± {int(hdi_prob * 100)}% HDI)")
    ax.set_title("Predictor importance")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, ax


__all__ = ["plot_predictor_importance"]
