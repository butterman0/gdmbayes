"""Posterior predictive check plot for fitted spGDMM models."""

import matplotlib.pyplot as plt
import numpy as np


def plot_ppc(
    idata,
    y_obs,
    var_name: str = "log_y",
    n_pp_samples: int = 200,
    figsize: tuple[float, float] = (6, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """Posterior predictive check: observed log-dissimilarity vs. replicates.

    Overlays ``n_pp_samples`` kernel-density estimates of individual posterior
    predictive replicates (light blue) against the observed distribution
    (solid black). Good fit is indicated by the observed KDE lying within the
    cloud of predictive KDEs.

    Parameters
    ----------
    idata : arviz.InferenceData
        Must contain a ``posterior_predictive`` group with a variable of
        shape ``(chain, draw, obs)`` on the log scale.
    y_obs : array-like, shape (n_obs,)
        Observed dissimilarities on the **original** scale. They are logged
        internally so they can be compared against ``var_name`` replicates.
    var_name : str, default "log_y"
        Name of the posterior predictive variable inside ``idata``. Matches
        the default response variable defined in ``spGDMM.build_model``.
    n_pp_samples : int, default 200
        Number of posterior predictive replicates to overlay.
    figsize : tuple, default (6, 4)
        Figure size.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    from scipy.stats import gaussian_kde

    pp = idata.posterior_predictive[var_name].values  # (chain, draw, obs)
    n_obs = pp.shape[-1]
    pp_flat = pp.reshape(-1, n_obs)

    rng = np.random.default_rng(0)
    n_sel = min(n_pp_samples, pp_flat.shape[0])
    idx = rng.choice(pp_flat.shape[0], size=n_sel, replace=False)
    pp_sel = pp_flat[idx]

    log_y_obs = np.log(np.asarray(y_obs, dtype=float))
    x_grid = np.linspace(
        min(log_y_obs.min(), pp_sel.min()) - 0.3,
        max(log_y_obs.max(), pp_sel.max()) + 0.3,
        300,
    )

    fig, ax = plt.subplots(figsize=figsize)

    for i, rep in enumerate(pp_sel):
        kde = gaussian_kde(rep)
        ax.plot(
            x_grid,
            kde(x_grid),
            color="steelblue",
            alpha=0.07,
            lw=0.8,
            label="Posterior predictive" if i == 0 else None,
        )

    kde_obs = gaussian_kde(log_y_obs)
    ax.plot(x_grid, kde_obs(x_grid), color="black", lw=2, label="Observed")

    ax.set_xlabel("log(Bray–Curtis dissimilarity)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Check")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


__all__ = ["plot_ppc"]
