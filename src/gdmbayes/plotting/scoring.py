"""CRPS skill-score visualisation for fitted spGDMM models."""

import matplotlib.pyplot as plt
import numpy as np


def crps_boxplot(
    y_test,
    y_pred,
    y_train,
    figsize: tuple[float, float] = (7, 4),
) -> tuple[plt.Figure, np.ndarray]:
    """Boxplots of per-site CRPS against a climatological null baseline.

    Two side-by-side panels:

    1. Raw CRPS for the model vs. the null (empirical distribution of the
       training observations as a forecast ensemble).
    2. CRPS skill score: ``1 - CRPS_model / CRPS_null``. Positive values mean
       the model beats climatology.

    All scores are computed on the **original** Bray–Curtis scale. If the
    caller wants log-scale scores they can pass logged ``y_test``, ``y_pred``
    and ``y_train`` directly.

    Parameters
    ----------
    y_test : array-like, shape (n_test,)
        Test observations.
    y_pred : array-like, shape (n_test, n_samples)
        Posterior predictive samples aligned to ``y_test``. Accepts
        ``xr.DataArray`` or numpy.
    y_train : array-like, shape (n_train,)
        Training observations used to build the climatological null.
    figsize : tuple, default (7, 4)
        Figure size.

    Returns
    -------
    fig, axes : matplotlib Figure and ndarray of Axes
    """
    from properscoring import crps_ensemble

    y_test = np.asarray(y_test, dtype=float)
    y_pred = y_pred.values if hasattr(y_pred, "values") else np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    crps_model = crps_ensemble(y_test, y_pred)
    crps_null = crps_ensemble(y_test, np.tile(y_train, (len(y_test), 1)))
    crps_skill = 1.0 - (crps_model / crps_null)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].boxplot(
        [crps_model, crps_null],
        tick_labels=["Model", "Null"],
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    axes[0].set_ylabel("CRPS (Bray–Curtis)")
    axes[0].set_title("CRPS Comparison")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].boxplot(
        [crps_skill],
        tick_labels=["Model"],
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="forestgreen", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_ylabel("CRPS skill (1 − CRPS/CRPS_null)")
    axes[1].set_title("CRPS Skill Score")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, axes


__all__ = ["crps_boxplot"]
