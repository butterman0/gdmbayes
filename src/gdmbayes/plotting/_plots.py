"""Plotting utilities for fitted spGDMM models."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
import matplotlib.pyplot as plt
from dms_variants.ispline import Isplines

if TYPE_CHECKING:
    from ..models.spgdmm import spGDMM


def plot_isplines(model: "spGDMM", features=None, hdi_prob=0.9, figsize=(6.5, 4)):
    """
    Plot I-spline effect curves for each environmental predictor.

    For each predictor, shows the individual basis-function contributions
    (thin coloured lines) and their weighted sum (black line) with a
    credible-interval band. Also reports the fitted alpha importance weight
    in the legend when alpha_importance=True.

    Parameters
    ----------
    model : fitted spGDMM
        A model on which .fit() has been called.
    features : list of str, optional
        Subset of feature names to plot. Defaults to all environmental
        features (distance excluded).
    hdi_prob : float
        Width of the highest-density interval band (default 0.90).
    figsize : tuple
        Figure size passed to matplotlib.

    Returns
    -------
    None (displays plots inline)
    """
    idata = model.idata
    prep = model.preprocessor

    beta = idata.posterior.beta  # (chain, draw, feature, basis_function)
    n_bases = beta.sizes["basis_function"]
    all_feats = list(beta.coords["feature"].values)

    # Drop distance / time predictors — they are not environmental effects
    skip = {"distance", "temporal_diff", "distance_euclidean", "binary_connectivity"}
    env_feats = [f for f in all_feats if f not in skip]
    if features is not None:
        env_feats = [f for f in env_feats if f in features]

    beta_mean = beta.mean(dim=["chain", "draw"])  # (feature, basis_function)
    hdi_vals = az.hdi(beta, hdi_prob=hdi_prob)
    beta_lo = hdi_vals.sel(hdi="lower").beta
    beta_hi = hdi_vals.sel(hdi="higher").beta

    has_alpha = "alpha" in idata.posterior
    if has_alpha:
        alpha_mean = idata.posterior.alpha.mean(dim=["chain", "draw"])

    predictor_mesh = prep.predictor_mesh_
    cfg = prep._get_config()
    deg = cfg.deg
    knots = cfg.knots

    for i, feat in enumerate(env_feats):
        mesh_col = predictor_mesh[i]
        x = np.linspace(mesh_col[0], mesh_col[-1], 300)

        spline = Isplines(deg, mesh_col, x)
        basis = np.column_stack([spline.I(j + 1) for j in range(n_bases)])

        bm = beta_mean.sel(feature=feat).values  # (n_bases,)
        blo = beta_lo.sel(feature=feat).values
        bhi = beta_hi.sel(feature=feat).values

        comp = basis * bm
        comp_lo = basis * blo
        comp_hi = basis * bhi
        total = comp.sum(axis=1)
        total_lo = comp_lo.sum(axis=1)
        total_hi = comp_hi.sum(axis=1)

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10(np.linspace(0, 0.9, n_bases))
        for j in range(n_bases):
            ax.plot(
                x,
                comp[:, j],
                lw=1,
                color=colors[j],
                label=f"I{j+1}  β={bm[j]:.2f}",
            )

        ax.plot(x, total, color="black", lw=2, label="Total")
        ax.fill_between(
            x,
            total_lo,
            total_hi,
            color="black",
            alpha=0.15,
            label=f"{int(hdi_prob * 100)}% HDI",
        )

        if has_alpha:
            a_val = float(alpha_mean.sel(feature=feat).values)
            ax.plot([], [], " ", label=f"α = {a_val:.2f}")

        # Mark internal knots
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
        ax.set_ylabel("Effect")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.show()


def plot_crps_comparison(
    y_test, y_pred, y_train, use_log=False, figsize=(7, 4)
):
    """
    Plot CRPS comparison between model and null baseline, plus CRPS skill scores.

    Creates two side-by-side boxplots:
    1. CRPS comparison: model vs null baseline
    2. CRPS skill scores: 1 - (CRPS_model / CRPS_null)

    Parameters
    ----------
    y_test : array-like
        Test observations (e.g., Bray-Curtis dissimilarity values).
    y_pred : array-like (n_test, n_samples)
        Posterior predictive samples from the model.
    y_train : array-like
        Training observations used for null model baseline.
    use_log : bool
        If False (default), compute CRPS on original scale (0-1 for Bray-Curtis).
        If True, compute CRPS on log-transformed values.
    figsize : tuple
        Figure size for the two subplots (width, height).

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
        The created figure and axes objects.
    """
    from properscoring import crps_ensemble

    # Prepare data on appropriate scale
    if use_log:
        y_test_scale = np.log(y_test)
        y_pred_scale = y_pred.values if hasattr(y_pred, "values") else y_pred
        y_train_scale = np.log(y_train)
        ylabel = "CRPS (log scale)"
    else:
        y_test_scale = y_test
        y_pred_scale = y_pred.values if hasattr(y_pred, "values") else y_pred
        y_train_scale = y_train
        ylabel = "CRPS (Bray–Curtis)"

    # Compute CRPS for model
    crps_model = crps_ensemble(y_test_scale, y_pred_scale)

    # Compute CRPS for null (use training data as forecast ensemble)
    crps_null = crps_ensemble(y_test_scale, np.tile(y_train_scale, (len(y_test_scale), 1)))

    # Compute CRPS skill score: 1 - (CRPS_model / CRPS_null)
    crps_skill = 1.0 - (crps_model / crps_null)

    # Create two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: CRPS comparison
    axes[0].boxplot(
        [crps_model, crps_null],
        tick_labels=["spGDMM", "Null"],
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    axes[0].set_ylabel(ylabel)
    axes[0].set_title("CRPS Comparison")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Right plot: CRPS skill scores
    axes[1].boxplot(
        [crps_skill],
        tick_labels=["spGDMM"],
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="forestgreen", alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
    )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_ylabel("CRPS Skill (1 − CRPS/CRPS_null)")
    axes[1].set_title("CRPS Skill Score")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    return fig, axes


def summarise_sampling(idata, var_names=None):
    """
    Return a tidy DataFrame of ESS / R-hat diagnostics and print a divergence
    count.

    Parameters
    ----------
    idata : arviz.InferenceData
        Result of a completed``model.fit()`` call.
    var_names : list of str, optional
        Parameters to include. Defaults to all variables in the posterior.

    Returns
    -------
    diag : pandas.DataFrame
        Columns: ``mean``, ``sd``, ``ess_bulk``, ``ess_tail``, ``r_hat``.
        Rows are individual scalar parameters (vectorised parameters are
        unpacked by ArviZ).
    """
    summary = az.summary(idata, var_names=var_names)
    keep = [c for c in ["mean", "sd", "ess_bulk", "ess_tail", "r_hat"] if c in summary.columns]
    diag = summary[keep].copy()

    # Divergence count from sample_stats
    n_div = 0
    if hasattr(idata, "sample_stats") and hasattr(idata.sample_stats, "diverging"):
        n_div = int(idata.sample_stats.diverging.values.sum())

    n_chains = idata.posterior.sizes.get("chain", 1)
    n_draws = idata.posterior.sizes.get("draw", 0)
    total = n_chains * n_draws

    print(f"Chains: {n_chains}  |  Draws/chain: {n_draws}  |  " f"Total draws: {total}  |  Divergences: {n_div}")

    # Flag problematic rows
    bad_rhat = diag["r_hat"] > 1.01 if "r_hat" in diag else pd.Series(dtype=bool)
    bad_ess = diag["ess_bulk"] < 100 if "ess_bulk" in diag else pd.Series(dtype=bool)
    n_bad_rhat = bad_rhat.sum()
    n_bad_ess = bad_ess.sum()
    if n_bad_rhat:
        print(f"  WARNING: {n_bad_rhat} parameter(s) with R-hat > 1.01")
    if n_bad_ess:
        print(f"  WARNING: {n_bad_ess} parameter(s) with ESS_bulk < 100")
    if not n_bad_rhat and not n_bad_ess and n_div == 0:
        print("  All diagnostics look healthy.")

    return diag


def plot_ppc(idata, y_obs, n_pp_samples=200, figsize=(6, 4)):
    """
    Posterior predictive check: observed log-dissimilarity distribution vs
    posterior predictive draws.

    Overlays ``n_pp_samples`` kernel-density estimates of individual posterior
    predictive replicates (light blue) against the observed distribution
    (solid black). Good model fit is indicated by the observed KDE lying
    within the cloud of predictive KDEs.

    Parameters
    ----------
    idata : arviz.InferenceData
        Must contain a ``posterior_predictive`` group with variable ``log_y``.
    y_obs : array-like, shape (n_obs,)
        Observed Bray-Curtis dissimilarities (original scale, not log).
    n_pp_samples : int
        Number of posterior predictive replicates to overlay (default 200).
    figsize : tuple
        Figure size.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    from scipy.stats import gaussian_kde

    pp = idata.posterior_predictive["log_y"].values  # (chain, draw, obs)
    n_chains, n_draws, n_obs = pp.shape
    pp_flat = pp.reshape(-1, n_obs)  # (total_draws, obs)

    rng = np.random.default_rng(0)
    idx = rng.choice(pp_flat.shape[0], size=min(n_pp_samples, pp_flat.shape[0]), replace=False)
    pp_sel = pp_flat[idx]  # (n_pp_samples, obs)

    log_y_obs = np.log(np.asarray(y_obs, dtype=float))
    x_grid = np.linspace(
        min(log_y_obs.min(), pp_sel.min()) - 0.3,
        max(log_y_obs.max(), pp_sel.max()) + 0.3,
        300,
    )

    fig, ax = plt.subplots(figsize=figsize)

    for i, rep in enumerate(pp_sel):
        kde = gaussian_kde(rep)
        ax.plot(x_grid, kde(x_grid), color="steelblue", alpha=0.07, lw=0.8, label="Posterior predictive" if i == 0 else None)

    kde_obs = gaussian_kde(log_y_obs)
    ax.plot(x_grid, kde_obs(x_grid), color="black", lw=2, label="Observed")

    ax.set_xlabel("log(Bray–Curtis dissimilarity)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Predictive Check")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig, ax


def rgb_from_biological_space(transformed_da: xr.DataArray) -> xr.DataArray:
    """
    PCA-based RGB map from a weighted biological-space DataArray.

    Parameters
    ----------
    transformed_da : xr.DataArray
        Output of spGDMM._predict_biological_space(). Dims (time, grid_cell, feature)
        where grid_cell is a MultiIndex coord with levels (yc, xc).

    Returns
    -------
    xr.DataArray
        Dims (time, xc, yc, rgb) with rgb in {R, G, B}, values in [0, 1].
    """
    from sklearn.decomposition import PCA

    tf = transformed_da.unstack("grid_cell")
    valid = ~tf.isnull().any(dim="feature")
    X_all = tf.where(valid).stack(sample=("time", "yc", "xc")).dropna(dim="sample")
    if X_all.sizes["sample"] == 0:
        raise ValueError("No fully observed rows available for PCA.")
    X_mat = X_all.transpose("sample", "feature").values

    pca = PCA(n_components=3).fit(X_mat)
    PC_ref = pca.transform(X_mat)
    pc_min, pc_max = PC_ref.min(axis=0), PC_ref.max(axis=0)
    pc_rng = np.where(pc_max > pc_min, pc_max - pc_min, 1.0)

    X_full = tf.transpose("time", "yc", "xc", "feature").stack(sample=("time", "yc", "xc"))
    X_valid = X_full.sel(sample=X_all["sample"]).transpose("sample", "feature").values
    PC_curr = pca.transform(X_valid)

    pcs_full = xr.DataArray(
        np.full((tf.sizes["time"], tf.sizes["yc"], tf.sizes["xc"], 3), np.nan, dtype=float),
        dims=("time", "yc", "xc", "pc"),
        coords={"time": tf["time"], "yc": tf["yc"], "xc": tf["xc"], "pc": range(3)},
    ).stack(sample=("time", "yc", "xc"))

    pcs_full.loc[dict(sample=X_all["sample"])] = xr.DataArray(
        PC_curr, dims=("sample", "pc"), coords={"sample": X_all["sample"], "pc": pcs_full["pc"]}
    )
    pcs_full = pcs_full.unstack("sample")

    pcs_norm = (
        pcs_full - xr.DataArray(pc_min, dims="pc", coords={"pc": pcs_full["pc"]})
    ) / xr.DataArray(pc_rng, dims="pc", coords={"pc": pcs_full["pc"]})
    pcs_norm = xr.where(
        xr.DataArray(pc_max == pc_min, dims="pc", coords={"pc": pcs_full["pc"]}),
        0.5,
        pcs_norm,
    )

    rgb_da = pcs_norm.assign_coords(pc=["R", "G", "B"]).rename(pc="rgb")
    return rgb_da.transpose("time", "xc", "yc", "rgb")


def rgb_biological_space(
    idata: az.InferenceData, X_pred: pd.DataFrame, metric: str = "median"
) -> xr.DataArray:
    """Compute RGB biological-space map from a fitted model's InferenceData.

    Equivalent to R's gdm.transform() + PCA + RGB colour assignment. Reconstructs
    the preprocessing state from ``idata.constant_data`` so no model object is needed.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData produced by ``spGDMM.fit()`` or loaded via ``spGDMM.load()``.
        Must contain ``constant_data`` (saved by ``_save_input_params``) and
        ``posterior`` with a ``beta`` variable of dims (feature, basis_function).
    X_pred : pd.DataFrame
        Site-level data with columns [xc, yc, time_idx, predictor1, ...].
        Index must be a MultiIndex with levels (yc, xc) so that grid_cell can
        be unstacked into a spatial grid.
    metric : str, default="median"
        Posterior summary statistic: "mean" or "median".

    Returns
    -------
    xr.DataArray
        Dims (time, xc, yc, rgb) with rgb in {R, G, B}, values in [0, 1].
    """
    from ..preprocessing.preprocessor import GDMPreprocessor

    preprocessor = GDMPreprocessor.from_xarray(idata.constant_data)

    beta = (
        idata.posterior.beta.mean(dim=["chain", "draw"])
        if metric == "mean"
        else idata.posterior.beta.median(dim=["chain", "draw"])
    )
    X_splined = preprocessor.transform(X_pred, biological_space=True).reshape(
        1, -1, beta.sizes["feature"], beta.sizes["basis_function"]
    )
    transformed = (
        xr.DataArray(
            X_splined,
            dims=("time", "grid_cell", "feature", "basis_function"),
            coords={
                "time": [0],
                "grid_cell": X_pred.index,
                "feature": beta["feature"].values,
                "basis_function": beta["basis_function"].values,
            },
        )
        * beta
    ).sum(dim="basis_function", skipna=False)
    return rgb_from_biological_space(transformed)


__all__ = [
    "plot_isplines",
    "plot_crps_comparison",
    "summarise_sampling",
    "plot_ppc",
    "rgb_biological_space",
    "rgb_from_biological_space",
]