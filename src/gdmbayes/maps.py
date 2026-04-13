"""Biological-space RGB maps for fitted spGDMM models.

Mirrors the R ``gdm`` workflow of ``gdm.transform`` → PCA → RGB colour
assignment for visualising community-composition gradients across space.
"""

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr


def rgb_from_biological_space(transformed_da: xr.DataArray) -> xr.DataArray:
    """PCA-based RGB map from a weighted biological-space DataArray.

    Parameters
    ----------
    transformed_da : xr.DataArray
        Output of ``spGDMM._predict_biological_space()``. Dims
        ``(time, grid_cell, feature)`` where ``grid_cell`` is a MultiIndex
        coord with levels ``(yc, xc)``.

    Returns
    -------
    xr.DataArray
        Dims ``(time, xc, yc, rgb)`` with rgb in ``{R, G, B}`` and values in
        ``[0, 1]``. Cells with any missing feature are NaN.
    """
    from sklearn.decomposition import PCA

    tf = transformed_da.unstack("grid_cell")
    valid = ~tf.isnull().any(dim="feature")
    X_valid = tf.where(valid).stack(sample=("time", "yc", "xc")).dropna(dim="sample")
    if X_valid.sizes["sample"] == 0:
        raise ValueError("No fully observed rows available for PCA.")

    X_mat = X_valid.transpose("sample", "feature").values
    PC = PCA(n_components=3).fit_transform(X_mat)

    pc_min = PC.min(axis=0)
    pc_max = PC.max(axis=0)
    pc_rng = np.where(pc_max > pc_min, pc_max - pc_min, 1.0)

    pcs_full = xr.DataArray(
        np.full((tf.sizes["time"], tf.sizes["yc"], tf.sizes["xc"], 3), np.nan, dtype=float),
        dims=("time", "yc", "xc", "pc"),
        coords={"time": tf["time"], "yc": tf["yc"], "xc": tf["xc"], "pc": range(3)},
    ).stack(sample=("time", "yc", "xc"))

    pcs_full.loc[dict(sample=X_valid["sample"])] = xr.DataArray(
        PC,
        dims=("sample", "pc"),
        coords={"sample": X_valid["sample"], "pc": pcs_full["pc"]},
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
    """Compute an RGB biological-space map from a fitted model's InferenceData.

    Equivalent to R's ``gdm.transform()`` + PCA + RGB colour assignment.
    Reconstructs the preprocessing state from ``idata.constant_data`` so no
    live model object is needed (works after ``spGDMM.load``).

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData produced by ``spGDMM.fit()`` or loaded via
        ``spGDMM.load()``. Must contain ``constant_data`` (saved by
        ``_save_input_params``) and a ``posterior.beta`` variable with dims
        ``(feature, basis_function)``.
    X_pred : pd.DataFrame
        Site-level data with columns ``[xc, yc, time_idx, predictor1, ...]``.
        Index must be a MultiIndex with levels ``(yc, xc)`` so that
        ``grid_cell`` can be unstacked into a spatial grid.
    metric : {"mean", "median"}, default "median"
        Posterior summary statistic used to collapse the ``beta`` draws.

    Returns
    -------
    xr.DataArray
        Dims ``(time, xc, yc, rgb)`` with rgb in ``{R, G, B}`` and values in
        ``[0, 1]``.
    """
    from .preprocessor import GDMPreprocessor

    preprocessor = GDMPreprocessor.from_xarray(idata.constant_data)

    beta_draws = idata.posterior.beta
    beta = (
        beta_draws.mean(dim=["chain", "draw"])
        if metric == "mean"
        else beta_draws.median(dim=["chain", "draw"])
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


__all__ = ["rgb_from_biological_space", "rgb_biological_space"]
