"""MCMC sampling diagnostics for fitted spGDMM models."""

import arviz as az
import pandas as pd


def summarise_sampling(
    idata,
    var_names=None,
    rhat_threshold: float = 1.01,
    ess_threshold: int = 100,
) -> pd.DataFrame:
    """Return a tidy DataFrame of ESS / R-hat diagnostics and print a divergence count.

    Parameters
    ----------
    idata : arviz.InferenceData
        Result of a completed ``model.fit()`` call.
    var_names : list of str, optional
        Parameters to include. Defaults to all variables in the posterior.
    rhat_threshold : float, default 1.01
        Values above this threshold are flagged as potentially problematic.
    ess_threshold : int, default 100
        Bulk ESS below this threshold is flagged as potentially problematic.

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

    n_div = 0
    if hasattr(idata, "sample_stats") and hasattr(idata.sample_stats, "diverging"):
        n_div = int(idata.sample_stats.diverging.values.sum())

    n_chains = idata.posterior.sizes.get("chain", 1)
    n_draws = idata.posterior.sizes.get("draw", 0)
    total = n_chains * n_draws

    print(
        f"Chains: {n_chains}  |  Draws/chain: {n_draws}  |  "
        f"Total draws: {total}  |  Divergences: {n_div}"
    )

    bad_rhat = diag["r_hat"] > rhat_threshold if "r_hat" in diag else pd.Series(dtype=bool)
    bad_ess = diag["ess_bulk"] < ess_threshold if "ess_bulk" in diag else pd.Series(dtype=bool)
    n_bad_rhat = int(bad_rhat.sum())
    n_bad_ess = int(bad_ess.sum())
    if n_bad_rhat:
        print(f"  WARNING: {n_bad_rhat} parameter(s) with R-hat > {rhat_threshold}")
    if n_bad_ess:
        print(f"  WARNING: {n_bad_ess} parameter(s) with ESS_bulk < {ess_threshold}")
    if not n_bad_rhat and not n_bad_ess and n_div == 0:
        print("  All diagnostics look healthy.")

    return diag


__all__ = ["summarise_sampling"]
