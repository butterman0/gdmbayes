"""Cross-validation helpers for gdmbayes."""

import numpy as np


def site_pairs(n_sites_total: int, site_idx) -> np.ndarray:
    """Return condensed pair indices for a subset of sites.

    Given a full dataset of ``n_sites_total`` sites whose pairwise dissimilarities
    are stored in a condensed vector ``y`` (length ``n_sites_total*(n_sites_total-1)//2``,
    upper-triangle order from ``np.triu_indices``), return the integer indices into
    ``y`` that correspond to pairs *within* ``site_idx``.

    Parameters
    ----------
    n_sites_total : int
        Total number of sites in the full dataset.
    site_idx : array-like of int
        Indices of the site subset (0-based, into the full site matrix X).

    Returns
    -------
    np.ndarray of int
        Indices into the condensed pair vector for pairs within ``site_idx``.

    Examples
    --------
    >>> import numpy as np
    >>> from gdmbayes import site_pairs
    >>> # 5 sites total; training on sites 0, 1, 3
    >>> idx = site_pairs(5, [0, 1, 3])
    >>> # y[idx] gives the 3 dissimilarities between sites 0-1, 0-3, 1-3
    """
    site_idx = np.asarray(site_idx)
    all_row, all_col = np.triu_indices(n_sites_total, k=1)
    s = set(site_idx.tolist())
    mask = np.array([r in s and c in s for r, c in zip(all_row, all_col)])
    return np.where(mask)[0]


__all__ = ["site_pairs"]
