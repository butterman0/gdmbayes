"""Built-in spatial effect functions for spGDMM.

Each function is called inside a PyMC model context after the GP latent
variable ``psi`` has been sampled, and returns a PyTensor vector to be added
to the linear predictor ``mu``.  Pass the string name or any callable with the
same signature to :class:`~gdmbayes.ModelConfig`:

.. code-block:: python

    from gdmbayes import ModelConfig, spatial_abs_diff

    # Built-in by name
    cfg = ModelConfig(spatial_effect="abs_diff")

    # Built-in by reference
    cfg = ModelConfig(spatial_effect=spatial_abs_diff)

    # Custom callable
    import pymc as pm
    def my_spatial(psi, row_ind, col_ind):
        return pm.math.tanh(psi[row_ind] - psi[col_ind])

    cfg = ModelConfig(spatial_effect=my_spatial)

**Signature**::

    fn(psi, row_ind, col_ind) -> effect

- ``psi``: PyTensor vector of length ``n_sites`` — GP latent values at each
  training site.
- ``row_ind``, ``col_ind``: integer arrays — upper-triangle pair indices
  produced by ``np.triu_indices(n_sites, k=1)``.
- Returns a PyTensor vector of length ``n_pairs`` to add to ``mu``.
"""

from collections.abc import Callable


def spatial_abs_diff(psi, row_ind, col_ind):
    """Spatial effect as the absolute difference of GP values between sites.

    Returns ``|psi[i] - psi[j]|`` for each pair ``(i, j)``.

    Uses the Python ``abs()`` builtin, which dispatches correctly to both
    numpy (``np.abs``) and PyTensor (``pytensor.tensor.abs``).
    """
    return abs(psi[row_ind] - psi[col_ind])


def spatial_squared_diff(psi, row_ind, col_ind):
    """Spatial effect as the squared absolute difference of GP values.

    Returns ``|psi[i] - psi[j]|²`` for each pair ``(i, j)``.
    """
    diff = psi[row_ind] - psi[col_ind]
    return diff ** 2


SPATIAL_FUNCTIONS: dict[str, Callable] = {
    "abs_diff": spatial_abs_diff,
    "squared_diff": spatial_squared_diff,
}
"""Registry mapping spatial effect string names to their functions.

Use this dict to inspect available options or extend the registry at runtime:

.. code-block:: python

    from gdmbayes import SPATIAL_FUNCTIONS
    print(list(SPATIAL_FUNCTIONS))  # ['abs_diff', 'squared_diff']
"""


__all__ = [
    "spatial_abs_diff",
    "spatial_squared_diff",
    "SPATIAL_FUNCTIONS",
]
