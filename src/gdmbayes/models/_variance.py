"""Built-in variance functions for spGDMM.

Each function is called inside a PyMC model context and returns a PyTensor
expression for the variance ``sigma²``.  Pass the string name or any callable
with the same signature to :class:`~gdmbayes.ModelConfig`:

.. code-block:: python

    from gdmbayes import ModelConfig, variance_polynomial

    # Built-in by name
    cfg = ModelConfig(variance="polynomial")

    # Built-in by reference
    cfg = ModelConfig(variance=variance_polynomial)

    # Custom callable
    import pymc as pm
    def my_variance(mu, X_sigma):
        beta_s = pm.HalfNormal("beta_s", sigma=1)
        return beta_s * pm.math.exp(mu)

    cfg = ModelConfig(variance=my_variance)

**Signature**::

    fn(mu, X_sigma) -> sigma2

- ``mu``: PyTensor vector — the linear predictor (mean) for each site pair.
- ``X_sigma``: np.ndarray of shape ``(n_pairs, k)`` or ``None`` — auxiliary
  covariates for the variance model (currently pairwise geographic distance).
- Returns a PyTensor scalar or vector representing ``sigma²``.
"""

from typing import Callable, Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def variance_homogeneous(mu, X_sigma):
    """Constant (homoscedastic) variance.

    Places an InverseGamma(1, 1) prior on a single shared ``sigma²``.
    """
    return pm.InverseGamma("sigma2", alpha=1, beta=1)


def variance_covariate_dependent(mu, X_sigma):
    """Variance as an exponential linear function of ``X_sigma``.

    If ``X_sigma`` is provided, fits ``sigma² = exp(X_sigma @ beta_sigma)``
    with Normal(0, 2) priors on ``beta_sigma``.  ``X_sigma`` may be a numpy
    array or a PyMC ``pm.Data`` variable (shape ``(n_pairs, k)``).  The number
    of columns ``k`` is read from the static shape when ``X_sigma`` is a
    symbolic tensor, falling back to 1 (the default: pairwise geographic distance).
    The linear predictor is clipped to [-20, 20] before exp() to prevent
    overflow during nutpie initialization.
    Falls back to :func:`variance_homogeneous` when ``X_sigma`` is None.

    """
    if X_sigma is not None:
        # Support both numpy arrays (shape known directly) and pm.Data/symbolic
        # tensors (static shape may be unknown, so evaluate to get the concrete shape).
        if isinstance(X_sigma, np.ndarray):
            n_cols = X_sigma.shape[1]
        else:
            n_cols = int(X_sigma.shape.eval()[1])
        beta_sigma = pm.Normal("beta_sigma", mu=0, sigma=10, shape=n_cols)
        return pm.math.exp(pt.clip(pm.math.dot(X_sigma, beta_sigma), -20, 20))
    return pm.InverseGamma("sigma2", alpha=1, beta=1)


def variance_polynomial(mu, X_sigma):
    """Variance as a cubic polynomial function of the mean ``mu``.

    Fits ``sigma² = exp(b0 + b1*mu + b2*mu² + b3*mu³)`` with
    Normal(0, 2) priors on all four coefficients.  The polynomial is clipped
    to [-20, 20] before exp() to prevent overflow during nutpie initialization.

    """
    beta_sigma = pm.Normal("beta_sigma", mu=0, sigma=10, shape=4)
    poly = (
        beta_sigma[0] + beta_sigma[1] * mu +
        beta_sigma[2] * mu ** 2 +
        beta_sigma[3] * mu ** 3
    )
    return pm.math.exp(pt.clip(poly, -20, 20))


VARIANCE_FUNCTIONS: Dict[str, Callable] = {
    "homogeneous": variance_homogeneous,
    "covariate_dependent": variance_covariate_dependent,
    "polynomial": variance_polynomial,
}
"""Registry mapping variance string names to their functions.

Use this dict to inspect available options or extend the registry at runtime:

.. code-block:: python

    from gdmbayes import VARIANCE_FUNCTIONS
    print(list(VARIANCE_FUNCTIONS))  # ['homogeneous', 'covariate_dependent', 'polynomial']
"""


__all__ = [
    "variance_homogeneous",
    "variance_covariate_dependent",
    "variance_polynomial",
    "VARIANCE_FUNCTIONS",
]
