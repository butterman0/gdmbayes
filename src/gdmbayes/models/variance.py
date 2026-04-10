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

from collections.abc import Callable

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
    with Normal(0, 10) priors on ``beta_sigma``.  ``X_sigma`` may be a numpy
    array or a PyMC ``pm.Data`` variable (shape ``(n_pairs, k)``).  The number
    of columns ``k`` is read from the static shape when ``X_sigma`` is a
    symbolic tensor, falling back to 1 (the default: pairwise geographic distance).
    The linear predictor is clipped to [-20, 20] before exp() to prevent
    overflow during nutpie initialization.
    Falls back to :func:`variance_homogeneous` when ``X_sigma`` is None.

    Uses a non-centered parameterization (``beta_sigma_raw ~ Normal(0, 1)``,
    ``beta_sigma = 10 * beta_sigma_raw``) to mitigate Neal's funnel geometry
    that arises when ``beta_sigma`` and ``beta`` interact through the likelihood.
    The implied prior on ``beta_sigma`` is unchanged: Normal(0, 10).
    """
    if X_sigma is not None:
        # Support both numpy arrays (shape known directly) and pm.Data/symbolic
        # tensors (static shape may be unknown, so evaluate to get the concrete shape).
        if isinstance(X_sigma, np.ndarray):
            n_cols = X_sigma.shape[1]
        else:
            n_cols = int(X_sigma.shape.eval()[1])
        # Non-centered parameterization: sample in unit-scale space to avoid
        # Neal's funnel between beta_sigma and beta.
        beta_sigma_raw = pm.Normal("beta_sigma_raw", mu=0, sigma=1, shape=n_cols)
        beta_sigma = pm.Deterministic("beta_sigma", 10 * beta_sigma_raw)
        return pm.math.exp(pt.clip(pm.math.dot(X_sigma, beta_sigma), -20, 20))
    return pm.InverseGamma("sigma2", alpha=1, beta=1)


def variance_polynomial(mu, X_sigma):
    """Variance as a cubic polynomial function of the mean ``mu``.

    Fits ``sigma² = exp(b0 + b1*mu + b2*mu² + b3*mu³)`` with
    Normal(0, 10) priors on all four coefficients.  The polynomial is clipped
    to [-20, 20] before exp() to prevent overflow during nutpie initialization.

    Uses a non-centered parameterization (``beta_sigma_raw ~ Normal(0, 1)``,
    ``beta_sigma = 10 * beta_sigma_raw``) to mitigate Neal's funnel geometry.
    The implied prior on ``beta_sigma`` is unchanged: Normal(0, 10).
    """
    # Non-centered parameterization: sample in unit-scale space.
    beta_sigma_raw = pm.Normal("beta_sigma_raw", mu=0, sigma=1, shape=4)
    beta_sigma = pm.Deterministic("beta_sigma", 10 * beta_sigma_raw)
    poly = (
        beta_sigma[0] + beta_sigma[1] * mu +
        beta_sigma[2] * mu ** 2 +
        beta_sigma[3] * mu ** 3
    )
    return pm.math.exp(pt.clip(poly, -20, 20))


VARIANCE_FUNCTIONS: dict[str, Callable] = {
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


def poly_fit(x: np.ndarray, degree: int = 3):
    """Fit orthogonal polynomials, replicating R's ``poly(x, degree)``.

    Returns ``(Z, alpha, norm2)`` where *Z* is an ``(n, degree)`` matrix of
    orthonormal columns (each with unit L2 norm), *alpha* and *norm2* are the
    three-term recurrence coefficients needed by :func:`poly_predict`.
    """
    x = np.asarray(x, dtype=float)
    xbar = float(x.mean())
    xc = x - xbar
    # Vandermonde including degree-0 constant column
    V = np.column_stack([xc ** k for k in range(degree + 1)])
    Q, R = np.linalg.qr(V)
    # R's poly() uses Q * diag(R), not plain Q, before renormalizing
    Z = Q * np.diag(R)[None, :]
    norm2 = (Z ** 2).sum(axis=0)  # length degree+1
    alpha = ((xc[:, None] * Z ** 2).sum(axis=0) / norm2 + xbar)[:degree]
    norm2 = np.concatenate([[1.0], norm2])  # length degree+2
    Z = Z / np.sqrt(norm2[1:])  # unit L2 norm columns
    Z = Z[:, 1:]  # drop intercept
    return Z, alpha, norm2


def poly_predict(x_new: np.ndarray, alpha: np.ndarray, norm2: np.ndarray):
    """Evaluate orthogonal polynomial basis on new data.

    Uses the three-term recurrence with coefficients from :func:`poly_fit`,
    matching R's ``predict.poly()`` behaviour.
    """
    x_new = np.asarray(x_new, dtype=float)
    degree = len(alpha)
    Z = np.zeros((len(x_new), degree + 1))
    Z[:, 0] = 1.0
    Z[:, 1] = x_new - alpha[0]
    for i in range(1, degree):
        Z[:, i + 1] = ((x_new - alpha[i]) * Z[:, i]
                        - (norm2[i + 1] / norm2[i]) * Z[:, i - 1])
    Z = Z / np.sqrt(norm2[1:])
    return Z[:, 1:]  # drop intercept


__all__ = [
    "variance_homogeneous",
    "variance_covariate_dependent",
    "variance_polynomial",
    "VARIANCE_FUNCTIONS",
    "poly_fit",
    "poly_predict",
]
