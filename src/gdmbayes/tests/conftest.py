"""Shared fixtures for gdmbayes tests."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from gdmbayes import GDM, GDMPreprocessor, ModelConfig, SamplerConfig, spGDMM


def _make_site_data(n_sites: int, seed: int = 42):
    """20-site-style DataFrame: xc, yc, time_idx, temp, depth + Bray-Curtis y."""
    np.random.seed(seed)
    X = pd.DataFrame(
        {
            "xc": np.random.uniform(0, 100, n_sites),
            "yc": np.random.uniform(0, 100, n_sites),
            "time_idx": np.zeros(n_sites, dtype=int),
            "temp": np.random.uniform(5, 20, n_sites),
            "depth": np.random.uniform(0, 200, n_sites),
        }
    )
    biomass = np.random.exponential(1, (n_sites, 10))
    y = np.clip(pdist(biomass, "braycurtis"), 1e-8, 1 - 1e-8)
    return X, y


@pytest.fixture(scope="session")
def site_data():
    """20 sites × 2 env predictors (temp, depth). Session-scoped, never mutated."""
    return _make_site_data(n_sites=20, seed=42)


@pytest.fixture(scope="session")
def site_data_small():
    """15 sites × 2 env predictors. For minimal-MCMC spGDMM fits."""
    return _make_site_data(n_sites=15, seed=7)


@pytest.fixture(scope="module")
def fitted_gdm(site_data):
    """Frequentist GDM fit once per test module."""
    X, y = site_data
    return GDM().fit(X, y)


@pytest.fixture(scope="module")
def fitted_spgdmm_fast(site_data_small):
    """Minimal-MCMC spGDMM fit (2 draws, 1 chain) shared across save/load + extract tests."""
    X, y = site_data_small
    model = spGDMM(
        preprocessor=GDMPreprocessor(deg=3, knots=2),
        model_config=ModelConfig(variance="homogeneous", spatial_effect="none"),
        sampler_config=SamplerConfig(
            draws=2, tune=2, chains=1, nuts_sampler="pymc", progressbar=False
        ),
    )
    model.fit(X, y)
    return model, X, y
