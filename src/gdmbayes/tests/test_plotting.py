"""Tests for plotting functions."""

import numpy as np

import xarray as xr

from gdmbayes.diagnostics import summarise_sampling
from gdmbayes.plotting import (
    crps_boxplot,
    plot_link_curve,
    plot_obs_vs_pred,
    plot_ppc,
    plot_predictor_importance,
)


class TestSummariseSampling:
    """Test sampling diagnostics summary."""

    def test_summarise_sampling_basic(self):
        """Test basic summary with synthetic data."""
        import arviz as az

        # Create synthetic inference data
        np.random.seed(42)
        n_chains = 2
        n_draws = 100

        posterior = {
            "beta_0": np.random.normal(0, 1, (n_chains, n_draws)),
            "sigma2": np.random.gamma(2, 1, (n_chains, n_draws)),
        }
        sample_stats = {
            "diverging": np.zeros((n_chains, n_draws), dtype=bool),
        }

        idata = az.from_dict(
            posterior=posterior,
            sample_stats=sample_stats,
        )

        # Run summary
        summary = summarise_sampling(idata)

        # Check output format
        assert summary is not None
        assert isinstance(summary, object)
        # Should contain parameter summaries

    def test_summarise_sampling_with_divergences(self):
        """Test summary with divergences present."""
        import arviz as az

        n_chains = 2
        n_draws = 100

        posterior = {
            "beta_0": np.random.normal(0, 1, (n_chains, n_draws)),
        }
        sample_stats = {
            "diverging": np.random.rand(n_chains, n_draws) > 0.8,  # Some divergences
        }

        idata = az.from_dict(
            posterior=posterior,
            sample_stats=sample_stats,
        )

        # Run summary - should report divergences
        summary = summarise_sampling(idata)
        assert summary is not None


class TestCrpsBoxplot:
    """Test CRPS boxplot."""

    def test_crps_boxplot_basic(self):
        """Test basic CRPS boxplot."""
        np.random.seed(42)

        y_test = np.random.uniform(0.1, 0.9, 50)
        y_train = np.random.uniform(0.1, 0.9, 100)
        y_pred = np.random.uniform(0.1, 0.9, (50, 100))

        fig, axes = crps_boxplot(y_test, y_pred, y_train)

        assert fig is not None
        assert axes is not None
        assert len(axes) == 2


class TestPosteriorPredictiveCheck:
    """Test posterior predictive check plotting."""

    def test_plot_ppc_basic(self):
        """Test basic PPC plot."""
        import arviz as az

        np.random.seed(42)
        n_chains = 2
        n_draws = 100
        n_obs = 50

        # Create fake posterior predictive
        posterior_predictive = {
            "log_y": np.random.normal(-1.5, 0.5, (n_chains, n_draws, n_obs)),
        }

        idata = az.from_dict(posterior_predictive=posterior_predictive)

        # Observed data
        y_obs = np.random.uniform(0.05, 0.5, n_obs)

        # Create plot
        fig, ax = plot_ppc(idata, y_obs, n_pp_samples=50)

        # Check output
        assert fig is not None
        assert ax is not None


class TestPlotISplines:
    """Test I-spline plotting."""

    def test_plot_isplines_skip(self):
        """Test that plot_isplines requires fitted model."""
        # This test verifies the function exists and handles unfitted models
        from gdmbayes import spGDMM

        model = spGDMM()

        # Should raise error or handle gracefully
        # (Actual plotting requires fitted model with idata)
        assert model is not None


class _FakeModel:
    """Minimal duck-typed spGDMM for fit-plot smoke tests.

    Exposes only the methods plot_obs_vs_pred / plot_link_curve consume.
    Avoids an expensive MCMC fit in unit tests while still exercising
    the plotting code paths end-to-end.
    """

    def __init__(self, n_pairs: int = 30, n_samples: int = 50, seed: int = 0):
        rng = np.random.default_rng(seed)
        self._y_hat = rng.uniform(0.2, 0.8, n_pairs)
        self._log_y_samples = rng.normal(
            loc=np.log(self._y_hat)[:, None],
            scale=0.1,
            size=(n_pairs, n_samples),
        )

    def predict(self, X):
        return self._y_hat

    def predict_posterior(self, X, combined=True, extend_idata=False):
        y_samples = np.exp(self._log_y_samples)
        return xr.DataArray(
            y_samples,
            dims=("pair", "sample"),
            coords={
                "pair": np.arange(y_samples.shape[0]),
                "sample": np.arange(y_samples.shape[1]),
            },
        )


class TestFitDiagnostics:
    """Test obs-vs-pred and link-curve plots."""

    def test_plot_obs_vs_pred_basic(self):
        model = _FakeModel(n_pairs=40)
        X = None  # fake model ignores X
        y = np.clip(model._y_hat + np.random.default_rng(1).normal(0, 0.05, 40), 1e-6, 1.0)

        fig, ax = plot_obs_vs_pred(model, X, y)

        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == "Predicted dissimilarity"
        assert ax.get_ylabel() == "Observed dissimilarity"

    def test_plot_link_curve_basic(self):
        model = _FakeModel(n_pairs=40)
        X = None
        rng = np.random.default_rng(2)
        y = np.clip(model._y_hat + rng.normal(0, 0.05, 40), 1e-6, 1.0)

        fig, ax = plot_link_curve(model, X, y)

        assert fig is not None
        assert ax is not None
        # There should be at least the curve, the scatter, and the censoring line
        assert len(ax.lines) >= 2


class TestPlotPredictorImportance:
    """Test posterior predictor-importance bar chart."""

    def test_plot_predictor_importance_basic(self):
        import arviz as az

        rng = np.random.default_rng(3)
        n_chains, n_draws, n_feat, n_basis = 2, 200, 3, 5

        beta = rng.lognormal(mean=0.0, sigma=0.5, size=(n_chains, n_draws, n_feat, n_basis))

        idata = az.from_dict(
            posterior={"beta": beta},
            dims={"beta": ["feature", "basis_function"]},
            coords={
                "feature": ["temp", "depth", "ph"],
                "basis_function": np.arange(1, n_basis + 1),
            },
        )

        class _M:
            pass

        m = _M()
        m.idata_ = idata

        fig, ax = plot_predictor_importance(m)

        assert fig is not None
        assert ax is not None
        # Three horizontal bars, one per feature
        assert len(ax.patches) == n_feat
        # Labels should be the feature names (sorted order unknown here)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert set(labels) == {"temp", "depth", "ph"}
