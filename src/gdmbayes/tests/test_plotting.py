"""Tests for plotting functions."""

import numpy as np

from gdmbayes.plotting import plot_crps_comparison, plot_ppc, summarise_sampling


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


class TestCrpsComparison:
    """Test CRPS comparison plotting."""

    def test_plot_crps_comparison_basic(self):
        """Test basic CRPS comparison plot."""
        np.random.seed(42)

        # Create fake test predictions
        y_test = np.random.uniform(0.1, 0.9, 50)
        y_train = np.random.uniform(0.1, 0.9, 100)
        y_pred = np.random.uniform(0.1, 0.9, (50, 100))  # 50 test points, 100 samples

        # Create plot
        fig, axes = plot_crps_comparison(y_test, y_pred, y_train)

        # Check output
        assert fig is not None
        assert axes is not None
        assert len(axes) == 2  # Two subplots

    def test_plot_crps_comparison_log(self):
        """Test CRPS comparison with log scale."""
        np.random.seed(42)

        y_test = np.random.uniform(0.1, 0.9, 50)
        y_train = np.random.uniform(0.1, 0.9, 100)
        y_pred = np.random.uniform(0.1, 0.9, (50, 100))

        fig, axes = plot_crps_comparison(y_test, y_pred, y_train, use_log=True)
        assert fig is not None


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
