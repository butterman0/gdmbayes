"""Tests for spGDMM model implementation."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from spgdmm import (
    spGDMM,
    ModelConfig,
    VarianceType,
    SpatialEffectType,
    SamplerConfig,
)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.deg == 3
        assert config.knots == 2
        assert config.mesh_choice == "percentile"
        assert config.distance_measure == "euclidean"
        assert config.alpha_importance is True
        assert config.variance_type == VarianceType.HOMOGENEOUS
        assert config.spatial_effect_type == SpatialEffectType.NONE

    def test_custom_variance_type(self):
        """Test CUSTOM variance type requires a callable."""
        config = ModelConfig(variance_type=VarianceType.CUSTOM)
        assert config.variance_type == VarianceType.CUSTOM
        assert config.custom_variance_fn is None

        def my_fn(mu, X_sigma):
            return mu

        config2 = ModelConfig(variance_type=VarianceType.CUSTOM, custom_variance_fn=my_fn)
        assert config2.custom_variance_fn is my_fn

    def test_custom_spatial_effect_type(self):
        """Test CUSTOM spatial effect type requires a callable."""
        config = ModelConfig(spatial_effect_type=SpatialEffectType.CUSTOM)
        assert config.spatial_effect_type == SpatialEffectType.CUSTOM
        assert config.custom_spatial_effect_fn is None

        def my_fn(psi, row_ind, col_ind):
            return psi[row_ind] - psi[col_ind]

        config2 = ModelConfig(
            spatial_effect_type=SpatialEffectType.CUSTOM,
            custom_spatial_effect_fn=my_fn,
        )
        assert config2.custom_spatial_effect_fn is my_fn

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ModelConfig(deg=4, knots=3)
        d = config.to_dict()
        assert d["deg"] == 4
        assert d["knots"] == 3
        assert d["variance_type"] == "homogeneous"

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        d = {"deg": 5, "knots": 2, "mesh_choice": "even", "distance_measure": "geodesic"}
        config = ModelConfig.from_dict(d)
        assert config.deg == 5
        assert config.knots == 2
        assert config.mesh_choice == "even"
        assert config.distance_measure == "geodesic"


class TestSamplerConfig:
    """Test SamplerConfig dataclass."""

    def test_default_sampler_config(self):
        """Test default sampler configuration."""
        config = SamplerConfig()
        assert config.draws == 1000
        assert config.tune == 1000
        assert config.chains == 4
        assert config.target_accept == 0.95
        assert config.nuts_sampler == "nutpie"

    def test_sampler_config_to_dict(self):
        """Test converting sampler config to dictionary."""
        config = SamplerConfig(draws=500, chains=2)
        d = config.to_dict()
        assert d["draws"] == 500
        assert d["chains"] == 2

    def test_sampler_config_from_dict(self):
        """Test creating sampler config from dictionary."""
        d = {"draws": 200, "tune": 50, "chains": 1, "random_seed": 42}
        config = SamplerConfig.from_dict(d)
        assert config.draws == 200
        assert config.tune == 50
        assert config.chains == 1
        assert config.random_seed == 42


class TestSpGDMM:
    """Test spGDMM model class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_sites = 20

        # Create site-level predictors
        X = pd.DataFrame({
            "xc": np.random.uniform(0, 100, n_sites),
            "yc": np.random.uniform(0, 100, n_sites),
            "time_idx": np.zeros(n_sites, dtype=int),
            "temp": np.random.uniform(5, 20, n_sites),
            "depth": np.random.uniform(0, 200, n_sites),
        })

        # Create biomass and compute dissimilarities
        biomass = np.random.exponential(1, (n_sites, 10))
        y = pdist(biomass, "braycurtis")
        y = np.clip(y, 1e-8, None)

        return X, y

    def test_model_initialization_default(self):
        """Test model initialization with defaults."""
        model = spGDMM()
        assert model._config.variance_type == VarianceType.HOMOGENEOUS
        assert model._config.spatial_effect_type == SpatialEffectType.NONE
        assert model._config.deg == 3
        assert model._config.knots == 2

    def test_model_from_config(self):
        """Test creating model from config."""
        config = ModelConfig(deg=4, knots=3)
        model = spGDMM(config=config)
        assert model._config.deg == 4
        assert model._config.knots == 3

    def test_output_var(self, sample_data):
        """Test output_var property."""
        X, y = sample_data
        model = spGDMM()
        model._generate_and_preprocess_model_data(X, y)
        assert model.output_var == "log_y"

    def test_get_default_model_config(self):
        """Test get_default_model_config method."""
        model = spGDMM()
        config = model.get_default_model_config()
        assert isinstance(config, dict)
        assert "deg" in config

    def test_get_default_sampler_config(self):
        """Test get_default_sampler_config method."""
        model = spGDMM()
        config = model.get_default_sampler_config()
        assert isinstance(config, dict)
        assert config["draws"] == 1000
        assert config["chains"] == 4

    def test_serializable_config(self):
        """Test _serializable_model_config property."""
        model = spGDMM()
        config = model._serializable_model_config
        assert isinstance(config, dict)
        assert "deg" in config
        assert "knots" in config

    def test_pw_distance(self):
        """Test pairwise distance calculation."""
        locations = np.array([[0, 0], [1, 0], [0, 1]])
        model = spGDMM()
        dists = model.pw_distance(locations, distance_measure="euclidean")
        assert len(dists) == 3  # n*(n-1)/2 for n=3
        assert np.all(dists >= 0)


class TestDataPreprocessing:
    """Test data preprocessing methods."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        n_sites = 20

        X = pd.DataFrame({
            "xc": np.random.uniform(0, 100, n_sites),
            "yc": np.random.uniform(0, 100, n_sites),
            "time_idx": np.zeros(n_sites, dtype=int),
            "temp": np.random.uniform(5, 20, n_sites),
        })

        biomass = np.random.exponential(1, (n_sites, 5))
        y = pdist(biomass, "braycurtis")
        y = np.clip(y, 1e-8, None)

        return X, y

    def test_generate_and_preprocess_model_data(self, sample_data):
        """Test data preprocessing."""
        X, y = sample_data
        model = spGDMM()
        model._generate_and_preprocess_model_data(X, y)

        # Check metadata is created
        assert model.metadata is not None
        assert model.training_metadata is not None

        # Check transformed data exists
        assert hasattr(model, "X_transformed")
        assert hasattr(model, "y_transformed")

    def test_build_model(self, sample_data):
        """Test model building."""
        X, y = sample_data
        model = spGDMM()
        model._generate_and_preprocess_model_data(X, y)
        model.build_model(model.X_transformed, model.y_transformed)

        assert model.model is not None
        # Check that key random variables exist in the model
        assert "beta_0" in model.model.named_vars
        # With alpha_importance=True, the model has beta, alpha, beta_dist for distance splines
        assert "beta" in model.model.named_vars