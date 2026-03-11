"""Tests for spGDMM model implementation."""

import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from spgdmm import (
    spGDMM,
    ModelConfig,
    SamplerConfig,
    PreprocessorConfig,
    GDMPreprocessor,
)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_config(self):
        """Test default model configuration — preprocessing fields are no longer present."""
        config = ModelConfig()
        assert config.alpha_importance is True
        assert config.variance == "homogeneous"
        assert config.spatial_effect == "none"
        # Preprocessing fields must NOT exist on ModelConfig any more
        assert not hasattr(config, "deg")
        assert not hasattr(config, "knots")
        assert not hasattr(config, "mesh_choice")
        assert not hasattr(config, "distance_measure")
        assert not hasattr(config, "extrapolation")

    def test_custom_variance_callable(self):
        """Test that a callable can be passed as variance."""
        def my_fn(mu, X_sigma):
            return mu

        config = ModelConfig(variance=my_fn)
        assert config.variance is my_fn

    def test_custom_spatial_effect_callable(self):
        """Test that a callable can be passed as spatial_effect."""
        def my_fn(psi, row_ind, col_ind):
            return psi[row_ind] - psi[col_ind]

        config = ModelConfig(spatial_effect=my_fn)
        assert config.spatial_effect is my_fn

    def test_invalid_variance_string(self):
        """Test that an invalid variance string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variance"):
            ModelConfig(variance="bad_name")

    def test_invalid_spatial_effect_string(self):
        """Test that an invalid spatial_effect string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown spatial_effect"):
            ModelConfig(spatial_effect="bad_name")

    def test_config_to_dict(self):
        """Test converting config to dictionary — only Bayesian model fields."""
        config = ModelConfig()
        d = config.to_dict()
        assert d["variance"] == "homogeneous"
        assert d["spatial_effect"] == "none"
        # Preprocessing keys must not be present
        assert "deg" not in d
        assert "knots" not in d
        assert "mesh_choice" not in d
        assert "distance_measure" not in d

    def test_config_from_dict(self):
        """Test creating config from dictionary — preprocessing keys emit DeprecationWarning."""
        import warnings as _warnings
        d = {"variance": "homogeneous", "spatial_effect": "none"}
        config = ModelConfig.from_dict(d)
        assert config.variance == "homogeneous"

        # Legacy keys should trigger a DeprecationWarning but not crash
        legacy_d = {"deg": 5, "knots": 2, "mesh_choice": "even", "distance_measure": "geodesic"}
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            config2 = ModelConfig.from_dict(legacy_d)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        # Result is still a valid ModelConfig
        assert isinstance(config2, ModelConfig)


class TestPreprocessorConfig:
    """Test PreprocessorConfig dataclass."""

    def test_default_config(self):
        """Test default preprocessor configuration."""
        cfg = PreprocessorConfig()
        assert cfg.deg == 3
        assert cfg.knots == 2
        assert cfg.mesh_choice == "percentile"
        assert cfg.distance_measure == "euclidean"
        assert cfg.extrapolation == "clip"
        assert cfg.custom_dist_mesh is None
        assert cfg.custom_predictor_mesh is None

    def test_to_dict(self):
        """Test to_dict excludes numpy arrays."""
        cfg = PreprocessorConfig(deg=4, knots=3, distance_measure="geodesic")
        d = cfg.to_dict()
        assert d["deg"] == 4
        assert d["knots"] == 3
        assert d["distance_measure"] == "geodesic"
        assert "custom_dist_mesh" not in d
        assert "custom_predictor_mesh" not in d

    def test_from_dict(self):
        """Test from_dict round-trip."""
        d = {"deg": 2, "knots": 1, "mesh_choice": "even", "extrapolation": "nan"}
        cfg = PreprocessorConfig.from_dict(d)
        assert cfg.deg == 2
        assert cfg.knots == 1
        assert cfg.mesh_choice == "even"
        assert cfg.extrapolation == "nan"

    def test_from_dict_defaults(self):
        """Test from_dict with empty dict gives defaults."""
        cfg = PreprocessorConfig.from_dict({})
        assert cfg.deg == 3
        assert cfg.extrapolation == "clip"


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
        assert model._config.variance == "homogeneous"
        assert model._config.spatial_effect == "none"
        # Preprocessing settings now live on the preprocessor
        assert model.preprocessor._get_config().deg == 3
        assert model.preprocessor._get_config().knots == 2

    def test_model_from_config(self):
        """Test creating model from preprocessor config."""
        prep_config = PreprocessorConfig(deg=4, knots=3)
        model = spGDMM(preprocessor=prep_config)
        assert model.preprocessor._get_config().deg == 4
        assert model.preprocessor._get_config().knots == 3

    def test_model_from_model_config(self):
        """Test creating model from ModelConfig."""
        model_config = ModelConfig(variance="polynomial")
        model = spGDMM(model_config=model_config)
        assert model._config.variance == "polynomial"

    def test_output_var(self, sample_data):
        """Test output_var property."""
        X, y = sample_data
        model = spGDMM()
        model._generate_and_preprocess_model_data(X, y)
        assert model.output_var == "log_y"

    def test_get_default_model_config(self):
        """Test get_default_model_config method returns Bayesian model fields."""
        model = spGDMM()
        config = model.get_default_model_config()
        assert isinstance(config, dict)
        assert "variance" in config
        assert "spatial_effect" in config
        # Preprocessing fields are not in ModelConfig
        assert "deg" not in config

    def test_get_default_sampler_config(self):
        """Test get_default_sampler_config method."""
        model = spGDMM()
        config = model.get_default_sampler_config()
        assert isinstance(config, dict)
        assert config["draws"] == 1000
        assert config["chains"] == 4

    def test_serializable_config(self):
        """Test _serializable_model_config property returns Bayesian model fields."""
        model = spGDMM()
        config = model._serializable_model_config
        assert isinstance(config, dict)
        assert "variance" in config
        # Preprocessing fields not in model config
        assert "deg" not in config
        assert "knots" not in config

    def test_pw_distance(self):
        """Test pairwise distance calculation."""
        locations = np.array([[0, 0], [1, 0], [0, 1]])
        model = spGDMM()
        dists = model.preprocessor.pw_distance(locations)
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


class TestExtrapolation:
    """Test extrapolation mode handling in _transform_for_prediction."""

    @pytest.fixture
    def fitted_model(self):
        """Set up a preprocessed model with a stub idata so _transform_for_prediction runs."""
        np.random.seed(0)
        n_sites = 15
        X = pd.DataFrame({
            "xc": np.linspace(0, 100, n_sites),
            "yc": np.linspace(0, 100, n_sites),
            "time_idx": np.zeros(n_sites, dtype=int),
            "temp": np.linspace(5, 20, n_sites),
        })
        biomass = np.random.exponential(1, (n_sites, 5))
        y = pdist(biomass, "braycurtis")
        y = np.clip(y, 1e-8, None)

        model = spGDMM()
        model._generate_and_preprocess_model_data(X, y)
        # Minimal stub so the fitted-check passes
        model.idata = az.from_dict({"posterior": {"dummy": np.ones((1, 1))}})
        return model, X

    def _make_oob_X(self, X_train):
        """Return a copy of X with one site's predictor pushed out of training range."""
        X_pred = X_train.copy()
        X_pred.iloc[0, 3] = X_train.iloc[:, 3].max() + 50  # far above max temp
        return X_pred

    def test_default_extrapolation_is_clip(self):
        """Extrapolation default lives on PreprocessorConfig."""
        cfg = PreprocessorConfig()
        assert cfg.extrapolation == "clip"

    def test_extrapolation_in_to_dict(self):
        cfg = PreprocessorConfig(extrapolation="error")
        d = cfg.to_dict()
        assert d["extrapolation"] == "error"

    def test_extrapolation_in_from_dict(self):
        cfg = PreprocessorConfig.from_dict({"extrapolation": "nan"})
        assert cfg.extrapolation == "nan"

    def test_from_dict_missing_extrapolation_defaults_to_clip(self):
        cfg = PreprocessorConfig.from_dict({})
        assert cfg.extrapolation == "clip"

    def test_clip_mode_warns(self, fitted_model):
        model, X = fitted_model
        X_oob = self._make_oob_X(X)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._transform_for_prediction(X_oob)
        assert any("clipped" in str(w.message).lower() for w in caught)

    def test_clip_mode_returns_array(self, fitted_model):
        model, X = fitted_model
        X_oob = self._make_oob_X(X)
        result = model._transform_for_prediction(X_oob)
        assert isinstance(result, np.ndarray)
        assert not np.isnan(result).any()

    def test_error_mode_raises(self, fitted_model):
        model, X = fitted_model
        model.preprocessor.config = PreprocessorConfig(extrapolation="error")
        X_oob = self._make_oob_X(X)
        with pytest.raises(ValueError, match="outside predictor_mesh bounds"):
            model._transform_for_prediction(X_oob)

    def test_error_mode_no_raise_in_range(self, fitted_model):
        model, X = fitted_model
        model.preprocessor.config = PreprocessorConfig(extrapolation="error")
        # Should not raise when all values are within training range
        model._transform_for_prediction(X)

    def test_nan_mode_produces_nans(self, fitted_model):
        model, X = fitted_model
        model.preprocessor.config = PreprocessorConfig(extrapolation="nan")
        X_oob = self._make_oob_X(X)
        result = model._transform_for_prediction(X_oob)
        # Some rows must contain NaN (pairs involving the out-of-range site)
        assert np.isnan(result).any()

    def test_nan_mode_inrange_pairs_are_finite(self, fitted_model):
        model, X = fitted_model
        model.preprocessor.config = PreprocessorConfig(extrapolation="nan")
        # All in-range — no NaN expected
        result = model._transform_for_prediction(X)
        assert not np.isnan(result).any()