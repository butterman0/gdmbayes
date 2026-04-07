"""Tests for spGDMM model implementation."""

import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from gdmbayes import (
    spGDMM,
    GDM,
    GDMModel,
    GDMResult,
    ModelConfig,
    SamplerConfig,
    PreprocessorConfig,
    GDMPreprocessor,
    site_pairs,
    holdout_pairs,
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
        """Test creating config from dictionary."""
        d = {"variance": "homogeneous", "spatial_effect": "none"}
        config = ModelConfig.from_dict(d)
        assert config.variance == "homogeneous"


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

    def test_n_features_in_not_set_before_preprocess(self):
        """n_features_in_ should not be set before _generate_and_preprocess_model_data."""
        model = spGDMM()
        assert not hasattr(model, "n_features_in_")

    def test_n_features_in_set_after_preprocess(self, sample_data):
        """n_features_in_ should equal the number of env predictors after preprocessing."""
        X, y = sample_data
        model = spGDMM()
        model._generate_and_preprocess_model_data(X, y)
        # sample_data has 2 env predictors: temp, depth
        assert model.n_features_in_ == 2

    def test_set_idata_attrs_sets_required_attrs(self, sample_data):
        """_set_idata_attrs must stamp id, model_type, version, model_config, sampler_config."""
        import json
        X, y = sample_data
        model = spGDMM()
        model._generate_and_preprocess_model_data(X, y)
        # Build a minimal stub idata with no posterior so we can call _set_idata_attrs
        stub = az.from_dict({"posterior": {"dummy": np.ones((1, 1))}})
        model.idata = stub  # needed so _save_input_params doesn't raise
        result = model._set_idata_attrs(stub)
        assert "id" in result.attrs
        assert "model_type" in result.attrs
        assert "version" in result.attrs
        assert "model_config" in result.attrs
        assert "sampler_config" in result.attrs
        # Values should be parseable JSON
        json.loads(result.attrs["model_config"])
        json.loads(result.attrs["sampler_config"])

    def test_site_pairs_count(self, sample_data):
        """site_pairs returns n_subset*(n_subset-1)//2 indices for a site subset."""
        X, y = sample_data
        n_sites = len(X)
        # Use sites 0..12 (13 sites → 78 pairs)
        subset = np.arange(13)
        idx = site_pairs(n_sites, subset)
        assert len(idx) == 13 * 12 // 2

    def test_site_level_cv_fit_predict(self, sample_data):
        """Fit GDM on training sites and predict on held-out test sites."""
        X, y = sample_data
        n_sites = len(X)
        # Split into first 15 train sites and last 5 test sites
        train_sites = np.arange(15)
        test_sites = np.arange(15, n_sites)

        train_pair_idx = site_pairs(n_sites, train_sites)
        test_pair_idx = site_pairs(n_sites, test_sites)

        X_train = X.iloc[train_sites].reset_index(drop=True)
        y_train = y[train_pair_idx]
        X_test = X.iloc[test_sites].reset_index(drop=True)

        m = GDM()
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

        expected_n_test_pairs = len(test_sites) * (len(test_sites) - 1) // 2
        assert preds.shape == (expected_n_test_pairs,)
        assert len(test_pair_idx) == expected_n_test_pairs


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
    """Test extrapolation mode handling in preprocessor.transform()."""

    @pytest.fixture
    def fitted_model(self):
        """Set up a preprocessed model."""
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
            model.preprocessor.transform(X_oob)
        assert any("clipped" in str(w.message).lower() for w in caught)

    def test_clip_mode_returns_array(self, fitted_model):
        model, X = fitted_model
        X_oob = self._make_oob_X(X)
        result = model.preprocessor.transform(X_oob)
        assert isinstance(result, np.ndarray)
        assert not np.isnan(result).any()

    def test_error_mode_raises(self, fitted_model):
        model, X = fitted_model
        model.preprocessor.config = PreprocessorConfig(extrapolation="error")
        X_oob = self._make_oob_X(X)
        with pytest.raises(ValueError, match="outside predictor_mesh bounds"):
            model.preprocessor.transform(X_oob)

    def test_error_mode_no_raise_in_range(self, fitted_model):
        model, X = fitted_model
        model.preprocessor.config = PreprocessorConfig(extrapolation="error")
        # Should not raise when all values are within training range
        model.preprocessor.transform(X)

    def test_nan_mode_produces_nans(self, fitted_model):
        model, X = fitted_model
        model.preprocessor.config = PreprocessorConfig(extrapolation="nan")
        X_oob = self._make_oob_X(X)
        result = model.preprocessor.transform(X_oob)
        # Some rows must contain NaN (pairs involving the out-of-range site)
        assert np.isnan(result).any()

    def test_nan_mode_inrange_pairs_are_finite(self, fitted_model):
        model, X = fitted_model
        model.preprocessor.config = PreprocessorConfig(extrapolation="nan")
        # All in-range — no NaN expected
        result = model.preprocessor.transform(X)
        assert not np.isnan(result).any()


# ---------------------------------------------------------------------------
# Helpers shared by sklearn interface and save/load tests
# ---------------------------------------------------------------------------

def _make_sample_data(n_sites=20, seed=42):
    np.random.seed(seed)
    X = pd.DataFrame({
        "xc": np.random.uniform(0, 100, n_sites),
        "yc": np.random.uniform(0, 100, n_sites),
        "time_idx": np.zeros(n_sites, dtype=int),
        "temp": np.random.uniform(5, 20, n_sites),
        "depth": np.random.uniform(0, 200, n_sites),
    })
    biomass = np.random.exponential(1, (n_sites, 10))
    y = np.clip(pdist(biomass, "braycurtis"), 1e-8, None)
    return X, y


class TestSpGDMMSklearnInterface:
    """Sklearn-style interface tests — no MCMC required."""

    @pytest.fixture
    def model(self):
        prep_cfg = PreprocessorConfig(deg=3, knots=2)
        model_cfg = ModelConfig(variance="homogeneous", spatial_effect="none")
        return spGDMM(preprocessor=prep_cfg, model_config=model_cfg)

    @pytest.fixture
    def preprocessed_model(self):
        X, y = _make_sample_data()
        prep_cfg = PreprocessorConfig(deg=3, knots=2)
        model_cfg = ModelConfig(variance="homogeneous", spatial_effect="none")
        m = spGDMM(preprocessor=prep_cfg, model_config=model_cfg)
        m._generate_and_preprocess_model_data(X, y)
        return m

    def test_get_params_returns_all_init_params(self, model):
        params = model.get_params()
        assert "preprocessor" in params
        assert "model_config" in params
        assert "sampler_config" in params

    def test_set_params_updates_model_config(self, model):
        new_cfg = {"variance": "polynomial", "spatial_effect": "none", "alpha_importance": True}
        result = model.set_params(model_config=new_cfg)
        assert result is model
        # The raw dict stored as model_config is updated
        assert model.model_config["variance"] == "polynomial"

    def test_clone_preserves_preprocessor_config(self, model):
        cloned = clone(model)
        assert cloned.preprocessor._get_config().deg == model.preprocessor._get_config().deg

    def test_clone_preserves_model_config(self, model):
        cloned = clone(model)
        assert cloned._config.variance == model._config.variance

    def test_clone_produces_unfitted_model(self, preprocessed_model):
        cloned = clone(preprocessed_model)
        assert cloned.idata is None
        assert not hasattr(cloned, "n_features_in_")

    def test_not_fitted_raises_check_is_fitted(self, model):
        with pytest.raises(NotFittedError):
            check_is_fitted(model)

    def test_fitted_after_preprocess(self, preprocessed_model):
        """check_is_fitted should NOT raise after we inject a stub posterior."""
        preprocessed_model.idata = az.from_dict({"posterior": {"dummy": np.ones((1, 1))}})
        # Should not raise
        check_is_fitted(preprocessed_model)

    def test_id_is_deterministic(self, model):
        assert model.id == model.id
        # Two models with identical config share the same id
        model2 = spGDMM(
            preprocessor=PreprocessorConfig(deg=3, knots=2),
            model_config=ModelConfig(variance="homogeneous", spatial_effect="none"),
        )
        assert model.id == model2.id

    def test_preprocessor_config_deg(self):
        model = spGDMM(preprocessor=PreprocessorConfig(deg=5))
        assert model.preprocessor.config.deg == 5

    def test_preprocessor_skips_refit(self):
        X, y = _make_sample_data()
        m = spGDMM()
        m._generate_and_preprocess_model_data(X, y)
        mesh_before = m.preprocessor.predictor_mesh_.copy()
        m._generate_and_preprocess_model_data(X, y)
        np.testing.assert_array_equal(m.preprocessor.predictor_mesh_, mesh_before)


class TestHoldoutCV:
    """Tests for masked-holdout CV (White et al. 2024 strategy)."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for holdout tests."""
        np.random.seed(42)
        n_sites = 20
        X = pd.DataFrame({
            "xc": np.random.uniform(0, 100, n_sites),
            "yc": np.random.uniform(0, 100, n_sites),
            "time_idx": np.zeros(n_sites, dtype=int),
            "temp": np.random.uniform(5, 20, n_sites),
            "depth": np.random.uniform(0, 200, n_sites),
        })
        biomass = np.random.exponential(1, (n_sites, 10))
        y = np.clip(pdist(biomass, "braycurtis"), 1e-8, None)
        return X, y

    def test_holdout_pairs_basic(self):
        """holdout_pairs returns pairs where either site is in test set."""
        n_sites = 5
        test_sites = [3, 4]
        idx = holdout_pairs(n_sites, test_sites)
        all_row, all_col = np.triu_indices(n_sites, k=1)
        for i in idx:
            assert all_row[i] in test_sites or all_col[i] in test_sites

    def test_holdout_pairs_count(self):
        """holdout_pairs with k test sites out of n should return n*(n-1)/2 - (n-k)*(n-k-1)/2."""
        n_sites = 10
        test_sites = [7, 8, 9]
        idx = holdout_pairs(n_sites, test_sites)
        n_train = n_sites - len(test_sites)
        expected = n_sites * (n_sites - 1) // 2 - n_train * (n_train - 1) // 2
        assert len(idx) == expected

    def test_holdout_pairs_disjoint_from_site_pairs(self):
        """holdout_pairs and site_pairs(train) should partition all pairs."""
        n_sites = 10
        test_sites = [7, 8, 9]
        train_sites = [i for i in range(n_sites) if i not in test_sites]
        hold_idx = holdout_pairs(n_sites, test_sites)
        train_idx = site_pairs(n_sites, train_sites)
        total = n_sites * (n_sites - 1) // 2
        assert len(hold_idx) + len(train_idx) == total
        assert len(set(hold_idx) & set(train_idx)) == 0

    def test_build_model_holdout(self, sample_data):
        """Model with holdout_mask has log_y_holdout as a free RV."""
        X, y = sample_data
        n_pairs = len(y)
        mask = np.zeros(n_pairs, dtype=bool)
        mask[:50] = True  # hold out first 50 pairs

        model = spGDMM(
            preprocessor=PreprocessorConfig(deg=3, knots=2),
            model_config=ModelConfig(variance="homogeneous", spatial_effect="none"),
        )
        model._generate_and_preprocess_model_data(X, y, holdout_mask=mask)
        model.build_model(model.X, model.y)

        assert "log_y_holdout" in model.model.named_vars
        free_rv_names = [v.name for v in model.model.free_RVs]
        assert "log_y_holdout" in free_rv_names

    def test_build_model_no_holdout_unchanged(self, sample_data):
        """Without holdout_mask, model should NOT have log_y_holdout."""
        X, y = sample_data
        model = spGDMM(
            preprocessor=PreprocessorConfig(deg=3, knots=2),
            model_config=ModelConfig(variance="homogeneous", spatial_effect="none"),
        )
        model._generate_and_preprocess_model_data(X, y)
        model.build_model(model.X, model.y)
        assert "log_y_holdout" not in model.model.named_vars

    def test_build_model_holdout_covariate_dependent(self, sample_data):
        """Holdout works with covariate_dependent variance (vector sigma2)."""
        X, y = sample_data
        n_pairs = len(y)
        mask = np.zeros(n_pairs, dtype=bool)
        mask[:50] = True

        model = spGDMM(
            preprocessor=PreprocessorConfig(deg=3, knots=2),
            model_config=ModelConfig(variance="covariate_dependent", spatial_effect="none"),
        )
        model._generate_and_preprocess_model_data(X, y, holdout_mask=mask)
        model.build_model(model.X, model.y)

        assert "log_y_holdout" in model.model.named_vars

    def test_fit_with_holdout_mask(self, sample_data):
        """Smoke test: fit with holdout_mask completes and extract_holdout_predictions works."""
        X, y = sample_data
        n_pairs = len(y)
        mask = np.zeros(n_pairs, dtype=bool)
        mask[:50] = True

        model = spGDMM(
            preprocessor=PreprocessorConfig(deg=3, knots=2),
            model_config=ModelConfig(variance="homogeneous", spatial_effect="none"),
            sampler_config=SamplerConfig(
                draws=2, tune=2, chains=1, nuts_sampler="pymc", progressbar=False
            ),
        )
        model.fit(X, y, holdout_mask=mask)
        result = model.extract_holdout_predictions()

        assert "hold_idx" in result
        assert "y_pred_mean" in result
        assert "y_pred_samples" in result
        assert len(result["hold_idx"]) == 50
        assert result["y_pred_mean"].shape == (50,)
        assert result["y_pred_samples"].shape[0] == 50
        # Predictions should be in [0, 1]
        assert np.all(result["y_pred_mean"] >= 0)
        assert np.all(result["y_pred_mean"] <= 1)


class TestSpGDMMSaveLoad:
    """Save/load round-trip tests — require MCMC (minimal sampling)."""

    _FAST_SAMPLER = SamplerConfig(
        draws=2, tune=2, chains=1, nuts_sampler="pymc", progressbar=False
    )

    @pytest.fixture(scope="module")
    def fitted_artifacts(self, tmp_path_factory):
        """Fit once per module; return (model, save_path, X, y)."""
        X, y = _make_sample_data(n_sites=15, seed=7)
        model = spGDMM(
            preprocessor=PreprocessorConfig(deg=3, knots=2),
            model_config=ModelConfig(variance="homogeneous", spatial_effect="none"),
            sampler_config=TestSpGDMMSaveLoad._FAST_SAMPLER,
        )
        model.fit(X, y)
        save_path = tmp_path_factory.mktemp("saveload") / "model.nc"
        model.save(str(save_path))
        return model, save_path, X, y

    def test_save_raises_before_fit(self, tmp_path):
        model = spGDMM(sampler_config=self._FAST_SAMPLER)
        with pytest.raises(RuntimeError):
            model.save(str(tmp_path / "model.nc"))

    def test_save_creates_file(self, fitted_artifacts):
        _, save_path, _, _ = fitted_artifacts
        assert save_path.exists()

    def test_load_returns_correct_type(self, fitted_artifacts):
        _, save_path, _, _ = fitted_artifacts
        loaded = spGDMM.load(str(save_path))
        assert isinstance(loaded, spGDMM)

    def test_load_preserves_model_config(self, fitted_artifacts):
        original, save_path, _, _ = fitted_artifacts
        loaded = spGDMM.load(str(save_path))
        assert loaded._config.variance == original._config.variance

    def test_load_preserves_sampler_config(self, fitted_artifacts):
        original, save_path, _, _ = fitted_artifacts
        loaded = spGDMM.load(str(save_path))
        assert loaded.sampler_config["draws"] == original.sampler_config["draws"]

    def test_load_preprocessor_is_fitted(self, fitted_artifacts):
        _, save_path, _, _ = fitted_artifacts
        loaded = spGDMM.load(str(save_path))
        assert hasattr(loaded.preprocessor, "n_predictors_")

    def test_load_preprocessor_mesh_matches(self, fitted_artifacts):
        original, save_path, _, _ = fitted_artifacts
        loaded = spGDMM.load(str(save_path))
        np.testing.assert_array_equal(
            loaded.preprocessor.predictor_mesh_,
            original.preprocessor.predictor_mesh_,
        )

    def test_load_id_matches(self, fitted_artifacts):
        original, save_path, _, _ = fitted_artifacts
        loaded = spGDMM.load(str(save_path))
        assert loaded.id == original.id

    def test_load_transform_consistency(self, fitted_artifacts):
        original, save_path, X, _ = fitted_artifacts
        loaded = spGDMM.load(str(save_path))
        orig_result = original.preprocessor.transform(X)
        load_result = loaded.preprocessor.transform(X)
        np.testing.assert_array_almost_equal(orig_result, load_result)

    def test_load_idata_has_posterior(self, fitted_artifacts):
        _, save_path, _, _ = fitted_artifacts
        loaded = spGDMM.load(str(save_path))
        assert "posterior" in loaded.idata


class TestGDM:
    """Tests for the frequentist GDM class."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 20
        X = pd.DataFrame({
            "xc": np.random.uniform(0, 100, n),
            "yc": np.random.uniform(0, 100, n),
            "time_idx": np.zeros(n),
            "temp": np.random.uniform(5, 20, n),
            "depth": np.random.uniform(0, 200, n),
        })
        biomass = np.random.exponential(1, (n, 10))
        y = pdist(biomass, "braycurtis").clip(1e-8, 1 - 1e-8)
        return X, y

    def test_fit_returns_self(self, sample_data):
        X, y = sample_data
        m = GDM()
        result = m.fit(X, y)
        assert result is m

    def test_fit_sets_coef(self, sample_data):
        X, y = sample_data
        m = GDM().fit(X, y)
        assert hasattr(m, "coef_")
        assert m.coef_.shape[0] > 0
        assert np.all(m.coef_ >= 0)  # NNLS guarantees non-negative

    def test_fit_sets_predictor_importance(self, sample_data):
        X, y = sample_data
        m = GDM().fit(X, y)
        assert hasattr(m, "predictor_importance_")
        assert "temp" in m.predictor_importance_
        assert "depth" in m.predictor_importance_
        assert all(v >= 0 for v in m.predictor_importance_.values())

    def test_fit_sets_explained(self, sample_data):
        X, y = sample_data
        m = GDM().fit(X, y)
        assert hasattr(m, "explained_")
        assert np.isfinite(m.explained_)
        assert m.explained_ <= 1.0

    def test_fit_sets_knots(self, sample_data):
        X, y = sample_data
        m = GDM().fit(X, y)
        assert hasattr(m, "knots_")
        assert "temp" in m.knots_
        assert "depth" in m.knots_

    def test_predict_shape(self, sample_data):
        X, y = sample_data
        m = GDM().fit(X, y)
        preds = m.predict(X)
        n = len(X)
        assert preds.shape == (n * (n - 1) // 2,)

    def test_predict_in_range(self, sample_data):
        X, y = sample_data
        m = GDM().fit(X, y)
        preds = m.predict(X)
        assert np.all(preds >= 0)
        assert np.all(preds < 1)

    def test_geo_false_excludes_distance(self, sample_data):
        X, y = sample_data
        m = GDM(geo=False).fit(X, y)
        assert "geo" not in m.predictor_importance_

    def test_geo_true_includes_distance(self, sample_data):
        X, y = sample_data
        m = GDM(geo=True).fit(X, y)
        assert "geo" in m.predictor_importance_
        assert "geo" in m.knots_

    def test_n_features_in(self, sample_data):
        X, y = sample_data
        m = GDM().fit(X, y)
        assert m.n_features_in_ == X.shape[1]

    def test_feature_names_in(self, sample_data):
        X, y = sample_data
        m = GDM().fit(X, y)
        np.testing.assert_array_equal(m.feature_names_in_, np.array(X.columns))

    def test_sklearn_clone(self, sample_data):
        from sklearn.base import clone
        m = GDM(splines=4, geo=True)
        cloned = clone(m)
        assert cloned.splines == 4
        assert cloned.geo is True
        assert not hasattr(cloned, "coef_")

    def test_sklearn_get_params(self):
        m = GDM(splines=4, knots=3, geo=True)
        params = m.get_params()
        assert params["splines"] == 4
        assert params["knots"] == 3
        assert params["geo"] is True

    def test_gdm_transform_shape(self, sample_data):
        X, y = sample_data
        m = GDM().fit(X, y)
        T = m.gdm_transform(X)
        n_sites = len(X)
        n_env = m.preprocessor_.n_predictors_ * m.preprocessor_.n_spline_bases_
        assert T.shape == (n_sites, n_env)

    def test_score_in_range(self, sample_data):
        X, y = sample_data
        m = GDM().fit(X, y)
        s = m.score(X, y)
        assert s <= 1.0

    def test_preprocessor_config_override(self, sample_data):
        X, y = sample_data
        cfg = PreprocessorConfig(deg=2, knots=3)
        m = GDM(preprocessor_config=cfg).fit(X, y)
        assert m.preprocessor_.n_spline_bases_ == 2 + 3

    def test_deviances_nonnegative(self, sample_data):
        X, y = sample_data
        m = GDM().fit(X, y)
        assert m.null_deviance_ >= 0
        assert m.model_deviance_ >= 0


class TestGDMModel:
    """Tests for the Bayesian GDMModel wrapper and GDMResult."""

    _FAST_SAMPLER = SamplerConfig(
        draws=2, tune=2, chains=1, nuts_sampler="pymc", progressbar=False
    )

    @pytest.fixture
    def sample_data(self):
        np.random.seed(7)
        n = 15
        X = pd.DataFrame({
            "xc": np.random.uniform(0, 100, n),
            "yc": np.random.uniform(0, 100, n),
            "time_idx": np.zeros(n, dtype=int),
            "temp": np.random.uniform(5, 20, n),
            "depth": np.random.uniform(0, 200, n),
        })
        biomass = np.random.exponential(1, (n, 8))
        y = pdist(biomass, "braycurtis").clip(1e-8, 1 - 1e-8)
        return X, y

    @pytest.fixture(scope="class")
    def fitted_result(self):
        """Fit once per class; return (model, result, X, y)."""
        np.random.seed(7)
        n = 15
        X = pd.DataFrame({
            "xc": np.random.uniform(0, 100, n),
            "yc": np.random.uniform(0, 100, n),
            "time_idx": np.zeros(n, dtype=int),
            "temp": np.random.uniform(5, 20, n),
            "depth": np.random.uniform(0, 200, n),
        })
        biomass = np.random.exponential(1, (n, 8))
        y = pdist(biomass, "braycurtis").clip(1e-8, 1 - 1e-8)
        model = GDMModel(
            geo=False,
            sampler_config=TestGDMModel._FAST_SAMPLER,
        )
        result = model.fit(X, y, dataname="test_data")
        return model, result, X, y

    def test_fit_returns_gdmresult(self, fitted_result):
        _, result, _, _ = fitted_result
        assert isinstance(result, GDMResult)

    def test_result_dataname(self, fitted_result):
        _, result, _, _ = fitted_result
        assert result.dataname == "test_data"

    def test_result_has_deviances(self, fitted_result):
        _, result, _, _ = fitted_result
        assert np.isfinite(result.gdmdeviance)
        assert np.isfinite(result.nulldeviance)
        assert result.gdmdeviance >= 0
        assert result.nulldeviance >= 0

    def test_result_explained_in_range(self, fitted_result):
        _, result, _, _ = fitted_result
        assert 0.0 <= result.explained <= 100.0

    def test_result_predictors_list(self, fitted_result):
        _, result, _, _ = fitted_result
        assert isinstance(result.predictors, list)
        assert len(result.predictors) > 0

    def test_result_coefficients_non_empty(self, fitted_result):
        _, result, _, _ = fitted_result
        assert isinstance(result.coefficients, dict)
        # Each predictor should have a list of coefficients (not empty [])
        for pred, coefs in result.coefficients.items():
            assert isinstance(coefs, list), f"coefficients[{pred!r}] is not a list"
            assert len(coefs) > 0, f"coefficients[{pred!r}] is empty"

    def test_result_knots_present(self, fitted_result):
        _, result, _, _ = fitted_result
        assert isinstance(result.knots, dict)
        assert len(result.knots) > 0

    def test_result_observed_shape(self, fitted_result):
        _, result, _, y = fitted_result
        assert result.observed.shape == y.shape

    def test_result_predicted_shape(self, fitted_result):
        _, result, _, y = fitted_result
        assert result.predicted.shape == y.shape

    def test_result_predicted_nonnegative(self, fitted_result):
        _, result, _, _ = fitted_result
        # exp(log_pred) is always non-negative; strict <1 is not guaranteed with
        # minimal MCMC sampling (posterior mean near 0 → exp(0) = 1)
        assert np.all(result.predicted >= 0)

    def test_result_idata_present(self, fitted_result):
        _, result, _, _ = fitted_result
        assert result.idata is not None
        assert "posterior" in result.idata

    def test_predict_returns_array(self, fitted_result):
        model, _, X, _ = fitted_result
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_predict_nonnegative(self, fitted_result):
        model, _, X, _ = fitted_result
        preds = model.predict(X)
        assert np.all(preds >= 0)

    def test_model_coefficients_property(self, fitted_result):
        model, _, _, _ = fitted_result
        coefs = model.coefficients
        assert isinstance(coefs, dict)
        assert len(coefs) > 0

    def test_model_explained_property(self, fitted_result):
        model, _, _, _ = fitted_result
        assert isinstance(model.explained, float)
        assert 0.0 <= model.explained <= 100.0

    def test_geo_true_includes_geographic(self):
        """With geo=True, 'geographic' should appear in knots."""
        np.random.seed(9)
        n = 15
        X = pd.DataFrame({
            "xc": np.random.uniform(0, 100, n),
            "yc": np.random.uniform(0, 100, n),
            "time_idx": np.zeros(n, dtype=int),
            "temp": np.random.uniform(5, 20, n),
        })
        biomass = np.random.exponential(1, (n, 5))
        y = pdist(biomass, "braycurtis").clip(1e-8, 1 - 1e-8)
        model = GDMModel(geo=True, sampler_config=TestGDMModel._FAST_SAMPLER)
        result = model.fit(X, y)
        assert result.geo is True
        assert "geographic" in result.knots

    def test_predict_before_fit_raises(self):
        model = GDMModel(sampler_config=TestGDMModel._FAST_SAMPLER)
        X = pd.DataFrame({"xc": [0], "yc": [0], "time_idx": [0], "temp": [10]})
        with pytest.raises(ValueError, match="fitted"):
            model.predict(X)

    def test_to_dict_round_trip(self, fitted_result):
        _, result, _, _ = fitted_result
        d = result.to_dict()
        result2 = GDMResult.from_dict(d)
        assert result2.dataname == result.dataname
        assert result2.explained == result.explained
        np.testing.assert_array_almost_equal(result2.predicted, result.predicted)