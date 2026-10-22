"""Tests for spGDMM (Bayesian) model and its configuration dataclasses."""

import json

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from gdmbayes import GDMPreprocessor, ModelConfig, SamplerConfig, spGDMM

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.alpha_importance is True
        assert config.variance == "homogeneous"
        assert config.spatial_effect == "none"
        # Preprocessing fields live on PreprocessorConfig, not ModelConfig
        for f in ("deg", "knots", "mesh_choice", "distance_measure", "extrapolation"):
            assert not hasattr(config, f)

    def test_custom_callable_variance(self):
        def my_fn(mu, X_sigma):
            return mu

        assert ModelConfig(variance=my_fn).variance is my_fn

    def test_custom_callable_spatial_effect(self):
        def my_fn(psi, row_ind, col_ind):
            return psi[row_ind] - psi[col_ind]

        assert ModelConfig(spatial_effect=my_fn).spatial_effect is my_fn

    def test_invalid_variance_string(self):
        with pytest.raises(ValueError, match="Unknown variance"):
            ModelConfig(variance="bad_name")

    def test_invalid_spatial_effect_string(self):
        with pytest.raises(ValueError, match="Unknown spatial_effect"):
            ModelConfig(spatial_effect="bad_name")

    def test_to_dict_excludes_preprocessing_fields(self):
        d = ModelConfig().to_dict()
        assert d["variance"] == "homogeneous"
        assert d["spatial_effect"] == "none"
        for f in ("deg", "knots", "mesh_choice", "distance_measure"):
            assert f not in d

    def test_from_dict_roundtrip(self):
        d = {"variance": "homogeneous", "spatial_effect": "abs_diff", "alpha_importance": False}
        config = ModelConfig.from_dict(d)
        assert config.variance == "homogeneous"
        assert config.spatial_effect == "abs_diff"
        assert config.alpha_importance is False


class TestSamplerConfig:
    def test_defaults(self):
        config = SamplerConfig()
        assert config.draws == 1000
        assert config.tune == 1000
        assert config.chains == 4
        assert config.target_accept == 0.95
        assert config.nuts_sampler == "nutpie"

    def test_from_dict_roundtrip(self):
        d = {"draws": 200, "tune": 50, "chains": 1, "random_seed": 42}
        config = SamplerConfig.from_dict(d)
        assert config.draws == 200
        assert config.tune == 50
        assert config.chains == 1
        assert config.random_seed == 42
        assert config.to_dict()["draws"] == 200


# ---------------------------------------------------------------------------
# spGDMM model basics
# ---------------------------------------------------------------------------


class TestSpGDMM:
    """Construction, config plumbing, and pre-fit data preprocessing."""

    def test_default_init(self):
        model = spGDMM()
        assert model._config.variance == "homogeneous"
        assert model._config.spatial_effect == "none"
        assert model.preprocessor.deg == 3
        assert model.preprocessor.knots == 2

    def test_custom_preprocessor(self):
        model = spGDMM(preprocessor=GDMPreprocessor(deg=4, knots=3))
        assert model.preprocessor.deg == 4
        assert model.preprocessor.knots == 3

    def test_custom_model_config(self):
        model = spGDMM(model_config=ModelConfig(variance="polynomial"))
        assert model._config.variance == "polynomial"

    def test_default_configs_getters(self):
        model = spGDMM()
        assert "variance" in model.get_default_model_config()
        assert model.get_default_sampler_config()["draws"] == 1000
        assert "variance" in model._serializable_model_config
        assert "deg" not in model._serializable_model_config

    def test_pw_distance(self):
        locations = np.array([[0, 0], [1, 0], [0, 1]])
        dists = spGDMM().preprocessor.pw_distance(locations)
        assert len(dists) == 3
        assert np.all(dists >= 0)

    def test_preprocess_sets_features_and_output_var(self, site_data):
        X, y = site_data
        model = spGDMM()
        assert not hasattr(model, "n_features_in_")
        model._generate_and_preprocess_model_data(X, y)
        assert model.n_features_in_ == 2  # temp, depth
        assert model.output_var == "log_y"
        assert model.X_GDM_ is not None
        assert model.log_y_ is not None

    def test_preprocessor_skips_refit(self, site_data):
        X, y = site_data
        m = spGDMM()
        m._generate_and_preprocess_model_data(X, y)
        mesh_before = m.preprocessor.predictor_mesh_.copy()
        m._generate_and_preprocess_model_data(X, y)
        np.testing.assert_array_equal(m.preprocessor.predictor_mesh_, mesh_before)

    def test_build_model_registers_rvs(self, site_data):
        X, y = site_data
        model = spGDMM()
        model._generate_and_preprocess_model_data(X, y)
        model.build_model()
        assert model.model_ is not None
        assert "beta_0" in model.model_.named_vars
        assert "beta" in model.model_.named_vars

    def test_set_idata_attrs_stamps_metadata(self, site_data):
        X, y = site_data
        model = spGDMM()
        model._generate_and_preprocess_model_data(X, y)
        stub = az.from_dict({"posterior": {"dummy": np.ones((1, 1))}})
        model.idata_ = stub
        result = model._set_idata_attrs(stub)
        for key in ("id", "model_type", "version", "model_config", "sampler_config"):
            assert key in result.attrs
        json.loads(result.attrs["model_config"])
        json.loads(result.attrs["sampler_config"])


# ---------------------------------------------------------------------------
# Sklearn interface (no MCMC required)
# ---------------------------------------------------------------------------


class TestSpGDMMSklearnInterface:
    @pytest.fixture
    def model(self):
        return spGDMM(
            preprocessor=GDMPreprocessor(deg=3, knots=2),
            model_config=ModelConfig(variance="homogeneous", spatial_effect="none"),
        )

    @pytest.fixture
    def preprocessed_model(self, model, site_data):
        X, y = site_data
        model._generate_and_preprocess_model_data(X, y)
        return model

    def test_get_params(self, model):
        params = model.get_params()
        assert {"preprocessor", "model_config", "sampler_config"} <= params.keys()

    def test_set_params_updates_model_config(self, model):
        new_cfg = {"variance": "polynomial", "spatial_effect": "none", "alpha_importance": True}
        assert model.set_params(model_config=new_cfg) is model
        assert model.model_config["variance"] == "polynomial"

    def test_clone_preserves_config_and_unfits(self, preprocessed_model):
        cloned = clone(preprocessed_model)
        assert cloned.preprocessor.deg == preprocessed_model.preprocessor.deg
        assert cloned._config.variance == preprocessed_model._config.variance
        assert cloned.idata_ is None
        assert not hasattr(cloned, "n_features_in_")

    def test_not_fitted_raises_check_is_fitted(self, model):
        with pytest.raises(NotFittedError):
            check_is_fitted(model)

    def test_fitted_after_stub_idata(self, preprocessed_model):
        preprocessed_model.idata_ = az.from_dict({"posterior": {"dummy": np.ones((1, 1))}})
        check_is_fitted(preprocessed_model)

    def test_fit_raises_on_y_none(self, model, site_data):
        X, _ = site_data
        with pytest.raises(ValueError, match="y is required"):
            model.fit(X, y=None)

    def test_id_deterministic_across_instances(self, model):
        model2 = spGDMM(
            preprocessor=GDMPreprocessor(deg=3, knots=2),
            model_config=ModelConfig(variance="homogeneous", spatial_effect="none"),
        )
        assert model.id == model2.id


# ---------------------------------------------------------------------------
# GP conditional prediction (requires real MCMC)
# ---------------------------------------------------------------------------


class TestGPConditionalPredict:
    """GP kriging at held-out sites. Needs a compiled PyMC model."""

    @pytest.fixture
    def train_test_data(self):
        np.random.seed(42)
        n_train, n_test = 15, 5
        n_total = n_train + n_test
        X_all = pd.DataFrame(
            {
                "xc": np.random.uniform(0, 100, n_total),
                "yc": np.random.uniform(0, 100, n_total),
                "time_idx": np.zeros(n_total, dtype=int),
                "temp": np.random.uniform(5, 20, n_total),
            }
        )
        biomass = np.random.exponential(1, (n_total, 10))
        y_all = np.clip(pdist(biomass, "braycurtis"), 1e-8, None)
        X_train = X_all.iloc[:n_train].reset_index(drop=True)
        X_test = X_all.iloc[n_train:].reset_index(drop=True)
        train_pair_idx = GDMPreprocessor.site_pairs(n_total, list(range(n_train)))
        test_pair_idx = GDMPreprocessor.site_pairs(n_total, list(range(n_train, n_total)))
        return X_train, y_all[train_pair_idx], X_test, y_all[test_pair_idx]

    def _make_model(self, spatial_effect):
        return spGDMM(
            preprocessor=GDMPreprocessor(deg=2, knots=1),
            model_config=ModelConfig(
                variance="homogeneous",
                spatial_effect=spatial_effect,
                alpha_importance=False,
            ),
            sampler_config=SamplerConfig(
                draws=2, tune=2, chains=1, nuts_sampler="pymc", progressbar=False
            ),
        )

    def test_gp_stored_only_when_spatial(self, train_test_data):
        import pymc as pm

        X_train, y_train, _, _ = train_test_data

        spatial = self._make_model("squared_diff")
        spatial.build_model(X_train, y_train)
        assert isinstance(spatial.gp_, pm.gp.Latent)

        none = self._make_model("none")
        none.build_model(X_train, y_train)
        assert none.gp_ is None

    def test_predict_new_sites_spatial(self, train_test_data):
        X_train, y_train, X_test, _ = train_test_data
        n_test = len(X_test)
        model = self._make_model("squared_diff")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (n_test * (n_test - 1) // 2,)
        assert np.all((y_pred >= 0) & (y_pred <= 1))

    def test_predict_new_sites_no_spatial(self, train_test_data):
        X_train, y_train, X_test, _ = train_test_data
        n_test = len(X_test)
        model = self._make_model("none")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (n_test * (n_test - 1) // 2,)
        assert np.all((y_pred >= 0) & (y_pred <= 1))

    def test_scoring_methods(self, train_test_data):
        """crps, score, explained_deviance return finite values with correct signs."""
        X_train, y_train, X_test, y_test = train_test_data
        model = self._make_model("none")
        model.sampler_config["draws"] = 4
        model.sampler_config["tune"] = 4
        model.fit(X_train, y_train)

        crps_val = model.crps(X_test, y_test)
        score_val = model.score(X_test, y_test)
        ed_val = model.explained_deviance(X_test, y_test)

        assert np.isfinite(crps_val) and crps_val >= 0.0
        assert np.isfinite(score_val) and score_val <= 0.0  # score() = -crps()
        assert np.isfinite(ed_val) and ed_val <= 1.0


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------


class TestSpGDMMSaveLoad:
    """Save/load. Single round-trip assertion block; all-or-nothing behaviour."""

    @pytest.fixture(scope="module")
    def saved_model(self, fitted_spgdmm_fast, tmp_path_factory):
        model, X, y = fitted_spgdmm_fast
        path = tmp_path_factory.mktemp("saveload") / "model.nc"
        model.save(str(path))
        return model, path, X

    def test_save_raises_before_fit(self, tmp_path):
        model = spGDMM(
            sampler_config=SamplerConfig(
                draws=2, tune=2, chains=1, nuts_sampler="pymc", progressbar=False
            )
        )
        with pytest.raises(RuntimeError):
            model.save(str(tmp_path / "model.nc"))

    def test_load_roundtrip(self, saved_model):
        original, save_path, X = saved_model
        assert save_path.exists()

        loaded = spGDMM.load(str(save_path))
        assert isinstance(loaded, spGDMM)
        assert loaded._config.variance == original._config.variance
        assert loaded.sampler_config["draws"] == original.sampler_config["draws"]
        assert hasattr(loaded.preprocessor, "n_predictors_")
        assert loaded.id == original.id
        assert "posterior" in loaded.idata_
        np.testing.assert_array_equal(
            loaded.preprocessor.predictor_mesh_, original.preprocessor.predictor_mesh_
        )
        np.testing.assert_array_almost_equal(
            loaded.preprocessor.transform(X), original.preprocessor.transform(X)
        )


# ---------------------------------------------------------------------------
# gdm_transform / ispline_extract (inspection methods)
# ---------------------------------------------------------------------------


class TestSpGDMMExtractMethods:
    def test_gdm_transform(self, fitted_spgdmm_fast):
        model, X, _ = fitted_spgdmm_fast
        result = model.gdm_transform(X)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (
            len(X),
            model.preprocessor.n_predictors_ * model.preprocessor.n_spline_bases_,
        )
        for col in result.columns:
            parts = col.rsplit("_", 1)
            assert len(parts) == 2 and parts[1].isdigit()

    def test_ispline_extract(self, fitted_spgdmm_fast):
        model, _, _ = fitted_spgdmm_fast
        result = model.ispline_extract()
        assert isinstance(result, dict)
        for pred in model.preprocessor.predictor_names_:
            assert pred in result
            assert "x" in result[pred]
            assert "y" in result[pred]
