"""Numerical regression test: ``gdmbayes.GDM`` vs the reference R ``gdm`` package.

The fixture ``data/r_gdm_southwest.json`` holds a fit of R ``gdm`` (1.6.0.7) on the
shipped Southwest Australia dataset (``examples/data/southwest``), produced by
``data/make_r_gdm_fixture.R`` with ``gdm::gdm(sp, geo = TRUE)`` and R's default
of 3 splines per predictor (knots at the 0/50/100 % quantiles).

Parameter mapping
-----------------
R's "3 splines" are quadratic I-splines on a 3-point mesh. In ``gdmbayes`` the
same basis is ``GDM(splines=2, knots=1)``: ``splines`` is the dms_variants
I-spline *order* (= polynomial degree) and the number of basis functions per
predictor is ``splines + knots`` = 3. With this mapping the two implementations
produce the same design matrix, so coefficients, deviances and fitted values
agree to floating-point precision and are all compared directly below.

Layout differences handled here:

* R returns the intercept separately and orders coefficients as
  ``[Geographic, pred_1, pred_2, ...]``; Python's ``coef_`` is
  ``[intercept, pred_1, pred_2, ..., geo]``.
* Python divides Euclidean coordinate distances by 1000, so the geo knots differ
  from R's by that factor (irrelevant to the fit: knots are quantiles).
* R reports ``explained`` in percent.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gdmbayes import GDM

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "data" / "r_gdm_southwest.json"
DATA_DIR = HERE.parents[2] / "examples" / "data" / "southwest"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists() or not DATA_DIR.exists(),
    reason=f"R gdm fixture ({FIXTURE}) or southwest data ({DATA_DIR}) not found",
)


@pytest.fixture(scope="module")
def ref():
    with open(FIXTURE) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def sw_data(ref):
    sites = pd.read_csv(DATA_DIR / "southwest_sites.csv")
    y = pd.read_csv(DATA_DIR / "southwest_y.csv")["y"].to_numpy()
    s = ref["settings"]
    X = sites.rename(columns={s["x_column"]: "xc", s["y_column"]: "yc"}).copy()
    X["time_idx"] = 0
    X = X[["xc", "yc", "time_idx", *s["env_predictors"]]]
    assert len(X) == s["n_sites"] and len(y) == s["n_pairs"]
    return X, y


@pytest.fixture(scope="module")
def fitted(sw_data, ref):
    X, y = sw_data
    n_splines = set(ref["settings"]["splines"])
    assert n_splines == {3}, "fixture must use R gdm's default 3 splines/predictor"
    # R: 3 quadratic I-splines per predictor  <=>  Python: order 2, 1 interior knot.
    return GDM(geo=True, splines=2, knots=1).fit(X, y)


def _r_coef_in_python_layout(ref, n_env):
    """Reorder R's [geo(3), env...(3 each)] + intercept to Python's coef_ layout."""
    r_coef = np.asarray(ref["coefficients"], dtype=float)
    n_geo = ref["settings"]["splines"][0]
    assert ref["predictors"][0] == "Geographic"
    return np.concatenate([[ref["intercept"]], r_coef[n_geo:], r_coef[:n_geo]])


def test_pair_order_matches_pdist(ref):
    """R's formatsitepair rows are in row-major upper-triangle (pdist) order."""
    n = ref["settings"]["n_sites"]
    k = ref["n_pairs_with_ids"]
    row, col = np.triu_indices(n, k=1)
    np.testing.assert_array_equal(ref["pair_s1"], row[:k])
    np.testing.assert_array_equal(ref["pair_s2"], col[:k])


def test_basis_size_matches_r(fitted, ref):
    assert fitted.preprocessor_.n_spline_bases_ == ref["settings"]["splines"][0]
    assert list(fitted.preprocessor_.predictor_names_) == ref["settings"]["env_predictors"]


def test_knots_match_r(fitted, ref):
    r_knots = np.asarray(ref["knots"], dtype=float)
    n_geo = ref["settings"]["splines"][0]
    env_names = ref["settings"]["env_predictors"]
    # R: [geo knots, pred_1 knots, ...]; Python keeps geo in km-ish units (/1000).
    np.testing.assert_allclose(fitted.knots_["geo"] * 1000.0, r_knots[:n_geo], rtol=1e-10)
    env_knots = np.concatenate([fitted.knots_[name] for name in env_names])
    np.testing.assert_allclose(env_knots, r_knots[n_geo:], rtol=1e-10)


def test_explained_deviance_matches_r(fitted, ref):
    assert fitted.explained_ == pytest.approx(ref["explained_percent"] / 100.0, abs=1e-8)
    assert fitted.null_deviance_ == pytest.approx(ref["null_deviance"], rel=1e-8)
    assert fitted.model_deviance_ == pytest.approx(ref["gdm_deviance"], rel=1e-8)


def test_coefficients_match_r(fitted, ref):
    n_env = len(ref["settings"]["env_predictors"])
    expected = _r_coef_in_python_layout(ref, n_env)
    assert fitted.coef_.shape == expected.shape
    np.testing.assert_allclose(fitted.coef_, expected, atol=1e-8)
    assert fitted.intercept_ == pytest.approx(ref["intercept"], abs=1e-8)


def test_predictions_match_r(fitted, sw_data, ref):
    X, _ = sw_data
    pred = fitted.predict(X)
    r_pred = np.asarray(ref["predicted"], dtype=float)
    assert pred.shape == r_pred.shape
    # Fixture predictions are rounded to 8 dp (max rounding error 5e-9).
    np.testing.assert_allclose(pred, r_pred, atol=1e-7)


def test_notebook_settings_differ_from_r_default(sw_data, ref):
    """Document that GDM(splines=3, knots=1) (4 bases) is NOT R gdm's default.

    The running-example notebook uses ``splines=3, knots=1``; that is a cubic
    I-spline basis with 4 functions per predictor, a strictly larger model than
    R's default (explained 0.7605 vs 0.7421 on this dataset). Guard against
    anyone "fixing" the regression test by matching the notebook instead.
    """
    X, y = sw_data
    model = GDM(geo=True, splines=3, knots=1).fit(X, y)
    assert model.preprocessor_.n_spline_bases_ == 4
    assert model.explained_ > ref["explained_percent"] / 100.0 + 0.01
