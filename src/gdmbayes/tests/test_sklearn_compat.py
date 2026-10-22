"""scikit-learn API conformance for the non-MCMC estimators.

GDM and GDMPreprocessor consume *site-level* X but produce *pairwise* outputs
(``n_sites * (n_sites - 1) / 2`` values), so every sklearn check that assumes
``len(y) == len(X)``, row-subset invariance, or row-wise ``predict``/``transform``
necessarily fails. Those checks are declared below with a reason, so that
``check_estimator`` still runs the remaining checks and any *new* failure is
caught.
"""

from sklearn.utils.estimator_checks import check_estimator

from gdmbayes import GDM, GDMPreprocessor

_PAIRWISE = "pairwise output: y has n_sites*(n_sites-1)/2 entries, not n_sites"
_SITE_TABLE = (
    "X must be a site table [xc, yc, time_idx, predictors...]; generic arrays are not valid"
)
_FEW_SITES = "needs at least 3 sites to form pairs and a spline mesh"

GDM_EXPECTED_FAILURES = {
    # y length / pairwise structure
    "check_fit_score_takes_y": _PAIRWISE,
    "check_estimators_fit_returns_self": _PAIRWISE,
    "check_regressors_train": _PAIRWISE,
    "check_regressor_data_not_an_array": _PAIRWISE,
    "check_regressors_int": _PAIRWISE,
    "check_supervised_y_2d": _PAIRWISE,
    "check_methods_sample_order_invariance": _PAIRWISE,
    "check_methods_subset_invariance": _PAIRWISE,
    "check_fit_idempotent": _PAIRWISE,
    "check_fit_check_is_fitted": _PAIRWISE,
    "check_pipeline_consistency": _PAIRWISE,
    "check_estimators_pickle": _PAIRWISE,
    "check_estimators_overwrite_params": _PAIRWISE,
    "check_dont_overwrite_parameters": _PAIRWISE,
    "check_dict_unchanged": _PAIRWISE,
    "check_n_features_in": _PAIRWISE,
    "check_n_features_in_after_fitting": _PAIRWISE,
    "check_fit2d_predict1d": _PAIRWISE,
    "check_readonly_memmap_input": _PAIRWISE,
    "check_f_contiguous_array_estimator": _PAIRWISE,
    "check_positive_only_tag_during_fit": _PAIRWISE,
    "check_regressors_no_decision_function": _PAIRWISE,
    "check_requires_y_none": _PAIRWISE,
    # input-format assumptions
    "check_estimators_dtypes": _SITE_TABLE,
    "check_complex_data": _SITE_TABLE,
    "check_dtype_object": _SITE_TABLE,
    "check_estimators_nan_inf": _SITE_TABLE,
    "check_estimators_empty_data_messages": _SITE_TABLE,
    "check_estimator_sparse_tag": _SITE_TABLE,
    "check_estimator_sparse_array": _SITE_TABLE,
    "check_estimator_sparse_matrix": _SITE_TABLE,
    "check_fit2d_1sample": _FEW_SITES,
    "check_fit2d_1feature": _SITE_TABLE,
}

PREPROCESSOR_EXPECTED_FAILURES = {
    "check_transformer_general": _PAIRWISE,
    "check_transformer_data_not_an_array": _PAIRWISE,
    "check_methods_sample_order_invariance": _PAIRWISE,
    "check_methods_subset_invariance": _PAIRWISE,
    "check_n_features_in_after_fitting": _PAIRWISE,
    "check_fit2d_predict1d": _PAIRWISE,
    "check_complex_data": _SITE_TABLE,
    "check_dtype_object": _SITE_TABLE,
    "check_estimators_nan_inf": _SITE_TABLE,
    "check_estimators_empty_data_messages": _SITE_TABLE,
    "check_estimator_sparse_tag": _SITE_TABLE,
    "check_estimator_sparse_array": _SITE_TABLE,
    "check_estimator_sparse_matrix": _SITE_TABLE,
    "check_fit2d_1sample": _FEW_SITES,
    "check_fit1d": _SITE_TABLE,
}


def test_gdm_check_estimator():
    check_estimator(GDM(), expected_failed_checks=GDM_EXPECTED_FAILURES)


def test_preprocessor_check_estimator():
    check_estimator(GDMPreprocessor(), expected_failed_checks=PREPROCESSOR_EXPECTED_FAILURES)
