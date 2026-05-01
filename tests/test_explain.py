"""SHAP explainability shape checks."""

import pytest

from m3_loan_default import data, explain, model


@pytest.fixture(scope="module")
def trained_and_shap():
    X, y = data.load()
    sample = X.sample(n=3_000, random_state=42)
    X_s = sample
    y_s = y.loc[sample.index]
    result = model.train(X_s, y_s, run_cv=False)
    shap_values = explain.compute_shap_values(result.model, result.X_test.head(200))
    return result, shap_values


def test_shap_shape_matches_input(trained_and_shap):
    result, shap_values = trained_and_shap
    assert shap_values.values.shape == (200, result.X_test.shape[1])


def test_top_features_returns_k_rows(trained_and_shap):
    result, shap_values = trained_and_shap
    top = explain.top_features(shap_values, result.X_test.head(200), k=5)
    assert len(top) == 5
    assert {"feature", "mean_abs_shap"}.issubset(top.columns)
