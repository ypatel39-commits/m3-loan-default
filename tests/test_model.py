"""Model training smoke test on a sub-sample (kept fast for CI)."""

import pytest

from m3_loan_default import data, model


@pytest.fixture(scope="module")
def trained():
    X, y = data.load()
    # Sub-sample for CI speed; AUC remains stable enough for a smoke check.
    sample = X.sample(n=5_000, random_state=42)
    X_s = sample
    y_s = y.loc[sample.index]
    return model.train(X_s, y_s, run_cv=False)


def test_training_returns_metrics(trained):
    assert "auc" in trained.metrics
    assert "precision" in trained.metrics
    assert "recall" in trained.metrics


def test_auc_above_random(trained):
    # Subsample run; we just want clearly better than chance.
    assert trained.metrics["auc"] > 0.65


def test_predictions_shape(trained):
    proba = trained.model.predict_proba(trained.X_test)
    assert proba.shape == (len(trained.X_test), 2)
