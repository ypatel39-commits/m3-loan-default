"""Data loading sanity checks."""

import pandas as pd
import pytest

from m3_loan_default import data


@pytest.fixture(scope="module")
def loaded():
    return data.load()


def test_load_returns_X_and_y(loaded):
    X, y = loaded
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_dataset_shape(loaded):
    X, y = loaded
    # UCI Credit Card Default has 30k rows and 23 features after dropping ID.
    assert len(X) > 25_000
    assert len(X) == len(y)
    assert X.shape[1] >= 20


def test_target_is_binary(loaded):
    _, y = loaded
    assert set(y.unique()).issubset({0, 1})
    # Default rate should land near the documented ~22%.
    rate = y.mean()
    assert 0.10 < rate < 0.40
