"""Smoke tests: package importable, version exposed."""

import m3_loan_default


def test_version_present():
    assert isinstance(m3_loan_default.__version__, str)
    assert len(m3_loan_default.__version__) > 0


def test_modules_importable():
    from m3_loan_default import data, model, explain  # noqa: F401
