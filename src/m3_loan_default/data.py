"""Data loader for the UCI Credit Card Default dataset.

Downloads the .xls file once into ``data/`` and caches a cleaned Parquet copy
on subsequent calls. Returns a feature matrix ``X`` and binary target ``y``
where ``y == 1`` means the borrower defaulted on next month's payment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import requests

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_PATH = DATA_DIR / "uci_credit_default.xls"
CLEAN_PATH = DATA_DIR / "uci_credit_default.csv"

TARGET_COL = "default_next_month"


def download(url: str = UCI_URL, dest: Path = RAW_PATH, force: bool = False) -> Path:
    """Download the raw UCI .xls file. Idempotent."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and drop the redundant ID column.

    The raw UCI .xls has placeholder headers ("X1", "X2", ..., "Y") on row 0
    and the human-readable names on row 1. Promote row 1 as the header.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.iloc[1]]
    df = df.iloc[2:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.rename(columns={"default payment next month": TARGET_COL})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df = df.dropna().reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def load(force_download: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for the UCI Credit Card Default dataset.

    Caches a cleaned Parquet to avoid re-parsing the .xls every call.
    """
    if CLEAN_PATH.exists() and not force_download:
        df = pd.read_csv(CLEAN_PATH)
    else:
        raw = download(force=force_download)
        df = pd.read_excel(raw, header=None)
        df = _clean(df)
        df.to_csv(CLEAN_PATH, index=False)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def feature_names() -> list[str]:
    """Convenience: list of feature column names without loading the dataset."""
    X, _ = load()
    return list(X.columns)


if __name__ == "__main__":
    X, y = load()
    print(f"shape: X={X.shape}, y={y.shape}, default_rate={y.mean():.4f}")
