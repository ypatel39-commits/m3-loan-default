"""SHAP explainability helpers for the trained XGBoost model.

Wraps ``shap.TreeExplainer`` so notebook + tests share one code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


def compute_shap_values(model: XGBClassifier, X: pd.DataFrame) -> shap.Explanation:
    """Return a SHAP ``Explanation`` object for the given rows."""
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def summary_plot(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    out_path: Optional[Path] = None,
    max_display: int = 15,
) -> Optional[Path]:
    """Save a SHAP summary (beeswarm) plot. Returns the saved path or None."""
    plt.figure()
    shap.summary_plot(
        shap_values.values, X, max_display=max_display, show=False
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path
    return None


def waterfall_plot(
    shap_values: shap.Explanation,
    row_idx: int = 0,
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    """Save a SHAP waterfall plot for a single prediction."""
    plt.figure()
    shap.plots.waterfall(shap_values[row_idx], show=False)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path
    return None


def top_features(
    shap_values: shap.Explanation, X: pd.DataFrame, k: int = 10
) -> pd.DataFrame:
    """Return the top-``k`` features ranked by mean(|SHAP value|)."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    df = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(k).reset_index(drop=True)
