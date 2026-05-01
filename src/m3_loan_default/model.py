"""XGBoost classifier for the UCI credit-default task.

Provides a single ``train`` entry point that returns the fitted model plus a
metrics dict (AUC, precision, recall, F1, accuracy, plus 5-fold CV AUC).
All randomness is seeded to ``RANDOM_STATE`` for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

RANDOM_STATE = 42


@dataclass
class TrainResult:
    model: XGBClassifier
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    metrics: dict


def make_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """Construct an XGBoost classifier with reasonable defaults."""
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def evaluate(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    return {
        "auc": float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "confusion_matrix": cm.tolist(),
    }


def cross_val_auc(model: XGBClassifier, X: pd.DataFrame, y: pd.Series, k: int = 5) -> float:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, scoring="roc_auc", cv=skf, n_jobs=-1)
    return float(np.mean(scores))


def train(X: pd.DataFrame, y: pd.Series, run_cv: bool = True) -> TrainResult:
    """Full training pipeline. Returns model + holdout metrics + optional CV AUC."""
    X_train, X_test, y_train, y_test = split(X, y)
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    spw = neg / pos if pos > 0 else 1.0

    model = make_model(scale_pos_weight=spw)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    if run_cv:
        # CV uses an unweighted base model on full data for an honest baseline.
        metrics["cv_auc_5fold"] = cross_val_auc(make_model(), X, y, k=5)

    return TrainResult(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metrics=metrics,
    )
