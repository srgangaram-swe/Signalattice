"""Machine-learning evaluation metrics (classification & regression).

Financial / portfolio metrics (Sharpe, drawdown, ...) live separately in
:mod:`quant_platform.risk.metrics`; this module covers predictive accuracy.
"""

from __future__ import annotations

import numpy as np

from quant_platform.logging_utils import get_logger

logger = get_logger(__name__)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and (if probabilities given) ROC-AUC."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:  # pragma: no cover - degenerate fold
            out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R^2 and directional accuracy (sign hit-rate)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    directional = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": directional,
    }


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return TN, FP, FN, TP counts for a binary problem."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(
        np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1]
    )
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
