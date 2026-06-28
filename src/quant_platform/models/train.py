"""Walk-forward training harness producing leakage-free out-of-sample signals.

The harness:

1. selects feature columns (honouring ``feature_blocklist``);
2. iterates :class:`TimeSeriesSplitter` folds (forward-chaining, embargoed);
3. fits a fresh model per fold on *past* data and predicts the held-out future;
4. aggregates out-of-sample (OOS) predictions and per-fold metrics;
5. fits a final model on all data for feature-importance reporting.

The OOS prediction frame is what the backtester consumes — every signal is
genuinely out-of-sample, which is what makes the backtest credible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.config import ModelConfig
from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.features.pipeline import (
    TARGET_DIRECTION,
    TARGET_RETURN,
    feature_columns,
)
from quant_platform.logging_utils import get_logger
from quant_platform.models.factory import build_estimator, supports_proba
from quant_platform.models.metrics import classification_metrics, regression_metrics
from quant_platform.models.splits import TimeSeriesSplitter

logger = get_logger(__name__)

SIGNAL_COL = "score"


@dataclass
class TrainResult:
    """Container for everything produced by a walk-forward training run."""

    predictions: pd.DataFrame  # date, ticker, score, y_true, fold
    metrics: dict[str, float]
    fold_metrics: list[dict[str, Any]]
    feature_importances: pd.Series
    feature_names: list[str]
    model: Any
    task: str
    n_splits: int
    extra: dict[str, Any] = field(default_factory=dict)


def _select_features(features: pd.DataFrame, config: ModelConfig) -> list[str]:
    cols = feature_columns(features)
    block = set(config.feature_blocklist)
    selected = [c for c in cols if c not in block]
    if not selected:
        raise ValueError("No feature columns left after applying feature_blocklist")
    return selected


def _make_pipeline(config: ModelConfig, seed: int):
    estimator = build_estimator(config, seed=seed)
    if config.standardize and config.type in {"logistic", "ridge", "lstm"}:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([("scaler", StandardScaler()), ("model", estimator)]), estimator
    return estimator, estimator


def _predict_scores(model, X: np.ndarray, task: str) -> np.ndarray:
    """Return a 1-D signal score: P(up) for classification, ŷ for regression."""
    if task == "classification":
        if supports_proba(model):
            proba = model.predict_proba(X)
            # Probability of the positive (up) class.
            return (
                proba[:, list(model.classes_).index(1.0)]
                if 1.0 in getattr(model, "classes_", [0.0, 1.0])
                else proba[:, -1]
            )
        # Fall back to decision_function or hard prediction.
        if hasattr(model, "decision_function"):
            d = model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-d))
        return model.predict(X).astype(float)
    return model.predict(X).astype(float)


def _importances(estimator, feature_names: list[str]) -> pd.Series:
    """Extract feature importances from common estimator types."""
    model = estimator
    # Unwrap a pipeline.
    if hasattr(estimator, "named_steps") and "model" in getattr(estimator, "named_steps", {}):
        model = estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        vals = np.abs(coef).ravel()
        if len(vals) != len(feature_names):  # multiclass safety
            vals = np.abs(coef).mean(axis=0)
    else:
        vals = np.full(len(feature_names), np.nan)
    s = pd.Series(vals, index=feature_names, name="importance")
    return s.sort_values(ascending=False)


def walk_forward_train(
    features: pd.DataFrame,
    config: ModelConfig,
    *,
    seed: int = 42,
) -> TrainResult:
    """Run walk-forward training/evaluation and return a :class:`TrainResult`."""
    features = features.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    feat_cols = _select_features(features, config)

    target_col = TARGET_DIRECTION if config.task == "classification" else TARGET_RETURN
    if target_col not in features.columns:
        raise ValueError(f"Target column '{target_col}' missing from feature frame")

    X_all = features[feat_cols].to_numpy(dtype=float)
    y_all = features[target_col].to_numpy(dtype=float)
    dates = features[DATE_COL]

    splitter = TimeSeriesSplitter(
        scheme=config.cv.scheme,
        n_splits=config.cv.n_splits,
        test_size=config.cv.test_size,
        min_train_size=config.cv.min_train_size,
        embargo=config.cv.embargo,
    )

    oos_rows = []
    fold_metrics: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dates)):
        model, _ = _make_pipeline(config, seed)
        y_train = y_all[train_idx]
        # Degenerate fold guard (classification with a single class).
        if config.task == "classification" and len(np.unique(y_train)) < 2:
            logger.warning("Fold %d skipped: training set has a single class", fold)
            continue
        model.fit(X_all[train_idx], y_train)
        scores = _predict_scores(model, X_all[test_idx], config.task)

        test_block = features.iloc[test_idx][[DATE_COL, TICKER_COL]].copy()
        test_block[SIGNAL_COL] = scores
        test_block["y_true"] = y_all[test_idx]
        test_block["fold"] = fold
        oos_rows.append(test_block)

        fm = _fold_metrics(y_all[test_idx], scores, config.task, fold)
        fold_metrics.append(fm)
        logger.info(
            "Fold %d: train=%d test=%d %s",
            fold,
            len(train_idx),
            len(test_idx),
            _fmt_metrics(fm),
        )

    if not oos_rows:
        raise RuntimeError("No folds produced predictions; check CV configuration.")

    predictions = pd.concat(oos_rows, ignore_index=True)

    # Aggregate OOS metrics across the full out-of-sample period.
    agg_metrics = _aggregate_metrics(predictions, config.task)

    # Final model on all data for importances + persistence.
    final_model, final_estimator = _make_pipeline(config, seed)
    final_model.fit(X_all, y_all)
    importances = _importances(final_model, feat_cols)

    logger.info(
        "Walk-forward complete: %d folds, OOS %s", len(fold_metrics), _fmt_metrics(agg_metrics)
    )
    return TrainResult(
        predictions=predictions,
        metrics=agg_metrics,
        fold_metrics=fold_metrics,
        feature_importances=importances,
        feature_names=feat_cols,
        model=final_model,
        task=config.task,
        n_splits=len(fold_metrics),
    )


def _fold_metrics(y_true, scores, task, fold) -> dict[str, Any]:
    if task == "classification":
        y_pred = (scores > 0.5).astype(int)
        m = classification_metrics(y_true, y_pred, scores)
    else:
        m = regression_metrics(y_true, scores)
    return {"fold": fold, **m}


def _aggregate_metrics(predictions: pd.DataFrame, task: str) -> dict[str, float]:
    y_true = predictions["y_true"].to_numpy()
    scores = predictions[SIGNAL_COL].to_numpy()
    if task == "classification":
        y_pred = (scores > 0.5).astype(int)
        return classification_metrics(y_true, y_pred, scores)
    return regression_metrics(y_true, scores)


def _fmt_metrics(m: dict[str, Any]) -> str:
    return " ".join(
        f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, (int, float)) and k != "fold"
    )


def save_model(result: TrainResult, path: str | Path) -> Path:
    """Persist the fitted final model + feature list via joblib."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "feature_names": result.feature_names,
            "task": result.task,
        },
        path,
    )
    logger.info("Saved model to %s", path)
    return path


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a previously persisted model bundle."""
    import joblib

    return joblib.load(Path(path))
