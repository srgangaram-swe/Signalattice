"""Tests for time-series splits, model factory and walk-forward training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.config import ModelConfig
from quant_platform.models.factory import build_estimator
from quant_platform.models.metrics import classification_metrics, regression_metrics
from quant_platform.models.splits import TimeSeriesSplitter
from quant_platform.models.train import walk_forward_train


def _panel_dates(n_dates=400, n_tickers=5):
    dates = pd.bdate_range("2018-01-01", periods=n_dates)
    return pd.Series(np.repeat(dates, n_tickers))


def test_splitter_no_overlap_and_forward():
    dates = _panel_dates(400, 5)
    splitter = TimeSeriesSplitter(
        scheme="walk_forward", n_splits=3, test_size=40, min_train_size=120, embargo=3
    )
    folds = list(splitter.split(dates))
    assert len(folds) == 3
    unique_dates = np.sort(dates.unique())
    for train_idx, test_idx in folds:
        train_dates = set(dates.iloc[train_idx])
        test_dates = set(dates.iloc[test_idx])
        # No leakage: every test date strictly after every train date.
        assert max(train_dates) < min(test_dates)
        # Embargo gap respected (in date-index space).
        train_max_pos = max(np.where(unique_dates == max(train_dates))[0])
        test_min_pos = min(np.where(unique_dates == min(test_dates))[0])
        assert test_min_pos - train_max_pos > 3


def test_expanding_grows_training_set():
    dates = _panel_dates(400, 4)
    splitter = TimeSeriesSplitter(
        scheme="expanding", n_splits=3, test_size=40, min_train_size=120, embargo=2
    )
    sizes = [len(tr) for tr, _ in splitter.split(dates)]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_splitter_raises_when_too_few_dates():
    dates = _panel_dates(50, 2)
    splitter = TimeSeriesSplitter(n_splits=5, test_size=40, min_train_size=120, embargo=5)
    with pytest.raises(ValueError):
        list(splitter.split(dates))


@pytest.mark.parametrize("mtype", ["logistic", "random_forest", "gradient_boosting"])
def test_factory_classification(mtype):
    cfg = ModelConfig(task="classification", type=mtype)
    est = build_estimator(cfg, seed=1)
    assert hasattr(est, "fit")
    assert hasattr(est, "predict")


def test_factory_xgboost_falls_back_when_absent():
    cfg = ModelConfig(task="classification", type="xgboost")
    est = build_estimator(cfg, seed=1)  # should not raise even without xgboost
    assert hasattr(est, "fit")


def test_classification_metrics_keys():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0, 1, 0, 0, 1])
    proba = np.array([0.2, 0.8, 0.4, 0.1, 0.9])
    m = classification_metrics(y, p, proba)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in m


def test_regression_metrics_directional():
    y = np.array([0.01, -0.02, 0.03, -0.01])
    p = np.array([0.02, -0.01, 0.01, 0.005])
    m = regression_metrics(y, p)
    assert 0.0 <= m["directional_accuracy"] <= 1.0
    assert m["rmse"] >= 0.0


def test_walk_forward_train_produces_oos_predictions(feature_frame, app_config):
    result = walk_forward_train(feature_frame, app_config.model, seed=7)
    assert not result.predictions.empty
    assert {"date", "ticker", "score", "y_true", "fold"}.issubset(result.predictions.columns)
    # scores look like probabilities for classification
    assert result.predictions["score"].between(0, 1).mean() > 0.95
    assert "accuracy" in result.metrics
    assert len(result.feature_importances) == len(result.feature_names)
    assert result.n_splits >= 1


def test_walk_forward_predictions_are_out_of_sample(feature_frame, app_config):
    """OOS predictions must cover only the tail (test) dates, not all history."""
    result = walk_forward_train(feature_frame, app_config.model, seed=7)
    pred_dates = pd.to_datetime(result.predictions["date"])
    all_dates = pd.to_datetime(feature_frame["date"])
    assert pred_dates.min() > all_dates.min()
