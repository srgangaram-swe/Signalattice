"""Modelling: time-series-safe CV, baseline strategies and ML models.

Public API:

- :class:`TimeSeriesSplitter` — walk-forward / expanding-window splits with embargo.
- :func:`build_estimator` — construct an sklearn-compatible model from config.
- :func:`walk_forward_train` — leakage-free walk-forward training producing
  out-of-sample predictions, metrics and feature importances.
- :func:`baseline_signal` — rules-based (non-ML) baseline signals.
"""

from __future__ import annotations

from quant_platform.models.baseline import available_baselines, baseline_signal
from quant_platform.models.factory import build_estimator
from quant_platform.models.splits import TimeSeriesSplitter
from quant_platform.models.train import TrainResult, walk_forward_train

__all__ = [
    "TimeSeriesSplitter",
    "build_estimator",
    "walk_forward_train",
    "TrainResult",
    "baseline_signal",
    "available_baselines",
]
