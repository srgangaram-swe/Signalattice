"""Modelling: time-series-safe CV, baseline strategies and ML models.

Public API:

- :class:`TimeSeriesSplitter` — walk-forward / expanding-window splits with embargo.
- :func:`build_estimator` — construct an sklearn-compatible model from config.
- :func:`walk_forward_train` — leakage-free walk-forward training producing
  out-of-sample predictions, metrics and feature importances.
- :func:`baseline_signal` — rules-based (non-ML) baseline signals.
- :func:`local_level_filter` — causal scalar state-space reference.
- :func:`dynamic_linear_filter` — time-varying-coefficient Kalman reference.
- :func:`ewma_variance` / :func:`garch11_variance` — one-step volatility baselines.
- :func:`shrinkage_covariance` — conditioned multivariate covariance baseline.
"""

from __future__ import annotations

from quant_platform.models.baseline import available_baselines, baseline_signal
from quant_platform.models.covariance import ShrinkageCovarianceResult, shrinkage_covariance
from quant_platform.models.factory import build_estimator
from quant_platform.models.splits import TimeSeriesSplitter
from quant_platform.models.state_space import (
    DynamicLinearConfig,
    DynamicLinearResult,
    GaussianIntervalDiagnostics,
    LocalLevelConfig,
    LocalLevelResult,
    dynamic_linear_filter,
    gaussian_interval_diagnostics,
    local_level_filter,
)
from quant_platform.models.train import TrainResult, walk_forward_train
from quant_platform.models.volatility import (
    ConditionalVarianceResult,
    ewma_variance,
    garch11_variance,
)

__all__ = [
    "TimeSeriesSplitter",
    "build_estimator",
    "walk_forward_train",
    "TrainResult",
    "baseline_signal",
    "available_baselines",
    "LocalLevelConfig",
    "LocalLevelResult",
    "DynamicLinearConfig",
    "DynamicLinearResult",
    "GaussianIntervalDiagnostics",
    "local_level_filter",
    "dynamic_linear_filter",
    "gaussian_interval_diagnostics",
    "ConditionalVarianceResult",
    "ewma_variance",
    "garch11_variance",
    "ShrinkageCovarianceResult",
    "shrinkage_covariance",
]
