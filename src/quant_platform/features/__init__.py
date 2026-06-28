"""Feature engineering for time-series and cross-sectional quant signals.

Public API:

- :func:`build_features` — turn a validated price panel into a model-ready
  feature matrix (per-ticker technical features + optional cross-sectional
  features + forward-looking targets), free of lookahead bias.
- individual indicator functions in :mod:`quant_platform.features.technical`.
"""

from __future__ import annotations

from quant_platform.features.pipeline import (
    FEATURE_PREFIX,
    build_features,
    feature_columns,
)

__all__ = ["build_features", "feature_columns", "FEATURE_PREFIX"]
