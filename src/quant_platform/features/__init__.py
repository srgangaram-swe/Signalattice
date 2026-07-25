"""Feature engineering for time-series and cross-sectional quant signals.

Public API:

- :func:`build_features` — turn a validated price panel into a model-ready
  feature matrix (per-ticker technical features + optional cross-sectional
  features + forward-looking targets), free of lookahead bias.
- individual indicator functions in :mod:`quant_platform.features.technical`.
- immutable registry, quality, feature-store, and resumable backfill contracts.
"""

from __future__ import annotations

from quant_platform.features.pipeline import (
    FEATURE_PREFIX,
    build_features,
    feature_columns,
)
from quant_platform.features.registry import FeatureRegistry, FeatureSpec
from quant_platform.features.store import (
    FeatureMaterializationRequest,
    FeatureOutputContract,
    FeatureStore,
)

__all__ = [
    "FEATURE_PREFIX",
    "FeatureMaterializationRequest",
    "FeatureOutputContract",
    "FeatureRegistry",
    "FeatureSpec",
    "FeatureStore",
    "build_features",
    "feature_columns",
]
