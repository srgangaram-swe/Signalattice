"""Feature engineering for time-series and cross-sectional quant signals.

Public API:

- :func:`build_features` — turn a validated price panel into a model-ready
  feature matrix (per-ticker technical features + optional cross-sectional
  features + forward-looking targets), free of lookahead bias.
- individual indicator functions in :mod:`quant_platform.features.technical`.
- immutable registry, quality, feature-store, and resumable backfill contracts.
- :func:`build_spectral_features` — opt-in causal spectral/time-frequency
  descriptors (SF-S3-MR1), disabled unless explicitly configured.
"""

from __future__ import annotations

from quant_platform.features.pipeline import (
    FEATURE_PREFIX,
    build_features,
    feature_columns,
)
from quant_platform.features.registry import FeatureRegistry, FeatureSpec
from quant_platform.features.spectral import (
    SPECTRAL_PREFIX,
    build_spectral_features,
    spectral_column_names,
    spectral_feature_registry,
)
from quant_platform.features.store import (
    FeatureMaterializationRequest,
    FeatureOutputContract,
    FeatureStore,
)

__all__ = [
    "FEATURE_PREFIX",
    "SPECTRAL_PREFIX",
    "FeatureMaterializationRequest",
    "FeatureOutputContract",
    "FeatureRegistry",
    "FeatureSpec",
    "FeatureStore",
    "build_features",
    "build_spectral_features",
    "feature_columns",
    "spectral_column_names",
    "spectral_feature_registry",
]
