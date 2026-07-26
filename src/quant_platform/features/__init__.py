"""Feature engineering for time-series and cross-sectional quant signals.

Public API:

- :func:`build_features` — turn a validated price panel into a model-ready
  feature matrix (per-ticker technical features + optional cross-sectional
  features + forward-looking targets), free of lookahead bias.
- individual indicator functions in :mod:`quant_platform.features.technical`.
- immutable registry, quality, feature-store, and resumable backfill contracts.
- :func:`build_spectral_features` — opt-in causal spectral/time-frequency
  descriptors (SF-S3-MR1), disabled unless explicitly configured.
- :func:`build_time_frequency_tensor` and :class:`TimeFrequencyStore` —
  opt-in spectrogram/scalogram tensors and their content-addressed store
  (SF-S3-MR2); tensors never join the feature matrix.
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
from quant_platform.features.time_frequency import (
    TimeFrequencyMetadata,
    TimeFrequencyTensor,
    build_time_frequency_tensor,
    normalize_tensor,
)
from quant_platform.features.time_frequency_store import TimeFrequencyStore

__all__ = [
    "FEATURE_PREFIX",
    "SPECTRAL_PREFIX",
    "FeatureMaterializationRequest",
    "FeatureOutputContract",
    "FeatureRegistry",
    "FeatureSpec",
    "FeatureStore",
    "TimeFrequencyMetadata",
    "TimeFrequencyStore",
    "TimeFrequencyTensor",
    "build_features",
    "build_spectral_features",
    "build_time_frequency_tensor",
    "feature_columns",
    "normalize_tensor",
    "spectral_column_names",
    "spectral_feature_registry",
]
