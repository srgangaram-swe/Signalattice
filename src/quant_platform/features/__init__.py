"""Feature engineering for time-series and cross-sectional quant signals.

Public API:

- :func:`build_features` — turn a validated price panel into a model-ready
  feature matrix (per-ticker technical features + optional cross-sectional
  features + forward-looking targets), free of lookahead bias.
- :func:`build_contract_features` — evaluate versioned, self-describing
  :class:`FeatureContract` objects (units, lookback, warm-up, missing-data
  policy, temporal availability, numerical behaviour) over a price panel.
- :func:`get_contract`, :func:`list_contracts`, :func:`contract_metadata_frame`
  — introspect the conventional-feature contract registry.
- individual indicator functions in :mod:`quant_platform.features.technical`.
- immutable registry, quality, feature-store, and resumable backfill contracts.
"""

from __future__ import annotations

from quant_platform.features.contracts import (
    CONTRACT_PREFIX,
    CONTRACT_REGISTRY,
    FeatureContract,
    FeatureFamily,
    MissingDataPolicy,
    NumericalRange,
    Scope,
    TemporalAvailability,
    Unit,
    build_contract_features,
    contract_metadata_frame,
    default_contracts,
    get_contract,
    liquidity_contracts,
    list_contracts,
    statistical_contracts,
    validate_contract_panel,
)
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
    "CONTRACT_PREFIX",
    "CONTRACT_REGISTRY",
    "FEATURE_PREFIX",
    "FeatureContract",
    "FeatureFamily",
    "FeatureMaterializationRequest",
    "FeatureOutputContract",
    "FeatureRegistry",
    "FeatureSpec",
    "FeatureStore",
    "MissingDataPolicy",
    "NumericalRange",
    "Scope",
    "TemporalAvailability",
    "Unit",
    "build_contract_features",
    "build_features",
    "contract_metadata_frame",
    "default_contracts",
    "feature_columns",
    "get_contract",
    "liquidity_contracts",
    "list_contracts",
    "statistical_contracts",
    "validate_contract_panel",
]
