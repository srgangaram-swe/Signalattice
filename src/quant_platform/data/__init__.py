"""Market-data ingestion, validation and storage.

Public API:

- :func:`ingest` — orchestrate fetch → validate → store → return a clean panel.
- :func:`load_processed` — load a previously ingested panel from Parquet.
- :class:`DataValidationError` — raised when ingested data fails schema checks.
"""

from __future__ import annotations

from quant_platform.data.ingest import ingest, load_processed
from quant_platform.data.schema import OHLCV_COLUMNS, PriceSchema
from quant_platform.data.signal_foundry_contract import (
    SignalFoundryContractError,
    export_signal_foundry_bundle,
    load_signal_foundry_bundle,
    validate_signal_foundry_bundle,
)
from quant_platform.data.validation import (
    DataValidationError,
    ValidationReport,
    validate_price_panel,
)

__all__ = [
    "DataValidationError",
    "OHLCV_COLUMNS",
    "PriceSchema",
    "SignalFoundryContractError",
    "ValidationReport",
    "export_signal_foundry_bundle",
    "ingest",
    "load_processed",
    "load_signal_foundry_bundle",
    "validate_price_panel",
    "validate_signal_foundry_bundle",
]
