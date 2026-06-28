"""Market-data ingestion, validation and storage.

Public API:

- :func:`ingest` — orchestrate fetch → validate → store → return a clean panel.
- :func:`load_processed` — load a previously ingested panel from Parquet.
- :class:`DataValidationError` — raised when ingested data fails schema checks.
"""

from __future__ import annotations

from quant_platform.data.ingest import ingest, load_processed
from quant_platform.data.schema import OHLCV_COLUMNS, PriceSchema
from quant_platform.data.validation import (
    DataValidationError,
    ValidationReport,
    validate_price_panel,
)

__all__ = [
    "ingest",
    "load_processed",
    "OHLCV_COLUMNS",
    "PriceSchema",
    "DataValidationError",
    "ValidationReport",
    "validate_price_panel",
]
