"""Canonical data schema definitions.

The platform standardises all ingested data onto a single tidy ("long") panel
format so that every downstream module makes the same assumptions:

* one row per ``(date, ticker)`` observation,
* a timezone-naive ``date`` column of pandas ``datetime64[ns]``,
* lower-snake-case OHLCV columns.

Keeping this contract in one place makes schema checks and refactors safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATE_COL = "date"
TICKER_COL = "ticker"

OHLCV_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

#: Required columns for the canonical long-format price panel.
PANEL_COLUMNS: tuple[str, ...] = (DATE_COL, TICKER_COL, *OHLCV_COLUMNS)

#: Expected pandas dtypes (as strings) for validation messages.
EXPECTED_DTYPES: dict[str, str] = {
    DATE_COL: "datetime64[ns]",
    TICKER_COL: "object",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "adj_close": "float64",
    "volume": "float64",
}


@dataclass(frozen=True)
class PriceSchema:
    """Lightweight, dependency-free schema describing the price panel."""

    date_col: str = DATE_COL
    ticker_col: str = TICKER_COL
    columns: tuple[str, ...] = field(default=PANEL_COLUMNS)
    price_columns: tuple[str, ...] = ("open", "high", "low", "close", "adj_close")
    volume_column: str = "volume"

    def required_columns(self) -> tuple[str, ...]:
        return self.columns


def coerce_panel_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a price panel to the canonical dtypes, returning a copy."""
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL]).dt.tz_localize(None)
    out[TICKER_COL] = out[TICKER_COL].astype(str)
    for col in OHLCV_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float64)
    return out.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
