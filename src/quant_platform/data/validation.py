"""Data validation layer.

We validate ingested data *before* it reaches feature engineering so that bad
data fails loudly and early rather than silently corrupting research results.
The checks are intentionally explicit (rather than relying on a heavyweight
dependency) so reviewers can see exactly what is asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_platform.data.schema import (
    DATE_COL,
    OHLCV_COLUMNS,
    TICKER_COL,
    PriceSchema,
)
from quant_platform.logging_utils import get_logger

logger = get_logger(__name__)


class DataValidationError(ValueError):
    """Raised when a price panel violates the expected schema/contract."""


@dataclass
class ValidationReport:
    """Structured result of validating a price panel."""

    n_rows: int = 0
    n_tickers: int = 0
    tickers: list[str] = field(default_factory=list)
    date_min: str | None = None
    date_max: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    nan_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"Validation: {status}",
            f"  rows={self.n_rows} tickers={self.n_tickers} "
            f"range=[{self.date_min} .. {self.date_max}]",
        ]
        if self.warnings:
            lines.append(f"  warnings ({len(self.warnings)}):")
            lines.extend(f"    - {w}" for w in self.warnings)
        if self.errors:
            lines.append(f"  errors ({len(self.errors)}):")
            lines.extend(f"    - {e}" for e in self.errors)
        return "\n".join(lines)


def validate_price_panel(
    df: pd.DataFrame,
    *,
    schema: PriceSchema | None = None,
    min_observations: int = 252,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate a long-format price panel against the canonical schema.

    Checks performed:

    1. required columns present;
    2. non-empty;
    3. no duplicate ``(date, ticker)`` rows;
    4. dates sorted & unique per ticker;
    5. prices strictly positive (no zero/negative);
    6. ``high >= low`` and ``high >= close >= low`` style sanity bounds;
    7. volume non-negative;
    8. each ticker has at least ``min_observations`` rows;
    9. NaN accounting (reported as warnings).

    Parameters
    ----------
    raise_on_error:
        When true (default) a :class:`DataValidationError` is raised if any
        hard error is found; otherwise the report is returned for inspection.
    """
    schema = schema or PriceSchema()
    report = ValidationReport()

    # 1. Required columns.
    missing = [c for c in schema.required_columns() if c not in df.columns]
    if missing:
        report.errors.append(f"missing required columns: {missing}")
        return _finalise(report, raise_on_error)

    # 2. Non-empty.
    report.n_rows = int(len(df))
    if report.n_rows == 0:
        report.errors.append("price panel is empty")
        return _finalise(report, raise_on_error)

    report.tickers = sorted(df[TICKER_COL].astype(str).unique().tolist())
    report.n_tickers = len(report.tickers)
    report.date_min = str(pd.to_datetime(df[DATE_COL]).min().date())
    report.date_max = str(pd.to_datetime(df[DATE_COL]).max().date())

    # 3. Duplicate (date, ticker).
    dup_mask = df.duplicated(subset=[DATE_COL, TICKER_COL], keep=False)
    if dup_mask.any():
        report.errors.append(f"{int(dup_mask.sum())} duplicate (date, ticker) rows")

    # NaN accounting.
    for col in OHLCV_COLUMNS:
        if col in df.columns:
            n_nan = int(df[col].isna().sum())
            report.nan_counts[col] = n_nan
            if n_nan:
                report.warnings.append(f"column '{col}' has {n_nan} NaNs")

    # Per-ticker checks.
    for ticker, g in df.groupby(TICKER_COL, sort=True):
        dates = pd.to_datetime(g[DATE_COL])
        # 4. sorted & unique per ticker.
        if not dates.is_monotonic_increasing:
            report.warnings.append(f"[{ticker}] dates not sorted ascending")
        if dates.duplicated().any():
            report.errors.append(f"[{ticker}] duplicate dates within ticker")

        # 8. minimum observations.
        if len(g) < min_observations:
            report.warnings.append(f"[{ticker}] only {len(g)} observations (< {min_observations})")

        # 5. positive prices.
        price_cols = [c for c in schema.price_columns if c in g.columns]
        prices = g[price_cols]
        nonpos = (prices <= 0) & prices.notna()
        if nonpos.to_numpy().any():
            n = int(nonpos.to_numpy().sum())
            report.errors.append(f"[{ticker}] {n} non-positive price values")

        # 6. OHLC bounds sanity.
        if {"high", "low"}.issubset(g.columns):
            bad_hl = (g["high"] < g["low"]) & g["high"].notna() & g["low"].notna()
            if bad_hl.any():
                report.errors.append(f"[{ticker}] {int(bad_hl.sum())} rows with high < low")

        # 7. volume non-negative.
        if schema.volume_column in g.columns:
            vol = g[schema.volume_column]
            bad_vol = (vol < 0) & vol.notna()
            if bad_vol.any():
                report.warnings.append(f"[{ticker}] {int(bad_vol.sum())} negative volume rows")

    return _finalise(report, raise_on_error)


def _finalise(report: ValidationReport, raise_on_error: bool) -> ValidationReport:
    if report.ok:
        logger.info("Data validation passed: %d rows, %d tickers", report.n_rows, report.n_tickers)
    else:
        logger.error("Data validation failed:\n%s", report.summary())
        if raise_on_error:
            raise DataValidationError(report.summary())
    return report


def assert_no_lookahead(features: pd.DataFrame, target: pd.Series) -> None:
    """Sanity check that a target is shifted into the future relative to features.

    This does not *prove* the absence of lookahead bias, but it catches the most
    common mistake — perfectly correlated, contemporaneous targets — by asserting
    the indices align and the target is not identical to a feature column.
    """
    if not features.index.equals(target.index):
        raise DataValidationError("feature/target indices are not aligned")
    for col in features.columns:
        if features[col].equals(target.astype(features[col].dtype, errors="ignore")):
            raise DataValidationError(f"target is identical to feature '{col}' — possible leakage")
    # Targets built from forward returns must not be perfectly explained by
    # contemporaneous returns; a correlation of exactly 1.0 is a red flag.
    if "log_return" in features.columns:
        corr = np.corrcoef(
            features["log_return"].fillna(0.0).to_numpy(),
            pd.to_numeric(target, errors="coerce").fillna(0.0).to_numpy(),
        )[0, 1]
        if abs(corr) > 0.999:
            raise DataValidationError(
                "target almost perfectly correlated with contemporaneous return"
            )
