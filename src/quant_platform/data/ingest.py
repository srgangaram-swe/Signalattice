"""Ingestion orchestration: fetch → validate → persist → load.

This module is the single entry point used by the CLI and pipeline. It handles
source selection (with graceful fallback to synthetic data), schema coercion,
validation, derived return columns and Parquet persistence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.config import DataConfig
from quant_platform.data.schema import DATE_COL, TICKER_COL, coerce_panel_dtypes
from quant_platform.data.sources import DataSourceError, fetch_stooq, fetch_yfinance
from quant_platform.data.synthetic import generate_synthetic_panel
from quant_platform.data.validation import validate_price_panel
from quant_platform.logging_utils import get_logger
from quant_platform.utils import ensure_dir, hash_dataframe, resolve_path

logger = get_logger(__name__)

PROCESSED_FILENAME = "price_panel.parquet"


def _add_return_columns(df: pd.DataFrame, price_field: str) -> pd.DataFrame:
    """Add simple & log returns computed per ticker from the chosen price field.

    Returns are computed *within* each ticker group and the first observation of
    each ticker is therefore ``NaN`` (no prior price) — this is correct and
    avoids spurious cross-ticker returns at group boundaries.
    """
    out = df.sort_values([TICKER_COL, DATE_COL]).copy()
    px = out[price_field]
    grp = out.groupby(TICKER_COL, sort=False)
    prev = grp[price_field].shift(1)
    out["return"] = px / prev - 1.0
    out["log_return"] = np.log(px / prev)
    # Guard against inf from zero prices (should already be filtered by validation).
    out[["return", "log_return"]] = out[["return", "log_return"]].replace([np.inf, -np.inf], np.nan)
    return out


def _fetch_raw(config: DataConfig) -> tuple[pd.DataFrame, str]:
    """Fetch raw data according to the configured source, with fallback.

    Returns a tuple of ``(panel, source_used)``.
    """
    source = config.source
    tickers = config.tickers
    order: list[str] = ["yfinance", "stooq", "synthetic"] if source == "auto" else [source]

    last_error: Exception | None = None
    for src in order:
        try:
            if src == "yfinance":
                panel = fetch_yfinance(
                    tickers,
                    config.start,
                    config.end,
                    retries=config.max_retries,
                    backoff=config.retry_backoff_seconds,
                )
            elif src == "stooq":
                panel = fetch_stooq(
                    tickers,
                    config.start,
                    config.end,
                    retries=config.max_retries,
                    backoff=config.retry_backoff_seconds,
                )
            elif src == "synthetic":
                panel = generate_synthetic_panel(
                    tickers,
                    benchmark=config.benchmark,
                    config=config.synthetic,
                    seed=42,
                )
            else:  # pragma: no cover - guarded by config validation
                raise DataSourceError(f"unknown source '{src}'")
            logger.info("Ingested data using source='%s'", src)
            return panel, src
        except DataSourceError as exc:
            last_error = exc
            logger.warning("Source '%s' unavailable: %s", src, exc)
            continue

    # If we get here every configured source failed.
    if source != "auto":
        raise DataSourceError(
            f"data source '{source}' failed: {last_error}. "
            "Set data.source='synthetic' or 'auto' for an offline fallback."
        )
    raise DataSourceError(f"all data sources failed; last error: {last_error}")


def ingest(
    config: DataConfig,
    *,
    base_dir: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Ingest market data and persist a validated, return-augmented panel.

    Parameters
    ----------
    config:
        Data configuration.
    base_dir:
        Base directory to resolve relative ``raw_dir``/``processed_dir`` against
        (defaults to the current working directory).
    force:
        Re-fetch even if a processed panel already exists on disk.

    Returns
    -------
    pandas.DataFrame
        The validated long-format panel with ``return`` and ``log_return``
        columns, sorted by ``(ticker, date)``.
    """
    processed_dir = ensure_dir(resolve_path(config.processed_dir, base_dir))
    raw_dir = ensure_dir(resolve_path(config.raw_dir, base_dir))
    out_path = processed_dir / PROCESSED_FILENAME

    if out_path.exists() and not force:
        logger.info("Loading cached processed panel from %s", out_path)
        return load_processed(processed_dir)

    panel, source_used = _fetch_raw(config)
    panel = coerce_panel_dtypes(panel)

    # Drop fully-empty rows and clip the date window.
    panel = panel.dropna(subset=["close"])
    if config.end:
        panel = panel[panel[DATE_COL] <= pd.to_datetime(config.end)]
    panel = panel[panel[DATE_COL] >= pd.to_datetime(config.start)]

    # Persist raw per-ticker parquet (audit trail / re-use).
    for ticker, g in panel.groupby(TICKER_COL):
        g.to_parquet(raw_dir / f"{ticker}.parquet", index=False)

    # Validate before deriving features.
    report = validate_price_panel(
        panel, min_observations=config.min_observations, raise_on_error=True
    )
    logger.info("\n%s", report.summary())

    # Derived returns.
    panel = _add_return_columns(panel, config.price_field)

    # Persist processed panel + a small metadata sidecar.
    panel.to_parquet(out_path, index=False)
    meta = {
        "source": source_used,
        "tickers": sorted(panel[TICKER_COL].unique().tolist()),
        "n_rows": int(len(panel)),
        "date_min": str(panel[DATE_COL].min().date()),
        "date_max": str(panel[DATE_COL].max().date()),
        "price_field": config.price_field,
        "data_hash": hash_dataframe(panel),
    }
    pd.Series(meta).to_json(processed_dir / "panel_metadata.json", indent=2)
    logger.info("Saved processed panel to %s (hash=%s)", out_path, meta["data_hash"])
    return panel


def load_processed(processed_dir: str | Path) -> pd.DataFrame:
    """Load a previously ingested processed panel from ``processed_dir``."""
    path = Path(processed_dir) / PROCESSED_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No processed panel at {path}. Run `ingest-data` first.")
    df = pd.read_parquet(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
