"""External market-data source adapters with retry/error handling.

Each adapter returns data already normalised onto the canonical long-format
schema. Network dependencies (``yfinance``, ``pandas_datareader``) are imported
lazily so the core package installs and runs without them; callers should be
prepared to fall back to synthetic data via :class:`DataSourceError`.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pandas as pd

from quant_platform.data.schema import DATE_COL, OHLCV_COLUMNS, TICKER_COL
from quant_platform.logging_utils import get_logger

logger = get_logger(__name__)


class DataSourceError(RuntimeError):
    """Raised when an external data source cannot return usable data."""


def _retry(
    fn: Callable[[], pd.DataFrame],
    *,
    retries: int,
    backoff: float,
    what: str,
) -> pd.DataFrame:
    """Call ``fn`` with exponential backoff, raising :class:`DataSourceError`."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            result = fn()
            if result is None or len(result) == 0:
                raise DataSourceError(f"empty result for {what}")
            return result
        except Exception as exc:  # noqa: BLE001 - we re-raise as DataSourceError
            last_exc = exc
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed for %s (%s); retrying in %.1fs",
                attempt,
                retries,
                what,
                exc,
                wait,
            )
            if attempt < retries:
                time.sleep(wait)
    raise DataSourceError(f"all {retries} attempts failed for {what}: {last_exc}")


def _normalise_yf_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Map a yfinance OHLCV frame onto the canonical schema."""
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    if "adj_close" not in df.columns and "close" in df.columns:
        # yfinance auto_adjust=True returns adjusted values in `close`.
        df["adj_close"] = df["close"]
    df = df.reset_index().rename(columns={"Date": DATE_COL, "index": DATE_COL})
    df[TICKER_COL] = ticker
    keep = [DATE_COL, TICKER_COL, *OHLCV_COLUMNS]
    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[keep]


def fetch_yfinance(
    tickers: list[str],
    start: str,
    end: str | None,
    *,
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """Fetch daily OHLCV from Yahoo Finance via :mod:`yfinance`.

    Raises
    ------
    DataSourceError
        If ``yfinance`` is not installed or no data could be retrieved.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dep
        raise DataSourceError(
            "yfinance is not installed; install with `pip install '.[data]'`"
        ) from exc

    def _download() -> pd.DataFrame:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if raw is None or len(raw) == 0:
            raise DataSourceError("yfinance returned no rows")
        frames = []
        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                if ticker in raw.columns.get_level_values(0):
                    sub = raw[ticker].dropna(how="all")
                    if len(sub):
                        frames.append(_normalise_yf_frame(sub, ticker))
        else:  # single ticker
            frames.append(_normalise_yf_frame(raw.dropna(how="all"), tickers[0]))
        if not frames:
            raise DataSourceError("yfinance returned no usable frames")
        return pd.DataFrame(pd.concat(frames, ignore_index=True))

    out = _retry(_download, retries=retries, backoff=backoff, what="yfinance download")
    logger.info("Fetched %d rows from yfinance for %d tickers", len(out), len(tickers))
    return out


def fetch_stooq(
    tickers: list[str],
    start: str,
    end: str | None,
    *,
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """Fetch daily data from Stooq via :mod:`pandas_datareader`.

    Stooq does not provide an explicit adjusted-close column, so ``adj_close``
    is set equal to ``close``.
    """
    try:
        from pandas_datareader import data as pdr
    except ImportError as exc:  # pragma: no cover - optional dep
        raise DataSourceError(
            "pandas-datareader is not installed; install with `pip install '.[data]'`"
        ) from exc

    def _download() -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            raw = pdr.DataReader(ticker, "stooq", start=start, end=end)
            if raw is None or len(raw) == 0:
                logger.warning("Stooq returned no data for %s", ticker)
                continue
            raw = raw.sort_index()
            raw = raw.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            raw["adj_close"] = raw["close"]
            raw = raw.reset_index().rename(columns={"Date": DATE_COL})
            raw[TICKER_COL] = ticker
            for col in OHLCV_COLUMNS:
                if col not in raw.columns:
                    raw[col] = pd.NA
            frames.append(raw[[DATE_COL, TICKER_COL, *OHLCV_COLUMNS]])
        if not frames:
            raise DataSourceError("stooq returned no usable frames")
        return pd.DataFrame(pd.concat(frames, ignore_index=True))

    out = _retry(_download, retries=retries, backoff=backoff, what="stooq download")
    logger.info("Fetched %d rows from stooq for %d tickers", len(out), len(tickers))
    return out
