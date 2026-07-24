"""Deterministic local benchmark for bundle export, validation, and as-of read."""

from __future__ import annotations

import json
import platform
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pyarrow

from quant_platform.data.signal_foundry_contract import (
    export_signal_foundry_bundle,
    load_signal_foundry_bundle,
    validate_signal_foundry_bundle,
)


def _panel(*, tickers: int = 32, dates: int = 1_500) -> pd.DataFrame:
    calendar = pd.bdate_range("2018-01-01", periods=dates)
    rows: list[dict[str, object]] = []
    for ticker_index in range(tickers):
        ticker = f"S{ticker_index:03d}"
        for date_index, date in enumerate(calendar):
            close = 50.0 + ticker_index + date_index * 0.01
            effective_at = pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=21)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close - 0.25,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000.0,
                    "effective_at": effective_at,
                    "available_at": effective_at + pd.Timedelta(hours=8),
                    "observed_at": pd.Timestamp("2026-07-23T00:00:00Z"),
                    "provider_updated_at": pd.Timestamp("2026-07-20T00:00:00Z"),
                    "instrument_id": ticker,
                    "currency": "USD",
                    "exchange_calendar": "XNYS",
                    "adjustment_state": "synthetic_benchmark",
                    "source": "synthetic_benchmark",
                    "source_table": "BENCH/OHLCV",
                }
            )
    return pd.DataFrame(rows)


def _timed[T](operation: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    value = operation()
    return value, time.perf_counter() - start


def main() -> None:
    frame = _panel()
    source_manifest = {
        "provider": "synthetic_benchmark",
        "request": {"table": "BENCH/OHLCV"},
        "request_hash": "3" * 64,
        "snapshot_hash": "4" * 64,
        "retrieved_at": "2026-07-23T00:00:00Z",
        "contains_api_key": False,
        "observations_redistributable": True,
        "point_in_time_limits": {},
    }
    tracemalloc.start()
    with tempfile.TemporaryDirectory(prefix="signal-foundry-contract-") as temporary:
        bundle, export_seconds = _timed(
            lambda: export_signal_foundry_bundle(
                frame,
                Path(temporary),
                source_manifest=source_manifest,
                producer_git_sha="0" * 40,
            )
        )
        _, validate_seconds = _timed(lambda: validate_signal_foundry_bundle(bundle))
        cutoff = "2022-01-01T00:00:00Z"
        filtered, read_seconds = _timed(lambda: load_signal_foundry_bundle(bundle, as_of=cutoff))
        bytes_on_disk = sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file())
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "benchmark": "signal_foundry_contract",
        "rows": len(frame),
        "tickers": int(frame["ticker"].nunique()),
        "filtered_rows": len(filtered),
        "export_seconds": export_seconds,
        "validate_seconds": validate_seconds,
        "as_of_read_seconds": read_seconds,
        "rows_per_export_second": len(frame) / export_seconds,
        "bundle_bytes": bytes_on_disk,
        "peak_tracemalloc_bytes": peak_bytes,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "limitations": [
            "single local process",
            "synthetic deterministic data",
            "warm filesystem state is not controlled",
            "not a provider-network benchmark",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
