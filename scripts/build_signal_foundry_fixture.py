"""Build the redistribution-safe Signal Foundry v1 contract fixture.

The fixture is deterministic and intentionally tiny. It gives AlphaForge and
other consumers a repository-native compatibility target without publishing
licensed provider observations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_platform.data.signal_foundry_contract import export_signal_foundry_bundle

OUTPUT_ROOT = Path("tests/fixtures/signal_foundry_v1")


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2023-12-28", periods=6)
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(("AAA", "SPY")):
        for day_index, date in enumerate(dates):
            close = 100.0 + ticker_index * 20.0 + day_index
            effective_at = pd.Timestamp(date, tz="UTC") + pd.Timedelta(hours=21)
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "adj_close": close - 0.25,
                    "volume": 1_000_000.0 + day_index * 10_000.0,
                    "effective_at": effective_at,
                    "available_at": effective_at + pd.Timedelta(hours=8),
                    "observed_at": pd.Timestamp("2026-07-23T00:00:00Z"),
                    "provider_updated_at": pd.Timestamp("2026-07-20T00:00:00Z"),
                    "instrument_id": ticker,
                    "currency": "USD",
                    "exchange_calendar": "XNYS",
                    "adjustment_state": "synthetic_fixture",
                    "source": "synthetic_provider_fixture",
                    "source_table": "TEST/OHLCV",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    bundle = export_signal_foundry_bundle(
        _panel(),
        OUTPUT_ROOT,
        source_manifest={
            "provider": "synthetic_provider_fixture",
            "request": {"table": "TEST/OHLCV"},
            "request_hash": "1" * 64,
            "snapshot_hash": "2" * 64,
            "retrieved_at": "2026-07-23T00:00:00Z",
            "adjustment_state": ["synthetic_fixture"],
            "availability_policy": {"available_at": "effective_at plus 8 hours"},
            "point_in_time_limits": {
                "historical_revisions_complete": True,
                "universe_membership_point_in_time": True,
                "corporate_actions_complete": True,
            },
            "contains_api_key": False,
            "observations_redistributable": True,
        },
        producer_git_sha="0" * 40,
    )
    pointer = OUTPUT_ROOT / "current.json"
    pointer.write_text(
        json.dumps({"schema_version": "1.0.0", "bundle_id": bundle.name}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(bundle)


if __name__ == "__main__":
    main()
