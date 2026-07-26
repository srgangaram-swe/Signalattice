"""Benchmark verified Nasdaq cache replay without credential or network access."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from quant_platform.config import load_config
from quant_platform.data.nasdaq_data_link import (
    HttpResponse,
    NasdaqDataLinkClient,
)


class _NoNetworkTransport:
    """Fail if cache-only replay attempts an external request."""

    def get(
        self,
        url: str,
        *,
        timeout: float,
        headers: Mapping[str, str],
    ) -> HttpResponse:
        raise AssertionError("cache replay attempted network access")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/nasdaq_wiki_sample.yaml"),
        help="Committed Signalattice configuration with an existing verified local cache.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=7,
        help="Number of measured cache replays (default: 7).",
    )
    arguments = parser.parse_args()
    if arguments.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    return arguments


def main() -> None:
    """Replay one immutable request repeatedly and emit machine-readable evidence."""
    arguments = _arguments()
    app_config = load_config(arguments.config)
    data_config = app_config.data
    cache_config = data_config.nasdaq_data_link.model_copy(update={"cache_mode": "cache_only"})
    client = NasdaqDataLinkClient(
        cache_config,
        transport=_NoNetworkTransport(),
        secret_resolver=lambda: None,
    )

    durations: list[float] = []
    snapshot_hashes: set[str] = set()
    row_counts: set[int] = set()
    tickers: set[int] = set()
    tracemalloc.start()
    for _ in range(arguments.repetitions):
        start = time.perf_counter()
        result = client.fetch(
            data_config.tickers,
            data_config.start,
            data_config.end,
        )
        durations.append(time.perf_counter() - start)
        snapshot_hashes.add(str(result.manifest["snapshot_hash"]))
        row_counts.add(len(result.panel))
        tickers.add(int(result.panel["ticker"].nunique()))
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if len(snapshot_hashes) != 1 or len(row_counts) != 1 or len(tickers) != 1:
        raise RuntimeError("cache replay was not stable across benchmark repetitions")
    row_count = row_counts.pop()
    result = {
        "benchmark": "nasdaq_data_link_cache_replay",
        "config": str(arguments.config),
        "repetitions": arguments.repetitions,
        "rows": row_count,
        "tickers": tickers.pop(),
        "snapshot_hash": snapshot_hashes.pop(),
        "seconds": {
            "minimum": min(durations),
            "median": statistics.median(durations),
            "mean": statistics.fmean(durations),
            "p95": float(np.percentile(durations, 95)),
            "maximum": max(durations),
        },
        "median_rows_per_second": row_count / statistics.median(durations),
        "peak_tracemalloc_bytes": peak_bytes,
        "network_requests": 0,
        "credential_available_to_client": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "limitations": [
            "single local process",
            "warm filesystem cache is not controlled",
            "one machine and one observed benchmark run",
            "measures verified local replay, not provider latency",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
