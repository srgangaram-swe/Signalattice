"""Deterministic local benchmark for the SF-S2-MR2 statistical feature contracts.

Reports wall time and peak memory for the volatility, distribution, and
dependence contracts across representative panel sizes. Several estimators
(Hurst, variance ratio, mutual information) run a per-window kernel and dominate
the cost, so panel sizes are more modest than the vectorised MR1 benchmark.
Fully offline and deterministic; laptop-scale numbers, not a distributed
benchmark.

Usage::

    python scripts/benchmark_statistical_contracts.py
"""

from __future__ import annotations

import json
import platform
import time
import tracemalloc
from collections.abc import Callable

import numpy as np
import pandas as pd

from quant_platform.features.contracts import build_contract_features, statistical_contracts

# (tickers, dates) grids; kept modest because Hurst/variance-ratio/mutual-
# information use per-window kernels (roughly O(rows * window)).
PANEL_SIZES: tuple[tuple[int, int], ...] = (
    (16, 504),
    (32, 1_008),
    (64, 1_260),
)


def _panel(tickers: int, dates: int, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range("2015-01-01", periods=dates)
    frames = []
    for index in range(tickers):
        close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, dates)))
        frames.append(
            pd.DataFrame(
                {
                    "date": calendar,
                    "ticker": f"S{index:04d}",
                    "open": close * (1.0 + rng.normal(0.0, 0.002, dates)),
                    "high": close * 1.005,
                    "low": close * 0.995,
                    "close": close,
                    "adj_close": close,
                    "volume": rng.uniform(1e6, 5e6, dates),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _timed[T](operation: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    value = operation()
    return value, time.perf_counter() - start


def main() -> None:
    contracts = statistical_contracts()
    n_contracts = len(contracts)
    cases = []
    for tickers, dates in PANEL_SIZES:
        frame = _panel(tickers, dates, seed=7)
        tracemalloc.start()
        features, seconds = _timed(lambda frame=frame: build_contract_features(frame, contracts))
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        cases.append(
            {
                "tickers": tickers,
                "dates": dates,
                "rows": len(frame),
                "feature_columns": len(features.columns) - 2,
                "seconds": seconds,
                "rows_per_second": len(frame) / seconds,
                "cell_updates_per_second": (len(frame) * n_contracts) / seconds,
                "peak_tracemalloc_bytes": peak_bytes,
            }
        )
    result = {
        "benchmark": "statistical_contracts",
        "n_contracts": n_contracts,
        "cases": cases,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "limitations": [
            "single local process",
            "synthetic deterministic data",
            "warm interpreter and cache state is not controlled",
            "laptop-scale; not a distributed benchmark",
            "per-window kernels (Hurst, variance ratio, mutual information) dominate",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
