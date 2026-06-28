"""Shared pytest fixtures.

The fixtures build a small, fully-offline synthetic dataset so the entire test
suite runs without network access in well under a minute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.config import AppConfig, SyntheticConfig
from quant_platform.data.synthetic import generate_synthetic_panel
from quant_platform.features.pipeline import build_features


@pytest.fixture(scope="session")
def tickers() -> list[str]:
    return ["SPY", "AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture(scope="session")
def synthetic_panel(tickers) -> pd.DataFrame:
    """A reproducible synthetic OHLCV panel with returns added."""
    cfg = SyntheticConfig(n_days=900, start="2018-01-01")
    panel = generate_synthetic_panel(tickers, benchmark="SPY", config=cfg, seed=7)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    grp = panel.groupby("ticker")["adj_close"]
    prev = grp.shift(1)
    panel["return"] = panel["adj_close"] / prev - 1.0
    panel["log_return"] = np.log(panel["adj_close"] / prev)
    return panel


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    """A small in-memory config tuned for fast tests."""
    return AppConfig.model_validate(
        {
            "project": {"name": "test", "seed": 7},
            "data": {
                "source": "synthetic",
                "tickers": ["SPY", "AAA", "BBB", "CCC", "DDD", "EEE"],
                "benchmark": "SPY",
                "start": "2018-01-01",
                "synthetic": {"n_days": 900, "start": "2018-01-01"},
                "min_observations": 100,
            },
            "model": {
                "task": "classification",
                "type": "logistic",
                "cv": {"n_splits": 3, "test_size": 60, "min_train_size": 200, "embargo": 2},
            },
            "backtest": {"strategy": "long_short", "signal": "model", "top_quantile": 0.34},
        }
    )


@pytest.fixture(scope="session")
def feature_frame(synthetic_panel, app_config) -> pd.DataFrame:
    """Engineered features + targets for the synthetic panel."""
    return build_features(
        synthetic_panel,
        app_config.features,
        benchmark="SPY",
        price_field="adj_close",
        forward_horizon=1,
    )


@pytest.fixture
def daily_returns() -> pd.Series:
    """A deterministic daily-return series for risk-metric tests."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=750)
    return pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
