"""Tests for synthetic data generation, validation and ingestion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.config import DataConfig
from quant_platform.data.ingest import ingest, load_processed
from quant_platform.data.schema import OHLCV_COLUMNS, PANEL_COLUMNS, coerce_panel_dtypes
from quant_platform.data.validation import DataValidationError, validate_price_panel


def test_synthetic_panel_schema(synthetic_panel):
    for col in PANEL_COLUMNS:
        assert col in synthetic_panel.columns
    assert synthetic_panel["adj_close"].gt(0).all()
    assert (synthetic_panel["high"] >= synthetic_panel["low"]).all()


def test_synthetic_is_deterministic(tickers):
    from quant_platform.config import SyntheticConfig
    from quant_platform.data.synthetic import generate_synthetic_panel

    cfg = SyntheticConfig(n_days=200)
    a = generate_synthetic_panel(tickers, benchmark="SPY", config=cfg, seed=11)
    b = generate_synthetic_panel(tickers, benchmark="SPY", config=cfg, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_validation_passes_clean_panel(synthetic_panel):
    report = validate_price_panel(synthetic_panel, min_observations=100)
    assert report.ok
    assert report.n_tickers == synthetic_panel["ticker"].nunique()


def test_validation_detects_negative_prices(synthetic_panel):
    bad = synthetic_panel.copy()
    bad.loc[bad.index[0], "close"] = -5.0
    with pytest.raises(DataValidationError):
        validate_price_panel(bad, min_observations=100)


def test_validation_detects_duplicates(synthetic_panel):
    bad = pd.concat([synthetic_panel, synthetic_panel.iloc[:1]], ignore_index=True)
    report = validate_price_panel(bad, min_observations=100, raise_on_error=False)
    assert not report.ok
    assert any("duplicate" in e for e in report.errors)


def test_validation_missing_column():
    df = pd.DataFrame({"date": [1], "ticker": ["X"]})
    report = validate_price_panel(df, raise_on_error=False)
    assert not report.ok


def test_coerce_dtypes_sorts_and_types(synthetic_panel):
    shuffled = synthetic_panel.sample(frac=1.0, random_state=1)
    coerced = coerce_panel_dtypes(shuffled)
    assert coerced["date"].is_monotonic_increasing is False  # sorted by (ticker, date)
    # within a ticker, dates ascending
    g = coerced[coerced["ticker"] == "AAA"]
    assert g["date"].is_monotonic_increasing
    for col in OHLCV_COLUMNS:
        assert coerced[col].dtype == np.float64


def test_ingest_synthetic_roundtrip(tmp_path):
    cfg = DataConfig(
        source="synthetic",
        tickers=["SPY", "AAA", "BBB"],
        benchmark="SPY",
        start="2019-01-01",
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
        min_observations=100,
        synthetic={"n_days": 400, "start": "2019-01-01"},
    )
    panel = ingest(cfg, base_dir=str(tmp_path))
    assert "return" in panel.columns
    assert "log_return" in panel.columns
    # first row per ticker has NaN return (no prior price). Use head(1), since
    # groupby.first() skips NaNs by design.
    first_rows = panel.sort_values(["ticker", "date"]).groupby("ticker").head(1)
    assert first_rows["return"].isna().all()
    # cache load returns identical data
    reloaded = load_processed(tmp_path / "processed")
    assert len(reloaded) == len(panel)
    assert (tmp_path / "processed" / "panel_metadata.json").exists()


def test_returns_have_no_inf(synthetic_panel):
    grp = synthetic_panel.groupby("ticker")["adj_close"]
    rets = synthetic_panel["adj_close"] / grp.shift(1) - 1.0
    assert not np.isinf(rets.dropna()).any()
