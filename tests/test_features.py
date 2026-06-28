"""Tests for technical indicators and the feature pipeline (no lookahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.features import technical as ta
from quant_platform.features.pipeline import (
    TARGET_DIRECTION,
    TARGET_RETURN,
    feature_columns,
)


@pytest.fixture
def price_series():
    rng = np.random.default_rng(3)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))


def test_log_and_simple_returns_close(price_series):
    simple = ta.simple_returns(price_series)
    log = ta.log_returns(price_series)
    # log ~ simple for small returns
    np.testing.assert_allclose(log.dropna(), np.log1p(simple.dropna()), rtol=1e-6)


def test_rsi_bounded(price_series):
    r = ta.rsi(price_series, 14).dropna()
    assert r.between(0, 100).all()


def test_rsi_all_gains_is_100():
    rising = pd.Series(np.arange(1, 100, dtype=float))
    r = ta.rsi(rising, 14).dropna()
    assert (r > 99).all()


def test_macd_columns(price_series):
    m = ta.macd(price_series)
    assert list(m.columns) == ["macd", "macd_signal", "macd_hist"]
    np.testing.assert_allclose(
        (m["macd"] - m["macd_signal"]).dropna(), m["macd_hist"].dropna(), atol=1e-9
    )


def test_bollinger_pctb_range(price_series):
    bb = ta.bollinger_bands(price_series, 20, 2.0)
    # %B is mostly within [0,1] but can exceed on breakouts; check it's finite
    assert np.isfinite(bb["bb_pctb"].dropna()).all()
    valid = bb.dropna(subset=["bb_upper", "bb_lower"])
    assert (valid["bb_upper"] >= valid["bb_lower"]).all()


def test_rolling_beta_of_self_is_one():
    rng = np.random.default_rng(5)
    mkt = pd.Series(rng.normal(0, 0.01, 300))
    beta = ta.rolling_beta(mkt, mkt, 63).dropna()
    np.testing.assert_allclose(beta, 1.0, atol=1e-8)


def test_drawdown_non_positive(price_series):
    dd = ta.drawdown(price_series)
    assert (dd <= 1e-12).all()


def test_momentum_definition(price_series):
    m = ta.momentum(price_series, 10)
    expected = price_series.iloc[20] / price_series.iloc[10] - 1.0
    assert np.isclose(m.iloc[20], expected)


def test_feature_pipeline_produces_features(feature_frame):
    feats = feature_columns(feature_frame)
    assert len(feats) > 20
    assert TARGET_DIRECTION in feature_frame.columns
    assert TARGET_RETURN in feature_frame.columns
    # no NaNs/inf remain after dropna
    assert not feature_frame[feats].isna().any().any()
    assert np.isfinite(feature_frame[feats].to_numpy()).all()


def test_target_direction_matches_forward_return(feature_frame):
    up = feature_frame[TARGET_RETURN] > 0
    assert (feature_frame[TARGET_DIRECTION] == up.astype(float)).all()


def test_no_lookahead_target_is_forward(synthetic_panel, app_config):
    """The target at t must equal the realised forward return t->t+1."""
    from quant_platform.features.pipeline import build_features

    feats = build_features(synthetic_panel, app_config.features, benchmark="SPY", forward_horizon=1)
    # Pick one ticker and verify against raw prices.
    raw = synthetic_panel[synthetic_panel["ticker"] == "AAA"].set_index("date")["adj_close"]
    sub = feats[feats["ticker"] == "AAA"].set_index("date")
    row = sub.iloc[10]
    d = sub.index[10]
    pos = raw.index.get_loc(d)
    expected = raw.iloc[pos + 1] / raw.iloc[pos] - 1.0
    assert np.isclose(row[TARGET_RETURN], expected, rtol=1e-6)


def test_cross_sectional_rank_features_present(feature_frame):
    cs = [c for c in feature_columns(feature_frame) if "cs_rank" in c]
    assert cs, "expected cross-sectional rank features"
