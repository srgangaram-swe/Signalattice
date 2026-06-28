"""Tests for the vectorized backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.backtest.engine import (
    _row_weights,
    monthly_return_table,
    run_backtest,
)
from quant_platform.config import BacktestConfig
from quant_platform.models.baseline import baseline_signal


def test_row_weights_long_short_dollar_neutral():
    scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    cfg = BacktestConfig(
        strategy="long_short", top_quantile=0.5, position_sizing="equal_weight", max_leverage=1.0
    )
    w = _row_weights(scores, cfg)
    assert np.isclose(w.sum(), 0.0, atol=1e-9)  # dollar-neutral
    assert np.isclose(w.abs().sum(), 1.0, atol=1e-9)  # gross == leverage
    assert w["D"] > 0 and w["C"] > 0  # top names long
    assert w["A"] < 0 and w["B"] < 0  # bottom names short


def test_row_weights_long_only_fully_invested():
    scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    cfg = BacktestConfig(
        strategy="long_only", top_quantile=0.5, position_sizing="equal_weight", max_leverage=1.0
    )
    w = _row_weights(scores, cfg)
    assert (w >= 0).all()
    assert np.isclose(w.sum(), 1.0, atol=1e-9)
    assert w["A"] == 0 and w["B"] == 0


def test_position_cap_enforced():
    scores = pd.Series({"A": 1.0, "B": 2.0})
    cfg = BacktestConfig(
        strategy="long_only", top_quantile=0.5, max_position_weight=0.6, max_leverage=1.0
    )
    w = _row_weights(scores, cfg)
    from quant_platform.backtest.engine import _apply_position_cap

    capped = _apply_position_cap(w.to_frame().T, 0.6)
    assert (capped.abs() <= 0.6 + 1e-9).all().all()


def test_monthly_return_table_shape():
    idx = pd.bdate_range("2020-01-01", periods=300)
    returns = pd.Series(np.full(len(idx), 0.001), index=idx)
    table = monthly_return_table(returns)
    assert "YEAR" in table.columns
    assert not table.empty


def test_backtest_runs_and_reports_stats(synthetic_panel):
    from quant_platform.config import FeatureConfig
    from quant_platform.features.pipeline import build_features

    feats = build_features(synthetic_panel, FeatureConfig(), benchmark="SPY", forward_horizon=1)
    score = baseline_signal(feats, "momentum")
    signals = feats[["date", "ticker"]].copy()
    signals["score"] = score.to_numpy()
    signals = signals.dropna(subset=["score"])

    cfg = BacktestConfig(strategy="long_short", signal="momentum", top_quantile=0.34)
    result = run_backtest(signals, synthetic_panel, cfg, benchmark="SPY")

    assert len(result.returns) > 100
    assert result.equity_curve.iloc[-1] > 0
    for key in ("sharpe", "cagr", "max_drawdown", "ann_volatility"):
        assert key in result.stats
    assert result.weights.shape[1] >= 1


def test_transaction_costs_reduce_returns(synthetic_panel):
    from quant_platform.config import FeatureConfig
    from quant_platform.features.pipeline import build_features

    feats = build_features(synthetic_panel, FeatureConfig(), benchmark="SPY", forward_horizon=1)
    score = baseline_signal(feats, "momentum")
    signals = feats[["date", "ticker"]].copy()
    signals["score"] = score.to_numpy()
    signals = signals.dropna(subset=["score"])

    cheap = run_backtest(
        signals,
        synthetic_panel,
        BacktestConfig(signal="momentum", cost_bps=0.0, slippage_bps=0.0, top_quantile=0.34),
        benchmark="SPY",
    )
    pricey = run_backtest(
        signals,
        synthetic_panel,
        BacktestConfig(signal="momentum", cost_bps=50.0, slippage_bps=50.0, top_quantile=0.34),
        benchmark="SPY",
    )
    assert cheap.equity_curve.iloc[-1] > pricey.equity_curve.iloc[-1]
    assert pricey.stats["total_cost_drag"] > cheap.stats["total_cost_drag"]
