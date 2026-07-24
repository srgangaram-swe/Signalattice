"""Tests for the vectorized backtesting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtest.engine import (
    _enforce_portfolio_limits,
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


def test_long_short_remains_flat_when_universe_cannot_support_both_sides():
    weights = _row_weights(pd.Series({"A": 1.0}), BacktestConfig(strategy="long_short"))
    assert weights.eq(0.0).all()


def test_rank_sizing_is_dollar_neutral_and_respects_gross_limit():
    config = BacktestConfig(strategy="long_short", position_sizing="rank", max_leverage=0.8)
    weights = _row_weights(pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}), config)
    assert np.isclose(weights.sum(), 0.0)
    assert np.isclose(weights.abs().sum(), 0.8)


def test_position_cap_enforced():
    scores = pd.Series({"A": 1.0, "B": 2.0})
    cfg = BacktestConfig(
        strategy="long_only", top_quantile=0.5, max_position_weight=0.6, max_leverage=1.0
    )
    w = _row_weights(scores, cfg)
    from quant_platform.backtest.engine import _apply_position_cap

    capped = _apply_position_cap(w.to_frame().T, 0.6)
    assert (capped.abs() <= 0.6 + 1e-9).all().all()


def test_limits_are_reapplied_after_overlay():
    weights = pd.DataFrame({"A": [1.5], "B": [-1.5]})
    cfg = BacktestConfig(max_position_weight=0.40, max_leverage=0.60)
    limited = _enforce_portfolio_limits(weights, cfg)
    assert limited.abs().max().max() <= 0.40
    assert limited.abs().sum(axis=1).iloc[0] <= 0.60


def test_position_caps_preserve_long_short_neutrality_by_scaling_down():
    weights = pd.DataFrame({"A": [0.25], "B": [0.25], "C": [-0.50]})
    limited = _enforce_portfolio_limits(
        weights,
        BacktestConfig(strategy="long_short", max_position_weight=0.25),
    )
    assert np.isclose(limited.sum(axis=1).iloc[0], 0.0)
    assert limited.abs().max().max() <= 0.25


def test_close_signal_observes_two_row_execution_lag():
    dates = pd.bdate_range("2024-01-01", periods=6)
    panel = pd.DataFrame(
        {
            "date": dates,
            "ticker": "A",
            "return": [0.00, 0.01, 0.03, -0.02, 0.04, 0.01],
        }
    )
    signals = pd.DataFrame({"date": dates[:4], "ticker": "A", "score": 1.0})
    cfg = BacktestConfig(
        strategy="long_only",
        max_leverage=1.0,
        max_position_weight=1.0,
        cost_bps=0.0,
        slippage_bps=0.0,
        execution_lag=2,
    )
    result = run_backtest(signals, panel, cfg, benchmark="A")
    assert result.returns.index[0] == dates[2]
    assert np.isclose(result.gross_returns.iloc[0], 0.03)
    assert np.isclose(result.weights.iloc[0, 0], 1.0)


def test_missing_signal_date_holds_position_and_market_calendar():
    dates = pd.bdate_range("2024-01-01", periods=7)
    panel = pd.DataFrame(
        {"date": dates, "ticker": "A", "return": np.arange(7, dtype=float) / 100.0}
    )
    signals = pd.DataFrame({"date": [dates[0], dates[2], dates[3]], "ticker": "A", "score": 1.0})
    cfg = BacktestConfig(
        strategy="long_only",
        max_position_weight=1.0,
        cost_bps=0.0,
        slippage_bps=0.0,
    )
    result = run_backtest(signals, panel, cfg, benchmark="A")
    assert dates[3] in result.returns.index
    assert np.isclose(result.weights.loc[dates[3], "A"], 1.0)


def test_missing_return_for_active_position_fails_closed():
    dates = pd.bdate_range("2024-01-01", periods=5)
    panel = pd.DataFrame({"date": dates, "ticker": "A", "return": [0.0, 0.01, np.nan, 0.01, 0.01]})
    signals = pd.DataFrame({"date": dates[:3], "ticker": "A", "score": 1.0})
    cfg = BacktestConfig(strategy="long_only", max_position_weight=1.0)
    with pytest.raises(ValueError, match="missing asset returns"):
        run_backtest(signals, panel, cfg, benchmark="A")


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
