"""Tests for decision-readiness and implementation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.config import BacktestConfig
from quant_platform.evaluation.decision import (
    break_even_cost_bps,
    cost_sensitivity,
    execution_delay_sensitivity,
    liquidity_capacity_table,
    readiness_gate,
    warm_inference_benchmark,
)


def _long_signals(dates: pd.DatetimeIndex, tickers: list[str], scores: list[list[float]]):
    wide = pd.DataFrame(scores, index=dates, columns=tickers)
    wide.index.name = "date"
    return wide.stack().rename("score").reset_index().rename(columns={"level_1": "ticker"})


def _return_panel(
    dates: pd.DatetimeIndex,
    tickers: list[str],
    returns: list[list[float]],
) -> pd.DataFrame:
    wide = pd.DataFrame(returns, index=dates, columns=tickers)
    wide.index.name = "date"
    return wide.stack().rename("return").reset_index().rename(columns={"level_1": "ticker"})


def test_cost_sensitivity_uses_total_one_way_bps():
    dates = pd.bdate_range("2025-01-01", periods=4)
    signals = _long_signals(dates, ["A"], [[1.0]] * 4)
    panel = _return_panel(dates, ["A"], [[0.0]] * 4)
    config = BacktestConfig(
        strategy="long_only",
        top_quantile=0.5,
        max_position_weight=1.0,
        cost_bps=9.0,
        slippage_bps=9.0,
    )

    table = cost_sensitivity(
        signals,
        panel,
        config,
        [0.0, 100.0],
        benchmark="A",
    )

    assert table["total_one_way_cost_bps"].tolist() == [0.0, 100.0]
    assert np.isclose(table.loc[0, "net_total_return"], 0.0)
    # Initial trade is 1.0 times AUM; 100 bps therefore removes exactly 1%.
    assert np.isclose(table.loc[1, "net_total_return"], -0.01)
    assert np.isclose(table.loc[1, "total_cost_drag"], 0.01)


def test_execution_delay_moves_signal_forward_without_lookahead():
    dates = pd.bdate_range("2025-01-01", periods=6)
    signals = _long_signals(
        dates,
        ["A", "B"],
        [
            [2.0, 1.0],  # t0: A
            [1.0, 2.0],  # t1: B
            [2.0, 1.0],  # t2: A
            [1.0, 2.0],  # t3: B
            [2.0, 1.0],
            [1.0, 2.0],
        ],
    )
    panel = _return_panel(
        dates,
        ["A", "B"],
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-0.2, 0.2],
            [0.3, -0.3],
            [-0.4, 0.4],
        ],
    )
    config = BacktestConfig(
        strategy="long_only",
        top_quantile=0.5,
        max_position_weight=1.0,
        cost_bps=0.0,
        slippage_bps=0.0,
    )

    table = execution_delay_sensitivity(
        signals,
        panel,
        config,
        2,
        benchmark="A",
    )

    assert table["additional_delay_bars"].tolist() == [0, 1, 2]
    assert table["configured_execution_lag_bars"].tolist() == [2, 2, 2]
    assert table["total_lag_bars"].tolist() == [2, 3, 4]
    # With one extra bar, t1/t2 signals become held for t4/t5: -30%, -40%.
    assert np.isclose(table.loc[1, "gross_total_return"], 0.7 * 0.6 - 1.0)
    # With two extra bars, t0/t1 signals become held for t4/t5: +30%, +40%.
    assert np.isclose(table.loc[2, "gross_total_return"], 1.3 * 1.4 - 1.0)


def test_break_even_cost_uses_gross_pnl_and_one_way_notional():
    observed = break_even_cost_bps(
        gross_returns=[0.01, 0.02],
        traded_notional=[50.0, 100.0],
        portfolio_value=100.0,
    )
    # Gross P&L is $3 on $150 traded: 2%, or 200 bps one way.
    assert np.isclose(observed, 200.0)


def test_liquidity_capacity_proxy_hand_calculation():
    dates = pd.bdate_range("2025-01-01", periods=2)
    panel = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "ticker": ["A", "B", "A", "B"],
            "adj_close": [10.0, 20.0, 10.0, 20.0],
            "volume": [100.0, 100.0, 100.0, 100.0],
        }
    )
    weights = pd.DataFrame(
        [[0.5, -0.5], [0.75, -0.25]],
        index=dates,
        columns=["A", "B"],
    )

    table = liquidity_capacity_table(
        panel,
        weights,
        [100.0, 200.0],
        liquidity_window=1,
        participation_limit=0.10,
    )

    # At $100 AUM participation is .05, .025, .025, .0125 across the four trades.
    assert np.isclose(table.loc[0, "median_participation_rate"], 0.025)
    assert np.isclose(table.loc[0, "max_participation_rate"], 0.05)
    assert np.isclose(table.loc[0, "total_traded_notional"], 150.0)
    assert np.isclose(table.loc[1, "max_participation_rate"], 0.10)
    assert np.isclose(table.loc[0, "capacity_at_participation_limit"], 200.0)
    assert table.loc[0, "proxy_label"].endswith("dollar_volume_proxy")


def test_liquidity_capacity_ignores_non_trade_cells_under_pandas_stack_semantics():
    dates = pd.bdate_range("2025-01-01", periods=2)
    panel = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "ticker": ["A", "B", "A", "B"],
            "close": [10.0, 20.0, 10.0, 20.0],
            "volume": [100.0, 100.0, 100.0, 100.0],
        }
    )
    # Only the first row is a trade; unchanged cells are intentionally masked.
    weights = pd.DataFrame([[0.5, -0.5], [0.5, -0.5]], index=dates, columns=["A", "B"])

    table = liquidity_capacity_table(panel, weights, [100.0], liquidity_window=1)

    numeric = table[
        [
            "median_participation_rate",
            "p95_participation_rate",
            "max_participation_rate",
            "capacity_at_participation_limit",
        ]
    ].to_numpy(dtype=float)
    assert np.isfinite(numeric).all()


def test_warm_inference_benchmark_schema_and_calls():
    inputs = np.arange(12, dtype=float).reshape(6, 2)
    calls: list[int] = []

    def predict(batch: np.ndarray) -> np.ndarray:
        calls.append(len(batch))
        return batch.sum(axis=1)

    table = warm_inference_benchmark(
        predict,
        inputs,
        [1, 4],
        warmup_runs=1,
        measured_runs=3,
    )

    assert calls == [1] * 4 + [4] * 4
    assert table["batch_size"].tolist() == [1, 4]
    required = {
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "throughput_rows_per_second",
    }
    assert required.issubset(table.columns)
    assert np.isfinite(table[list(required)].to_numpy()).all()
    assert (table["throughput_rows_per_second"] > 0.0).all()


def test_readiness_gate_exposes_independent_failures():
    result = readiness_gate(
        expected_calibration_error=0.04,
        break_even_one_way_cost_bps=25.0,
        positive_fold_fraction=0.75,
        p95_latency_ms=75.0,
    )

    assert result["verdict"] == "NOT_READY"
    assert result["passed_count"] == 3
    assert result["criterion_count"] == 4
    latency = next(
        criterion
        for criterion in result["criteria"]
        if criterion["criterion"] == "operational_latency"
    )
    assert latency == {
        "criterion": "operational_latency",
        "metric": "warm_p95_latency_ms",
        "status": "FAIL",
        "passed": False,
        "observed": 75.0,
        "threshold": 50.0,
        "operator": "<=",
    }


def test_readiness_gate_checks_capacity_at_declared_aum():
    result = readiness_gate(
        expected_calibration_error=0.01,
        break_even_one_way_cost_bps=25.0,
        positive_fold_fraction=0.75,
        p95_latency_ms=5.0,
        p95_participation_rate=0.03,
        max_p95_participation_rate=0.01,
    )

    capacity = next(
        item for item in result["criteria"] if item["criterion"] == "liquidity_capacity"
    )
    assert capacity["status"] == "FAIL"
    assert capacity["observed"] == 0.03
    assert capacity["threshold"] == 0.01
    assert result["criterion_count"] == 5
