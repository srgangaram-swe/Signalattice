"""Research diagnostics connecting forecasts to executable decisions.

These utilities are deliberately conservative about timing and terminology:

* transaction-cost scenarios are expressed as *total one-way* basis points;
* an execution delay moves a signal forward in time before the backtest runs;
* the capacity estimate is labeled as a dollar-volume proxy, not an execution
  model; and
* the readiness gate reports independent criteria instead of obscuring failure
  modes behind a single score.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd

from quant_platform.backtest.engine import BacktestResult, run_backtest
from quant_platform.config import BacktestConfig, RiskConfig
from quant_platform.data.schema import DATE_COL, TICKER_COL


class ReadinessCriterion(TypedDict):
    """One auditable deployment-readiness criterion."""

    criterion: str
    metric: str
    status: str
    passed: bool
    observed: float
    threshold: float
    operator: str


class ReadinessGateResult(TypedDict):
    """Serializable output of :func:`readiness_gate`."""

    overall_pass: bool
    verdict: str
    criteria: list[ReadinessCriterion]
    passed_count: int
    criterion_count: int


def _scenario_row(result: BacktestResult) -> dict[str, float | int]:
    """Flatten the common economic outputs from a backtest result."""
    net_returns = result.returns.to_numpy(dtype=float)
    gross_returns = result.gross_returns.to_numpy(dtype=float)
    equity = result.equity_curve.to_numpy(dtype=float)
    return {
        "periods": len(result.returns),
        "net_total_return": float(np.prod(1.0 + net_returns) - 1.0),
        "gross_total_return": float(np.prod(1.0 + gross_returns) - 1.0),
        "terminal_equity": float(equity[-1]),
        "cagr": float(result.stats.get("cagr", np.nan)),
        "sharpe": float(result.stats.get("sharpe", np.nan)),
        "max_drawdown": float(result.stats.get("max_drawdown", np.nan)),
        "annual_turnover": float(result.stats.get("annual_turnover", np.nan)),
        "total_cost_drag": float(result.stats.get("total_cost_drag", np.nan)),
    }


def _validated_nonnegative_values(values: Sequence[float], *, name: str) -> list[float]:
    """Return finite, nonnegative scenario values in caller-provided order."""
    converted = [float(value) for value in values]
    if not converted:
        raise ValueError(f"{name} must contain at least one value")
    if any(not np.isfinite(value) or value < 0.0 for value in converted):
        raise ValueError(f"{name} values must be finite and nonnegative")
    return converted


def cost_sensitivity(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
    config: BacktestConfig,
    total_cost_bps: Sequence[float],
    *,
    benchmark: str,
    risk_config: RiskConfig | None = None,
    score_col: str = "score",
) -> pd.DataFrame:
    """Backtest a grid of total one-way implementation costs.

    Each scenario replaces ``config.cost_bps + config.slippage_bps`` with the
    supplied total.  The full amount is assigned to ``cost_bps`` and slippage
    is set to zero, preventing accidental double counting.  The underlying
    engine charges that rate against one-way traded notional (portfolio
    turnover).
    """
    scenarios = _validated_nonnegative_values(total_cost_bps, name="total_cost_bps")
    rows: list[dict[str, float | int]] = []
    for one_way_bps in scenarios:
        scenario_config = config.model_copy(update={"cost_bps": one_way_bps, "slippage_bps": 0.0})
        result = run_backtest(
            signals,
            panel,
            scenario_config,
            benchmark=benchmark,
            risk_config=risk_config,
            score_col=score_col,
        )
        row: dict[str, float | int] = {"total_one_way_cost_bps": one_way_bps}
        row.update(_scenario_row(result))
        rows.append(row)
    return pd.DataFrame(rows)


def execution_delay_sensitivity(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
    config: BacktestConfig,
    additional_delay_bars: int | Sequence[int],
    *,
    benchmark: str,
    risk_config: RiskConfig | None = None,
    score_col: str = "score",
) -> pd.DataFrame:
    """Measure performance under incremental delay beyond the configured lag.

    A delay of ``d`` assigns the score observed at close ``t`` to target-weight
    date ``t + d``.  :func:`run_backtest` then applies its conservative
    ``config.execution_lag``.  An integer requests the baseline plus every
    additional delay from one through that value; a sequence requests those
    specific additional delays.  A zero-delay baseline is always included.

    All scenarios are evaluated on the common target-date window beginning
    after the maximum additional delay, so performance differences are not
    caused by unequal start dates.  No backward fill or future value is used.
    """
    if isinstance(additional_delay_bars, bool):
        raise TypeError("additional_delay_bars must be an integer or sequence")
    if isinstance(additional_delay_bars, (int, np.integer)):
        max_delay = int(additional_delay_bars)
        if max_delay < 1:
            raise ValueError("integer additional_delay_bars must be at least one")
        delays = list(range(max_delay + 1))
    else:
        delays = []
        for delay in additional_delay_bars:
            if isinstance(delay, bool) or not isinstance(delay, (int, np.integer)):
                raise TypeError("additional delay values must be integers")
            if int(delay) < 0:
                raise ValueError("additional delay values must be nonnegative")
            delays.append(int(delay))
        if not delays:
            raise ValueError("additional_delay_bars must not be empty")
        delays = sorted({0, *delays})
        max_delay = max(delays)
    required = {DATE_COL, TICKER_COL, score_col}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"signals missing required columns: {sorted(missing)}")

    scores_wide = signals.pivot_table(
        index=DATE_COL,
        columns=TICKER_COL,
        values=score_col,
        aggfunc="last",
    ).sort_index()
    if len(scores_wide.index) <= max_delay + config.execution_lag:
        raise ValueError("signals do not contain enough bars for the requested delays")
    common_dates = scores_wide.index[max_delay:]

    rows: list[dict[str, float | int]] = []
    for delay in delays:
        delayed_wide = scores_wide.shift(delay).loc[common_dates]
        delayed_series = cast(
            pd.Series,
            delayed_wide.stack(future_stack=True).dropna(),
        )
        delayed = delayed_series.rename(score_col).reset_index()
        result = run_backtest(
            delayed,
            panel,
            config,
            benchmark=benchmark,
            risk_config=risk_config,
            score_col=score_col,
        )
        row: dict[str, float | int] = {
            "additional_delay_bars": delay,
            "configured_execution_lag_bars": config.execution_lag,
            "total_lag_bars": delay + config.execution_lag,
        }
        row.update(_scenario_row(result))
        rows.append(row)
    return pd.DataFrame(rows)


def _positional_series(values: pd.Series | Sequence[float], *, name: str) -> pd.Series:
    """Convert one-dimensional numeric input without applying label alignment."""
    if isinstance(values, pd.Series):
        series = values.reset_index(drop=True).astype(float)
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        series = pd.Series(array, dtype=float)
    if series.empty:
        raise ValueError(f"{name} must not be empty")
    return series


def break_even_cost_bps(
    gross_returns: pd.Series | Sequence[float],
    traded_notional: pd.Series | Sequence[float],
    *,
    portfolio_value: float | pd.Series | Sequence[float] = 1.0,
) -> float:
    """Return the one-way cost rate that consumes aggregate gross P&L.

    ``gross_returns`` are period portfolio returns and ``traded_notional`` is
    the one-way absolute dollar notional traded in the corresponding periods.
    A scalar or period-specific ``portfolio_value`` converts each return into
    gross dollars.  When turnover fractions are supplied instead of dollars,
    leave ``portfolio_value=1`` so numerator and denominator share normalized
    AUM units.

    The result solves the arithmetic implementation equation
    ``sum(gross P&L) - rate * sum(traded notional) = 0``.  A negative result
    correctly indicates a strategy that loses money before costs.
    """
    gross = _positional_series(gross_returns, name="gross_returns")
    traded = _positional_series(traded_notional, name="traded_notional")
    if len(gross) != len(traded):
        raise ValueError("gross_returns and traded_notional must have equal length")

    if isinstance(portfolio_value, (int, float, np.integer, np.floating)):
        capital = pd.Series(float(portfolio_value), index=gross.index, dtype=float)
    else:
        capital = _positional_series(portfolio_value, name="portfolio_value")
        if len(capital) != len(gross):
            raise ValueError("portfolio_value must be scalar or match gross_returns")

    frame = pd.concat(
        [
            gross.rename("gross_return"),
            traded.rename("traded_notional"),
            capital.rename("portfolio_value"),
        ],
        axis=1,
    ).dropna()
    if frame.empty:
        raise ValueError("inputs have no complete observations")
    numeric = frame.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("inputs must contain only finite values or missing observations")
    if (frame["traded_notional"] < 0.0).any():
        raise ValueError("traded_notional must be nonnegative one-way absolute notional")
    if (frame["portfolio_value"] <= 0.0).any():
        raise ValueError("portfolio_value must be positive")

    total_traded = float(frame["traded_notional"].sum())
    if total_traded <= 0.0:
        raise ValueError("total traded_notional must be positive")
    gross_pnl = float((frame["gross_return"] * frame["portfolio_value"]).sum())
    return gross_pnl / total_traded * 10_000.0


def liquidity_capacity_table(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    aum_scenarios: Sequence[float],
    *,
    price_col: str | None = None,
    volume_col: str = "volume",
    liquidity_window: int = 20,
    participation_limit: float = 0.10,
) -> pd.DataFrame:
    """Estimate trade participation and capacity across AUM scenarios.

    The diagnostic uses absolute target-weight changes times AUM as required
    traded notional.  Liquidity is the trailing median of ``price * volume``
    through the target-weight date.  This is explicitly an ex-post
    *dollar-volume proxy*: it does not model spread, order-book depth, market
    impact, auction availability, or intraday volume profiles.

    The first weight row is treated as a trade from cash.  ``weights`` must be a
    date-indexed, ticker-column matrix such as ``BacktestResult.weights``.
    """
    aums = _validated_nonnegative_values(aum_scenarios, name="aum_scenarios")
    if any(aum <= 0.0 for aum in aums):
        raise ValueError("aum_scenarios values must be positive")
    if isinstance(liquidity_window, bool) or not isinstance(liquidity_window, int):
        raise TypeError("liquidity_window must be an integer")
    if liquidity_window < 1:
        raise ValueError("liquidity_window must be at least one")
    if not np.isfinite(participation_limit) or participation_limit <= 0.0:
        raise ValueError("participation_limit must be finite and positive")
    if weights.empty:
        raise ValueError("weights must not be empty")
    if not weights.index.is_unique or not weights.columns.is_unique:
        raise ValueError("weights index and columns must be unique")

    selected_price = price_col
    if selected_price is None:
        selected_price = "close" if "close" in panel.columns else "adj_close"
    required = {DATE_COL, TICKER_COL, selected_price, volume_col}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    market = panel[[DATE_COL, TICKER_COL, selected_price, volume_col]].copy()
    market["_dollar_volume"] = pd.to_numeric(
        market[selected_price], errors="coerce"
    ) * pd.to_numeric(market[volume_col], errors="coerce")
    dollar_volume = market.pivot_table(
        index=DATE_COL,
        columns=TICKER_COL,
        values="_dollar_volume",
        aggfunc="last",
    ).sort_index()
    dollar_volume = dollar_volume.where(dollar_volume > 0.0)
    liquidity = dollar_volume.rolling(liquidity_window, min_periods=1).median()
    liquidity = liquidity.reindex(index=weights.index, columns=weights.columns)

    numeric_weights = weights.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    weight_changes = numeric_weights.diff()
    weight_changes.iloc[0] = numeric_weights.iloc[0]
    absolute_changes = weight_changes.abs()
    trade_mask = absolute_changes > 0.0
    valid_mask = trade_mask & liquidity.notna()
    trade_count = int(trade_mask.to_numpy().sum())
    observation_count = int(valid_mask.to_numpy().sum())
    coverage = observation_count / trade_count if trade_count else 0.0

    participation_per_dollar = (absolute_changes / liquidity).where(valid_mask)
    finite_coefficients = participation_per_dollar.stack().dropna().to_numpy(dtype=float)
    max_coefficient = float(np.max(finite_coefficients)) if finite_coefficients.size else np.nan
    capacity = (
        participation_limit / max_coefficient
        if np.isfinite(max_coefficient) and max_coefficient > 0.0
        else np.nan
    )
    total_weight_turnover = float(absolute_changes.to_numpy(dtype=float).sum())
    proxy_label = f"trailing_{liquidity_window}_bar_median_dollar_volume_proxy"

    rows: list[dict[str, float | int | str]] = []
    for aum in aums:
        traded_notional = absolute_changes * aum
        participation = participation_per_dollar * aum
        observed = participation.stack().dropna().to_numpy(dtype=float)
        if observed.size:
            median_participation = float(np.median(observed))
            p95_participation = float(np.percentile(observed, 95))
            max_participation = float(np.max(observed))
            share_above = float(np.mean(observed > participation_limit))
        else:
            median_participation = np.nan
            p95_participation = np.nan
            max_participation = np.nan
            share_above = np.nan
        rows.append(
            {
                "aum": aum,
                "proxy_label": proxy_label,
                "price_column": selected_price,
                "liquidity_window_bars": liquidity_window,
                "participation_limit": participation_limit,
                "trade_count": trade_count,
                "liquidity_observation_count": observation_count,
                "liquidity_coverage": coverage,
                "total_weight_turnover": total_weight_turnover,
                "total_traded_notional": float(traded_notional.to_numpy().sum()),
                "median_participation_rate": median_participation,
                "p95_participation_rate": p95_participation,
                "max_participation_rate": max_participation,
                "share_trades_above_limit": share_above,
                "capacity_at_participation_limit": float(capacity),
            }
        )
    return pd.DataFrame(rows)


def _deterministic_batch(inputs: Any, batch_size: int) -> Any:
    """Take or repeat rows deterministically while preserving pandas metadata."""
    try:
        input_length = len(inputs)
    except TypeError as exc:  # pragma: no cover - defensive API validation
        raise TypeError("inputs must be a sized, row-indexable object") from exc
    if input_length < 1:
        raise ValueError("inputs must contain at least one row")
    positions = np.arange(batch_size) % input_length
    if isinstance(inputs, (pd.DataFrame, pd.Series)):
        return inputs.iloc[positions]
    array = np.asarray(inputs)
    if array.ndim == 0:
        raise ValueError("inputs must contain a row dimension")
    return array[positions]


def warm_inference_benchmark(
    predict_fn: Callable[[Any], Any],
    inputs: Any,
    batch_sizes: Sequence[int],
    *,
    warmup_runs: int = 5,
    measured_runs: int = 30,
    synchronize: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Benchmark warm synchronous inference with deterministic input batches.

    Rows are selected cyclically from ``inputs``, making every invocation use
    the same batch for a given size.  Warmup calls occur before timing.  Pass a
    backend synchronization callback for asynchronous accelerators (for
    example, a CUDA synchronize function) so recorded latency includes actual
    completion time.
    """
    if not callable(predict_fn):
        raise TypeError("predict_fn must be callable")
    if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int):
        raise TypeError("warmup_runs must be an integer")
    if isinstance(measured_runs, bool) or not isinstance(measured_runs, int):
        raise TypeError("measured_runs must be an integer")
    if warmup_runs < 1 or measured_runs < 1:
        raise ValueError("warmup_runs and measured_runs must be at least one")

    sizes: list[int] = []
    for size in batch_sizes:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise TypeError("batch_sizes values must be integers")
        if int(size) < 1:
            raise ValueError("batch_sizes values must be positive")
        sizes.append(int(size))
    if not sizes:
        raise ValueError("batch_sizes must contain at least one value")

    rows: list[dict[str, float | int | str]] = []
    for batch_size in sizes:
        batch = _deterministic_batch(inputs, batch_size)
        for _ in range(warmup_runs):
            predict_fn(batch)
            if synchronize is not None:
                synchronize()

        elapsed_ns: list[int] = []
        for _ in range(measured_runs):
            started = time.perf_counter_ns()
            predict_fn(batch)
            if synchronize is not None:
                synchronize()
            elapsed_ns.append(max(time.perf_counter_ns() - started, 1))

        latency_ms = np.asarray(elapsed_ns, dtype=float) / 1_000_000.0
        total_seconds = float(np.sum(elapsed_ns)) / 1_000_000_000.0
        rows.append(
            {
                "batch_size": batch_size,
                "warmup_runs": warmup_runs,
                "measured_runs": measured_runs,
                "timer": "perf_counter_ns",
                "mean_latency_ms": float(np.mean(latency_ms)),
                "p50_latency_ms": float(np.percentile(latency_ms, 50)),
                "p95_latency_ms": float(np.percentile(latency_ms, 95)),
                "p99_latency_ms": float(np.percentile(latency_ms, 99)),
                "throughput_rows_per_second": batch_size * measured_runs / total_seconds,
            }
        )
    return pd.DataFrame(rows)


def readiness_gate(
    *,
    expected_calibration_error: float,
    break_even_one_way_cost_bps: float,
    positive_fold_fraction: float,
    p95_latency_ms: float,
    roc_auc: float | None = None,
    net_sharpe: float | None = None,
    p95_participation_rate: float | None = None,
    max_calibration_error: float = 0.05,
    min_roc_auc: float = 0.52,
    min_net_sharpe: float = 0.50,
    min_break_even_cost_bps: float = 10.0,
    min_positive_fold_fraction: float = 0.60,
    max_p95_latency_ms: float = 50.0,
    max_p95_participation_rate: float = 0.10,
) -> ReadinessGateResult:
    """Apply explicit gates for research-to-decision readiness.

    Missing or non-finite observed metrics fail their criterion.  Thresholds
    remain configurable because acceptable calibration, cost headroom,
    cross-validation stability, and latency are mandate- and venue-specific.
    The result is JSON-friendly and contains no opaque weighted score.
    """
    thresholds = {
        "max_calibration_error": float(max_calibration_error),
        "min_break_even_cost_bps": float(min_break_even_cost_bps),
        "min_positive_fold_fraction": float(min_positive_fold_fraction),
        "max_p95_latency_ms": float(max_p95_latency_ms),
        "min_roc_auc": float(min_roc_auc),
        "min_net_sharpe": float(min_net_sharpe),
        "max_p95_participation_rate": float(max_p95_participation_rate),
    }
    if any(not np.isfinite(value) for value in thresholds.values()):
        raise ValueError("readiness thresholds must be finite")
    if thresholds["max_calibration_error"] < 0.0:
        raise ValueError("max_calibration_error must be nonnegative")
    if thresholds["min_break_even_cost_bps"] < 0.0:
        raise ValueError("min_break_even_cost_bps must be nonnegative")
    if not 0.0 <= thresholds["min_positive_fold_fraction"] <= 1.0:
        raise ValueError("min_positive_fold_fraction must be between zero and one")
    if thresholds["max_p95_latency_ms"] <= 0.0:
        raise ValueError("max_p95_latency_ms must be positive")
    if thresholds["max_p95_participation_rate"] <= 0.0:
        raise ValueError("max_p95_participation_rate must be positive")

    specifications = [
        (
            "predictive_calibration",
            "expected_calibration_error",
            float(expected_calibration_error),
            thresholds["max_calibration_error"],
            "<=",
        ),
        (
            "economic_robustness",
            "break_even_one_way_cost_bps",
            float(break_even_one_way_cost_bps),
            thresholds["min_break_even_cost_bps"],
            ">=",
        ),
        (
            "stability",
            "positive_walk_forward_fold_fraction",
            float(positive_fold_fraction),
            thresholds["min_positive_fold_fraction"],
            ">=",
        ),
        (
            "operational_latency",
            "warm_p95_latency_ms",
            float(p95_latency_ms),
            thresholds["max_p95_latency_ms"],
            "<=",
        ),
    ]
    if roc_auc is not None:
        specifications.append(
            (
                "predictive_discrimination",
                "walk_forward_roc_auc",
                float(roc_auc),
                thresholds["min_roc_auc"],
                ">=",
            )
        )
    if net_sharpe is not None:
        specifications.append(
            (
                "net_economic_quality",
                "net_sharpe",
                float(net_sharpe),
                thresholds["min_net_sharpe"],
                ">=",
            )
        )
    if p95_participation_rate is not None:
        specifications.append(
            (
                "liquidity_capacity",
                "p95_dollar_volume_participation_rate",
                float(p95_participation_rate),
                thresholds["max_p95_participation_rate"],
                "<=",
            )
        )

    criteria: list[ReadinessCriterion] = []
    for criterion, metric, observed, threshold, operator in specifications:
        finite = bool(np.isfinite(observed))
        passed = finite and (observed <= threshold if operator == "<=" else observed >= threshold)
        criteria.append(
            {
                "criterion": criterion,
                "metric": metric,
                "status": "PASS" if passed else "FAIL",
                "passed": passed,
                "observed": observed,
                "threshold": threshold,
                "operator": operator,
            }
        )

    overall_pass = all(item["passed"] for item in criteria)
    passed_count = sum(item["passed"] for item in criteria)
    return {
        "overall_pass": overall_pass,
        "verdict": "READY" if overall_pass else "NOT_READY",
        "criteria": criteria,
        "passed_count": passed_count,
        "criterion_count": len(criteria),
    }
