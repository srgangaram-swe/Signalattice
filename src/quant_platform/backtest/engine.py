"""Vectorized, cross-sectional backtesting engine.

Conventions (chosen to avoid lookahead bias)
---------------------------------------------
* A signal observed after the **close of day t** sets target weights ``W[t]``.
* With the default two-row execution lag, the strategy waits through the next
  close and first earns the close-to-close return ending on ``t+2``. This is a
  conservative daily-bar convention that never assumes a fill at an observed
  close.
* Trading incurs turnover-based transaction costs and slippage in the first
  return interval for which the new weights are active.

Therefore the realised strategy return on day ``t+1`` is::

    net[t+L] = sum_i W[t,i] * R[t+L,i] - turnover(W[t], W[t-1]) * fee

Implemented with simple pandas ``shift`` operations so the whole backtest is
vectorised (no per-day Python loop over the P&L).

Supported strategies: ``long_only`` and ``long_short`` (dollar-neutral).
Supported sizing: ``equal_weight``, ``rank`` and ``vol_target`` (a trailing-vol
leverage overlay applied on top of the base weights, using only past data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.config import BacktestConfig, RiskConfig
from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.logging_utils import get_logger
from quant_platform.risk.analytics import exposure_summary
from quant_platform.risk.metrics import (
    TRADING_DAYS,
    drawdown_series,
    performance_summary,
)

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Structured backtest output."""

    returns: pd.Series  # net strategy returns (realised dates)
    gross_returns: pd.Series
    equity_curve: pd.Series
    benchmark_returns: pd.Series
    benchmark_equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    drawdown: pd.Series
    exposures: pd.DataFrame
    stats: dict[str, float]
    benchmark_stats: dict[str, float]
    monthly_returns: pd.DataFrame
    trade_summary: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------


def _row_weights(scores: pd.Series, cfg: BacktestConfig) -> pd.Series:
    """Construct target weights for a single date from a cross-section of scores."""
    s = scores.dropna()
    weights = pd.Series(0.0, index=scores.index)
    n = len(s)
    if n == 0:
        return weights

    lev = cfg.max_leverage
    k = max(1, int(round(n * cfg.top_quantile)))

    if cfg.signal_selection == "probability_threshold":
        longs = s[s >= cfg.long_threshold].nlargest(k).index
        if cfg.strategy == "long_only":
            if len(longs):
                weights.loc[longs] = lev / len(longs)
            return weights
        shorts = s[s <= cfg.short_threshold].nsmallest(k).index
        # Remain flat when calibration confidence does not support both sides;
        # otherwise a missing side silently turns a long/short policy directional.
        if not len(longs) or not len(shorts):
            return weights
        weights.loc[longs] = (lev / 2.0) / len(longs)
        weights.loc[shorts] = -(lev / 2.0) / len(shorts)
        return weights

    if cfg.position_sizing == "rank":
        ranks = s.rank(pct=True)
        if cfg.strategy == "long_only":
            raw = (ranks - (1.0 - cfg.top_quantile)).clip(lower=0.0)
            if raw.sum() > 0:
                weights.loc[raw.index] = lev * raw / raw.sum()
        else:  # long_short
            raw = ranks - ranks.mean()
            denom = raw.abs().sum()
            if denom > 0:
                weights.loc[raw.index] = lev * raw / denom
        return weights

    # equal_weight (and base for vol_target)
    ordered = s.sort_values(ascending=False)
    longs = ordered.index[:k]
    if cfg.strategy == "long_only":
        weights.loc[longs] = lev / len(longs)
    else:  # long_short, dollar-neutral
        shorts = ordered.index[-k:]
        # Guard against overlap when the universe is tiny.
        short_names = [t for t in shorts if t not in set(longs)]
        if not short_names:
            return weights
        half = lev / 2.0
        weights.loc[longs] = half / len(longs)
        weights.loc[short_names] = -half / len(short_names)
    return weights


def _apply_position_cap(weights: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    """Clip individual position weights to ``±max_weight``.

    We deliberately *do not* renormalise gross exposure back up afterwards:
    redistributing the clipped excess proportionally can push other names back
    over the cap, so we accept a (usually tiny) reduction in gross exposure when
    a cap binds. This guarantees ``|w_i| <= max_weight`` for every position.
    """
    if max_weight <= 0:
        return weights
    return weights.clip(lower=-max_weight, upper=max_weight)


def _build_weights(scores_wide: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Build the full (date x ticker) target-weight matrix from scores."""
    weights = scores_wide.apply(lambda row: _row_weights(row, cfg), axis=1)
    weights = weights.reindex(columns=scores_wide.columns).fillna(0.0)
    return _enforce_portfolio_limits(weights, cfg)


def _enforce_portfolio_limits(weights: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Enforce both per-name and portfolio gross limits without scaling up."""
    limited = _apply_position_cap(weights, cfg.max_position_weight)
    if cfg.strategy == "long_short":
        long_gross = limited.clip(lower=0.0).sum(axis=1)
        short_gross = -limited.clip(upper=0.0).sum(axis=1)
        balanced_gross = pd.concat([long_gross, short_gross], axis=1).min(axis=1)
        long_scale = (balanced_gross / long_gross.replace(0.0, np.nan)).fillna(0.0)
        short_scale = (balanced_gross / short_gross.replace(0.0, np.nan)).fillna(0.0)
        limited = limited.clip(lower=0.0).mul(long_scale, axis=0) + limited.clip(upper=0.0).mul(
            short_scale, axis=0
        )
    gross = limited.abs().sum(axis=1)
    scale = (cfg.max_leverage / gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    return limited.mul(scale, axis=0)


def _apply_rebalance_threshold(weights: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Apply a causal per-position no-trade band to desired target weights."""
    if threshold <= 0 or weights.empty:
        return weights
    rows: list[pd.Series] = []
    previous = pd.Series(0.0, index=weights.columns)
    for _, desired in weights.iterrows():
        update = (desired - previous).abs() >= threshold
        current = previous.where(~update, desired)
        rows.append(current)
        previous = current
    return pd.DataFrame(rows, index=weights.index, columns=weights.columns)


def _vol_target_overlay(
    base_returns: pd.Series,
    cfg: BacktestConfig,
    *,
    window: int = 63,
    max_leverage_cap: float = 3.0,
) -> pd.Series:
    """Trailing-volatility leverage multiplier targeting an annualised vol.

    Uses realised vol of the *base* (unlevered) strategy returns up to ``t-1``
    (shifted) so the overlay is causal / leakage-free.
    """
    realized = base_returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(
        TRADING_DAYS
    )
    realized = realized.shift(1)  # only use information available before the day
    lev = (cfg.vol_target_annual / realized).clip(upper=max_leverage_cap)
    return pd.Series(lev.fillna(1.0), index=base_returns.index, dtype=float)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _pivot(panel: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if panel.duplicated([DATE_COL, TICKER_COL]).any():
        raise ValueError("price panel must contain at most one row per (date, ticker)")
    # ``pivot`` retains dates whose values are all missing; ``pivot_table``
    # silently drops them and can therefore hide an unavailable held return.
    return panel.pivot(index=DATE_COL, columns=TICKER_COL, values=value_col)


def run_backtest(
    signals: pd.DataFrame,
    panel: pd.DataFrame,
    config: BacktestConfig,
    *,
    benchmark: str,
    risk_config: RiskConfig | None = None,
    score_col: str = "score",
) -> BacktestResult:
    """Run a vectorized cross-sectional backtest.

    Parameters
    ----------
    signals:
        Long-format frame with ``date``, ``ticker`` and a score column. Higher
        score => more bullish. Typically the out-of-sample model predictions or
        a baseline signal.
    panel:
        Price panel with a ``return`` column (output of ingestion).
    config:
        Backtest configuration (strategy, costs, sizing...).
    benchmark:
        Ticker used as the buy-and-hold benchmark.
    risk_config:
        Used for annualisation / VaR confidence in the stats block.
    """
    risk_config = risk_config or RiskConfig()
    fee = (config.cost_bps + config.slippage_bps) / 1e4

    required_signal_cols = {DATE_COL, TICKER_COL, score_col}
    missing_signal_cols = required_signal_cols.difference(signals.columns)
    if missing_signal_cols:
        raise ValueError(f"signals missing required columns: {sorted(missing_signal_cols)}")
    if signals.duplicated([DATE_COL, TICKER_COL]).any():
        raise ValueError("signals must contain at most one score per (date, ticker)")

    # Wide matrices.
    scores_wide = signals.pivot(index=DATE_COL, columns=TICKER_COL, values=score_col)
    returns_wide = _pivot(panel, "return")
    scores_wide.index = pd.to_datetime(scores_wide.index)
    returns_wide.index = pd.to_datetime(returns_wide.index)

    # Restrict to the signal window and align. The calendar extends by the
    # execution lag so the final observed decisions can realize a return.
    scores_wide = scores_wide.sort_index()
    returns_wide = returns_wide.sort_index()
    # Use the benchmark column from the full universe if present.
    common_cols = [c for c in scores_wide.columns if c in returns_wide.columns]
    if not common_cols:
        raise ValueError("signals and price panel have no tickers in common")
    scores_wide = scores_wide[common_cols]

    return_dates = returns_wide.index
    signal_start = int(return_dates.searchsorted(scores_wide.index.min(), side="left"))
    signal_end = int(return_dates.searchsorted(scores_wide.index.max(), side="right"))
    calendar_end = min(len(return_dates), signal_end + config.execution_lag)
    calendar = return_dates[signal_start:calendar_end]
    if len(calendar) <= config.execution_lag:
        raise ValueError("not enough return dates after the signal window for execution")

    # Build desired weights only when a signal exists. Missing signal dates hold
    # the previous target rather than disappearing from the P&L calendar.
    desired = _build_weights(scores_wide, config).reindex(calendar).ffill().fillna(0.0)
    desired = _apply_rebalance_threshold(desired, config.rebalance_threshold)
    desired = _enforce_portfolio_limits(desired, config)

    # Close-derived decisions become held weights only after the declared lag.
    weights = desired.shift(config.execution_lag)
    active = weights.notna().any(axis=1)
    weights = weights.loc[active].fillna(0.0)
    aligned_returns = returns_wide.reindex(index=weights.index, columns=weights.columns)
    unavailable = aligned_returns.isna() & weights.abs().gt(1e-12)
    if unavailable.any().any():
        examples = unavailable.stack()[lambda values: values].index.tolist()[:5]
        raise ValueError(f"missing asset returns for active positions; examples={examples}")
    gross = (weights * aligned_returns).sum(axis=1, min_count=1)

    # Optional volatility-target leverage overlay (causal).
    if config.position_sizing == "vol_target":
        overlay = _vol_target_overlay(gross, config)
        weights = weights.mul(overlay, axis=0)
        weights = _enforce_portfolio_limits(weights, config)
        gross = (weights * aligned_returns).sum(axis=1, min_count=1)

    # Turnover & costs.
    previous = weights.shift(1).fillna(0.0)
    turnover = (weights - previous).abs().sum(axis=1)
    cost = turnover * fee
    net = gross - cost

    equity = config.initial_capital * (1.0 + net).cumprod()

    # Benchmark: buy & hold over the same window.
    if benchmark in returns_wide.columns:
        bench_ret = returns_wide[benchmark].reindex(net.index)
        if bench_ret.isna().any():
            raise ValueError(f"benchmark '{benchmark}' has missing returns in evaluation window")
    else:
        raise ValueError(f"benchmark '{benchmark}' missing from price panel")
    bench_equity = config.initial_capital * (1.0 + bench_ret).cumprod()

    # Stats.
    stats = performance_summary(
        net,
        benchmark_returns=bench_ret,
        confidence=risk_config.var_confidence,
        periods_per_year=risk_config.trading_days,
    )
    stats["avg_turnover"] = float(turnover.mean())
    stats["annual_turnover"] = float(turnover.mean() * risk_config.trading_days)
    stats["total_cost_drag"] = float(cost.sum())
    bench_stats = performance_summary(
        bench_ret,
        confidence=risk_config.var_confidence,
        periods_per_year=risk_config.trading_days,
    )

    result = BacktestResult(
        returns=net,
        gross_returns=gross,
        equity_curve=equity,
        benchmark_returns=bench_ret,
        benchmark_equity=bench_equity,
        weights=weights,
        turnover=turnover,
        costs=cost,
        drawdown=drawdown_series(net),
        exposures=exposure_summary(weights),
        stats=stats,
        benchmark_stats=bench_stats,
        monthly_returns=monthly_return_table(net),
        trade_summary=_trade_summary(weights, turnover, net),
        config=config.model_dump(),
    )
    logger.info(
        "Backtest done: Sharpe=%.2f CAGR=%.2f%% MaxDD=%.2f%% (vs bench Sharpe=%.2f)",
        stats["sharpe"],
        stats["cagr"] * 100.0,
        stats["max_drawdown"] * 100.0,
        bench_stats["sharpe"],
    )
    return result


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Pivot daily returns into a Year x Month table of monthly compounded returns."""
    r = pd.Series(returns).dropna()
    if r.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(r.index)
    monthly = (1.0 + r).groupby([idx.year, idx.month]).prod() - 1.0
    monthly.index = monthly.index.set_names(["year", "month"])
    table = monthly.unstack("month")
    # Append a full-year compounded column.
    table["YEAR"] = (1.0 + table.fillna(0.0)).prod(axis=1) - 1.0
    month_names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    table = table.rename(columns=month_names)
    return table


def _trade_summary(
    weights: pd.DataFrame, turnover: pd.Series, returns: pd.Series
) -> dict[str, Any]:
    """Summary statistics describing trading activity and P&L distribution."""
    active = weights.abs() > 1e-8
    return {
        "n_days": int(len(returns)),
        "avg_positions": float(active.sum(axis=1).mean()),
        "avg_long_positions": float((weights > 1e-8).sum(axis=1).mean()),
        "avg_short_positions": float((weights < -1e-8).sum(axis=1).mean()),
        "avg_gross_exposure": float(weights.abs().sum(axis=1).mean()),
        "avg_net_exposure": float(weights.sum(axis=1).mean()),
        "avg_daily_turnover": float(turnover.mean()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "pct_positive_days": float((returns > 0).mean()),
    }
