"""Vectorized, cross-sectional backtesting engine.

Conventions (chosen to avoid lookahead bias)
---------------------------------------------
* A signal observed at the **close of day t** sets the target weights ``W[t]``.
* Those weights are held over **day t+1** and earn ``R[t+1]`` (next-day return).
* Trading from ``W[t-1]`` to ``W[t]`` incurs turnover-based transaction costs +
  slippage, charged against the day the new position becomes active.

Therefore the realised strategy return on day ``t+1`` is::

    net[t+1] = sum_i W[t,i] * R[t+1,i]  -  turnover(W[t], W[t-1]) * fee

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

    if cfg.position_sizing == "rank":
        ranks = s.rank(pct=True)
        if cfg.strategy == "long_only":
            raw = (ranks - (1.0 - cfg.top_quantile)).clip(lower=0.0)
            if raw.sum() > 0:
                weights.loc[raw.index] = lev * raw / raw.sum()
        else:  # long_short
            raw = ranks - 0.5  # demeaned tilt
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
        shorts = [t for t in shorts if t not in set(longs)]
        half = lev / 2.0
        weights.loc[longs] = half / len(longs)
        if shorts:
            weights.loc[shorts] = -half / len(shorts)
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
    weights = _apply_position_cap(weights, cfg.max_position_weight)
    return weights


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
    return lev.fillna(1.0)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _pivot(panel: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return panel.pivot_table(index=DATE_COL, columns=TICKER_COL, values=value_col)


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

    # Wide matrices.
    scores_wide = signals.pivot_table(index=DATE_COL, columns=TICKER_COL, values=score_col)
    returns_wide = _pivot(panel, "return")

    # Restrict to the signal window and align.
    scores_wide = scores_wide.sort_index()
    returns_wide = returns_wide.reindex(
        index=scores_wide.index.union(returns_wide.index)
    ).sort_index()
    # Use the benchmark column from the full universe if present.
    common_cols = [c for c in scores_wide.columns if c in returns_wide.columns]
    scores_wide = scores_wide[common_cols]

    # Target weights from scores.
    weights = _build_weights(scores_wide, config)

    # Align returns to weight dates (forward return earned next day).
    aligned_returns = returns_wide.reindex(columns=weights.columns)
    # Realised strategy return at date t uses weights from t-1.
    w_lag = weights.shift(1)
    gross = (w_lag * aligned_returns.reindex(index=weights.index)).sum(axis=1)

    # Optional volatility-target leverage overlay (causal).
    if config.position_sizing == "vol_target":
        overlay = _vol_target_overlay(gross, config)
        weights = weights.mul(overlay, axis=0)
        w_lag = weights.shift(1)
        gross = (w_lag * aligned_returns.reindex(index=weights.index)).sum(axis=1)

    # Turnover & costs.
    turnover = (weights - weights.shift(1)).abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost = turnover.shift(1).fillna(0.0) * fee
    net = (gross - cost).fillna(0.0)

    # Trim leading all-NaN/zero warmup (first row has no prior weights).
    net = net.iloc[1:]
    gross = gross.iloc[1:]
    turnover = turnover.iloc[1:]
    weights = weights.iloc[1:]

    equity = config.initial_capital * (1.0 + net).cumprod()

    # Benchmark: buy & hold over the same window.
    if benchmark in returns_wide.columns:
        bench_ret = returns_wide[benchmark].reindex(net.index).fillna(0.0)
    else:
        logger.warning("Benchmark '%s' missing; using zero benchmark", benchmark)
        bench_ret = pd.Series(0.0, index=net.index)
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
