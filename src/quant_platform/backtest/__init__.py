"""Vectorized backtesting engine.

Public API:

- :func:`run_backtest` — turn a panel of signals + prices into a fully-costed
  strategy P&L, weights, exposures and performance statistics.
- :class:`BacktestResult` — structured results container.
"""

from __future__ import annotations

from quant_platform.backtest.engine import BacktestResult, run_backtest

__all__ = ["run_backtest", "BacktestResult"]
