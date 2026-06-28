"""Risk and performance analytics.

Public API:

- :func:`performance_summary` — headline performance + risk metrics for a
  return series (Sharpe, Sortino, CAGR, max drawdown, VaR/CVaR, beta, ...).
- individual metric functions (e.g. :func:`sharpe_ratio`, :func:`value_at_risk`).
- :func:`correlation_matrix`, :func:`stress_test` — portfolio-level analytics.
"""

from __future__ import annotations

from quant_platform.risk.analytics import (
    correlation_matrix,
    exposure_summary,
    rolling_sharpe,
    stress_test,
)
from quant_platform.risk.metrics import (
    annualized_return,
    annualized_volatility,
    beta,
    cagr,
    calmar_ratio,
    conditional_value_at_risk,
    drawdown_series,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)

__all__ = [
    "performance_summary",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "cagr",
    "max_drawdown",
    "drawdown_series",
    "value_at_risk",
    "conditional_value_at_risk",
    "beta",
    "correlation_matrix",
    "rolling_sharpe",
    "exposure_summary",
    "stress_test",
]
