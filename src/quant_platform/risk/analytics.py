"""Portfolio-level risk analytics: correlations, exposures, stress tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.logging_utils import get_logger
from quant_platform.risk.metrics import TRADING_DAYS, value_at_risk

logger = get_logger(__name__)


def correlation_matrix(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation of asset returns (wide: date x ticker)."""
    return returns_wide.corr()


def rolling_sharpe(
    returns: pd.Series, window: int = 126, periods_per_year: int = TRADING_DAYS
) -> pd.Series:
    """Rolling annualised Sharpe ratio of a strategy/return series."""
    r = pd.Series(returns).astype(float)
    mean = r.rolling(window, min_periods=max(2, window // 2)).mean()
    std = r.rolling(window, min_periods=max(2, window // 2)).std()
    return (mean / std.replace(0.0, np.nan)) * np.sqrt(periods_per_year)


def exposure_summary(weights: pd.DataFrame) -> pd.DataFrame:
    """Summarise portfolio exposures over time from a weights matrix.

    ``weights`` is a wide (date x ticker) frame of portfolio weights (longs
    positive, shorts negative). Returns per-date gross/net/long/short exposure
    and the number of active long/short positions.
    """
    longs = weights.clip(lower=0.0)
    shorts = weights.clip(upper=0.0)
    out = pd.DataFrame(index=weights.index)
    out["gross_exposure"] = weights.abs().sum(axis=1)
    out["net_exposure"] = weights.sum(axis=1)
    out["long_exposure"] = longs.sum(axis=1)
    out["short_exposure"] = shorts.sum(axis=1)
    out["n_long"] = (weights > 0).sum(axis=1)
    out["n_short"] = (weights < 0).sum(axis=1)
    return out


def stress_test(
    returns: pd.Series,
    scenarios: dict[str, float],
    *,
    beta_to_market: float = 1.0,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Apply simple scenario shocks to a strategy return series.

    Two scenario families are supported (inferred from the scenario name):

    * **equity shocks** (name contains ``"equity"``): a one-off market move of
      ``shock``; estimated P&L = ``beta_to_market * shock``.
    * **vol shocks** (name contains ``"vol"``): historical VaR re-scaled by the
      ``shock`` multiplier (e.g. a 2x volatility spike).

    Returns a tidy frame of scenario → estimated impact.
    """
    base_var = value_at_risk(returns, confidence, method="historical")
    rows = []
    for name, shock in scenarios.items():
        lname = name.lower()
        if "equity" in lname or "market" in lname or "drawdown" in lname:
            impact = beta_to_market * shock
            kind = "equity_shock"
        elif "vol" in lname:
            impact = -base_var * shock
            kind = "vol_shock"
        else:
            impact = shock
            kind = "custom"
        rows.append({"scenario": name, "type": kind, "shock": shock, "estimated_pnl": impact})
    df = pd.DataFrame(rows)
    logger.info("Computed %d stress scenarios", len(df))
    return df
