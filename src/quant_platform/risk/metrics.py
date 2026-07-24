"""Performance and risk metrics for return series.

All functions take a periodic (typically daily) simple-return series and use a
configurable ``periods_per_year`` for annualisation (252 trading days by
default). Conventions follow standard quant practice:

* Sharpe / Sortino assume an annual risk-free rate that is converted to the
  per-period rate before subtraction.
* Volatility is annualised by ``sqrt(periods_per_year)``.
* VaR/CVaR are reported as **positive loss magnitudes** at the given confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns).dropna().astype(float)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric annualised return (CAGR-equivalent from periodic returns)."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    values = r.to_numpy(dtype=float)
    growth = float(np.prod(1.0 + values))
    years = len(r) / periods_per_year
    if years <= 0 or growth <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def cagr(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate from an equity curve (level series)."""
    eq = _clean(equity_curve)
    values = eq.to_numpy(dtype=float)
    if len(values) < 2 or values[0] <= 0:
        return float("nan")
    total_growth = float(values[-1] / values[0])
    years = len(eq) / periods_per_year
    if years <= 0 or total_growth <= 0:
        return float("nan")
    return float(total_growth ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sharpe ratio. ``risk_free`` is an *annual* rate."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    rf_per = risk_free / periods_per_year
    excess = r - rf_per
    std = excess.std(ddof=1)
    # Guard against (near-)constant series: a vanishing denominator is not a
    # meaningful Sharpe ratio (daily vol below 1e-12 is numerical noise).
    if not np.isfinite(std) or std < 1e-12:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sortino ratio (downside-deviation-adjusted return)."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    rf_per = risk_free / periods_per_year
    excess = r - rf_per
    downside = excess[excess < 0]
    dd = np.sqrt((downside**2).mean()) if len(downside) else 0.0
    if not np.isfinite(dd) or dd < 1e-12:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown path (<= 0) from the cumulative-return equity curve."""
    r = _clean(returns)
    if r.empty:
        return r
    # Include initial capital (1.0) when establishing the running peak. Without
    # it an immediate loss is incorrectly reported as a zero drawdown because
    # the first depressed equity value becomes its own high-water mark.
    equity = (1.0 + r).cumprod().to_numpy(dtype=float)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    return pd.Series(equity / peaks - 1.0, index=r.index, name="drawdown")


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else float("nan")


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised return divided by the absolute max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return annualized_return(returns, periods_per_year) / abs(mdd)


def value_at_risk(
    returns: pd.Series, confidence: float = 0.95, method: str = "historical"
) -> float:
    """Value-at-Risk as a positive loss magnitude at ``confidence``.

    ``method`` is ``"historical"`` (empirical quantile) or ``"gaussian"``
    (parametric normal). A return of 0.02 means "with 95% confidence the
    one-period loss will not exceed 2%".
    """
    r = _clean(returns)
    if r.empty:
        return float("nan")
    alpha = 1.0 - confidence
    if method == "gaussian":
        from scipy.stats import norm

        z = norm.ppf(alpha)
        var = -(r.mean() + z * r.std(ddof=1))
    else:
        var = -np.quantile(r, alpha)
    return float(max(var, 0.0))


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR): mean loss beyond the VaR threshold."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    alpha = 1.0 - confidence
    threshold = np.quantile(r, alpha)
    tail = r[r <= threshold]
    if tail.empty:
        return float("nan")
    return float(max(-tail.mean(), 0.0))


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Beta of a return series to a benchmark (cov / var)."""
    df = pd.concat([_clean(returns), _clean(benchmark_returns)], axis=1, join="inner").dropna()
    if len(df) < 2:
        return float("nan")
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])
    var_b = cov[1, 1]
    if var_b == 0:
        return float("nan")
    return float(cov[0, 1] / var_b)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def performance_summary(
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    risk_free: float = 0.0,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """Compute a dictionary of headline performance & risk statistics."""
    r = _clean(returns)
    summary = {
        "cagr": annualized_return(r, periods_per_year),
        "ann_return": annualized_return(r, periods_per_year),
        "ann_volatility": annualized_volatility(r, periods_per_year),
        "sharpe": sharpe_ratio(r, risk_free, periods_per_year),
        "sortino": sortino_ratio(r, risk_free, periods_per_year),
        "calmar": calmar_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "var_95": value_at_risk(r, confidence, method="historical"),
        "cvar_95": conditional_value_at_risk(r, confidence),
        "hit_rate": hit_rate(r),
        "skew": (
            float(np.asarray(pd.Series(r.to_numpy(dtype=float)).skew()).item())
            if len(r) > 2
            else float("nan")
        ),
        "kurtosis": (
            float(np.asarray(pd.Series(r.to_numpy(dtype=float)).kurtosis()).item())
            if len(r) > 3
            else float("nan")
        ),
        "n_periods": float(len(r)),
    }
    if benchmark_returns is not None:
        summary["beta"] = beta(r, benchmark_returns)
        bench_ann = annualized_return(benchmark_returns, periods_per_year)
        summary["alpha_ann"] = summary["ann_return"] - summary["beta"] * bench_ann
    return summary
