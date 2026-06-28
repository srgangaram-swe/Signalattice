"""Vectorised technical-indicator primitives.

Every function operates on a single asset's time-ordered :class:`pandas.Series`
and returns a same-indexed Series. All windows are **trailing** (they use only
information available up to and including the current bar), which is the key to
avoiding lookahead bias: a feature at time *t* never peeks at *t+1*.

These primitives are intentionally small and individually unit-tested; the
:mod:`quant_platform.features.pipeline` module composes them per ticker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_returns(price: pd.Series, periods: int = 1) -> pd.Series:
    """Simple percentage return over ``periods`` bars."""
    return price.pct_change(periods=periods)


def log_returns(price: pd.Series, periods: int = 1) -> pd.Series:
    """Continuously-compounded (log) return over ``periods`` bars."""
    return np.log(price / price.shift(periods))


def rolling_volatility(
    returns: pd.Series, window: int = 21, *, annualize: bool = True
) -> pd.Series:
    """Trailing standard deviation of returns, optionally annualised."""
    vol = returns.rolling(window, min_periods=max(2, window // 2)).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def realized_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualised realized volatility = sqrt(sum of squared returns) scaled."""
    rv = returns.pow(2).rolling(window, min_periods=max(2, window // 2)).sum()
    return np.sqrt(rv * (TRADING_DAYS / window))


def rolling_sharpe(returns: pd.Series, window: int = 63, *, annualize: bool = True) -> pd.Series:
    """Trailing Sharpe ratio (risk-free assumed 0) over ``window`` bars."""
    mean = returns.rolling(window, min_periods=max(2, window // 2)).mean()
    std = returns.rolling(window, min_periods=max(2, window // 2)).std()
    sharpe = mean / std.replace(0.0, np.nan)
    if annualize:
        sharpe = sharpe * np.sqrt(TRADING_DAYS)
    return sharpe


def moving_average(price: pd.Series, window: int = 20) -> pd.Series:
    """Simple moving average."""
    return price.rolling(window, min_periods=max(2, window // 2)).mean()


def ema(price: pd.Series, span: int = 12) -> pd.Series:
    """Exponential moving average."""
    return price.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def ma_ratio(price: pd.Series, window: int = 50) -> pd.Series:
    """Price relative to its moving average (a normalised trend feature)."""
    ma = moving_average(price, window)
    return price / ma - 1.0


def rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing, scaled to [0, 100]."""
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing == EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 (only gains) RSI is 100.
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Returns a DataFrame with ``macd``, ``macd_signal`` and ``macd_hist`` columns.
    """
    ema_fast = ema(price, fast)
    ema_slow = ema(price, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def bollinger_bands(price: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands.

    Returns ``bb_upper``, ``bb_lower``, ``bb_pctb`` (position within the bands,
    0=lower 1=upper) and ``bb_bandwidth`` (band width / middle band).
    """
    mid = moving_average(price, window)
    std = price.rolling(window, min_periods=max(2, window // 2)).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = upper - lower
    pctb = (price - lower) / width.replace(0.0, np.nan)
    bandwidth = width / mid.replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pctb": pctb,
            "bb_bandwidth": bandwidth,
        }
    )


def momentum(price: pd.Series, window: int = 63, *, skip: int = 0) -> pd.Series:
    """Total return over ``window`` bars, optionally skipping the most recent
    ``skip`` bars (the classic 12-1 momentum convention uses skip=21)."""
    return price.shift(skip) / price.shift(window) - 1.0


def zscore(series: pd.Series, window: int = 21) -> pd.Series:
    """Rolling z-score — the canonical mean-reversion signal."""
    mean = series.rolling(window, min_periods=max(2, window // 2)).mean()
    std = series.rolling(window, min_periods=max(2, window // 2)).std()
    return (series - mean) / std.replace(0.0, np.nan)


def volume_features(volume: pd.Series, price: pd.Series, window: int = 21) -> pd.DataFrame:
    """Volume-based features: dollar volume, volume z-score and relative volume."""
    dollar_vol = volume * price
    vol_z = zscore(volume, window)
    rel_vol = volume / volume.rolling(window, min_periods=max(2, window // 2)).mean()
    return pd.DataFrame(
        {
            "log_dollar_volume": np.log1p(dollar_vol),
            "volume_zscore": vol_z,
            "relative_volume": rel_vol,
        }
    )


def rolling_beta(
    asset_returns: pd.Series, market_returns: pd.Series, window: int = 63
) -> pd.Series:
    """Rolling beta of an asset to the market = cov(a, m) / var(m)."""
    cov = asset_returns.rolling(window, min_periods=max(2, window // 2)).cov(market_returns)
    var = market_returns.rolling(window, min_periods=max(2, window // 2)).var()
    return cov / var.replace(0.0, np.nan)


def drawdown(price: pd.Series) -> pd.Series:
    """Drawdown from the running peak (<= 0)."""
    running_max = price.cummax()
    return price / running_max - 1.0


def rolling_max_drawdown(price: pd.Series, window: int = 252) -> pd.Series:
    """Worst drawdown observed within a trailing ``window``."""

    def _mdd(x: np.ndarray) -> float:
        peak = np.maximum.accumulate(x)
        return float(np.min(x / peak - 1.0))

    return price.rolling(window, min_periods=max(2, window // 2)).apply(_mdd, raw=True)
