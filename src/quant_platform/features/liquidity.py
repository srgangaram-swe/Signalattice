"""Causal volume, liquidity, spread, and impact proxies (SF-S2-MR3).

Every estimator operates on a single asset's time-ordered series with trailing
windows ending at bar *t*, emits ``NaN`` until a full window is available, and
propagates ``NaN`` (never imputes) through gaps.

These are **daily-bar proxies**. They approximate liquidity, spread, and price
impact from open/high/low/close/volume; they are not observed order-book spread,
depth, or executable capacity, and each carries an interpretation limit
documented on its contract. Zero-volume and non-positive-price bars yield ``NaN``
for the affected estimate rather than an infinity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 3 - 2*sqrt(2), the Corwin-Schultz normalisation constant.
_CORWIN_SCHULTZ_CONST = 3.0 - 2.0 * np.sqrt(2.0)


def _positive(series: pd.Series) -> pd.Series:
    """Return a float copy with non-positive entries replaced by NaN."""
    values = series.to_numpy(dtype=float)
    return pd.Series(np.where(values > 0.0, values, np.nan), index=series.index, name=series.name)


def rolling_log_dollar_volume(close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """Trailing mean of ``log(close * volume)`` — a liquidity *level* proxy.

    Source fields: close, volume. Units: natural-log dollars. A bar with
    non-positive traded value contributes ``NaN``.
    """
    dollar_volume = _positive(close * volume)
    log_dollar: pd.Series = np.log(dollar_volume)
    return log_dollar.rolling(window, min_periods=window).mean()


def volume_change(volume: pd.Series, periods: int = 1) -> pd.Series:
    """Log change in volume over ``periods`` bars (a volume-momentum proxy).

    Source field: volume. Units: log ratio. Zero-volume bars yield ``NaN``.
    """
    positive_volume = _positive(volume)
    log_volume: pd.Series = np.log(positive_volume)
    return log_volume - log_volume.shift(periods)


def relative_volume(volume: pd.Series, window: int = 21) -> pd.Series:
    """Volume divided by its trailing mean (unitless volume intensity, >= 0)."""
    mean_volume = volume.rolling(window, min_periods=window).mean()
    return volume / mean_volume.replace(0.0, np.nan)


def amihud_illiquidity(
    close: pd.Series, volume: pd.Series, window: int = 21, *, scale: float = 1e6
) -> pd.Series:
    """Amihud illiquidity: trailing mean of ``|return| / dollar_volume``.

    A price-impact proxy — larger means a given dollar traded moves the price
    more. Source fields: close, volume. Units: return per ``scale`` dollars
    (default ``scale = 1e6``, i.e. per \\$1M). Zero-volume bars yield ``NaN`` for
    that bar. Interpretation limit: it is a coarse average impact, not a measured
    order-book impact curve.
    """
    returns = close.pct_change().abs()
    dollar_volume = _positive(close * volume)
    daily = returns / dollar_volume * scale
    return daily.rolling(window, min_periods=window).mean()


def volume_imbalance(close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """Signed-volume imbalance over the trailing window, in ``[-1, 1]``.

    ``(up-day volume - down-day volume) / total volume`` using the sign of the
    close-to-close return as the trade-direction proxy. Source fields: close,
    volume. A zero-volume window yields ``NaN``. Interpretation limit: return
    sign is a crude proxy for signed order flow.
    """
    direction = np.sign(close.pct_change().to_numpy(dtype=float))
    signed = pd.Series(direction, index=close.index) * volume
    numerator = signed.rolling(window, min_periods=window).sum()
    denominator = volume.rolling(window, min_periods=window).sum()
    return (numerator / denominator.replace(0.0, np.nan)).clip(-1.0, 1.0)


def corwin_schultz_spread(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Corwin-Schultz (2012) high-low bid-ask spread estimator (fractional, >= 0).

    Uses each adjacent day pair ``(t-1, t)`` (so the estimate at *t* is causal),
    with negative per-pair estimates floored at zero, then averaged over the
    trailing window. Source fields: high, low. Interpretation limit: a
    statistical spread proxy that assumes the high and low are a buy and a sell;
    it is not the observed quoted spread.
    """
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    n = high_values.shape[0]
    log_hl_sq = np.log(high_values / low_values) ** 2

    beta = np.full(n, np.nan)
    gamma = np.full(n, np.nan)
    beta[1:] = log_hl_sq[1:] + log_hl_sq[:-1]
    two_day_high = np.maximum(high_values[1:], high_values[:-1])
    two_day_low = np.minimum(low_values[1:], low_values[:-1])
    gamma[1:] = np.log(two_day_high / two_day_low) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CORWIN_SCHULTZ_CONST - np.sqrt(
        gamma / _CORWIN_SCHULTZ_CONST
    )
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = np.where(spread < 0.0, 0.0, spread)
    return pd.Series(spread, index=high.index).rolling(window, min_periods=window).mean()


def roll_spread(close: pd.Series, window: int = 21) -> pd.Series:
    """Roll (1984) implied effective spread from serial covariance of returns.

    ``spread = 2 * sqrt(-Cov(dp_t, dp_{t-1}))`` over the trailing window, where
    ``dp`` is the log return; when the serial covariance is non-negative the
    estimator is undefined and returned as ``0``. Fractional units, ``>= 0``.
    Source field: close. Interpretation limit: assumes bid-ask bounce is the only
    source of negative serial correlation.
    """
    log_returns: pd.Series = np.log(_positive(close)).diff()
    covariance = log_returns.rolling(window, min_periods=window).cov(log_returns.shift(1))
    spread: pd.Series = 2.0 * np.sqrt((-covariance).clip(lower=0.0))
    return spread


def turnover(volume: pd.Series, shares_outstanding: pd.Series, window: int = 21) -> pd.Series:
    """Trailing mean share turnover ``volume / shares_outstanding`` (>= 0).

    Requires a ``shares_outstanding`` series in addition to OHLCV; this is the
    contract's explicit extra data requirement. Units: fraction of shares
    outstanding traded per bar. Non-positive share counts yield ``NaN``.
    """
    rate = volume / _positive(shares_outstanding)
    return rate.rolling(window, min_periods=window).mean()
