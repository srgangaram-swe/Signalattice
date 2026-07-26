"""Causal volatility, distributional, and dependence estimators (SF-S2-MR2).

Every estimator operates on a single asset's time-ordered series and uses only
**trailing** windows ending at bar *t* (`min_periods == window`), so a value at
*t* never reads *t+1*. Estimators emit ``NaN`` until a full window is available
and propagate ``NaN`` through genuine gaps rather than imputing.

Three families live here:

* **Volatility** — range-based (Parkinson, Garman-Klass, Rogers-Satchell) and
  exponentially weighted estimators, complementing the close-to-close
  estimators already in :mod:`quant_platform.features.technical`.
* **Distribution** — rolling skewness, excess kurtosis, downside deviation, and
  a robust median-absolute-deviation dispersion.
* **Dependence / memory** — autocorrelation, partial autocorrelation, the Hurst
  exponent, the Lo-MacKinlay variance ratio, rolling correlation, and a
  binned mutual-information estimator.

These are conventional statistical baselines. They make no predictive claim; the
contract layer in :mod:`quant_platform.features.contracts` wraps them with units,
warm-up, and numerical metadata.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quant_platform.features.technical import TRADING_DAYS

# ---------------------------------------------------------------------------
# Volatility estimators (annualised by default)
# ---------------------------------------------------------------------------


def _annualise(vol: pd.Series, annualize: bool) -> pd.Series:
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def parkinson_volatility(
    high: pd.Series, low: pd.Series, window: int = 21, *, annualize: bool = True
) -> pd.Series:
    """Parkinson high-low range volatility.

    ``sigma^2 = 1/(4 ln 2) * mean( (ln(H/L))^2 )`` over the trailing window.
    Assumes positive prices and continuous trading; underestimates when gaps or
    jumps dominate. More efficient than close-to-close under those assumptions.
    """
    log_hl = np.log((high / low).to_numpy(dtype=float))
    factor = 1.0 / (4.0 * np.log(2.0))
    var = pd.Series(factor * log_hl**2, index=high.index).rolling(window, min_periods=window).mean()
    return _annualise(np.sqrt(var), annualize)


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    *,
    annualize: bool = True,
) -> pd.Series:
    """Garman-Klass OHLC volatility.

    ``sigma^2 = mean( 0.5 (ln(H/L))^2 - (2 ln2 - 1)(ln(C/O))^2 )``. The windowed
    variance is clipped at zero before the square root to absorb small-sample
    negativity. Assumes no overnight drift/jumps between close and next open.
    """
    log_hl = np.log((high / low).to_numpy(dtype=float))
    log_co = np.log((close / open_).to_numpy(dtype=float))
    term = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    var = pd.Series(term, index=high.index).rolling(window, min_periods=window).mean()
    return _annualise(np.sqrt(var.clip(lower=0.0)), annualize)


def rogers_satchell_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    *,
    annualize: bool = True,
) -> pd.Series:
    """Rogers-Satchell OHLC volatility (drift-independent).

    ``sigma^2 = mean( ln(H/C)ln(H/O) + ln(L/C)ln(L/O) )``. Each per-bar term is
    non-negative, so unlike Garman-Klass it stays valid under a trending drift.
    """
    log_hc = np.log((high / close).to_numpy(dtype=float))
    log_ho = np.log((high / open_).to_numpy(dtype=float))
    log_lc = np.log((low / close).to_numpy(dtype=float))
    log_lo = np.log((low / open_).to_numpy(dtype=float))
    term = log_hc * log_ho + log_lc * log_lo
    var = pd.Series(term, index=high.index).rolling(window, min_periods=window).mean()
    return _annualise(np.sqrt(var.clip(lower=0.0)), annualize)


def ewma_volatility(returns: pd.Series, span: int = 21, *, annualize: bool = True) -> pd.Series:
    """Exponentially weighted (RiskMetrics-style) volatility of returns."""
    var = (returns**2).ewm(span=span, adjust=False, min_periods=span).mean()
    return _annualise(np.sqrt(var), annualize)


# ---------------------------------------------------------------------------
# Distributional statistics (on returns)
# ---------------------------------------------------------------------------


def rolling_skewness(returns: pd.Series, window: int = 63) -> pd.Series:
    """Trailing sample skewness (bias-adjusted Fisher-Pearson). Needs >= 3 obs."""
    return returns.rolling(window, min_periods=window).skew()


def rolling_kurtosis(returns: pd.Series, window: int = 63) -> pd.Series:
    """Trailing sample **excess** kurtosis (0 for a normal). Needs >= 4 obs."""
    return returns.rolling(window, min_periods=window).kurt()


def downside_deviation(
    returns: pd.Series, window: int = 63, *, annualize: bool = True
) -> pd.Series:
    """Root-mean-square of negative returns over the trailing window (>= 0)."""
    negative = returns.clip(upper=0.0)
    dd = np.sqrt((negative**2).rolling(window, min_periods=window).mean())
    return _annualise(dd, annualize)


def median_absolute_deviation(returns: pd.Series, window: int = 63) -> pd.Series:
    """Robust dispersion: median(|x - median(x)|) over the trailing window.

    Insensitive to outliers, unlike the standard deviation. Returned in the same
    units as ``returns`` (not scaled to a normal-consistent estimate).
    """

    def _mad(values: np.ndarray) -> float:
        return float(np.median(np.abs(values - np.median(values))))

    return returns.rolling(window, min_periods=window).apply(_mad, raw=True)


# ---------------------------------------------------------------------------
# Dependence and memory estimators
# ---------------------------------------------------------------------------


def _rolling_pearson(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    cov = a.rolling(window, min_periods=window).cov(b)
    std_a = a.rolling(window, min_periods=window).std()
    std_b = b.rolling(window, min_periods=window).std()
    corr = cov / (std_a * std_b).replace(0.0, np.nan)
    return corr.clip(-1.0, 1.0)


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    """Trailing Pearson correlation between two aligned series, clipped to [-1, 1]."""
    return _rolling_pearson(a, b, window)


def rolling_autocorrelation(returns: pd.Series, window: int = 63, lag: int = 1) -> pd.Series:
    """Trailing lag-``lag`` autocorrelation of ``returns`` (in [-1, 1])."""
    if lag < 1:
        raise ValueError("autocorrelation lag must be >= 1")
    return _rolling_pearson(returns, returns.shift(lag), window)


def partial_autocorrelation_lag2(returns: pd.Series, window: int = 63) -> pd.Series:
    """Trailing lag-2 partial autocorrelation.

    Closed form from the first two autocorrelations:
    ``pacf(2) = (r2 - r1^2) / (1 - r1^2)``. The denominator is guarded and the
    result clipped to [-1, 1].
    """
    r1 = rolling_autocorrelation(returns, window, lag=1)
    r2 = rolling_autocorrelation(returns, window, lag=2)
    denom = (1.0 - r1**2).replace(0.0, np.nan)
    return ((r2 - r1**2) / denom).clip(-1.0, 1.0)


def hurst_exponent(
    price: pd.Series, window: int = 128, *, lags: Sequence[int] = (2, 4, 8, 16, 32)
) -> pd.Series:
    """Trailing Hurst exponent of ``log(price)`` via the structure function.

    For each lag ``L`` the standard deviation of ``L``-step log-price differences
    inside the window is computed; the Hurst exponent is the slope of
    ``log(std)`` on ``log(L)``. ``H ~ 0.5`` is a random walk, ``> 0.5`` trending,
    ``< 0.5`` mean-reverting. Complexity is ``O(window * len(lags))`` per bar.
    Only lags strictly smaller than the window are used.
    """
    usable = tuple(lag for lag in lags if 0 < lag < window)
    if len(usable) < 2:
        raise ValueError("hurst_exponent needs at least two lags smaller than the window")
    log_lags = np.log(np.asarray(usable, dtype=float))

    def _hurst(values: np.ndarray) -> float:
        magnitudes = []
        kept = []
        for index, lag in enumerate(usable):
            diffs = values[lag:] - values[:-lag]
            spread = float(np.std(diffs))
            if spread > 0.0:
                magnitudes.append(np.log(spread))
                kept.append(log_lags[index])
        if len(kept) < 2:
            return float("nan")
        slope = np.polyfit(np.asarray(kept), np.asarray(magnitudes), 1)[0]
        return float(slope)

    log_price = np.log(price.to_numpy(dtype=float))
    return (
        pd.Series(log_price, index=price.index)
        .rolling(window, min_periods=window)
        .apply(_hurst, raw=True)
    )


def variance_ratio(returns: pd.Series, window: int = 63, q: int = 5) -> pd.Series:
    """Trailing Lo-MacKinlay variance ratio ``VR(q)`` (overlapping estimator).

    ``VR(q) = Var(q-period return) / (q * Var(1-period return))``. ``VR ~ 1``
    under a random walk, ``> 1`` trending, ``< 1`` mean-reverting. Undefined when
    the one-period variance is zero.
    """
    if q < 2:
        raise ValueError("variance_ratio requires q >= 2")

    def _vr(values: np.ndarray) -> float:
        n = values.shape[0]
        mean = values.mean()
        var_one = values.var(ddof=1)
        if var_one <= 0.0 or n <= q:
            return float("nan")
        cumulative = np.cumsum(values)
        q_sums = cumulative[q - 1 :] - np.concatenate(([0.0], cumulative[:-q]))
        # `scale` already carries the factor q, so var_q is the per-period
        # variance of the q-period return; VR is var_q / var_one directly.
        scale = q * (n - q + 1) * (1.0 - q / n)
        var_q = np.sum((q_sums - q * mean) ** 2) / scale
        return float(var_q / var_one)

    return returns.rolling(window, min_periods=window).apply(_vr, raw=True)


def _mutual_information_bits(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    counts, _, _ = np.histogram2d(x, y, bins=bins)
    total = counts.sum()
    if total <= 0:  # pragma: no cover - a finite window always has positive mass
        return float("nan")
    joint = counts / total
    marginal_x = joint.sum(axis=1, keepdims=True)
    marginal_y = joint.sum(axis=0, keepdims=True)
    outer = marginal_x * marginal_y
    mask = (joint > 0) & (outer > 0)
    return float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask])))


def rolling_mutual_information(
    a: pd.Series, b: pd.Series, window: int = 63, *, bins: int = 8
) -> pd.Series:
    """Trailing binned mutual information (nats) between two aligned series.

    Discretises each window of ``a`` and ``b`` into a ``bins x bins`` histogram
    and returns the plug-in mutual information (``>= 0``). Captures nonlinear
    dependence the Pearson correlation misses. A window containing any missing
    value yields ``NaN``. Plug-in MI is positively biased for small samples and
    many bins; keep ``bins`` small relative to ``window``.
    """
    values_a = a.to_numpy(dtype=float)
    values_b = b.to_numpy(dtype=float)
    n = values_a.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if n >= window:
        windows_a = np.lib.stride_tricks.sliding_window_view(values_a, window)
        windows_b = np.lib.stride_tricks.sliding_window_view(values_b, window)
        for index in range(windows_a.shape[0]):
            x = windows_a[index]
            y = windows_b[index]
            if np.isfinite(x).all() and np.isfinite(y).all():
                out[index + window - 1] = _mutual_information_bits(x, y, bins)
    return pd.Series(out, index=a.index, name=a.name)
