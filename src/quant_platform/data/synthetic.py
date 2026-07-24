"""Synthetic market-data generator.

Used as a deterministic, offline fallback when network data sources are
unavailable (e.g. in CI or air-gapped environments). The generator produces a
realistic *factor-structured* panel: a common market factor drives each name
through a per-ticker beta, plus idiosyncratic noise. The default process embeds
small, declared AR(1) momentum and mean-reversion effects so causal-model tests
can recover a known edge. Setting both autocorrelation parameters to zero gives
a null directional process. Synthetic results remain engineering evidence, not
evidence of live-market profitability.

The output conforms exactly to the canonical schema in
:mod:`quant_platform.data.schema`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.config import SyntheticConfig
from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.logging_utils import get_logger

logger = get_logger(__name__)


def _ar1_shocks(noise: np.ndarray, coefficient: float) -> np.ndarray:
    """Create a stationary, unit-variance AR(1) path from IID standard noise."""
    values = np.empty_like(noise, dtype=float)
    values[0] = noise[0]
    innovation_scale = np.sqrt(1.0 - coefficient**2)
    for idx in range(1, len(noise)):
        values[idx] = coefficient * values[idx - 1] + innovation_scale * noise[idx]
    return values


def generate_synthetic_panel(
    tickers: list[str],
    *,
    benchmark: str,
    config: SyntheticConfig,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic synthetic OHLCV panel.

    Parameters
    ----------
    tickers:
        Symbols to generate. The ``benchmark`` is treated as the market factor
        proxy (beta ~ 1, no idiosyncratic alpha) so downstream beta/correlation
        features are meaningful.
    config:
        :class:`~quant_platform.config.SyntheticConfig` controlling horizon and
        return moments.
    seed:
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n = int(config.n_days)
    # Business-day calendar.
    dates = pd.bdate_range(start=config.start, periods=n)
    dt = 1.0 / 252.0

    # --- common market factor (geometric Brownian motion) ---
    mkt_mu = config.annual_drift
    mkt_sigma = config.market_vol
    # Add mild volatility clustering via a slow-moving vol regime.
    regime = 1.0 + 0.5 * np.sin(np.linspace(0, 6 * np.pi, n)) ** 2
    mkt_shocks = _ar1_shocks(rng.standard_normal(n), config.market_autocorrelation) * regime
    mkt_ret = (mkt_mu - 0.5 * mkt_sigma**2) * dt + mkt_sigma * np.sqrt(dt) * mkt_shocks

    frames: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        is_bench = ticker == benchmark
        if is_bench:
            beta = 1.0
            alpha = 0.0
            idio_sigma = 0.0
            ret = mkt_ret.copy()
        else:
            # Deterministic-but-varied parameters per ticker.
            t_rng = np.random.default_rng(seed + 1000 * (i + 1))
            beta = float(np.clip(t_rng.normal(config.market_beta_mean, 0.35), 0.1, 2.2))
            alpha = float(t_rng.normal(0.0, 0.02)) * dt
            idio_sigma = float(np.clip(t_rng.normal(config.annual_vol, 0.05), 0.08, 0.6))
            idio = (
                _ar1_shocks(t_rng.standard_normal(n), config.idiosyncratic_autocorrelation) * regime
            )
            ret = (
                alpha + beta * mkt_ret + idio_sigma * np.sqrt(dt) * idio - 0.5 * idio_sigma**2 * dt
            )

        # Build a price series from returns.
        start_price = float(20 + 380 * rng.random())
        close = start_price * np.exp(np.cumsum(ret))

        # Construct plausible OHLC around close.
        intraday = np.abs(rng.normal(0, idio_sigma if idio_sigma else mkt_sigma, n)) * np.sqrt(dt)
        open_ = close * (1.0 + rng.normal(0, 0.002, n))
        high = np.maximum(open_, close) * (1.0 + intraday)
        low = np.minimum(open_, close) * (1.0 - intraday)
        # Adjusted close: apply a small steady dividend drag so adj != close.
        div_factor = np.exp(-np.cumsum(np.full(n, 0.015 * dt)))
        adj_close = close * div_factor

        # Volume: lognormal with a level proportional to |return| (activity).
        base_vol = float(10 ** rng.uniform(5.5, 7.0))
        volume = base_vol * np.exp(rng.normal(0, 0.4, n)) * (1.0 + 3.0 * np.abs(ret))

        frame = pd.DataFrame(
            {
                DATE_COL: dates,
                TICKER_COL: ticker,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "adj_close": adj_close,
                "volume": np.round(volume),
            }
        )
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    logger.info(
        "Generated synthetic panel: %d tickers x %d days (seed=%d)",
        len(tickers),
        n,
        seed,
    )
    return panel
