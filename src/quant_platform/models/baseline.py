"""Rules-based (non-ML) baseline signals.

A credible research project always compares ML models against simple, robust
baselines — if a gradient-boosted model can't beat 12-1 momentum after costs,
that's an important finding. These functions return a ``signal`` score per
``(date, ticker)`` row aligned to the feature frame's index. Higher score =>
more bullish. The backtester converts scores into positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.features.cross_sectional import cross_sectional_rank
from quant_platform.features.pipeline import FEATURE_PREFIX
from quant_platform.logging_utils import get_logger

logger = get_logger(__name__)


def available_baselines() -> list[str]:
    return ["momentum", "ma_crossover", "mean_reversion"]


def momentum_signal(features: pd.DataFrame, *, lookback_col: str | None = None) -> pd.Series:
    """Cross-sectional momentum: rank names by 12-1 momentum each day.

    Returns a score in ``[0, 1]`` (percentile rank), so the backtester can go
    long the top names and short the bottom.
    """
    col = lookback_col or f"{FEATURE_PREFIX}mom_12_1"
    if col not in features.columns:
        col = f"{FEATURE_PREFIX}mom_252"
    return cross_sectional_rank(features, col).rename("signal")


def ma_crossover_signal(features: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """Trend-following: +1 when fast MA above slow MA, else -1.

    Uses the price-relative MA ratio features; the sign of (fast_ratio -
    slow_ratio) approximates a golden/death-cross trend filter.
    """
    fast_col = f"{FEATURE_PREFIX}ma_ratio_{fast}"
    slow_col = f"{FEATURE_PREFIX}ma_ratio_{slow}"
    if fast_col not in features.columns or slow_col not in features.columns:
        # Fall back to any two available MA-ratio columns.
        ma_cols = sorted(
            (c for c in features.columns if c.startswith(f"{FEATURE_PREFIX}ma_ratio_")),
            key=lambda c: int(c.rsplit("_", 1)[1]),
        )
        if len(ma_cols) < 2:
            raise ValueError("ma_crossover requires at least two MA-ratio features")
        fast_col, slow_col = ma_cols[0], ma_cols[-1]
    diff = features[fast_col].to_numpy(dtype=float) - features[slow_col].to_numpy(dtype=float)
    signal = pd.Series(np.sign(diff), index=features.index, dtype=float)
    return signal.fillna(0.0).rename("signal")


def mean_reversion_signal(features: pd.DataFrame) -> pd.Series:
    """Short-term reversal: buy recent losers, sell recent winners.

    Cross-sectional rank of the negative 5-day return.
    """
    col = f"{FEATURE_PREFIX}reversal_5"
    if col not in features.columns:
        raise ValueError("mean_reversion requires the reversal_5 feature")
    return cross_sectional_rank(features, col).rename("signal")


def baseline_signal(features: pd.DataFrame, kind: str) -> pd.Series:
    """Dispatch to a named baseline signal generator."""
    logger.info("Computing baseline signal: %s", kind)
    if kind == "momentum":
        return momentum_signal(features)
    if kind == "ma_crossover":
        return ma_crossover_signal(features)
    if kind == "mean_reversion":
        return mean_reversion_signal(features)
    raise ValueError(f"Unknown baseline '{kind}'. Available: {available_baselines()}")
