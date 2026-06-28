"""Cross-sectional feature transforms.

Cross-sectional features compare a name to its peers *on the same date* — the
backbone of relative-value and factor strategies (e.g. "rank stocks by 12-1
momentum each day, go long the top decile, short the bottom"). These transforms
operate across tickers within each date and introduce no lookahead because they
only use contemporaneous information.
"""

from __future__ import annotations

import pandas as pd

from quant_platform.data.schema import DATE_COL


def cross_sectional_rank(panel: pd.DataFrame, column: str, *, pct: bool = True) -> pd.Series:
    """Rank ``column`` across tickers within each date.

    Parameters
    ----------
    pct:
        If true, return percentile ranks in ``[0, 1]`` (recommended — scale-free
        and robust to outliers); otherwise integer ranks.
    """
    return panel.groupby(DATE_COL)[column].rank(pct=pct, method="average")


def cross_sectional_zscore(panel: pd.DataFrame, column: str) -> pd.Series:
    """Standardise ``column`` across tickers within each date (demean / unit std)."""
    grp = panel.groupby(DATE_COL)[column]
    mean = grp.transform("mean")
    std = grp.transform("std")
    return (panel[column] - mean) / std.replace(0.0, pd.NA)


def cross_sectional_demean(panel: pd.DataFrame, column: str) -> pd.Series:
    """Subtract the cross-sectional mean (market-neutralise a signal)."""
    grp = panel.groupby(DATE_COL)[column]
    return panel[column] - grp.transform("mean")
