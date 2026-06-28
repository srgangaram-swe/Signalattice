"""Feature pipeline: price panel -> model-ready feature matrix + targets.

Design principles
-----------------
* **No lookahead bias.** Features at bar *t* use only data up to *t*; the target
  is a *forward* return spanning *t+1 .. t+h*. The two never overlap.
* **No leakage across tickers.** All trailing transforms are computed *within*
  each ticker via ``groupby``; cross-sectional transforms are computed *within*
  each date.
* **Reproducible & typed.** Output columns are namespaced (``f_*`` for features,
  ``target_*`` for labels) so downstream code can select them unambiguously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.config import FeatureConfig
from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.features import technical as ta
from quant_platform.features.cross_sectional import cross_sectional_rank, cross_sectional_zscore
from quant_platform.logging_utils import get_logger

logger = get_logger(__name__)

FEATURE_PREFIX = "f_"
TARGET_RETURN = "target_forward_return"
TARGET_DIRECTION = "target_direction"


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (``f_*``) present in ``df``."""
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def _per_ticker_features(
    g: pd.DataFrame,
    cfg: FeatureConfig,
    price_field: str,
    market_returns: pd.Series,
) -> pd.DataFrame:
    """Compute all trailing technical features for a single ticker group."""
    price = g[price_field]
    ret = g["return"]
    out: dict[str, pd.Series] = {}

    # Returns at multiple horizons.
    out[f"{FEATURE_PREFIX}ret_1d"] = ret
    out[f"{FEATURE_PREFIX}logret_1d"] = g["log_return"]
    out[f"{FEATURE_PREFIX}ret_5d"] = ta.simple_returns(price, 5)

    # Volatility family.
    for w in cfg.vol_windows:
        out[f"{FEATURE_PREFIX}vol_{w}"] = ta.rolling_volatility(ret, w)
    out[f"{FEATURE_PREFIX}realized_vol_{cfg.realized_vol_window}"] = ta.realized_volatility(
        ret, cfg.realized_vol_window
    )

    # Rolling risk-adjusted return.
    out[f"{FEATURE_PREFIX}sharpe_{cfg.sharpe_window}"] = ta.rolling_sharpe(ret, cfg.sharpe_window)

    # Trend: moving averages (as price-relative ratios) and EMAs.
    for w in cfg.ma_windows:
        out[f"{FEATURE_PREFIX}ma_ratio_{w}"] = ta.ma_ratio(price, w)
    for s in cfg.ema_windows:
        out[f"{FEATURE_PREFIX}ema_ratio_{s}"] = price / ta.ema(price, s) - 1.0

    # Oscillators.
    out[f"{FEATURE_PREFIX}rsi_{cfg.rsi_window}"] = ta.rsi(price, cfg.rsi_window)
    macd_df = ta.macd(price, cfg.macd.fast, cfg.macd.slow, cfg.macd.signal)
    for col in macd_df.columns:
        out[f"{FEATURE_PREFIX}{col}"] = macd_df[col]

    # Bollinger bands (keep the scale-free features only).
    bb = ta.bollinger_bands(price, cfg.bollinger.window, cfg.bollinger.n_std)
    out[f"{FEATURE_PREFIX}bb_pctb"] = bb["bb_pctb"]
    out[f"{FEATURE_PREFIX}bb_bandwidth"] = bb["bb_bandwidth"]

    # Momentum & mean reversion.
    for w in cfg.momentum_windows:
        out[f"{FEATURE_PREFIX}mom_{w}"] = ta.momentum(price, w)
    out[f"{FEATURE_PREFIX}mom_12_1"] = ta.momentum(price, 252, skip=21)
    out[f"{FEATURE_PREFIX}reversal_5"] = -ta.simple_returns(price, 5)
    out[f"{FEATURE_PREFIX}zscore_21"] = ta.zscore(price, 21)

    # Volume features.
    vf = ta.volume_features(g["volume"], price, cfg.realized_vol_window)
    for col in vf.columns:
        out[f"{FEATURE_PREFIX}{col}"] = vf[col]

    # Beta to market.
    mkt = g[DATE_COL].map(market_returns)
    mkt.index = g.index
    out[f"{FEATURE_PREFIX}beta_{cfg.beta_window}"] = ta.rolling_beta(ret, mkt, cfg.beta_window)

    # Drawdown features.
    out[f"{FEATURE_PREFIX}drawdown"] = ta.drawdown(price)
    out[f"{FEATURE_PREFIX}max_dd_{cfg.drawdown_window}"] = ta.rolling_max_drawdown(
        price, cfg.drawdown_window
    )

    features = pd.DataFrame(out, index=g.index)
    return features


def _add_targets(panel: pd.DataFrame, price_field: str, forward_horizon: int) -> pd.DataFrame:
    """Append forward-return regression target and binary direction target.

    The forward return spans ``t+1 .. t+horizon`` and is computed per ticker so
    it never crosses ticker boundaries. The label at the last ``horizon`` rows of
    each ticker is ``NaN`` (the future is unknown) and is dropped before training.
    """
    out = panel.copy()
    grp = out.groupby(TICKER_COL, sort=False)[price_field]
    fwd = grp.shift(-forward_horizon) / out[price_field] - 1.0
    out[TARGET_RETURN] = fwd
    out[TARGET_DIRECTION] = np.where(fwd.isna(), np.nan, (fwd > 0).astype(float))
    return out


def build_features(
    panel: pd.DataFrame,
    config: FeatureConfig,
    *,
    benchmark: str,
    price_field: str = "adj_close",
    forward_horizon: int = 1,
) -> pd.DataFrame:
    """Build the full feature matrix and targets from a validated price panel.

    Parameters
    ----------
    panel:
        Long-format price panel with ``return``/``log_return`` columns
        (output of :func:`quant_platform.data.ingest`).
    config:
        Feature-engineering configuration.
    benchmark:
        Ticker used as the market proxy for beta and (optionally) excluded from
        the cross-section ranking universe sizing.
    price_field:
        Which price column features are derived from (``adj_close`` recommended).
    forward_horizon:
        Horizon (in bars) for the prediction target.

    Returns
    -------
    pandas.DataFrame
        Long-format frame with metadata columns, ``f_*`` features and
        ``target_*`` labels. Rows with missing features/targets are dropped when
        ``config.dropna`` is true.
    """
    if panel.empty:
        raise ValueError("Cannot build features from an empty panel")

    panel = panel.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)

    # Market return series indexed by date (for beta).
    market = (
        panel.loc[panel[TICKER_COL] == benchmark, [DATE_COL, "return"]]
        .drop_duplicates(DATE_COL)
        .set_index(DATE_COL)["return"]
    )
    if market.empty:
        logger.warning("Benchmark '%s' not found; beta features will be NaN", benchmark)
        market = pd.Series(dtype=float)

    feature_frames = []
    for _ticker, g in panel.groupby(TICKER_COL, sort=False):
        feats = _per_ticker_features(g, config, price_field, market)
        feature_frames.append(feats)
    features = pd.concat(feature_frames).sort_index()

    enriched = pd.concat([panel, features], axis=1)

    # Targets (forward-looking, leakage-free).
    enriched = _add_targets(enriched, price_field, forward_horizon)

    # Cross-sectional features (computed across tickers per date).
    if config.cross_sectional:
        cs_base = [
            f"{FEATURE_PREFIX}mom_12_1",
            f"{FEATURE_PREFIX}reversal_5",
            f"{FEATURE_PREFIX}vol_{config.vol_windows[-1]}",
            f"{FEATURE_PREFIX}rsi_{config.rsi_window}",
        ]
        for col in cs_base:
            if col in enriched.columns:
                base = col.replace(FEATURE_PREFIX, "")
                enriched[f"{FEATURE_PREFIX}cs_rank_{base}"] = cross_sectional_rank(enriched, col)
                enriched[f"{FEATURE_PREFIX}cs_z_{base}"] = pd.to_numeric(
                    cross_sectional_zscore(enriched, col), errors="coerce"
                )

    # Replace inf and (optionally) drop incomplete rows.
    feat_cols = feature_columns(enriched)
    enriched[feat_cols] = enriched[feat_cols].replace([np.inf, -np.inf], np.nan)

    n_before = len(enriched)
    if config.dropna:
        enriched = enriched.dropna(subset=[*feat_cols, TARGET_DIRECTION, TARGET_RETURN])
    enriched = enriched.reset_index(drop=True)
    logger.info(
        "Built %d features for %d tickers (%d -> %d rows after dropna=%s)",
        len(feat_cols),
        enriched[TICKER_COL].nunique(),
        n_before,
        len(enriched),
        config.dropna,
    )
    return enriched
