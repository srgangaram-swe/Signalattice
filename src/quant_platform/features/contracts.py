"""Versioned, self-describing feature contracts (Signal Foundry SF-S2-MR1..MR3).

A :class:`FeatureContract` pairs a *causal* computation kernel with the metadata
a research platform needs to reason about a feature without reading its code:
its family, unit, the panel columns it consumes, its lookback and warm-up, the
sampling frequency it assumes, its missing-data policy, when the value becomes
available relative to the bar it is stamped on, and its numerical behaviour.

Design invariants
-----------------
* **Causal.** Every contract reads only information available up to and including
  bar *t*; a feature at *t* never peeks at *t+1*. This is the leakage boundary
  the whole platform depends on.
* **Full-window warm-up.** A contract emits ``NaN`` until it has observed a
  complete lookback window for that asset. Partial-window values are suppressed
  so a feature means the same thing at every timestamp and across assets. The
  first defined value for an asset appears at position ``warmup - 1``.
* **Fail closed.** Malformed input (missing columns, non-finite ``inf`` values,
  duplicate ``(date, ticker)`` keys) raises before any feature is computed.
  Genuine gaps (``NaN``) propagate rather than being silently imputed.
* **No new mathematics.** Kernels compose the audited primitives in
  :mod:`quant_platform.features.technical`,
  :mod:`quant_platform.features.statistics`,
  :mod:`quant_platform.features.liquidity`, and
  :mod:`quant_platform.features.cross_sectional`; this module adds the contract,
  metadata, and registry layer only.

The public entry point is :func:`build_contract_features`, which turns a
validated long price panel into a namespaced feature matrix. :func:`get_contract`
and :func:`contract_metadata_frame` expose the registry for introspection and
documentation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.features import liquidity as lq
from quant_platform.features import statistics as st
from quant_platform.features import technical as ta
from quant_platform.features.cross_sectional import (
    cross_sectional_rank,
    cross_sectional_zscore,
)

#: Canonical price field for conventional features (split/dividend adjusted).
PRICE_FIELD = "adj_close"

#: Namespace prefix for contract-produced feature columns.
CONTRACT_PREFIX = "fc_"


class Scope(StrEnum):
    """Where a feature draws its cross-section of information from."""

    PER_ASSET = "per_asset"  # computed within one asset's own time series
    CROSS_SECTIONAL = "cross_sectional"  # computed across assets within one date


class FeatureFamily(StrEnum):
    """Conventional feature families covered by this slice."""

    RETURN = "return"
    MOMENTUM = "momentum"
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY = "volatility"
    DISTRIBUTION = "distribution"
    DEPENDENCE = "dependence"
    LIQUIDITY = "liquidity"


class Unit(StrEnum):
    """Measurement unit of a feature's output."""

    SIMPLE_RETURN = "simple_return"  # fractional change, 0.01 == 1%
    LOG_RETURN = "log_return"  # continuously compounded
    RATIO_DEVIATION = "ratio_deviation"  # value / reference - 1, dimensionless
    ZSCORE = "zscore"  # standardised, dimensionless
    RANK_PCT = "rank_pct"  # cross-sectional percentile in [0, 1]
    LOG_RESIDUAL = "log_residual"  # residual in log-price units (~fractional)
    ANNUALIZED_VOL = "annualized_vol"  # annualised standard-deviation units
    CORRELATION = "correlation"  # Pearson-style coefficient in [-1, 1]
    INFORMATION_NATS = "information_nats"  # mutual information in nats (>= 0)
    DIMENSIONLESS = "dimensionless"  # pure number (skew, kurtosis, Hurst, ratio)
    LOG_DOLLAR_VOLUME = "log_dollar_volume"  # natural log of traded dollar value
    ILLIQUIDITY = "illiquidity"  # Amihud: return per million dollars traded
    SPREAD_FRACTION = "spread_fraction"  # estimated bid-ask spread as a fraction


class MissingDataPolicy(StrEnum):
    """How a contract responds to missing observations."""

    PROPAGATE_NAN = "propagate_nan"  # NaN in -> NaN out; never imputed


class TemporalAvailability(StrEnum):
    """When the value is knowable relative to the bar it is stamped on."""

    CAUSAL = "causal"  # uses only bars up to and including t


@dataclass(frozen=True)
class NumericalRange:
    """Closed interval a feature's defined values are guaranteed to lie in."""

    lower: float | None
    upper: float | None

    def contains(self, values: pd.Series) -> bool:
        """True if every finite value in ``values`` lies within the range."""
        finite = values.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        below = self.lower is not None and bool(np.any(finite < self.lower))
        above = self.upper is not None and bool(np.any(finite > self.upper))
        return not (below or above)


# A kernel maps a validated, date-sorted long panel to a panel-aligned Series.
FeatureKernel = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class FeatureContract:
    """A single conventional feature plus the metadata that describes it."""

    name: str
    version: str
    family: FeatureFamily
    scope: Scope
    unit: Unit
    description: str
    inputs: tuple[str, ...]
    params: tuple[tuple[str, int], ...]
    lookback: int
    warmup: int
    kernel: FeatureKernel = field(compare=False, repr=False)
    frequency: str = "daily"
    temporal_availability: TemporalAvailability = TemporalAvailability.CAUSAL
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.PROPAGATE_NAN
    numerical_range: NumericalRange | None = None

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Compute the feature for a validated, date-sorted long ``panel``.

        The panel must already satisfy :func:`validate_contract_panel` (this is
        what :func:`build_contract_features` guarantees): required columns
        present, sorted by ``(ticker, date)``, and a default range index. The
        returned Series is aligned to ``panel.index``.
        """
        missing = [column for column in self.inputs if column not in panel.columns]
        if missing:
            raise ValueError(
                f"contract '{self.name}' requires columns {missing} absent from the panel"
            )
        return self.kernel(panel)

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the contract."""
        return {
            "name": self.name,
            "version": self.version,
            "family": self.family.value,
            "scope": self.scope.value,
            "unit": self.unit.value,
            "description": self.description,
            "inputs": list(self.inputs),
            "params": dict(self.params),
            "lookback": self.lookback,
            "warmup": self.warmup,
            "frequency": self.frequency,
            "temporal_availability": self.temporal_availability.value,
            "missing_data_policy": self.missing_data_policy.value,
            "numerical_range": (
                None
                if self.numerical_range is None
                else {"lower": self.numerical_range.lower, "upper": self.numerical_range.upper}
            ),
        }


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------


def _mask_warmup(series: pd.Series, warmup: int) -> pd.Series:
    """Suppress the first ``warmup - 1`` values so only full-window output remains."""
    if warmup > 1:
        series = series.copy()
        series.iloc[: warmup - 1] = np.nan
    return series


def _per_asset(panel: pd.DataFrame, transform: Callable[[pd.DataFrame], pd.Series]) -> pd.Series:
    """Apply a per-asset ``transform`` within each ticker and realign to ``panel``."""
    parts: list[pd.Series] = []
    for _ticker, group in panel.groupby(TICKER_COL, sort=False):
        parts.append(transform(group))
    combined = pd.concat(parts) if parts else pd.Series(dtype=float)
    return combined.reindex(panel.index)


def _cross_sectional_base(panel: pd.DataFrame, window: int) -> pd.Series:
    """Warm-up-masked per-asset momentum used as a cross-sectional rank/z base."""
    return _per_asset(
        panel,
        lambda group: _mask_warmup(ta.momentum(group[PRICE_FIELD], window), window),
    )


# ---------------------------------------------------------------------------
# Contract builders
# ---------------------------------------------------------------------------


def _return_contract(periods: int) -> FeatureContract:
    label = "1d" if periods == 1 else f"{periods}d"

    def kernel(panel: pd.DataFrame) -> pd.Series:
        return _per_asset(
            panel,
            lambda group: _mask_warmup(ta.simple_returns(group[PRICE_FIELD], periods), periods),
        )

    return FeatureContract(
        name=f"ret_{label}",
        version="1.0.0",
        family=FeatureFamily.RETURN,
        scope=Scope.PER_ASSET,
        unit=Unit.SIMPLE_RETURN,
        description=f"Simple {label} adjusted-close return.",
        inputs=(PRICE_FIELD,),
        params=(("periods", periods),),
        lookback=periods,
        warmup=periods,
        kernel=kernel,
    )


def _log_return_contract() -> FeatureContract:
    def kernel(panel: pd.DataFrame) -> pd.Series:
        return _per_asset(
            panel,
            lambda group: _mask_warmup(ta.log_returns(group[PRICE_FIELD], 1), 1),
        )

    return FeatureContract(
        name="logret_1d",
        version="1.0.0",
        family=FeatureFamily.RETURN,
        scope=Scope.PER_ASSET,
        unit=Unit.LOG_RETURN,
        description="One-bar continuously-compounded (log) adjusted-close return.",
        inputs=(PRICE_FIELD,),
        params=(("periods", 1),),
        lookback=1,
        warmup=1,
        kernel=kernel,
    )


def _rate_of_change_contract(window: int) -> FeatureContract:
    def kernel(panel: pd.DataFrame) -> pd.Series:
        return _per_asset(
            panel,
            lambda group: _mask_warmup(ta.simple_returns(group[PRICE_FIELD], window), window),
        )

    return FeatureContract(
        name=f"roc_{window}",
        version="1.0.0",
        family=FeatureFamily.MOMENTUM,
        scope=Scope.PER_ASSET,
        unit=Unit.SIMPLE_RETURN,
        description=f"Rate of change over {window} bars (fractional).",
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        kernel=kernel,
    )


def _momentum_contract(window: int, *, skip: int = 0) -> FeatureContract:
    name = f"mom_{window}" if skip == 0 else f"mom_{window}_{skip}"
    detail = "" if skip == 0 else f", skipping the most recent {skip} bars"
    return FeatureContract(
        name=name,
        version="1.0.0",
        family=FeatureFamily.MOMENTUM,
        scope=Scope.PER_ASSET,
        unit=Unit.SIMPLE_RETURN,
        description=f"Total return over {window} bars{detail}.",
        inputs=(PRICE_FIELD,),
        params=(("window", window), ("skip", skip)),
        lookback=window,
        warmup=window,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(ta.momentum(group[PRICE_FIELD], window, skip=skip), window),
        ),
    )


def _ma_distance_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"ma_dist_{window}",
        version="1.0.0",
        family=FeatureFamily.TREND,
        scope=Scope.PER_ASSET,
        unit=Unit.RATIO_DEVIATION,
        description=f"Price relative to its {window}-bar simple moving average (price/MA - 1).",
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(ta.ma_ratio(group[PRICE_FIELD], window), window),
        ),
    )


def _ema_distance_contract(span: int) -> FeatureContract:
    def kernel(panel: pd.DataFrame) -> pd.Series:
        return _per_asset(
            panel,
            lambda group: _mask_warmup(
                group[PRICE_FIELD] / ta.ema(group[PRICE_FIELD], span) - 1.0, span
            ),
        )

    return FeatureContract(
        name=f"ema_dist_{span}",
        version="1.0.0",
        family=FeatureFamily.TREND,
        scope=Scope.PER_ASSET,
        unit=Unit.RATIO_DEVIATION,
        description=(
            f"Price relative to its span-{span} exponential moving average "
            "(price/EMA - 1). The EMA has infinite memory; warm-up is the span."
        ),
        inputs=(PRICE_FIELD,),
        params=(("span", span),),
        lookback=span,
        warmup=span,
        kernel=kernel,
    )


def _zscore_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"zscore_{window}",
        version="1.0.0",
        family=FeatureFamily.MEAN_REVERSION,
        scope=Scope.PER_ASSET,
        unit=Unit.ZSCORE,
        description=f"Rolling {window}-bar z-score of adjusted close (mean-reversion signal).",
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(ta.zscore(group[PRICE_FIELD], window), window),
        ),
    )


def _reversal_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"reversal_{window}",
        version="1.0.0",
        family=FeatureFamily.MEAN_REVERSION,
        scope=Scope.PER_ASSET,
        unit=Unit.SIMPLE_RETURN,
        description=f"Negated {window}-bar return (short-horizon reversal signal).",
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(-ta.simple_returns(group[PRICE_FIELD], window), window),
        ),
    )


def _vwap_deviation_contract(window: int) -> FeatureContract:
    def kernel(panel: pd.DataFrame) -> pd.Series:
        return _per_asset(
            panel,
            lambda group: _mask_warmup(
                ta.vwap_deviation(
                    group["high"], group["low"], group["close"], group["volume"], window
                ),
                window,
            ),
        )

    return FeatureContract(
        name=f"vwap_dev_{window}",
        version="1.0.0",
        family=FeatureFamily.MEAN_REVERSION,
        scope=Scope.PER_ASSET,
        unit=Unit.RATIO_DEVIATION,
        description=f"Close relative to its {window}-bar VWAP (close/VWAP - 1).",
        inputs=("high", "low", "close", "volume"),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        kernel=kernel,
    )


def _residual_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"resid_{window}",
        version="1.0.0",
        family=FeatureFamily.MEAN_REVERSION,
        scope=Scope.PER_ASSET,
        unit=Unit.LOG_RESIDUAL,
        description=(f"Residual of adjusted close from a rolling {window}-bar log-linear trend."),
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: ta.rolling_regression_residual(group[PRICE_FIELD], window),
        ),
    )


def _cross_sectional_rank_contract(window: int) -> FeatureContract:
    def kernel(panel: pd.DataFrame) -> pd.Series:
        base = _cross_sectional_base(panel, window)
        frame = pd.DataFrame({DATE_COL: panel[DATE_COL].to_numpy(), "_base": base.to_numpy()})
        return cross_sectional_rank(frame, "_base")

    return FeatureContract(
        name=f"cs_rank_mom_{window}",
        version="1.0.0",
        family=FeatureFamily.MOMENTUM,
        scope=Scope.CROSS_SECTIONAL,
        unit=Unit.RANK_PCT,
        description=(
            f"Cross-sectional percentile rank of {window}-bar momentum across the "
            "universe on each date."
        ),
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        numerical_range=NumericalRange(lower=0.0, upper=1.0),
        kernel=kernel,
    )


def _cross_sectional_zscore_contract(window: int) -> FeatureContract:
    def kernel(panel: pd.DataFrame) -> pd.Series:
        base = _cross_sectional_base(panel, window)
        frame = pd.DataFrame({DATE_COL: panel[DATE_COL].to_numpy(), "_base": base.to_numpy()})
        return pd.to_numeric(cross_sectional_zscore(frame, "_base"), errors="coerce")

    return FeatureContract(
        name=f"cs_z_mom_{window}",
        version="1.0.0",
        family=FeatureFamily.MOMENTUM,
        scope=Scope.CROSS_SECTIONAL,
        unit=Unit.ZSCORE,
        description=(
            f"Cross-sectional z-score of {window}-bar momentum across the universe " "on each date."
        ),
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        kernel=kernel,
    )


# ---------------------------------------------------------------------------
# Statistical contract builders (SF-S2-MR2): volatility, distribution, dependence
# ---------------------------------------------------------------------------

_NON_NEGATIVE = NumericalRange(lower=0.0, upper=None)
_CORRELATION_RANGE = NumericalRange(lower=-1.0, upper=1.0)


def _returns(group: pd.DataFrame) -> pd.Series:
    """One-bar simple adjusted-close return of a single-asset group."""
    return ta.simple_returns(group[PRICE_FIELD], 1)


def _market_return(panel: pd.DataFrame) -> pd.Series:
    """Equal-weight cross-sectional mean of one-bar returns (market proxy).

    Self-contained: needs no designated benchmark ticker. On a single-name date
    it degenerates to that name's own return.
    """
    asset_returns = _per_asset(panel, _returns)
    frame = pd.DataFrame(
        {DATE_COL: panel[DATE_COL].to_numpy(), "_ret": asset_returns.to_numpy()},
        index=panel.index,
    )
    return frame.groupby(DATE_COL)["_ret"].transform("mean")


def _parkinson_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"parkinson_vol_{window}",
        version="1.0.0",
        family=FeatureFamily.VOLATILITY,
        scope=Scope.PER_ASSET,
        unit=Unit.ANNUALIZED_VOL,
        description=f"Annualised Parkinson high-low range volatility over {window} bars.",
        inputs=("high", "low"),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(
                st.parkinson_volatility(group["high"], group["low"], window), window
            ),
        ),
    )


def _garman_klass_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"garman_klass_vol_{window}",
        version="1.0.0",
        family=FeatureFamily.VOLATILITY,
        scope=Scope.PER_ASSET,
        unit=Unit.ANNUALIZED_VOL,
        description=f"Annualised Garman-Klass OHLC volatility over {window} bars.",
        inputs=("open", "high", "low", "close"),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(
                st.garman_klass_volatility(
                    group["open"], group["high"], group["low"], group["close"], window
                ),
                window,
            ),
        ),
    )


def _rogers_satchell_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"rogers_satchell_vol_{window}",
        version="1.0.0",
        family=FeatureFamily.VOLATILITY,
        scope=Scope.PER_ASSET,
        unit=Unit.ANNUALIZED_VOL,
        description=f"Annualised drift-independent Rogers-Satchell volatility over {window} bars.",
        inputs=("open", "high", "low", "close"),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(
                st.rogers_satchell_volatility(
                    group["open"], group["high"], group["low"], group["close"], window
                ),
                window,
            ),
        ),
    )


def _ewma_vol_contract(span: int) -> FeatureContract:
    return FeatureContract(
        name=f"ewma_vol_{span}",
        version="1.0.0",
        family=FeatureFamily.VOLATILITY,
        scope=Scope.PER_ASSET,
        unit=Unit.ANNUALIZED_VOL,
        description=f"Annualised exponentially weighted return volatility (span {span}).",
        inputs=(PRICE_FIELD,),
        params=(("span", span),),
        lookback=span,
        warmup=span,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(st.ewma_volatility(_returns(group), span), span),
        ),
    )


def _skewness_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"skew_{window}",
        version="1.0.0",
        family=FeatureFamily.DISTRIBUTION,
        scope=Scope.PER_ASSET,
        unit=Unit.DIMENSIONLESS,
        description=f"Rolling {window}-bar sample skewness of returns.",
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(st.rolling_skewness(_returns(group), window), window),
        ),
    )


def _kurtosis_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"kurt_{window}",
        version="1.0.0",
        family=FeatureFamily.DISTRIBUTION,
        scope=Scope.PER_ASSET,
        unit=Unit.DIMENSIONLESS,
        description=f"Rolling {window}-bar excess kurtosis of returns.",
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(st.rolling_kurtosis(_returns(group), window), window),
        ),
    )


def _downside_deviation_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"downside_dev_{window}",
        version="1.0.0",
        family=FeatureFamily.DISTRIBUTION,
        scope=Scope.PER_ASSET,
        unit=Unit.ANNUALIZED_VOL,
        description=f"Annualised downside deviation of returns over {window} bars.",
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(st.downside_deviation(_returns(group), window), window),
        ),
    )


def _mad_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"mad_{window}",
        version="1.0.0",
        family=FeatureFamily.DISTRIBUTION,
        scope=Scope.PER_ASSET,
        unit=Unit.SIMPLE_RETURN,
        description=f"Robust median absolute deviation of returns over {window} bars.",
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(
                st.median_absolute_deviation(_returns(group), window), window
            ),
        ),
    )


def _autocorrelation_contract(window: int, lag: int) -> FeatureContract:
    warmup = window + lag
    return FeatureContract(
        name=f"autocorr_{lag}_{window}",
        version="1.0.0",
        family=FeatureFamily.DEPENDENCE,
        scope=Scope.PER_ASSET,
        unit=Unit.CORRELATION,
        description=f"Rolling {window}-bar lag-{lag} return autocorrelation.",
        inputs=(PRICE_FIELD,),
        params=(("window", window), ("lag", lag)),
        lookback=warmup,
        warmup=warmup,
        numerical_range=_CORRELATION_RANGE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(
                st.rolling_autocorrelation(_returns(group), window, lag), warmup
            ),
        ),
    )


def _partial_autocorrelation_contract(window: int) -> FeatureContract:
    warmup = window + 2
    return FeatureContract(
        name=f"pacf_2_{window}",
        version="1.0.0",
        family=FeatureFamily.DEPENDENCE,
        scope=Scope.PER_ASSET,
        unit=Unit.CORRELATION,
        description=f"Rolling {window}-bar lag-2 partial autocorrelation of returns.",
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=warmup,
        warmup=warmup,
        numerical_range=_CORRELATION_RANGE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(
                st.partial_autocorrelation_lag2(_returns(group), window), warmup
            ),
        ),
    )


def _hurst_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"hurst_{window}",
        version="1.0.0",
        family=FeatureFamily.DEPENDENCE,
        scope=Scope.PER_ASSET,
        unit=Unit.DIMENSIONLESS,
        description=(
            f"Rolling {window}-bar Hurst exponent of log price via the structure "
            "function (~0.5 random walk, >0.5 trending, <0.5 mean-reverting)."
        ),
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(st.hurst_exponent(group[PRICE_FIELD], window), window),
        ),
    )


def _variance_ratio_contract(window: int, q: int) -> FeatureContract:
    return FeatureContract(
        name=f"var_ratio_{q}_{window}",
        version="1.0.0",
        family=FeatureFamily.DEPENDENCE,
        scope=Scope.PER_ASSET,
        unit=Unit.DIMENSIONLESS,
        description=f"Rolling {window}-bar Lo-MacKinlay variance ratio at horizon q={q}.",
        inputs=(PRICE_FIELD,),
        params=(("window", window), ("q", q)),
        lookback=window,
        warmup=window,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(st.variance_ratio(_returns(group), window, q), window),
        ),
    )


def _market_beta_contract(window: int) -> FeatureContract:
    warmup = window + 1

    def kernel(panel: pd.DataFrame) -> pd.Series:
        market = _market_return(panel)

        def per(group: pd.DataFrame) -> pd.Series:
            beta = ta.rolling_beta(_returns(group), market.loc[group.index], window)
            return _mask_warmup(beta, warmup)

        return _per_asset(panel, per)

    return FeatureContract(
        name=f"beta_mkt_{window}",
        version="1.0.0",
        family=FeatureFamily.DEPENDENCE,
        scope=Scope.PER_ASSET,
        unit=Unit.DIMENSIONLESS,
        description=(f"Rolling {window}-bar beta of returns to the equal-weight market proxy."),
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=warmup,
        warmup=warmup,
        kernel=kernel,
    )


def _market_correlation_contract(window: int) -> FeatureContract:
    warmup = window + 1

    def kernel(panel: pd.DataFrame) -> pd.Series:
        market = _market_return(panel)

        def per(group: pd.DataFrame) -> pd.Series:
            corr = st.rolling_correlation(_returns(group), market.loc[group.index], window)
            return _mask_warmup(corr, warmup)

        return _per_asset(panel, per)

    return FeatureContract(
        name=f"corr_mkt_{window}",
        version="1.0.0",
        family=FeatureFamily.DEPENDENCE,
        scope=Scope.PER_ASSET,
        unit=Unit.CORRELATION,
        description=(
            f"Rolling {window}-bar correlation of returns to the equal-weight market proxy."
        ),
        inputs=(PRICE_FIELD,),
        params=(("window", window),),
        lookback=warmup,
        warmup=warmup,
        numerical_range=_CORRELATION_RANGE,
        kernel=kernel,
    )


def _market_mutual_information_contract(window: int, bins: int) -> FeatureContract:
    warmup = window + 1

    def kernel(panel: pd.DataFrame) -> pd.Series:
        market = _market_return(panel)

        def per(group: pd.DataFrame) -> pd.Series:
            info = st.rolling_mutual_information(
                _returns(group), market.loc[group.index], window, bins=bins
            )
            return _mask_warmup(info, warmup)

        return _per_asset(panel, per)

    return FeatureContract(
        name=f"mutual_info_mkt_{window}",
        version="1.0.0",
        family=FeatureFamily.DEPENDENCE,
        scope=Scope.PER_ASSET,
        unit=Unit.INFORMATION_NATS,
        description=(
            f"Rolling {window}-bar binned mutual information between returns and the "
            "equal-weight market proxy."
        ),
        inputs=(PRICE_FIELD,),
        params=(("window", window), ("bins", bins)),
        lookback=warmup,
        warmup=warmup,
        numerical_range=_NON_NEGATIVE,
        kernel=kernel,
    )


def statistical_contracts() -> tuple[FeatureContract, ...]:
    """The frozen SF-S2-MR2 volatility, distribution, and dependence contracts."""
    return (
        # Volatility.
        _parkinson_contract(21),
        _garman_klass_contract(21),
        _rogers_satchell_contract(21),
        _ewma_vol_contract(21),
        # Distribution.
        _skewness_contract(63),
        _kurtosis_contract(63),
        _downside_deviation_contract(63),
        _mad_contract(63),
        # Dependence / memory.
        _autocorrelation_contract(63, 1),
        _partial_autocorrelation_contract(63),
        _hurst_contract(128),
        _variance_ratio_contract(63, 5),
        _market_beta_contract(63),
        _market_correlation_contract(63),
        _market_mutual_information_contract(63, 8),
    )


# ---------------------------------------------------------------------------
# Liquidity contract builders (SF-S2-MR3): volume, liquidity, spread, impact
# ---------------------------------------------------------------------------


def _volume_change_contract(periods: int) -> FeatureContract:
    warmup = periods + 1
    return FeatureContract(
        name=f"volume_change_{periods}",
        version="1.0.0",
        family=FeatureFamily.LIQUIDITY,
        scope=Scope.PER_ASSET,
        unit=Unit.DIMENSIONLESS,
        description=f"Log change in volume over {periods} bar(s).",
        inputs=("volume",),
        params=(("periods", periods),),
        lookback=warmup,
        warmup=warmup,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(lq.volume_change(group["volume"], periods), warmup),
        ),
    )


def _dollar_volume_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"dollar_volume_{window}",
        version="1.0.0",
        family=FeatureFamily.LIQUIDITY,
        scope=Scope.PER_ASSET,
        unit=Unit.LOG_DOLLAR_VOLUME,
        description=f"Trailing {window}-bar mean of log traded dollar value (liquidity level).",
        inputs=("close", "volume"),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(
                lq.rolling_log_dollar_volume(group["close"], group["volume"], window), window
            ),
        ),
    )


def _relative_volume_contract(window: int) -> FeatureContract:
    return FeatureContract(
        name=f"rel_volume_{window}",
        version="1.0.0",
        family=FeatureFamily.LIQUIDITY,
        scope=Scope.PER_ASSET,
        unit=Unit.DIMENSIONLESS,
        description=f"Volume relative to its trailing {window}-bar mean.",
        inputs=("volume",),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(lq.relative_volume(group["volume"], window), window),
        ),
    )


def _amihud_contract(window: int, scale: int = 1_000_000) -> FeatureContract:
    warmup = window + 1
    return FeatureContract(
        name=f"amihud_{window}",
        version="1.0.0",
        family=FeatureFamily.LIQUIDITY,
        scope=Scope.PER_ASSET,
        unit=Unit.ILLIQUIDITY,
        description=(
            f"Amihud illiquidity: trailing {window}-bar mean of |return| per "
            f"{scale:,} dollars traded (price-impact proxy)."
        ),
        inputs=("close", "volume"),
        params=(("window", window), ("scale", scale)),
        lookback=warmup,
        warmup=warmup,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(
                lq.amihud_illiquidity(group["close"], group["volume"], window, scale=scale),
                warmup,
            ),
        ),
    )


def _volume_imbalance_contract(window: int) -> FeatureContract:
    warmup = window + 1
    return FeatureContract(
        name=f"volume_imbalance_{window}",
        version="1.0.0",
        family=FeatureFamily.LIQUIDITY,
        scope=Scope.PER_ASSET,
        unit=Unit.DIMENSIONLESS,
        description=f"Signed-volume imbalance over {window} bars (return-sign proxy), in [-1, 1].",
        inputs=("close", "volume"),
        params=(("window", window),),
        lookback=warmup,
        warmup=warmup,
        numerical_range=_CORRELATION_RANGE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(
                lq.volume_imbalance(group["close"], group["volume"], window), warmup
            ),
        ),
    )


def _corwin_schultz_contract(window: int) -> FeatureContract:
    warmup = window + 1
    return FeatureContract(
        name=f"corwin_schultz_{window}",
        version="1.0.0",
        family=FeatureFamily.LIQUIDITY,
        scope=Scope.PER_ASSET,
        unit=Unit.SPREAD_FRACTION,
        description=f"Trailing {window}-bar Corwin-Schultz high-low spread estimate (fraction).",
        inputs=("high", "low"),
        params=(("window", window),),
        lookback=warmup,
        warmup=warmup,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(
                lq.corwin_schultz_spread(group["high"], group["low"], window), warmup
            ),
        ),
    )


def _roll_spread_contract(window: int) -> FeatureContract:
    warmup = window + 2
    return FeatureContract(
        name=f"roll_spread_{window}",
        version="1.0.0",
        family=FeatureFamily.LIQUIDITY,
        scope=Scope.PER_ASSET,
        unit=Unit.SPREAD_FRACTION,
        description=f"Trailing {window}-bar Roll implied effective spread (fraction).",
        inputs=("close",),
        params=(("window", window),),
        lookback=warmup,
        warmup=warmup,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(lq.roll_spread(group["close"], window), warmup),
        ),
    )


def _turnover_contract(window: int) -> FeatureContract:
    """Turnover requires a ``shares_outstanding`` column beyond OHLCV.

    It is registered (discoverable via :func:`get_contract`) but deliberately
    excluded from :func:`default_contracts`, so a plain OHLCV panel still builds;
    request it explicitly and supply the extra column.
    """
    return FeatureContract(
        name=f"turnover_{window}",
        version="1.0.0",
        family=FeatureFamily.LIQUIDITY,
        scope=Scope.PER_ASSET,
        unit=Unit.DIMENSIONLESS,
        description=(
            f"Trailing {window}-bar mean share turnover (volume / shares_outstanding). "
            "Requires a shares_outstanding column."
        ),
        inputs=("volume", "shares_outstanding"),
        params=(("window", window),),
        lookback=window,
        warmup=window,
        numerical_range=_NON_NEGATIVE,
        kernel=lambda panel: _per_asset(
            panel,
            lambda group: _mask_warmup(
                lq.turnover(group["volume"], group["shares_outstanding"], window), window
            ),
        ),
    )


def liquidity_contracts() -> tuple[FeatureContract, ...]:
    """The frozen SF-S2-MR3 OHLCV-computable volume, liquidity, and spread contracts.

    Turnover is excluded here because it needs an extra ``shares_outstanding``
    column; obtain it via ``get_contract("turnover_21")``.
    """
    return (
        _volume_change_contract(1),
        _dollar_volume_contract(21),
        _relative_volume_contract(21),
        _amihud_contract(21),
        _volume_imbalance_contract(21),
        _corwin_schultz_contract(21),
        _roll_spread_contract(21),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def default_contracts() -> tuple[FeatureContract, ...]:
    """The frozen default set of SF-S2-MR1, MR2, and MR3 feature contracts.

    Contains only contracts computable from a canonical OHLCV panel. Turnover
    (which needs ``shares_outstanding``) is registered but not included here.
    """
    return (
        # Returns.
        _return_contract(1),
        _return_contract(5),
        _return_contract(21),
        _log_return_contract(),
        # Momentum / rate of change.
        _rate_of_change_contract(10),
        _momentum_contract(63),
        _momentum_contract(126),
        _momentum_contract(252),
        _momentum_contract(252, skip=21),
        # Trend.
        _ma_distance_contract(20),
        _ma_distance_contract(50),
        _ma_distance_contract(200),
        _ema_distance_contract(12),
        _ema_distance_contract(26),
        # Mean reversion.
        _zscore_contract(21),
        _reversal_contract(5),
        _vwap_deviation_contract(21),
        _residual_contract(63),
        # Cross-sectional.
        _cross_sectional_rank_contract(63),
        _cross_sectional_zscore_contract(63),
        # Volatility, distribution, and dependence (SF-S2-MR2).
        *statistical_contracts(),
        # Volume, liquidity, spread, and impact (SF-S2-MR3).
        *liquidity_contracts(),
    )


def _build_registry() -> dict[str, FeatureContract]:
    # The registry is discoverable via get_contract/list_contracts and includes
    # data-dependent contracts (turnover) that are not in the default OHLCV set.
    registry: dict[str, FeatureContract] = {}
    for contract in (*default_contracts(), _turnover_contract(21)):
        if contract.name in registry:  # pragma: no cover - guards developer error
            raise ValueError(f"duplicate feature contract name '{contract.name}'")
        registry[contract.name] = contract
    return registry


CONTRACT_REGISTRY: dict[str, FeatureContract] = _build_registry()


def list_contracts() -> tuple[str, ...]:
    """Return the registered contract names in deterministic definition order."""
    return tuple(CONTRACT_REGISTRY.keys())


def get_contract(name: str) -> FeatureContract:
    """Look up a contract by name, raising ``KeyError`` if it is unknown."""
    try:
        return CONTRACT_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown feature contract '{name}'; known contracts: {list_contracts()}"
        ) from exc


def contract_metadata_frame(
    contracts: Iterable[FeatureContract] | None = None,
) -> pd.DataFrame:
    """Return a table of contract metadata for documentation and inspection."""
    chosen = list(contracts) if contracts is not None else list(default_contracts())
    return pd.DataFrame([contract.metadata() for contract in chosen])


# ---------------------------------------------------------------------------
# Panel validation and the public build entry point
# ---------------------------------------------------------------------------


def validate_contract_panel(panel: pd.DataFrame, required_inputs: Sequence[str]) -> pd.DataFrame:
    """Validate and normalise a long price panel for contract computation.

    Returns a defensively-copied, ``(ticker, date)``-sorted frame with a default
    range index. Fails closed on missing keys/inputs, non-finite ``inf`` values,
    and duplicate ``(date, ticker)`` observations.
    """
    if panel.empty:
        raise ValueError("cannot compute feature contracts from an empty panel")

    required = [DATE_COL, TICKER_COL, *required_inputs]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")

    out = panel.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])

    if out.duplicated([DATE_COL, TICKER_COL]).any():
        raise ValueError("panel contains duplicate (date, ticker) observations")

    for column in required_inputs:
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"input column '{column}' contains non-finite (inf) values")

    return out.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)


def build_contract_features(
    panel: pd.DataFrame,
    contracts: Iterable[FeatureContract] | None = None,
    *,
    prefix: str = CONTRACT_PREFIX,
) -> pd.DataFrame:
    """Compute a namespaced feature matrix from a long price panel.

    Parameters
    ----------
    panel:
        Long-format OHLCV panel (see :mod:`quant_platform.data.schema`).
    contracts:
        Contracts to evaluate; defaults to :func:`default_contracts`.
    prefix:
        Column-name prefix for produced features (default ``"fc_"``).

    Returns
    -------
    pandas.DataFrame
        ``date``, ``ticker`` and one ``{prefix}{name}`` column per contract,
        row-aligned to the validated (sorted) panel. Values are ``NaN`` during
        each feature's warm-up window and wherever inputs were missing.
    """
    chosen = list(contracts) if contracts is not None else list(default_contracts())
    if not chosen:
        raise ValueError("at least one feature contract is required")

    required = sorted({column for contract in chosen for column in contract.inputs})
    validated = validate_contract_panel(panel, required)

    result = validated[[DATE_COL, TICKER_COL]].copy()
    for contract in chosen:
        result[f"{prefix}{contract.name}"] = contract.compute(validated).to_numpy(dtype=float)
    return result
