"""Causal spectral transform and descriptor engine (SF-S3-MR1).

Turns a validated price panel into versioned, leakage-safe frequency-domain
features. This module is the composition layer: :mod:`spectral_transforms`
supplies the causal mathematics, :mod:`spectral_descriptors` reduces a spectrum
to interpretable numbers, :class:`~quant_platform.config.SpectralConfig` declares
the window, and this module decides *what* is analysed and *how it is
registered*.

Four properties are enforced here rather than left to the caller:

**Causality.** Every value at bar *t* is a function of ``x[t-L+1 .. t]`` only.
Windows are trailing, never centred. The first ``L-1`` bars of each ticker are
NaN — a warm-up, not a zero — so a partially converged descriptor cannot be
mistaken for a converged one, and per-ticker grouping means one asset's history
can never enter another's window.

**A declared window contract.** :class:`~quant_platform.config.SpectralWindow`
records length, segment length, hop, overlap, padding, FFT size, sampling
frequency, frequency unit, detrend, taper, and warm-up, and every one of those
is copied onto every emitted feature's registry spec.

**Scale-free outputs.** Every descriptor is invariant to the amplitude of its
input channel (see :mod:`spectral_descriptors`), so the registered specs declare
``normalization="none"`` truthfully and carry no fitted state — there is no
train/test boundary here to get wrong. :func:`fit_training_normalizer` exists for
callers who nonetheless want standardized columns, and it fits strictly inside a
supplied training interval and records that interval.

**Bounded compute.** Window length, FFT size, and the total window tensor are
capped, so a configuration mistake fails fast instead of exhausting memory in the
middle of a backfill.

Order-imbalance channels named in the sprint plan are **not** implemented: the
canonical panel carries OHLCV only, with no quote or trade-side data. Emitting a
proxy and calling it imbalance would be inventing evidence.
"""

from __future__ import annotations

import hashlib
import inspect
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quant_platform.config import SpectralConfig
from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.features import spectral_descriptors as sd
from quant_platform.features import spectral_transforms as st
from quant_platform.features import technical as ta
from quant_platform.features.registry import (
    FeatureRegistry,
    FeatureSpec,
    FittedTransformState,
    semantic_hash,
)

FloatArray = NDArray[np.float64]

#: Prefix for every column this engine emits.
SPECTRAL_PREFIX = "f_spec_"

#: Ceiling on the total causal-window tensor (bars x length x channels). This is
#: a refusal threshold, not a tuning knob: a request past it indicates a
#: configuration mistake, and accepting it silently would turn a typo into an
#: out-of-memory kill part-way through a long backfill.
MAX_WINDOW_CELLS = 200_000_000

#: Panel columns each analysis channel derives from.
CHANNEL_INPUTS: dict[str, tuple[str, ...]] = {
    "return": ("return",),
    "volatility": ("return",),
    "volume": ("volume",),
    "residual": ("return",),
}


def spectral_column_names(config: SpectralConfig) -> tuple[str, ...]:
    """Return every feature column the configuration emits, in stable order."""
    names: list[str] = []
    for channel in config.channels:
        for descriptor in sd.descriptor_column_names(config.descriptors):
            names.append(f"{SPECTRAL_PREFIX}{channel}_{descriptor}")
        names.append(f"{SPECTRAL_PREFIX}{channel}_flux")
        if channel in config.wavelet_channels:
            names.append(f"{SPECTRAL_PREFIX}{channel}_cwt_peak_period")
            names.append(f"{SPECTRAL_PREFIX}{channel}_cwt_concentration")
            for level in range(1, config.dwt_levels + 1):
                names.append(f"{SPECTRAL_PREFIX}{channel}_dwt_d{level}")
            names.append(f"{SPECTRAL_PREFIX}{channel}_dwt_approx")
    return tuple(names)


def channel_lookback(config: SpectralConfig, channel: str) -> int:
    """Total bars of history one channel's descriptors depend on.

    The spectral window sits on top of whatever trailing transform builds the
    channel, so a volatility or residual channel needs strictly more warm-up
    than a raw-return channel. Understating this is how a feature ends up
    materialized from a partition that never contained enough history.
    """
    extra = {
        "return": 0,
        "volume": 0,
        "volatility": config.volatility_window - 1,
        "residual": config.beta_window - 1,
    }[channel]
    return config.window.length + extra


def morlet_scales(config: SpectralConfig) -> FloatArray:
    """Return Morlet scales matching the configured target periods."""
    periods = np.asarray(config.cwt_periods, dtype=np.float64)
    return np.asarray(config.morlet_omega0 * periods / (2.0 * np.pi), dtype=np.float64)


def spectral_implementation_hash() -> str:
    """Hash the transform, descriptor, and channel-construction sources.

    Any change to the mathematics changes the identity of every feature it
    produces, so a materialized column can never be silently reinterpreted under
    a new implementation.
    """
    source = "\n".join(
        (
            inspect.getsource(st),
            inspect.getsource(sd),
            inspect.getsource(_channel_series),
            inspect.getsource(_channel_features),
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _channel_series(
    group: pd.DataFrame,
    channel: str,
    config: SpectralConfig,
    market_returns: pd.Series,
) -> pd.Series:
    """Build one analysis channel for a single ticker, from trailing data only.

    * ``return`` — the raw return series.
    * ``volatility`` — trailing standard deviation of returns, so the engine
      describes structure *in the volatility process* rather than in returns.
    * ``volume`` — ``log1p(volume)``, compressing the heavy right tail so one
      exceptional print cannot dominate a window's spectrum.
    * ``residual`` — market-relative return ``r - beta * r_market`` using the
      trailing rolling beta already defined for the conventional features.
      Removing the common factor is what stops every asset's spectrum from
      being a mildly rescaled copy of the index's.
    """
    returns = group["return"]
    if channel == "return":
        return returns
    if channel == "volatility":
        return returns.rolling(config.volatility_window, min_periods=config.volatility_window).std()
    if channel == "volume":
        return pd.Series(np.log1p(group["volume"].to_numpy(dtype=np.float64)), index=group.index)
    market = group[DATE_COL].map(market_returns)
    market.index = group.index
    beta = ta.rolling_beta(returns, market, config.beta_window)
    return returns - beta * market


def _channel_features(
    values: FloatArray, config: SpectralConfig, channel: str
) -> dict[str, FloatArray]:
    """Compute every descriptor for one channel of one ticker."""
    window = config.window
    frequencies = np.asarray(window.frequencies(), dtype=np.float64)
    windows = st.causal_windows(values, window.length)
    psd = st.welch_psd(
        windows,
        segment_length=window.segment_length,
        hop=window.hop,
        n_fft=window.n_fft,
        sampling_frequency=window.sampling_frequency,
        taper_name=window.taper,
        detrend=window.detrend,
    )
    columns = {
        f"{SPECTRAL_PREFIX}{channel}_{name}": series
        for name, series in sd.describe(psd, frequencies, config.descriptors).items()
    }
    magnitudes = st.stft(
        windows,
        segment_length=window.segment_length,
        hop=window.hop,
        n_fft=window.n_fft,
        taper_name=window.taper,
        detrend=window.detrend,
    )
    columns[f"{SPECTRAL_PREFIX}{channel}_flux"] = sd.spectral_flux(magnitudes)

    if channel in config.wavelet_channels:
        power = st.cwt_power(windows, morlet_scales(config), config.morlet_omega0)
        total = power.sum(axis=-1)
        valid = np.isfinite(power).all(axis=-1) & (total > 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = power / np.where(total > 0.0, total, np.nan)[:, None]
        periods = np.asarray(config.cwt_periods, dtype=np.float64)
        peak = np.argmax(np.where(np.isfinite(power), power, -np.inf), axis=-1)
        columns[f"{SPECTRAL_PREFIX}{channel}_cwt_peak_period"] = np.where(
            valid, periods[peak], np.nan
        )
        columns[f"{SPECTRAL_PREFIX}{channel}_cwt_concentration"] = np.where(
            valid, np.max(np.where(np.isfinite(share), share, -np.inf), axis=-1), np.nan
        )
        energies = st.dwt_energies(windows, config.wavelet, config.dwt_levels)
        for level in range(config.dwt_levels):
            columns[f"{SPECTRAL_PREFIX}{channel}_dwt_d{level + 1}"] = energies[:, level]
        columns[f"{SPECTRAL_PREFIX}{channel}_dwt_approx"] = energies[:, -1]
    return columns


def build_spectral_features(
    panel: pd.DataFrame,
    config: SpectralConfig,
    *,
    benchmark: str,
) -> pd.DataFrame:
    """Compute causal spectral features for every ticker in a price panel.

    Args:
        panel: Long-format panel carrying ``date``, ``ticker``, ``return``, and
            ``volume``.
        config: Engine configuration; must be enabled.
        benchmark: Ticker used as the market proxy for the residual channel.

    Returns:
        A frame indexed exactly like ``panel``, holding only ``f_spec_*``
        columns in :func:`spectral_column_names` order. Warm-up bars are NaN.

    Raises:
        ValueError: If the engine is disabled, the panel is empty or missing a
            required column, the benchmark is absent while the residual channel
            is requested, or the request exceeds the compute ceiling.
    """
    if not config.enabled:
        raise ValueError("spectral engine is disabled; enable it before building features")
    if panel.empty:
        raise ValueError("cannot build spectral features from an empty panel")
    required = {DATE_COL, TICKER_COL, "return", "volume"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")

    cells = len(panel) * config.window.length * len(config.channels)
    if cells > MAX_WINDOW_CELLS:
        raise ValueError(
            f"requested spectral window tensor of {cells} cells exceeds the "
            f"{MAX_WINDOW_CELLS} ceiling; reduce window length, channels, or panel size"
        )

    ordered = panel.sort_values([TICKER_COL, DATE_COL])
    market = (
        ordered.loc[ordered[TICKER_COL] == benchmark, [DATE_COL, "return"]]
        .drop_duplicates(DATE_COL)
        .set_index(DATE_COL)["return"]
    )
    if market.empty and "residual" in config.channels:
        # Fail closed: an all-NaN residual channel would look like a warm-up
        # artefact rather than a missing benchmark.
        raise ValueError(
            f"residual channel requires benchmark {benchmark!r}, which is absent from the panel"
        )

    frames: list[pd.DataFrame] = []
    for _ticker, group in ordered.groupby(TICKER_COL, sort=False):
        columns: dict[str, FloatArray] = {}
        for channel in config.channels:
            values = _channel_series(group, channel, config, market).to_numpy(dtype=np.float64)
            columns.update(_channel_features(values, config, channel))
        frames.append(pd.DataFrame(columns, index=group.index))
    features = pd.concat(frames).reindex(panel.index)
    return features.loc[:, list(spectral_column_names(config))]


def spectral_feature_registry(
    config: SpectralConfig,
    *,
    dropna: bool = True,
) -> FeatureRegistry:
    """Describe every column the spectral engine emits.

    Each spec records the full window contract in ``parameters``, so a stored
    feature can be traced back to the exact transform geometry that produced it.

    ``normalization`` is ``"none"`` because every descriptor is scale-free by
    construction — see :mod:`spectral_descriptors`. That is a substantive claim,
    not an unexamined default: it is what makes these columns free of fitted
    state, and therefore free of any train/test boundary to violate.

    Raises:
        ValueError: If the engine is disabled.
    """
    if not config.enabled:
        raise ValueError("cannot register features for a disabled spectral engine")
    implementation = spectral_implementation_hash()
    missing_policy: Literal["drop", "preserve"] = "drop" if dropna else "preserve"
    window_parameters = _window_parameters(config)
    specs: list[FeatureSpec] = []
    for channel in config.channels:
        lookback = channel_lookback(config, channel)
        prefix = f"{SPECTRAL_PREFIX}{channel}_"
        # The residual channel consumes the benchmark's returns as well as the
        # asset's, so it carries a cross-asset dependency the others do not.
        risk: Literal["low", "medium", "high"] = "medium" if channel == "residual" else "low"
        for name in spectral_column_names(config):
            if not name.startswith(prefix):
                continue
            specs.append(
                FeatureSpec(
                    name=name,
                    family="spectral",
                    input_columns=CHANNEL_INPUTS[channel],
                    parameters={**window_parameters, "channel": channel},
                    lookback_bars=lookback,
                    warmup_bars=lookback,
                    normalization="none",
                    missing_policy=missing_policy,
                    sampling_frequency="1d",
                    leakage_risk=risk,
                    implementation_sha256=implementation,
                )
            )
    return FeatureRegistry(specs)


def _window_parameters(config: SpectralConfig) -> dict[str, str | int | float | bool | None]:
    """Return the JSON-safe window and descriptor parameters for a spec."""
    window = config.window
    return {
        "length": window.length,
        "segment_length": window.segment_length,
        "hop": window.hop,
        "overlap": window.overlap,
        "n_segments": window.n_segments,
        "n_fft": window.n_fft,
        "padding": window.padding,
        "sampling_frequency": window.sampling_frequency,
        "frequency_unit": window.frequency_unit,
        "warmup_bars": window.warmup_bars,
        "detrend": window.detrend,
        "taper": window.taper,
        "rolloff_quantile": config.descriptors.rolloff_quantile,
        "concentration_bins": config.descriptors.concentration_bins,
        "bands": ",".join(band.name for band in config.descriptors.bands),
    }


def fit_training_normalizer(
    features: pd.DataFrame,
    dates: pd.Series,
    *,
    fit_start: date,
    fit_end: date,
) -> tuple[pd.Series, pd.Series, FittedTransformState]:
    """Fit standardization statistics on a closed training interval only.

    The engine's own descriptors need no normalization, so this exists for
    callers who want standardized inputs for a scale-sensitive model. It is an
    explicit, interval-recording function precisely because the tempting
    alternative — standardizing over the whole frame — is a leakage bug that
    produces better validation numbers and no warning at all.

    Args:
        features: Spectral feature columns.
        dates: Observation dates aligned row-wise to ``features``.
        fit_start: First date (inclusive) admitted to the fit.
        fit_end: Last date (inclusive) admitted to the fit.

    Returns:
        ``(mean, std, state)``, where ``state`` records the fit interval, the
        sample count, and a hash of the fitted statistics.

    Raises:
        ValueError: If the interval is inverted, the inputs are misaligned, or
            the interval selects no usable rows.
    """
    if fit_end < fit_start:
        raise ValueError("fit_end must be on or after fit_start")
    if len(features) != len(dates):
        raise ValueError("features and dates must describe the same observations")
    stamps = pd.to_datetime(dates).dt.date
    selected = ((stamps >= fit_start) & (stamps <= fit_end)).to_numpy()
    training = features.loc[selected]
    if training.empty or not bool(training.notna().any().any()):
        raise ValueError("training interval selected no usable spectral observations")
    mean = training.mean()
    std = training.std(ddof=0)
    payload = {
        "mean": {str(name): _finite_or_none(value) for name, value in mean.items()},
        "std": {str(name): _finite_or_none(value) for name, value in std.items()},
    }
    state = FittedTransformState(
        method="spectral_zscore",
        state_sha256=semantic_hash(payload),
        fit_start=fit_start,
        fit_end=fit_end,
        sample_count=int(len(training)),
    )
    return mean, std, state


def _finite_or_none(value: object) -> float | None:
    """Return a JSON-safe finite float, or ``None`` for a degenerate statistic."""
    number = float(value)  # type: ignore[arg-type]
    return number if np.isfinite(number) else None
