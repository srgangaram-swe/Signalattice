"""Versioned time-frequency tensors: spectrograms and scalograms (SF-S3-MR2).

SF-S3-MR1 reduced a causal window to a handful of scalars. This module keeps the
*whole* time-frequency surface instead — a dense tensor per observation — for
consumers that want the structure rather than a summary.

A tensor is a different kind of artifact from a feature column, and is treated
as one: it is never joined into the feature matrix, it is content-addressed and
cached separately, and it carries its own immutable metadata, missingness mask,
and normalization state.

Shape is always ``(n_samples, n_channels, n_frequency, n_time)``:

* **samples** — one per ``(ticker, date)`` observation, in the order of the
  supplied alignment index.
* **channels** — return / volatility / volume / residual, sharing one frequency
  grid so a multi-channel tensor is genuinely stackable rather than a ragged
  collection glued together.
* **frequency** — one-sided FFT bins (spectrogram) or Morlet scales expressed as
  periods (scalogram), always ascending.
* **time** — sub-segments *inside* the causal window, oldest first, so index
  ``-1`` is always the most recent slice.

Four properties are enforced here:

**Causality.** Every slice comes from :mod:`spectral_transforms`' trailing
windows. Warm-up samples are not padded with anything — they are masked out and
their values are NaN. There is no centred window and no forward fill.

**An explicit mask.** ``mask[i, c]`` is ``False`` when that sample's window was
incomplete or contained a NaN. A consumer that ignores the mask sees NaN, not a
plausible zero; there is no silent imputation anywhere.

**Train-only normalization.** Power spans orders of magnitude and is heavily
right-skewed, so the default normalization is log-power followed by a per
``(channel, frequency)`` z-score. Its statistics are fit on an explicitly
supplied training interval and recorded as a
:class:`~quant_platform.features.registry.FittedTransformState`. Fitting over
the whole tensor is the leakage bug this design exists to make impossible.

**Bounded size.** Cell count and serialized bytes are both capped before
anything is allocated or written.

Rendering lives in :mod:`quant_platform.reporting.time_frequency_plots`. Per the
issue's non-goals, an image is an inspection aid; it is not evidence of
predictability, and nothing here trains an image model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_platform.config import TimeFrequencyConfig
from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.features import spectral_transforms as st
from quant_platform.features.registry import SHA256_RE, FittedTransformState, semantic_hash
from quant_platform.features.spectral import _channel_series

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: Tensor schema version. Bumped when the axis meaning or ordering changes.
TENSOR_SCHEMA_VERSION = "1.0.0"

#: Floor applied before taking a logarithm of power. Power can legitimately
#: underflow to zero in a quiet band; ``log(0)`` would produce ``-inf`` and
#: poison every downstream statistic, so the floor is stated, recorded in the
#: metadata, and applied identically at fit and apply time.
LOG_POWER_FLOOR = 1e-300


class TimeFrequencyError(RuntimeError):
    """Base class for time-frequency tensor failures."""


class TensorIntegrityError(TimeFrequencyError):
    """Raised when stored bytes do not match their recorded digest."""


class TensorBoundsError(TimeFrequencyError):
    """Raised when a request exceeds a declared resource ceiling."""


class _Contract(BaseModel):
    """Strict immutable base for persisted tensor contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TimeFrequencyMetadata(_Contract):
    """Immutable description of one materialized tensor.

    Everything needed to interpret, reproduce, and verify the tensor lives here;
    the arrays themselves carry no self-describing information at all, which is
    exactly why the metadata must be complete and hash-bound to them.
    """

    schema_version: Literal["1.0.0"] = "1.0.0"
    representation: Literal["spectrogram", "scalogram"]
    channels: tuple[str, ...] = Field(min_length=1)
    #: Frequency axis in cycles per bar (spectrogram) or periods in bars
    #: (scalogram). Named by `frequency_axis` so the unit is never ambiguous.
    frequency_axis: Literal["cycles_per_bar", "period_bars"]
    frequency_values: tuple[float, ...] = Field(min_length=1)
    window_parameters: dict[str, str | int | float | bool | None]
    n_samples: int = Field(ge=1)
    n_time: int = Field(ge=1)
    coverage_start: date
    coverage_end: date
    tickers: tuple[str, ...] = Field(min_length=1)
    normalization: Literal["none", "train_log_zscore"]
    log_power_floor: float | None = None
    fitted_state: FittedTransformState | None = None
    observed_fraction: float = Field(ge=0.0, le=1.0)
    values_sha256: str
    mask_sha256: str

    @field_validator("values_sha256", "mask_sha256")
    @classmethod
    def _full_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("tensor digests must be lowercase full SHA-256 digests")
        return value

    @field_validator("frequency_values")
    @classmethod
    def _ascending_finite(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("frequency axis values must be finite")
        if not np.all(np.diff(array) > 0.0):
            raise ValueError("frequency axis values must be strictly ascending")
        return values

    @field_validator("channels", "tickers")
    @classmethod
    def _unique_identifiers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("identifiers must be unique")
        if any(not value or not value.replace("_", "").isalnum() for value in values):
            raise ValueError("identifiers must be non-empty alphanumeric")
        return values

    @model_validator(mode="after")
    def _consistent(self) -> TimeFrequencyMetadata:
        if self.coverage_end < self.coverage_start:
            raise ValueError("coverage_end must be on or after coverage_start")
        expects_state = self.normalization == "train_log_zscore"
        if expects_state and self.fitted_state is None:
            raise ValueError("train_log_zscore normalization requires a recorded fitted state")
        if not expects_state and self.fitted_state is not None:
            raise ValueError("fitted_state is only valid for a fitted normalization")
        if expects_state and self.log_power_floor is None:
            raise ValueError("a log normalization must record its power floor")
        if self.representation == "scalogram" and self.frequency_axis != "period_bars":
            raise ValueError("a scalogram's frequency axis is expressed in periods")
        if self.representation == "spectrogram" and self.frequency_axis != "cycles_per_bar":
            raise ValueError("a spectrogram's frequency axis is expressed in cycles per bar")
        return self

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return the declared tensor shape."""
        return (self.n_samples, len(self.channels), len(self.frequency_values), self.n_time)

    @property
    def identity(self) -> str:
        """Return the deterministic semantic identity of this tensor."""
        return semantic_hash(self.model_dump(mode="json"))


@dataclass(frozen=True)
class TimeFrequencyTensor:
    """A materialized time-frequency tensor with its mask and metadata.

    ``values`` and ``mask`` are plain arrays rather than model fields because a
    pydantic contract cannot validate a hundred megabytes of float meaningfully;
    integrity is instead bound by the digests recorded in ``metadata`` and
    checked by :meth:`verify`.
    """

    values: FloatArray
    mask: BoolArray
    metadata: TimeFrequencyMetadata

    def __post_init__(self) -> None:
        if self.values.shape != self.metadata.shape:
            raise TimeFrequencyError(
                f"tensor shape {self.values.shape} does not match metadata {self.metadata.shape}"
            )
        if self.mask.shape != self.metadata.shape[:2]:
            raise TimeFrequencyError(
                f"mask shape {self.mask.shape} does not match "
                f"(n_samples, n_channels) = {self.metadata.shape[:2]}"
            )

    def verify(self) -> None:
        """Re-hash the arrays and compare against the recorded digests.

        Raises:
            TensorIntegrityError: If either array no longer matches its digest.
        """
        if digest_array(self.values) != self.metadata.values_sha256:
            raise TensorIntegrityError("tensor values do not match their recorded digest")
        if digest_array(self.mask) != self.metadata.mask_sha256:
            raise TensorIntegrityError("tensor mask does not match its recorded digest")

    @property
    def nbytes(self) -> int:
        """Return the in-memory size of the arrays."""
        return int(self.values.nbytes + self.mask.nbytes)


def digest_array(array: np.ndarray) -> str:
    """Return a SHA-256 over an array's dtype, shape, and C-ordered bytes.

    dtype and shape are folded in deliberately: identical bytes reinterpreted
    under a different dtype or reshaped to different axes are a *different*
    tensor, and a digest that could not tell them apart would be useless as an
    integrity check.
    """
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}|{contiguous.shape}".encode()
    return hashlib.sha256(header + contiguous.tobytes()).hexdigest()


def _frequency_axis(config: TimeFrequencyConfig) -> tuple[str, FloatArray]:
    """Return the axis name and ascending grid for the configured representation."""
    if config.representation == "spectrogram":
        grid = np.asarray(config.window.frequencies(), dtype=np.float64)
        return "cycles_per_bar", grid
    return "period_bars", np.asarray(config.scalogram_periods, dtype=np.float64)


def _channel_surface(values: FloatArray, config: TimeFrequencyConfig) -> FloatArray:
    """Return the ``(n_bars, n_frequency, n_time)`` surface for one channel."""
    window = config.window
    windows = st.causal_windows(values, window.length)
    if config.representation == "spectrogram":
        magnitudes = st.stft(
            windows,
            segment_length=window.segment_length,
            hop=window.hop,
            n_fft=window.n_fft,
            taper_name=window.taper,
            detrend=window.detrend,
        )
        # STFT returns (bars, time, frequency); the tensor contract puts
        # frequency before time so a rendered image reads as frequency-by-time.
        return np.asarray(np.swapaxes(magnitudes**2, 1, 2), dtype=np.float64)
    return _scalogram_surface(windows, config)


def _scalogram_surface(windows: FloatArray, config: TimeFrequencyConfig) -> FloatArray:
    """Return causal Morlet power per scale across sub-window time positions.

    The scalogram's time axis is built by evaluating the causal wavelet at
    ``n_time`` positions stepping back from the most recent bar, so every column
    is itself a causal estimate over a strictly earlier sub-window. Stepping
    *back* rather than interpolating forward is what keeps the last column the
    only one that touches the evaluation bar.
    """
    window = config.window
    scales = config.morlet_omega0 * np.asarray(config.scalogram_periods, dtype=np.float64)
    scales = scales / (2.0 * np.pi)
    n_time = window.n_segments
    positions = [window.length - 1 - offset * window.hop for offset in range(n_time)][::-1]
    columns = []
    for end in positions:
        # Sub-window ending `end` bars into the causal window, same length as a
        # Welch segment so the two representations describe the same support.
        start = end - window.segment_length + 1
        sub = windows[:, start : end + 1]
        columns.append(st.cwt_power(sub, scales, config.morlet_omega0))
    return np.asarray(np.stack(columns, axis=-1), dtype=np.float64)


def build_time_frequency_tensor(
    panel: pd.DataFrame,
    config: TimeFrequencyConfig,
    *,
    benchmark: str,
) -> tuple[TimeFrequencyTensor, pd.DataFrame]:
    """Build a causal, masked, multi-channel time-frequency tensor.

    Args:
        panel: Long-format panel with ``date``, ``ticker``, ``return``, ``volume``.
        config: Enabled tensor configuration.
        benchmark: Market proxy ticker for the residual channel.

    Returns:
        ``(tensor, index)`` where ``index`` is the ``(date, ticker)`` alignment
        frame in tensor-sample order. The index is returned rather than embedded
        so a caller can join it without parsing metadata, and so metadata stays
        bounded regardless of panel size.

    Raises:
        ValueError: If the configuration is disabled or the panel is unusable.
        TensorBoundsError: If the request exceeds the declared cell ceiling.
    """
    if not config.enabled:
        raise ValueError("time-frequency tensors are disabled; enable them before building")
    if panel.empty:
        raise ValueError("cannot build a tensor from an empty panel")
    required = {DATE_COL, TICKER_COL, "return", "volume"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")

    axis_name, frequency_values = _frequency_axis(config)
    n_time = config.window.n_segments
    cells = len(panel) * len(config.channels) * len(frequency_values) * n_time
    if cells > config.max_tensor_cells:
        raise TensorBoundsError(
            f"requested tensor of {cells} cells exceeds the {config.max_tensor_cells} ceiling; "
            "reduce channels, frequency resolution, window length, or panel size"
        )

    ordered = panel.sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)
    market = (
        ordered.loc[ordered[TICKER_COL] == benchmark, [DATE_COL, "return"]]
        .drop_duplicates(DATE_COL)
        .set_index(DATE_COL)["return"]
    )
    if market.empty and "residual" in config.channels:
        raise ValueError(
            f"residual channel requires benchmark {benchmark!r}, which is absent from the panel"
        )

    surfaces: list[FloatArray] = []
    for _ticker, group in ordered.groupby(TICKER_COL, sort=False):
        per_channel = []
        for channel in config.channels:
            series = _channel_series(group, channel, _spectral_view(config), market)
            per_channel.append(_channel_surface(series.to_numpy(dtype=np.float64), config))
        surfaces.append(np.stack(per_channel, axis=1))
    values = np.concatenate(surfaces, axis=0)

    # A sample/channel is observed only when its entire surface is finite: a
    # partially finite window is an incomplete estimate, not a usable one.
    mask = np.isfinite(values).all(axis=(2, 3))
    values = np.where(mask[:, :, None, None], values, np.nan)

    index = ordered.loc[:, [DATE_COL, TICKER_COL]].copy()
    dates = pd.to_datetime(index[DATE_COL])
    metadata = TimeFrequencyMetadata(
        representation=config.representation,
        channels=tuple(config.channels),
        frequency_axis=axis_name,  # type: ignore[arg-type]
        frequency_values=tuple(float(value) for value in frequency_values),
        window_parameters=_window_parameters(config),
        n_samples=values.shape[0],
        n_time=n_time,
        coverage_start=dates.min().date(),
        coverage_end=dates.max().date(),
        tickers=tuple(sorted(ordered[TICKER_COL].astype(str).unique())),
        normalization="none",
        observed_fraction=float(mask.mean()),
        values_sha256=digest_array(values),
        mask_sha256=digest_array(mask),
    )
    return TimeFrequencyTensor(values=values, mask=mask, metadata=metadata), index


def _spectral_view(config: TimeFrequencyConfig):  # type: ignore[no-untyped-def]
    """Adapt the tensor config to the channel-builder's expected shape.

    ``_channel_series`` needs only the two trailing-transform windows, so a small
    shim keeps one channel definition shared between the descriptor engine and
    the tensor builder — two implementations of "what is the volatility channel"
    would eventually disagree.
    """
    from quant_platform.config import SpectralConfig

    return SpectralConfig(
        enabled=True,
        window=config.window,
        channels=list(config.channels),
        volatility_window=config.volatility_window,
        beta_window=config.beta_window,
    )


def _window_parameters(config: TimeFrequencyConfig) -> dict[str, str | int | float | bool | None]:
    """Return the JSON-safe window contract recorded in tensor metadata."""
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
        "representation": config.representation,
        "morlet_omega0": config.morlet_omega0,
    }


def normalize_tensor(
    tensor: TimeFrequencyTensor,
    index: pd.DataFrame,
    *,
    fit_start: date,
    fit_end: date,
    floor: float = LOG_POWER_FLOOR,
) -> TimeFrequencyTensor:
    """Return a log-power, per-bin z-scored tensor fit on a training interval.

    Power is non-negative and heavily right-skewed across orders of magnitude, so
    a raw z-score would be dominated by a handful of high-power windows. Taking a
    logarithm first turns multiplicative spread into additive spread, which is
    what makes a per-bin z-score meaningful.

    Statistics are computed **only** from observed samples whose date falls in
    ``[fit_start, fit_end]``, and the interval, sample count, and a digest of the
    statistics are recorded in the returned metadata.

    Raises:
        ValueError: If the interval is inverted, misaligned with the tensor, or
            selects no observed samples.
    """
    if fit_end < fit_start:
        raise ValueError("fit_end must be on or after fit_start")
    if len(index) != tensor.metadata.n_samples:
        raise ValueError("index and tensor must describe the same samples")
    if tensor.metadata.normalization != "none":
        raise ValueError("tensor is already normalized; normalize the raw tensor instead")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("log power floor must be finite and positive")

    stamps = pd.to_datetime(index[DATE_COL]).dt.date
    selected = ((stamps >= fit_start) & (stamps <= fit_end)).to_numpy()
    if not selected.any():
        raise ValueError("training interval selected no samples")

    logged = np.log(np.maximum(tensor.values, floor))
    training = logged[selected]
    training_mask = tensor.mask[selected]
    if not training_mask.any():
        raise ValueError("training interval selected no observed samples")

    # Statistics per (channel, frequency, time) bin, over observed samples only.
    weights = training_mask[:, :, None, None]
    counts = weights.sum(axis=0)
    if int(counts.min()) < 2:
        raise ValueError(
            "every (channel, frequency, time) bin needs at least two observed "
            "training samples to estimate a standard deviation"
        )
    safe = np.where(weights, training, 0.0)
    mean = safe.sum(axis=0) / counts
    variance = (np.where(weights, (training - mean) ** 2, 0.0)).sum(axis=0) / counts
    std = np.sqrt(variance)
    # A bin with no variation carries no information; dividing by ~0 would
    # amplify float noise into a large standardized value.
    std = np.where(std > 0.0, std, 1.0)

    standardized = np.where(tensor.mask[:, :, None, None], (logged - mean) / std, np.nan)
    state = FittedTransformState(
        method="time_frequency_log_zscore",
        state_sha256=semantic_hash(
            {"mean": digest_array(mean), "std": digest_array(std), "floor": float(floor)}
        ),
        fit_start=fit_start,
        fit_end=fit_end,
        sample_count=int(selected.sum()),
    )
    metadata = tensor.metadata.model_copy(
        update={
            "normalization": "train_log_zscore",
            "log_power_floor": float(floor),
            "fitted_state": state,
            "values_sha256": digest_array(standardized),
        }
    )
    return TimeFrequencyTensor(values=standardized, mask=tensor.mask, metadata=metadata)
