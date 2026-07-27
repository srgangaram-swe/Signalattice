"""One comparison contract for every representation family (SF-S3-MR3).

The sprint's whole question is whether an advanced representation beats a
conventional one. That comparison is only meaningful if the families differ in
*exactly one* respect — the representation — and in nothing else. Two teams
comparing "wavelets vs raw features" while one detrends and the other does not,
or one uses 64-bar windows and the other 60, have measured their preprocessing.

This module removes that failure mode structurally. Every family here consumes
**the same** :func:`~quant_platform.features.spectral_transforms.causal_windows`
output, built once from one
:class:`~quant_platform.config.SpectralWindow`. There is no per-family
preprocessing argument to get wrong, because there is no per-family
preprocessing at all.

Families:

* ``raw`` — scale-free time-domain shape statistics of the window.
* ``spectral`` — SF-S3-MR1's Welch descriptors.
* ``wavelet`` — SF-S3-MR1's discrete-wavelet band energies.
* ``emd`` — adaptive-mode descriptors from Empirical Mode Decomposition.
* ``vmd`` — adaptive-mode descriptors from Variational Mode Decomposition.

Every emitted descriptor is **scale-free**, for the same reason as in
SF-S3-MR1: a scale-dependent descriptor would encode recent volatility, which
the conventional features already carry, and would make the families
incomparable across regimes rather than across representations.

Adaptive decomposition is expensive — orders of magnitude more so than an FFT —
so :class:`~quant_platform.config.RepresentationConfig` exposes a ``stride``.
Descriptors are computed every ``stride`` bars and the intervening bars are NaN:
an honest gap, never a forward fill, which would smear a later window's
information backwards.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_platform.config import RepresentationConfig
from quant_platform.features import spectral_descriptors as sd
from quant_platform.features import spectral_transforms as st
from quant_platform.features.decomposition import (
    Decomposition,
    DecompositionError,
    emd,
    vmd,
)

FloatArray = NDArray[np.float64]

#: Families comparable under this contract, in stable order.
REPRESENTATION_FAMILIES: tuple[str, ...] = ("raw", "spectral", "wavelet", "emd", "vmd")

#: Descriptors emitted per adaptive mode, in stable order.
MODE_DESCRIPTORS: tuple[str, ...] = (
    "energy_fraction",
    "dominant_period",
    "spectral_entropy",
    "stability",
)


def mode_descriptors(decomposition: Decomposition, *, n_reported: int) -> dict[str, float]:
    """Reduce a decomposition to a fixed-width, scale-free descriptor vector.

    A decomposition produces a variable number of modes, but a feature vector
    must have a fixed width. Modes beyond ``n_reported`` are folded into the
    residual accounting and missing modes are NaN rather than zero — a zero
    energy fraction and "this mode did not exist" are different facts.

    Per mode:

    * ``energy_fraction`` — share of total signal energy.
    * ``dominant_period`` — bars per cycle at the mode's spectral peak. This is
      what makes an adaptive mode interpretable: without it, "mode 2" names a
      position in an algorithm's output rather than a timescale.
    * ``spectral_entropy`` — how tonal the mode is. A genuine intrinsic mode is
      narrowband; high entropy is the signature of mode mixing, and reporting it
      is what lets an unstable mode be rejected instead of used.
    * ``stability`` — ratio of the second half's energy to the first half's,
      log-compressed and bounded. An intermittent mode that exists only in part
      of the window is not a component of the whole window.

    Plus ``residual_energy_fraction`` and ``max_cross_correlation``: orthogonal
    modes should barely correlate, so a high value means the decomposition has
    split one component across several modes.
    """
    if n_reported < 1:
        raise DecompositionError("n_reported must be at least one")
    modes = decomposition.modes
    total = float(np.sum(decomposition.reconstruction**2))
    columns: dict[str, float] = {}
    for index in range(n_reported):
        prefix = f"mode{index + 1}_"
        if index >= modes.shape[0] or total <= 0.0:
            for name in MODE_DESCRIPTORS:
                columns[prefix + name] = float("nan")
            continue
        mode = modes[index]
        columns[prefix + "energy_fraction"] = float(np.sum(mode**2) / total)
        columns[prefix + "dominant_period"] = _dominant_period(mode)
        columns[prefix + "spectral_entropy"] = _mode_entropy(mode)
        columns[prefix + "stability"] = _stability(mode)
    columns["residual_energy_fraction"] = float(decomposition.report.residual_energy_fraction)
    columns["max_cross_correlation"] = _max_cross_correlation(modes)
    columns["converged"] = float(decomposition.report.converged)
    return columns


def _dominant_period(mode: FloatArray) -> float:
    """Return the mode's peak period in bars, or NaN when it has no oscillation."""
    centred = mode - mode.mean()
    if not np.any(centred):
        return float("nan")
    spectrum = np.abs(np.fft.rfft(centred))
    frequencies = np.fft.rfftfreq(centred.size)
    # Bin 0 is the mean, already removed; taking its argmax would be meaningless.
    peak = int(np.argmax(spectrum[1:])) + 1
    frequency = float(frequencies[peak])
    return 1.0 / frequency if frequency > 0.0 else float("nan")


def _mode_entropy(mode: FloatArray) -> float:
    """Return the mode's normalized spectral entropy in ``[0, 1]``."""
    centred = mode - mode.mean()
    power = np.abs(np.fft.rfft(centred)) ** 2
    total = float(power.sum())
    if total <= 0.0 or power.size < 2:
        return float("nan")
    mass = power / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(mass > 0.0, mass * np.log(mass), 0.0)
    return float(-terms.sum() / np.log(power.size))


def _stability(mode: FloatArray) -> float:
    """Return a bounded log-ratio of late to early energy, in ``[-1, 1]``.

    ``0`` means the mode's energy is evenly spread across the window; ``+1`` that
    it exists only at the end, ``-1`` only at the start. ``tanh`` of the log ratio
    keeps a mode that is absent from one half from producing an infinity.
    """
    half = mode.size // 2
    if half < 1:
        return float("nan")
    early = float(np.sum(mode[:half] ** 2))
    late = float(np.sum(mode[half:] ** 2))
    if early <= 0.0 and late <= 0.0:
        return float("nan")
    if early <= 0.0:
        return 1.0
    if late <= 0.0:
        return -1.0
    return float(np.tanh(np.log(late / early)))


def _max_cross_correlation(modes: FloatArray) -> float:
    """Return the largest absolute correlation between distinct modes."""
    if modes.shape[0] < 2:
        return float("nan")
    centred = modes - modes.mean(axis=1, keepdims=True)
    norms = np.sqrt((centred**2).sum(axis=1))
    usable = norms > 0.0
    if usable.sum() < 2:
        return float("nan")
    normalized = centred[usable] / norms[usable][:, None]
    gram = np.abs(normalized @ normalized.T)
    np.fill_diagonal(gram, 0.0)
    return float(gram.max())


def _raw_descriptors(window: FloatArray) -> dict[str, float]:
    """Scale-free time-domain shape statistics of one window.

    The conventional-representation arm. Every statistic is standardized by the
    window's own standard deviation, so the arm carries *shape* rather than
    amplitude — matching every other family and keeping the comparison about
    representation rather than volatility.
    """
    centred = window - window.mean()
    deviation = float(np.std(centred))
    if deviation <= 0.0:
        return {
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "autocorr_1": float("nan"),
            "autocorr_5": float("nan"),
            "variance_ratio_2": float("nan"),
        }
    standardized = centred / deviation
    columns = {
        "skewness": float(np.mean(standardized**3)),
        "kurtosis": float(np.mean(standardized**4) - 3.0),
    }
    for lag in (1, 5):
        columns[f"autocorr_{lag}"] = _autocorrelation(standardized, lag)
    # Variance ratio: 1 under a random walk, >1 trending, <1 mean-reverting.
    aggregated = standardized[: standardized.size // 2 * 2].reshape(-1, 2).sum(axis=1)
    single = float(np.var(standardized))
    columns["variance_ratio_2"] = (
        float(np.var(aggregated) / (2.0 * single)) if single > 0.0 else float("nan")
    )
    return columns


def _autocorrelation(standardized: FloatArray, lag: int) -> float:
    if standardized.size <= lag:
        return float("nan")
    head, tail = standardized[:-lag], standardized[lag:]
    denominator = float(np.sum(standardized**2))
    return float(np.sum(head * tail) / denominator) if denominator > 0.0 else float("nan")


def family_descriptor_names(family: str, config: RepresentationConfig) -> tuple[str, ...]:
    """Return the descriptor names one family emits, in stable order."""
    if family == "raw":
        return ("skewness", "kurtosis", "autocorr_1", "autocorr_5", "variance_ratio_2")
    if family == "spectral":
        return sd.descriptor_column_names(config.descriptors)
    if family == "wavelet":
        return tuple(
            [f"dwt_d{level}" for level in range(1, config.dwt_levels + 1)] + ["dwt_approx"]
        )
    if family in {"emd", "vmd"}:
        names = [
            f"mode{index + 1}_{descriptor}"
            for index in range(config.n_reported_modes)
            for descriptor in MODE_DESCRIPTORS
        ]
        return tuple(names + ["residual_energy_fraction", "max_cross_correlation", "converged"])
    raise ValueError(f"unknown representation family {family!r}; have {REPRESENTATION_FAMILIES}")


def build_representation_descriptors(
    values: FloatArray, family: str, config: RepresentationConfig
) -> dict[str, FloatArray]:
    """Compute one family's descriptors over identical causal windows.

    Args:
        values: One asset's channel series.
        family: One of :data:`REPRESENTATION_FAMILIES`.
        config: Shared window, descriptor, and decomposition settings.

    Returns:
        A dict keyed by :func:`family_descriptor_names`, each a ``(len(values),)``
        array. Warm-up bars and bars skipped by ``stride`` are NaN.

    Raises:
        ValueError: If the family is unknown.
    """
    names = family_descriptor_names(family, config)
    window = config.window
    windows = st.causal_windows(np.asarray(values, dtype=np.float64), window.length)
    n_bars = windows.shape[0]

    if family == "spectral":
        psd = st.welch_psd(
            windows,
            segment_length=window.segment_length,
            hop=window.hop,
            n_fft=window.n_fft,
            sampling_frequency=window.sampling_frequency,
            taper_name=window.taper,
            detrend=window.detrend,
        )
        frequencies = np.asarray(window.frequencies(), dtype=np.float64)
        return sd.describe(psd, frequencies, config.descriptors)
    if family == "wavelet":
        energies = st.dwt_energies(windows, config.wavelet, config.dwt_levels)
        return {name: energies[:, index] for index, name in enumerate(names)}

    # Per-window families: raw statistics and the adaptive decompositions. These
    # cannot be vectorized across bars, so they honour `stride`.
    columns = {name: np.full(n_bars, np.nan, dtype=np.float64) for name in names}
    evaluated = range(window.length - 1, n_bars, config.stride)
    for position in evaluated:
        row = windows[position]
        if not np.isfinite(row).all():
            continue
        computed = _window_descriptors(row, family, config)
        for name, value in computed.items():
            columns[name][position] = value
    return columns


def _window_descriptors(
    row: FloatArray, family: str, config: RepresentationConfig
) -> dict[str, float]:
    """Compute one window's descriptors for a per-window family."""
    if family == "raw":
        return _raw_descriptors(row)
    detrended = st.detrend_segments(row[None, :], config.window.detrend)[0]
    try:
        if family == "emd":
            decomposition = emd(
                detrended,
                max_modes=config.max_modes,
                sd_tolerance=config.sd_tolerance,
                max_sift_iterations=config.max_sift_iterations,
            )
        else:
            decomposition = vmd(
                detrended,
                n_modes=config.n_reported_modes,
                alpha=config.vmd_alpha,
                tau=config.vmd_tau,
                max_iterations=config.vmd_max_iterations,
            )
    except DecompositionError:
        # A window the algorithm cannot decompose (degenerate or too short after
        # detrending) yields missing descriptors, never fabricated ones.
        return {}
    return mode_descriptors(decomposition, n_reported=config.n_reported_modes)


def shared_window_identity(config: RepresentationConfig) -> dict[str, str | int | float | bool]:
    """Return the preprocessing contract every family shares.

    Published so a comparison can *prove* the arms differ only in their
    representation. If two runs disagree here, their descriptors were not
    computed on the same folds and are not comparable, whatever the metrics say.
    """
    window = config.window
    return {
        "length": window.length,
        "segment_length": window.segment_length,
        "hop": window.hop,
        "n_fft": window.n_fft,
        "detrend": window.detrend,
        "taper": window.taper,
        "sampling_frequency": window.sampling_frequency,
        "frequency_unit": window.frequency_unit,
        "warmup_bars": window.warmup_bars,
        "stride": config.stride,
    }
