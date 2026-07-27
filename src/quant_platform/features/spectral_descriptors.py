"""Scale-free descriptors of a power spectrum (SF-S3-MR1).

A power spectral density is a vector; a model wants a handful of interpretable
numbers. This module defines that reduction, with one governing design rule:

**Every descriptor here is invariant to the amplitude of the input signal.**

That is not a stylistic preference. A raw band power scales with the variance of
the series, so it would encode "this asset was volatile this month" — a fact the
volatility features already carry, and one whose distribution shifts violently
between regimes. By normalizing the spectrum to a probability mass over
frequency before reducing it, every descriptor answers the question the spectral
representation is actually for: *how is the variation distributed across
frequencies*, independent of how much of it there was. The practical consequence
is that these columns need no fitted normalization to be comparable across
assets, and therefore carry no train/test leakage surface at all.

The one genuinely scale-dependent quantity, total band power, is deliberately
emitted only in relative form. If an absolute level is ever wanted, it must go
through a train-fitted normalization with a recorded fit interval.

Conventions:

* ``psd`` is ``(n_bars, n_freqs)`` and non-negative; ``frequencies`` is
  ``(n_freqs,)`` in cycles per bar.
* A window whose spectrum is all zero (a constant input) has no frequency
  distribution to describe, so every descriptor is NaN rather than a
  convenient-looking zero.
* NaN in, NaN out. No descriptor imputes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_platform.config import DescriptorConfig, FrequencyBand

FloatArray = NDArray[np.float64]

#: Descriptor column names, in stable emission order. Downstream schemas depend
#: on this tuple; appending is a minor version, reordering or removing is not.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "centroid",
    "bandwidth",
    "entropy",
    "flatness",
    "rolloff",
    "peak_frequency",
    "sparsity",
    "concentration",
)


def descriptor_column_names(config: DescriptorConfig) -> tuple[str, ...]:
    """Return every descriptor name this configuration emits, in stable order.

    Callers depend on this order to build column layouts and registry specs, so
    it is derived from one place rather than reconstructed at each use site.
    """
    band_powers = tuple(f"band_{band.name}" for band in config.bands)
    ratios = tuple(
        f"ratio_{first.name}_{second.name}"
        for first, second in zip(config.bands, config.bands[1:], strict=False)
    )
    return DESCRIPTOR_NAMES + band_powers + ratios


def normalized_spectrum(psd: FloatArray) -> FloatArray:
    """Return the spectrum as a probability mass over frequency.

    Windows with zero or non-finite total power become all-NaN rows, which is
    what makes every downstream descriptor fail closed on a degenerate window.
    """
    total = psd.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mass = psd / np.where(total > 0.0, total, np.nan)
    return np.asarray(mass, dtype=np.float64)


def spectral_centroid(mass: FloatArray, frequencies: FloatArray) -> FloatArray:
    """First moment of the normalized spectrum: the centre of gravity in Hz-per-bar."""
    return np.asarray(mass @ frequencies, dtype=np.float64)


def spectral_bandwidth(
    mass: FloatArray, frequencies: FloatArray, centroid: FloatArray
) -> FloatArray:
    """Square root of the second central moment: how spread the energy is."""
    deviation = frequencies[None, :] - centroid[:, None]
    return np.asarray(np.sqrt((mass * deviation**2).sum(axis=-1)), dtype=np.float64)


def spectral_entropy(mass: FloatArray) -> FloatArray:
    """Shannon entropy of the normalized spectrum, scaled to ``[0, 1]``.

    Normalizing by ``log(n_freqs)`` makes the value comparable across different
    FFT lengths — without it, changing ``n_fft`` would silently change the
    feature's scale and break any model trained on the old configuration.
    ``1`` is a flat (white) spectrum, ``0`` a single pure tone.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(mass > 0.0, mass * np.log(mass), 0.0)
    n_freqs = mass.shape[-1]
    if n_freqs < 2:
        return np.full(mass.shape[0], np.nan, dtype=np.float64)
    entropy = -terms.sum(axis=-1) / np.log(n_freqs)
    # A row that was all-NaN must stay NaN, not become 0 via the `where` above.
    invalid = ~np.isfinite(mass).all(axis=-1)
    return np.asarray(np.where(invalid, np.nan, entropy), dtype=np.float64)


def spectral_flatness(psd: FloatArray) -> FloatArray:
    """Wiener entropy: geometric mean over arithmetic mean of the spectrum.

    ``1`` is perfectly flat, values near ``0`` mean the energy is concentrated
    in a few bins. Unlike entropy this is a ratio of means, so it reacts to deep
    spectral valleys as strongly as to peaks; an exactly-zero bin drives it to
    zero, which is the mathematically correct answer and is preserved rather
    than floored.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        geometric = np.exp(np.log(psd).mean(axis=-1))
        arithmetic = psd.mean(axis=-1)
        flatness = geometric / np.where(arithmetic > 0.0, arithmetic, np.nan)
    return np.asarray(flatness, dtype=np.float64)


def spectral_rolloff(mass: FloatArray, frequencies: FloatArray, quantile: float) -> FloatArray:
    """Lowest frequency below which ``quantile`` of the energy lies."""
    cumulative = np.cumsum(mass, axis=-1)
    reached = cumulative >= quantile
    # A row of all-NaN never reaches the quantile; argmax would return 0 and
    # quietly report the DC bin, so those rows are masked back to NaN.
    index = np.argmax(reached, axis=-1)
    result = frequencies[index]
    return np.asarray(np.where(reached.any(axis=-1), result, np.nan), dtype=np.float64)


def peak_frequency(mass: FloatArray, frequencies: FloatArray) -> FloatArray:
    """Frequency of the largest spectral bin (the dominant frequency)."""
    valid = np.isfinite(mass).all(axis=-1)
    index = np.argmax(np.where(np.isfinite(mass), mass, -np.inf), axis=-1)
    return np.asarray(np.where(valid, frequencies[index], np.nan), dtype=np.float64)


def spectral_sparsity(psd: FloatArray) -> FloatArray:
    """Hoyer sparsity of the spectrum in ``[0, 1]``.

    ``(sqrt(K) - ||p||_1 / ||p||_2) / (sqrt(K) - 1)``: ``0`` when every bin
    carries equal power, ``1`` when a single bin carries all of it. It is
    scale-free by construction (a ratio of norms) and, unlike entropy,
    responds to the *number* of active bins rather than the shape of the
    distribution over them.
    """
    n_freqs = psd.shape[-1]
    if n_freqs < 2:
        return np.full(psd.shape[0], np.nan, dtype=np.float64)
    l1 = np.abs(psd).sum(axis=-1)
    l2 = np.sqrt((psd**2).sum(axis=-1))
    root = np.sqrt(n_freqs)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = l1 / np.where(l2 > 0.0, l2, np.nan)
    return np.asarray((root - ratio) / (root - 1.0), dtype=np.float64)


def energy_concentration(mass: FloatArray, bins: int) -> FloatArray:
    """Fraction of total energy held by the ``bins`` strongest frequencies."""
    n_freqs = mass.shape[-1]
    take = min(bins, n_freqs)
    ordered = np.sort(np.where(np.isfinite(mass), mass, -np.inf), axis=-1)[:, -take:]
    valid = np.isfinite(mass).all(axis=-1)
    return np.asarray(np.where(valid, ordered.sum(axis=-1), np.nan), dtype=np.float64)


def relative_band_power(
    mass: FloatArray, frequencies: FloatArray, band: FrequencyBand
) -> FloatArray:
    """Fraction of total energy inside a half-open frequency band ``[low, high)``.

    The band ending at Nyquist is closed at the top. The one-sided axis stops
    there, so a half-open top edge would drop the Nyquist bin from every band —
    leaving a set of bands that silently fails to sum to one, and quietly
    discarding the highest-frequency evidence in the window.
    """
    upper_is_nyquist = band.high >= float(frequencies[-1])
    if upper_is_nyquist:
        selected = frequencies >= band.low
    else:
        selected = (frequencies >= band.low) & (frequencies < band.high)
    if not selected.any():
        # An empty band means the configuration and the FFT resolution disagree;
        # report NaN rather than a structural zero that looks like real evidence.
        return np.full(mass.shape[0], np.nan, dtype=np.float64)
    return np.asarray(mass[:, selected].sum(axis=-1), dtype=np.float64)


def spectral_flux(magnitudes: FloatArray) -> FloatArray:
    """Mean normalized change between consecutive short-time spectra.

    This is the only descriptor that needs the STFT rather than a single
    averaged spectrum: it measures how much the *shape* of the spectrum moves
    within the causal window, which separates a stably oscillating regime from
    one whose frequency content is reorganizing. Each slice is normalized before
    differencing, so the result reports shape change rather than amplitude
    change — the latter is already volatility.

    Args:
        magnitudes: ``(n_bars, n_segments, n_freqs)`` short-time magnitudes.

    Returns:
        ``(n_bars,)``, NaN when fewer than two slices exist.
    """
    if magnitudes.shape[1] < 2:
        return np.full(magnitudes.shape[0], np.nan, dtype=np.float64)
    totals = magnitudes.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        shapes = magnitudes / np.where(totals > 0.0, totals, np.nan)
    difference = np.diff(shapes, axis=1)
    return np.asarray(np.sqrt((difference**2).sum(axis=-1)).mean(axis=-1), dtype=np.float64)


def describe(
    psd: FloatArray, frequencies: FloatArray, config: DescriptorConfig
) -> dict[str, FloatArray]:
    """Reduce a power spectrum to the full named descriptor set.

    Args:
        psd: ``(n_bars, n_freqs)`` non-negative power spectral density.
        frequencies: ``(n_freqs,)`` bin frequencies in cycles per bar.
        config: Roll-off quantile, concentration width, and band definitions.

    Returns:
        A dict keyed by :meth:`DescriptorConfig.column_names`, each value a
        ``(n_bars,)`` float array.

    Raises:
        ValueError: If ``psd`` is not two-dimensional or its trailing axis does
            not match ``frequencies``.
    """
    if psd.ndim != 2:
        raise ValueError(f"psd must be two-dimensional, got shape {psd.shape}")
    if psd.shape[-1] != frequencies.shape[0]:
        raise ValueError(
            f"psd has {psd.shape[-1]} bins but {frequencies.shape[0]} frequencies were supplied"
        )
    mass = normalized_spectrum(psd)
    centroid = spectral_centroid(mass, frequencies)
    columns: dict[str, FloatArray] = {
        "centroid": centroid,
        "bandwidth": spectral_bandwidth(mass, frequencies, centroid),
        "entropy": spectral_entropy(mass),
        "flatness": spectral_flatness(psd),
        "rolloff": spectral_rolloff(mass, frequencies, config.rolloff_quantile),
        "peak_frequency": peak_frequency(mass, frequencies),
        "sparsity": spectral_sparsity(psd),
        "concentration": energy_concentration(mass, config.concentration_bins),
    }
    powers = {band.name: relative_band_power(mass, frequencies, band) for band in config.bands}
    for name, values in powers.items():
        columns[f"band_{name}"] = values
    for first, second in zip(config.bands, config.bands[1:], strict=False):
        denominator = powers[second.name]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = powers[first.name] / np.where(denominator > 0.0, denominator, np.nan)
        columns[f"ratio_{first.name}_{second.name}"] = np.asarray(ratio, dtype=np.float64)
    return columns
