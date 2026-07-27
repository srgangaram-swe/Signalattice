"""Causal rolling spectral transforms (SF-S3-MR1).

Every transform here answers the same question at every bar *t*: what does the
frequency content of the **trailing** window ``x[t-L+1 .. t]`` look like? Nothing
in this module may read ``x[t+1]`` or later, and no window is centred on *t* —
a centred window is the classic way a time-frequency feature acquires lookahead
without anyone noticing, because the leakage is buried in a convolution kernel
rather than in an obvious ``shift(-1)``.

Assumptions this module makes, and which the caller inherits:

* **Bar time, not calendar time.** The series is treated as uniformly sampled in
  *bars*. Frequencies are therefore ``cycles_per_bar``: a "period of 5" means
  five trading bars, not five calendar days. Weekends and holidays do not make
  the sampling irregular in this coordinate; they make bar time a non-linear
  reparameterization of calendar time, which is a modelling choice, not an
  error, and is why the frequency unit is named explicitly in the contract.
* **Missing data fails closed.** A window containing a NaN produces NaN rather
  than an imputed spectrum. Interpolating inside a spectral window invents
  frequency content that the market never produced.

The algorithms (Welch averaging, the STFT segmentation, the causal Morlet
wavelet, and the Daubechies filter bank) are implemented here rather than
delegated, so their windowing, normalization, and edge behaviour are auditable;
``scipy.signal`` is used as an *independent reference* in the tests, not as the
implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Name of an analysis taper. A taper trades frequency resolution for dynamic
#: range: the boxcar has the narrowest main lobe and the worst side lobes
#: (-13 dB), so a strong low frequency smears across the whole spectrum and
#: corrupts every descriptor computed from it. Hann (-31 dB) is the default.
TaperName = str

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

#: Daubechies orthogonal scaling filters, low-pass coefficients h[k].
#:
#: `haar` (1 vanishing moment) and `db2` (2 vanishing moments, 4 taps, commonly
#: printed as the "Daubechies-4" coefficients). Stated literally so the filter
#: bank can be checked against the published values by eye.
_SQRT2 = np.sqrt(2.0)
_SQRT3 = np.sqrt(3.0)
DAUBECHIES_FILTERS: dict[str, FloatArray] = {
    "haar": np.array([1.0 / _SQRT2, 1.0 / _SQRT2], dtype=np.float64),
    "db2": np.array(
        [
            (1.0 + _SQRT3) / (4.0 * _SQRT2),
            (3.0 + _SQRT3) / (4.0 * _SQRT2),
            (3.0 - _SQRT3) / (4.0 * _SQRT2),
            (1.0 - _SQRT3) / (4.0 * _SQRT2),
        ],
        dtype=np.float64,
    ),
}


def taper(name: TaperName, length: int) -> FloatArray:
    """Return a symmetric analysis taper of ``length`` samples.

    Args:
        name: One of ``boxcar``, ``hann``, ``hamming``, ``blackman``.
        length: Taper length in samples; must be positive.

    Raises:
        ValueError: If ``name`` is unknown or ``length`` is not positive.
    """
    if length < 1:
        raise ValueError(f"taper length must be >= 1, got {length}")
    if name == "boxcar":
        return np.ones(length, dtype=np.float64)
    # Periodic-symmetric form (sym=True), matching scipy.signal.get_window(..., fftbins=False)
    # for the tapers used here, so the differential tests compare like with like.
    if length == 1:
        return np.ones(1, dtype=np.float64)
    n = np.arange(length, dtype=np.float64)
    ratio = n / (length - 1)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * ratio)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * ratio)
    if name == "blackman":
        return 0.42 - 0.5 * np.cos(2.0 * np.pi * ratio) + 0.08 * np.cos(4.0 * np.pi * ratio)
    raise ValueError(f"unknown taper {name!r}; expected boxcar, hann, hamming, or blackman")


#: Relative size below which a detrended residual is rounding error, not signal.
#:
#: Detrending a constant series does not give exactly zero — ``4.2 - mean(4.2)``
#: leaves ~1e-16 of floating-point residue. Left alone, that residue normalizes
#: into a perfectly plausible-looking spectrum computed entirely from rounding
#: error, which is precisely the kind of silent fabricated evidence a research
#: platform must not produce. Residuals this small relative to the segment's own
#: magnitude are collapsed to exact zero so the window reports as degenerate.
CONSTANT_RELATIVE_TOLERANCE = 1e-12


def detrend_segments(segments: FloatArray, method: str) -> FloatArray:
    """Remove a constant or linear trend from each segment's last axis.

    A non-zero mean puts the entire signal energy in the DC bin and dominates
    every normalized descriptor; a linear drift leaks across the low-frequency
    bins. Detrending is therefore part of the transform contract, not an
    optional convenience.

    A segment whose residual is negligible relative to its own magnitude is
    collapsed to exact zero — see :data:`CONSTANT_RELATIVE_TOLERANCE`.

    Raises:
        ValueError: If ``method`` is not ``none``, ``mean``, or ``linear``.
    """
    if method == "none":
        return segments
    if method == "mean":
        residual = segments - segments.mean(axis=-1, keepdims=True)
    elif method == "linear":
        length = segments.shape[-1]
        if length < 2:
            residual = segments - segments.mean(axis=-1, keepdims=True)
        else:
            # Closed-form least-squares line fit; cheaper and more stable than
            # solving a normal-equation system per segment.
            index = np.arange(length, dtype=np.float64)
            centred = index - index.mean()
            denominator = float((centred**2).sum())
            slope = (segments * centred).sum(axis=-1, keepdims=True) / denominator
            intercept = segments.mean(axis=-1, keepdims=True) - slope * index.mean()
            residual = segments - (slope * index + intercept)
    else:
        raise ValueError(f"unknown detrend {method!r}; expected none, mean, or linear")
    return _collapse_rounding_residue(segments, residual)


def _collapse_rounding_residue(original: FloatArray, residual: FloatArray) -> FloatArray:
    """Zero out residuals that are pure floating-point noise."""
    scale = np.abs(original).max(axis=-1, keepdims=True)
    residual_scale = np.abs(residual).max(axis=-1, keepdims=True)
    degenerate = residual_scale <= CONSTANT_RELATIVE_TOLERANCE * scale
    return np.asarray(np.where(degenerate, 0.0, residual), dtype=np.float64)


def causal_windows(values: FloatArray, length: int) -> FloatArray:
    """Return trailing windows: row ``t`` is ``values[t-length+1 .. t]``.

    Rows before the warm-up boundary (``t < length - 1``) are filled with NaN
    rather than partially populated, so an incompletely warmed feature can never
    be mistaken for a converged one.

    Returns:
        A ``(len(values), length)`` array. The result is a fresh, writable copy;
        the strided view is never handed out, because a caller mutating it would
        silently corrupt overlapping windows.

    Raises:
        ValueError: If ``length`` is not positive.
    """
    if length < 1:
        raise ValueError(f"window length must be >= 1, got {length}")
    n_bars = values.shape[0]
    out = np.full((n_bars, length), np.nan, dtype=np.float64)
    if n_bars >= length:
        view = np.lib.stride_tricks.sliding_window_view(values, length)
        out[length - 1 :] = view
    return out


def segment_windows(windows: FloatArray, segment_length: int, hop: int) -> FloatArray:
    """Split each causal window into overlapping sub-segments.

    Args:
        windows: ``(n_bars, length)`` trailing windows.
        segment_length: Samples per sub-segment.
        hop: Advance between consecutive sub-segments; ``overlap`` is
            ``segment_length - hop``.

    Returns:
        ``(n_bars, n_segments, segment_length)``.

    Raises:
        ValueError: If the segmentation parameters do not fit the window.
    """
    length = windows.shape[-1]
    if not 1 <= segment_length <= length:
        raise ValueError(f"segment_length must be in [1, {length}], got {segment_length}")
    if not 1 <= hop <= segment_length:
        raise ValueError(f"hop must be in [1, {segment_length}], got {hop}")
    n_segments = 1 + (length - segment_length) // hop
    starts = np.arange(n_segments) * hop
    index = starts[:, None] + np.arange(segment_length)[None, :]
    return windows[:, index]


def welch_psd(
    windows: FloatArray,
    *,
    segment_length: int,
    hop: int,
    n_fft: int,
    sampling_frequency: float,
    taper_name: TaperName = "hann",
    detrend: str = "mean",
) -> FloatArray:
    """One-sided Welch power spectral density per causal window.

    Welch's method trades frequency resolution for variance: a single
    periodogram of an ``L``-sample window is an inconsistent estimator whose
    variance does not shrink as ``L`` grows, so averaging ``K`` tapered
    sub-periodograms cuts the variance by roughly ``K`` at the cost of a main
    lobe ``L/segment_length`` times wider. For noisy financial series the
    variance reduction is worth far more than the resolution.

    The density scaling is ``|X_k|^2 / (fs * sum(w^2))`` with interior bins
    doubled for the one-sided fold, so the result integrates to the signal
    variance and is directly comparable to ``scipy.signal.welch(...,
    scaling="density")`` — which the tests assert.

    Returns:
        ``(n_bars, n_fft // 2 + 1)`` non-negative PSD, NaN where the window was
        incomplete or contained a NaN.
    """
    if n_fft < segment_length:
        raise ValueError(f"n_fft must be >= segment_length, got {n_fft} < {segment_length}")
    if sampling_frequency <= 0.0 or not np.isfinite(sampling_frequency):
        raise ValueError("sampling_frequency must be finite and positive")
    segments = segment_windows(windows, segment_length, hop)
    segments = detrend_segments(segments, detrend)
    window_taper = taper(taper_name, segment_length)
    spectra = np.fft.rfft(segments * window_taper, n=n_fft, axis=-1)
    power = np.abs(spectra) ** 2
    power /= sampling_frequency * float((window_taper**2).sum())
    # One-sided fold: every bin except DC and (for even n_fft) Nyquist stands in
    # for a conjugate pair, so it carries twice the density.
    if n_fft % 2 == 0:
        power[..., 1:-1] *= 2.0
    else:
        power[..., 1:] *= 2.0
    return np.asarray(power.mean(axis=1), dtype=np.float64)


def amplitude_spectrum(
    windows: FloatArray,
    *,
    n_fft: int,
    taper_name: TaperName = "hann",
    detrend: str = "mean",
) -> FloatArray:
    """Single-window one-sided amplitude spectrum ``|rfft(w * x)|``.

    The plain rolling FFT: maximum frequency resolution, maximum variance. Kept
    distinct from :func:`welch_psd` because the two answer different questions
    and must never be silently substituted for one another.
    """
    tapered = detrend_segments(windows, detrend) * taper(taper_name, windows.shape[-1])
    return np.asarray(np.abs(np.fft.rfft(tapered, n=n_fft, axis=-1)), dtype=np.float64)


def stft(
    windows: FloatArray,
    *,
    segment_length: int,
    hop: int,
    n_fft: int,
    taper_name: TaperName = "hann",
    detrend: str = "mean",
) -> FloatArray:
    """Short-time Fourier magnitudes inside each causal window.

    Returns:
        ``(n_bars, n_segments, n_fft // 2 + 1)`` magnitudes, ordered oldest
        sub-segment first, so index ``-1`` is always the most recent slice.
    """
    segments = detrend_segments(segment_windows(windows, segment_length, hop), detrend)
    spectra = np.fft.rfft(segments * taper(taper_name, segment_length), n=n_fft, axis=-1)
    return np.asarray(np.abs(spectra), dtype=np.float64)


def rfft_frequencies(n_fft: int, sampling_frequency: float) -> FloatArray:
    """Return one-sided bin frequencies in the sampling unit (cycles per bar)."""
    return np.asarray(np.fft.rfftfreq(n_fft, d=1.0 / sampling_frequency), dtype=np.float64)


def causal_morlet(length: int, scale: float, omega0: float = 6.0) -> ComplexArray:
    """Return a causal (past-only) complex Morlet analysis filter.

    The standard Morlet ``pi^(-1/4) e^(i w0 t) e^(-t^2/2)`` is symmetric about
    its centre, so a rolling convolution with it reads the future. Truncating it
    to the past is what makes it usable in production research, and that
    truncation has two consequences that are corrected here rather than ignored:

    1. **Admissibility.** A truncated wavelet no longer integrates to zero, so it
       would respond to a constant offset. The discrete mean is subtracted to
       restore the zero-mean property on the truncated support.
    2. **Normalization.** Energy is renormalized to unity *after* truncation, so
       responses are comparable across scales instead of decaying with the
       fraction of the wavelet that was cut away.

    The filter is indexed by lag into the past: element ``k`` multiplies the
    sample ``k`` bars before the evaluation bar.

    Args:
        length: Filter length in bars (the causal support).
        scale: Wavelet scale in bars; larger scales resolve lower frequencies.
        omega0: Dimensionless central frequency. The admissibility-motivated
            conventional value is 6.0.
    """
    if length < 2:
        raise ValueError(f"morlet length must be >= 2, got {length}")
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("morlet scale must be finite and positive")
    lag = np.arange(length, dtype=np.float64)
    t = -lag / scale
    psi = np.pi**-0.25 * np.exp(1j * omega0 * t) * np.exp(-(t**2) / 2.0)
    psi = psi - psi.mean()
    energy = float(np.sqrt(np.sum(np.abs(psi) ** 2)))
    if energy == 0.0:  # pragma: no cover - only reachable for a degenerate scale
        raise ValueError("morlet filter collapsed to zero energy")
    return np.asarray(psi / energy, dtype=np.complex128)


def morlet_scale_frequencies(scales: FloatArray, omega0: float = 6.0) -> FloatArray:
    """Map Morlet scales to centre frequencies in cycles per bar.

    ``f = omega0 / (2 * pi * s)``. Published without approximation because a
    scale index that cannot be converted to a frequency is not interpretable
    evidence.
    """
    return np.asarray(omega0 / (2.0 * np.pi * np.asarray(scales, dtype=np.float64)))


def cwt_power(windows: FloatArray, scales: FloatArray, omega0: float = 6.0) -> FloatArray:
    """Causal continuous-wavelet power at the evaluation bar, per scale.

    The coefficient is the correlation of the trailing window with the causal
    Morlet filter, evaluated at the most recent bar. That position sits at the
    edge of the analysed support — inside the cone of influence — so the
    estimate is edge-affected by construction. This is inherent to any causal
    wavelet estimate and is documented rather than hidden: a centred estimate
    would be cleaner and would read the future.

    Returns:
        ``(n_bars, n_scales)`` non-negative power.
    """
    length = windows.shape[-1]
    filters = np.stack([causal_morlet(length, float(scale), omega0) for scale in scales])
    # Column 0 of `windows` is the oldest bar; the filter is indexed by lag, so
    # reverse the window to align lag 0 with the evaluation bar.
    coefficients = windows[:, ::-1] @ filters.conj().T
    return np.asarray(np.abs(coefficients) ** 2, dtype=np.float64)


def _quadrature_filter(low_pass: FloatArray) -> FloatArray:
    """Return the orthogonal high-pass mirror ``g[k] = (-1)^k h[N-1-k]``."""
    taps = low_pass.size
    signs = np.array([(-1.0) ** k for k in range(taps)], dtype=np.float64)
    return np.asarray(signs * low_pass[::-1], dtype=np.float64)


def dwt_energies(windows: FloatArray, wavelet: str, levels: int) -> FloatArray:
    """Relative energy per discrete-wavelet detail level plus the approximation.

    A Mallat cascade: at each level the signal is convolved with the orthogonal
    scaling (low-pass) and wavelet (high-pass) filters and decimated by two,
    with **periodic** extension. Periodic extension is chosen deliberately —
    it is the only extension that keeps the transform exactly orthogonal, which
    makes Parseval's identity (total coefficient energy equals total input
    energy) an exact invariant the tests assert rather than a tolerance. The
    price is wrap-around contamination between the oldest and newest samples of
    a window; every sample involved is still strictly in the past, so this is an
    edge artefact, not a causality violation.

    Energies are returned as **fractions of total window energy**, which makes
    them scale-free and therefore comparable across assets and volatility
    regimes without any fitted normalization.

    Returns:
        ``(n_bars, levels + 1)``: detail energy fractions for levels 1..N
        followed by the final approximation energy fraction.

    Raises:
        ValueError: If the wavelet is unknown, or the window length is not
            divisible by ``2 ** levels`` (which would force a ragged cascade).
    """
    if wavelet not in DAUBECHIES_FILTERS:
        raise ValueError(
            f"unknown wavelet {wavelet!r}; expected one of {sorted(DAUBECHIES_FILTERS)}"
        )
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    length = windows.shape[-1]
    if length % (2**levels) != 0:
        raise ValueError(
            f"window length {length} must be divisible by 2**levels ({2**levels}) "
            "so every cascade stage decimates evenly"
        )
    low_pass = DAUBECHIES_FILTERS[wavelet]
    high_pass = _quadrature_filter(low_pass)

    approximation = windows
    detail_energy: list[FloatArray] = []
    for _ in range(levels):
        low, high = _cascade_stage(approximation, low_pass, high_pass)
        detail_energy.append((high**2).sum(axis=-1))
        approximation = low
    total = (windows**2).sum(axis=-1)
    # A constant window has zero AC energy; report NaN rather than 0/0, so a
    # degenerate window is visibly missing instead of silently "all approximation".
    safe_total = np.where(total > 0.0, total, np.nan)
    columns = [*detail_energy, (approximation**2).sum(axis=-1)]
    return np.asarray(np.stack(columns, axis=-1) / safe_total[:, None], dtype=np.float64)


def _cascade_stage(
    signal: FloatArray, low_pass: FloatArray, high_pass: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """One decimating filter-bank stage under periodic extension."""
    length = signal.shape[-1]
    taps = low_pass.size
    # Periodic extension: index k of the output taps back over wrapped samples.
    out_length = length // 2
    offsets = np.arange(taps)
    starts = 2 * np.arange(out_length)
    index = (starts[:, None] + offsets[None, :]) % length
    gathered = signal[..., index]
    low = gathered @ low_pass
    high = gathered @ high_pass
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
