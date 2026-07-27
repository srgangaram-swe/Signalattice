"""Bounded adaptive signal decompositions: EMD, EEMD, CEEMDAN, VMD (SF-S3-MR3).

Fixed transforms (Fourier, wavelets) project a series onto a basis chosen in
advance. Adaptive decompositions derive their basis *from the data*, which is
what makes them attractive for non-stationary series — and what makes them
dangerous: the basis is a function of the sample, so an unstable decomposition
produces modes that look like structure and are an artefact of the particular
noise realization.

Every algorithm here is therefore built around three refusals:

**Nothing runs unbounded.** Sifting iterations, mode counts, ensemble sizes, and
ADMM iterations all have explicit ceilings. An adaptive method with no iteration
cap can spin on a pathological window forever.

**Non-convergence is never silent.** Every call returns a
:class:`DecompositionReport` stating how it terminated — the stopping criterion
that fired, iterations used against the budget, and an explicit ``converged``
flag. The issue's non-goal is exactly this: silently accepting non-convergence.

**Randomness is seeded and recorded.** EEMD and CEEMDAN add noise. Each
realization draws from a named child stream of a recorded root seed, so a run is
reproducible bit-for-bit and the seed is part of the evidence.

Reconstruction is the governing invariant: for every method, the modes plus the
residual sum back to the input to floating-point tolerance. A decomposition that
does not reconstruct is not a decomposition, and the tests assert it for all four.

References implemented from the primary literature:

* EMD — Huang et al. (1998), sifting with cubic-spline extrema envelopes and the
  Cauchy-type standard-deviation stopping criterion.
* EEMD — Wu & Huang (2009), noise-assisted ensemble averaging.
* CEEMDAN — Torres et al. (2011), complete ensemble with adaptive noise.
* VMD — Dragomiretskiy & Zosso (2014), ADMM in the frequency domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

FloatArray = NDArray[np.float64]

#: Hard ceilings. Refusal thresholds, not tuning knobs.
MAX_MODES = 20
MAX_SIFT_ITERATIONS = 2_000
MAX_ENSEMBLES = 500
MAX_VMD_ITERATIONS = 5_000
MAX_SERIES_LENGTH = 100_000

#: Minimum samples for a decomposition to mean anything. Below this the extrema
#: envelopes are interpolated through so few knots that the "modes" describe the
#: spline, not the series.
MIN_SERIES_LENGTH = 16

StoppingReason = Literal[
    "amplitude_tolerance",
    "monotonic_residual",
    "max_modes",
    "max_iterations",
    "dual_tolerance",
    "degenerate_input",
]


class DecompositionError(ValueError):
    """Raised when a decomposition request is unusable or out of bounds."""


@dataclass(frozen=True)
class DecompositionReport:
    """How one decomposition terminated.

    Attributes:
        method: ``emd`` / ``eemd`` / ``ceemdan`` / ``vmd``.
        n_modes: Modes produced, excluding the residual.
        iterations: Sifting or ADMM iterations actually consumed.
        max_iterations: The configured iteration budget.
        converged: ``True`` when a stopping criterion fired, ``False`` when the
            budget ran out. Spending the whole budget is not convergence.
        stopping_reason: Which criterion ended the run.
        residual_energy_fraction: Energy left in the final residual, as a
            fraction of the input's. A large value means the modes explain
            little and the decomposition should not be read as a description.
        reconstruction_error: Max absolute difference between the input and the
            summed modes plus residual. Published rather than assumed.
        seed: Root seed for noise-assisted methods, else ``None``.
        n_ensembles: Realizations averaged, for noise-assisted methods.
        noise_std: Noise amplitude as a fraction of the input's standard
            deviation, for noise-assisted methods.
    """

    method: str
    n_modes: int
    iterations: int
    max_iterations: int
    converged: bool
    stopping_reason: StoppingReason
    residual_energy_fraction: float
    reconstruction_error: float
    seed: int | None = None
    n_ensembles: int | None = None
    noise_std: float | None = None

    def __post_init__(self) -> None:
        if self.n_modes < 0 or self.iterations < 0:
            raise ValueError("decomposition counts must be non-negative")
        if self.iterations > self.max_iterations:
            raise ValueError("iterations cannot exceed the configured budget")
        if not np.isfinite(self.residual_energy_fraction) or self.residual_energy_fraction < 0.0:
            raise ValueError("residual energy fraction must be finite and non-negative")
        if not np.isfinite(self.reconstruction_error) or self.reconstruction_error < 0.0:
            raise ValueError("reconstruction error must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record for run manifests."""
        return {
            "method": self.method,
            "n_modes": self.n_modes,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "converged": self.converged,
            "stopping_reason": self.stopping_reason,
            "residual_energy_fraction": self.residual_energy_fraction,
            "reconstruction_error": self.reconstruction_error,
            "seed": self.seed,
            "n_ensembles": self.n_ensembles,
            "noise_std": self.noise_std,
        }


@dataclass(frozen=True)
class Decomposition:
    """Modes, residual, and termination evidence from one decomposition.

    ``modes`` is ``(n_modes, n_samples)``.

    **Mode order is not a frequency guarantee.** EMD sifts highest-frequency
    first and VMD is explicitly sorted that way, so for those two the index is
    also a frequency rank. The noise-assisted variants offer no such guarantee:
    injecting noise at each stage can produce a later mode with a *higher*
    dominant frequency than an earlier one. That is why every mode carries a
    measured ``dominant_period`` descriptor — a consumer should order and
    interpret modes by that, never by trusting the index to mean a timescale.
    """

    modes: FloatArray
    residual: FloatArray
    report: DecompositionReport

    def __post_init__(self) -> None:
        if self.modes.ndim != 2:
            raise ValueError("modes must be a (n_modes, n_samples) array")
        if self.residual.shape != self.modes.shape[1:]:
            raise ValueError("residual length must match the mode length")

    @property
    def reconstruction(self) -> FloatArray:
        """Return the summed modes plus residual."""
        return np.asarray(self.modes.sum(axis=0) + self.residual, dtype=np.float64)


def _validate_series(values: FloatArray) -> FloatArray:
    """Return a validated 1-D finite series, or fail closed."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise DecompositionError(f"series must be one-dimensional, got shape {array.shape}")
    if array.size < MIN_SERIES_LENGTH:
        raise DecompositionError(
            f"series must hold at least {MIN_SERIES_LENGTH} samples, got {array.size}"
        )
    if array.size > MAX_SERIES_LENGTH:
        raise DecompositionError(
            f"series length {array.size} exceeds the {MAX_SERIES_LENGTH} ceiling"
        )
    if not np.isfinite(array).all():
        raise DecompositionError("series contains non-finite values; decomposition fails closed")
    return array


def _extrema(values: FloatArray) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return indices of interior local maxima and minima."""
    difference = np.diff(values)
    # A plateau has zero slope; treating it as neither rising nor falling keeps
    # a flat stretch from registering as a dense run of extrema.
    sign = np.sign(difference)
    maxima = np.flatnonzero((sign[:-1] > 0) & (sign[1:] < 0)) + 1
    minima = np.flatnonzero((sign[:-1] < 0) & (sign[1:] > 0)) + 1
    return maxima.astype(np.int64), minima.astype(np.int64)


def _mirror_extend(
    indices: NDArray[np.int64], values: FloatArray, length: int
) -> tuple[FloatArray, FloatArray]:
    """Mirror the outermost extrema beyond both ends of the series.

    A cubic spline through interior extrema only is wildly unconstrained at the
    boundaries, and the resulting envelope error propagates inward through every
    subsequent sifting iteration — the classic EMD end effect. Reflecting the
    two outermost extrema about each endpoint pins the spline without inventing
    amplitude that the series does not contain.
    """
    positions = indices.astype(np.float64)
    amplitudes = values[indices]
    if positions.size >= 2:
        left = np.array([-positions[1], -positions[0]], dtype=np.float64)
        left_values = np.array([amplitudes[1], amplitudes[0]], dtype=np.float64)
        right = np.array(
            [2.0 * (length - 1) - positions[-1], 2.0 * (length - 1) - positions[-2]],
            dtype=np.float64,
        )
        right_values = np.array([amplitudes[-1], amplitudes[-2]], dtype=np.float64)
    else:
        left = np.array([-positions[0]], dtype=np.float64)
        left_values = amplitudes[:1]
        right = np.array([2.0 * (length - 1) - positions[-1]], dtype=np.float64)
        right_values = amplitudes[-1:]
    extended_positions = np.concatenate([left, positions, right])
    extended_values = np.concatenate([left_values, amplitudes, right_values])
    order = np.argsort(extended_positions, kind="stable")
    unique_positions, unique_index = np.unique(extended_positions[order], return_index=True)
    return unique_positions, extended_values[order][unique_index]


def _envelope_mean(values: FloatArray) -> FloatArray | None:
    """Return the mean of the upper and lower extrema envelopes, or ``None``.

    ``None`` means the series has too few extrema to build envelopes, which is
    the signal that sifting has reached a monotonic residual.
    """
    length = values.size
    maxima, minima = _extrema(values)
    if maxima.size < 1 or minima.size < 1:
        return None
    grid = np.arange(length, dtype=np.float64)
    upper_x, upper_y = _mirror_extend(maxima, values, length)
    lower_x, lower_y = _mirror_extend(minima, values, length)
    if upper_x.size < 2 or lower_x.size < 2:
        return None
    upper = CubicSpline(upper_x, upper_y, extrapolate=True)(grid)
    lower = CubicSpline(lower_x, lower_y, extrapolate=True)(grid)
    return np.asarray((upper + lower) / 2.0, dtype=np.float64)


def _sift(
    values: FloatArray, *, sd_tolerance: float, max_iterations: int
) -> tuple[FloatArray, int, bool]:
    """Sift one intrinsic mode function out of ``values``.

    The Cauchy-type stopping criterion of Huang et al.: iterate until the
    relative change between successive candidates falls below ``sd_tolerance``.
    Returns ``(imf, iterations, converged)``; ``converged`` is ``False`` when the
    iteration budget was exhausted instead.
    """
    candidate = values.copy()
    for iteration in range(1, max_iterations + 1):
        mean = _envelope_mean(candidate)
        if mean is None:
            return candidate, iteration, True
        following = candidate - mean
        denominator = float(np.sum(candidate**2))
        if denominator <= 0.0:
            return following, iteration, True
        deviation = float(np.sum((candidate - following) ** 2) / denominator)
        candidate = following
        if deviation < sd_tolerance:
            return candidate, iteration, True
    return candidate, max_iterations, False


def emd(
    values: FloatArray,
    *,
    max_modes: int = 8,
    sd_tolerance: float = 0.2,
    max_sift_iterations: int = 100,
) -> Decomposition:
    """Empirical Mode Decomposition (Huang et al., 1998).

    Repeatedly sifts the highest-frequency intrinsic mode out of the residual
    until the residual is monotonic (no extrema left to build an envelope from)
    or the mode budget is spent.

    Args:
        values: Finite 1-D series.
        max_modes: Mode ceiling, at most :data:`MAX_MODES`.
        sd_tolerance: Cauchy stopping threshold for each sift.
        max_sift_iterations: Per-mode sifting budget.

    Raises:
        DecompositionError: On an unusable series or an out-of-range bound.
    """
    series = _validate_series(values)
    _check_bounds(max_modes=max_modes, max_sift_iterations=max_sift_iterations)
    if sd_tolerance <= 0.0 or not np.isfinite(sd_tolerance):
        raise DecompositionError("sd_tolerance must be finite and positive")

    residual = series.copy()
    modes: list[FloatArray] = []
    iterations = 0
    converged = True
    reason: StoppingReason = "monotonic_residual"
    for _ in range(max_modes):
        if _envelope_mean(residual) is None:
            reason = "monotonic_residual"
            break
        mode, used, mode_converged = _sift(
            residual, sd_tolerance=sd_tolerance, max_iterations=max_sift_iterations
        )
        iterations += used
        converged = converged and mode_converged
        if not mode_converged:
            reason = "max_iterations"
        modes.append(mode)
        residual = residual - mode
    else:
        # The residual still has extrema, so the decomposition is truncated
        # rather than finished. Spending a budget is not convergence.
        reason = "max_modes"
        converged = False

    return _finalize(
        series,
        modes,
        residual,
        method="emd",
        iterations=iterations,
        max_iterations=max_modes * max_sift_iterations,
        converged=converged,
        reason=reason,
    )


def eemd(
    values: FloatArray,
    *,
    n_ensembles: int = 50,
    noise_std: float = 0.2,
    seed: int = 42,
    max_modes: int = 8,
    sd_tolerance: float = 0.2,
    max_sift_iterations: int = 100,
) -> Decomposition:
    """Ensemble EMD (Wu & Huang, 2009).

    Plain EMD suffers **mode mixing**: an intermittent component makes a single
    IMF carry oscillations of very different scales, because sifting follows the
    extrema it happens to find. Adding white noise populates the whole scale
    space uniformly, so each realization's sifting is anchored by a dense,
    unbiased set of extrema; averaging across realizations then cancels the added
    noise at rate ``1/sqrt(n_ensembles)`` while the true components survive.

    The residual noise is the price: it does not vanish, so EEMD's modes do not
    reconstruct the input exactly the way EMD's do. This implementation returns
    the residual as ``input - sum(modes)``, which makes reconstruction exact by
    construction and pushes the ensemble's leftover noise into the residual where
    it is visible in ``residual_energy_fraction`` rather than hidden.

    Args:
        n_ensembles: Realizations to average, at most :data:`MAX_ENSEMBLES`.
        noise_std: Added-noise amplitude as a fraction of the series' standard
            deviation.
        seed: Root seed; realization ``i`` draws from a named child stream.
    """
    series = _validate_series(values)
    _check_bounds(
        max_modes=max_modes,
        max_sift_iterations=max_sift_iterations,
        n_ensembles=n_ensembles,
    )
    if noise_std < 0.0 or not np.isfinite(noise_std):
        raise DecompositionError("noise_std must be finite and non-negative")
    if seed < 0:
        raise DecompositionError("seed must be non-negative")

    scale = float(np.std(series)) * noise_std
    accumulated = np.zeros((max_modes, series.size), dtype=np.float64)
    counts = np.zeros(max_modes, dtype=np.float64)
    iterations = 0
    converged = True
    for realization in range(n_ensembles):
        noise = _child_stream(seed, "eemd", realization).normal(0.0, 1.0, series.size) * scale
        trial = emd(
            series + noise,
            max_modes=max_modes,
            sd_tolerance=sd_tolerance,
            max_sift_iterations=max_sift_iterations,
        )
        iterations += trial.report.iterations
        converged = converged and trial.report.converged
        for index, mode in enumerate(trial.modes):
            accumulated[index] += mode
            counts[index] += 1.0
    used = counts > 0.0
    modes = [accumulated[index] / counts[index] for index in np.flatnonzero(used)]
    residual = series - np.sum(modes, axis=0) if modes else series.copy()
    return _finalize(
        series,
        modes,
        residual,
        method="eemd",
        iterations=iterations,
        max_iterations=n_ensembles * max_modes * max_sift_iterations,
        converged=converged,
        reason="amplitude_tolerance" if converged else "max_iterations",
        seed=seed,
        n_ensembles=n_ensembles,
        noise_std=noise_std,
    )


def ceemdan(
    values: FloatArray,
    *,
    n_ensembles: int = 30,
    noise_std: float = 0.2,
    seed: int = 42,
    max_modes: int = 8,
    sd_tolerance: float = 0.2,
    max_sift_iterations: int = 100,
) -> Decomposition:
    """Complete EEMD with Adaptive Noise (Torres et al., 2011).

    EEMD averages *independent* decompositions, so its modes do not sum back to
    the signal and the mode count can differ between realizations. CEEMDAN fixes
    both by extracting one mode at a time from a **shared** residual: at stage
    *k* the ensemble decomposes ``residual + beta_k * E_k(noise_i)`` and averages
    only the first IMF, then subtracts it. Reconstruction is therefore exact by
    construction, and every realization contributes to the same mode index.

    ``beta_k`` scales the injected noise to the current residual's amplitude, so
    later stages — whose residuals are small — are not swamped by noise sized for
    the original series.
    """
    series = _validate_series(values)
    _check_bounds(
        max_modes=max_modes,
        max_sift_iterations=max_sift_iterations,
        n_ensembles=n_ensembles,
    )
    if noise_std < 0.0 or not np.isfinite(noise_std):
        raise DecompositionError("noise_std must be finite and non-negative")
    if seed < 0:
        raise DecompositionError("seed must be non-negative")

    residual = series.copy()
    modes: list[FloatArray] = []
    iterations = 0
    converged = True
    reason: StoppingReason = "monotonic_residual"
    for stage in range(max_modes):
        if _envelope_mean(residual) is None:
            reason = "monotonic_residual"
            break
        scale = float(np.std(residual)) * noise_std
        accumulated = np.zeros(series.size, dtype=np.float64)
        for realization in range(n_ensembles):
            noise = (
                _child_stream(seed, "ceemdan", stage * MAX_ENSEMBLES + realization).normal(
                    0.0, 1.0, series.size
                )
                * scale
            )
            mode, used, mode_converged = _sift(
                residual + noise,
                sd_tolerance=sd_tolerance,
                max_iterations=max_sift_iterations,
            )
            iterations += used
            converged = converged and mode_converged
            if not mode_converged:
                reason = "max_iterations"
            accumulated += mode
        mode_mean = accumulated / n_ensembles
        modes.append(mode_mean)
        residual = residual - mode_mean
    else:
        # As in EMD: a spent mode budget leaves an undecomposed residual.
        reason = "max_modes"
        converged = False

    return _finalize(
        series,
        modes,
        residual,
        method="ceemdan",
        iterations=iterations,
        max_iterations=max_modes * n_ensembles * max_sift_iterations,
        converged=converged,
        reason=reason,
        seed=seed,
        n_ensembles=n_ensembles,
        noise_std=noise_std,
    )


def vmd(
    values: FloatArray,
    *,
    n_modes: int = 4,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tolerance: float = 1e-7,
    max_iterations: int = 500,
) -> Decomposition:
    """Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014).

    Unlike EMD's greedy sifting, VMD solves a single variational problem: find
    ``n_modes`` band-limited components, each compact around its own centre
    frequency, whose sum reproduces the signal. That makes it far more robust to
    noise and free of mode mixing by construction — at the cost of having to
    choose the mode count in advance, which EMD infers.

    The solution alternates, in the frequency domain:

    * each mode is a Wiener-filtered residual, ``(f - sum_{i != k} u_i +
      lambda/2) / (1 + alpha (omega - omega_k)^2)``, so ``alpha`` directly sets
      the bandwidth penalty — larger means narrower modes;
    * each centre frequency moves to its mode's power centroid;
    * the dual variable ``lambda`` enforces exact reconstruction when
      ``tau > 0``. It defaults to ``0`` (noise-tolerant mode), which relaxes the
      constraint; the leftover appears in the residual rather than being hidden.

    The series is mirror-extended before transforming so the periodic FFT does
    not wrap the end of the window onto its start.
    """
    series = _validate_series(values)
    if not 1 <= n_modes <= MAX_MODES:
        raise DecompositionError(f"n_modes must be in [1, {MAX_MODES}], got {n_modes}")
    if not 1 <= max_iterations <= MAX_VMD_ITERATIONS:
        raise DecompositionError(
            f"max_iterations must be in [1, {MAX_VMD_ITERATIONS}], got {max_iterations}"
        )
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise DecompositionError("alpha must be finite and positive")
    if tau < 0.0 or not np.isfinite(tau):
        raise DecompositionError("tau must be finite and non-negative")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise DecompositionError("tolerance must be finite and positive")

    length = series.size
    mirrored = np.concatenate(
        [series[length // 2 : 0 : -1], series, series[-2 : -length // 2 - 2 : -1]]
    )
    total = mirrored.size
    frequencies = np.fft.fftfreq(total)
    spectrum = np.fft.fft(mirrored)

    modes_hat = np.zeros((n_modes, total), dtype=np.complex128)
    # Log-spaced initialization across the resolvable band. Linear spacing puts
    # almost every starting centre frequency in the top decade, so on a signal
    # whose components span decades — the normal case for market data, where a
    # weekly and an annual cycle differ by ~50x — several modes start above every
    # true component, converge onto the same one, and the decomposition silently
    # reports a duplicated mode. This is the initialization sensitivity VMD is
    # known for; log spacing is the standard remedy and is used deliberately.
    lowest = 2.0 / total
    omega = np.geomspace(lowest, 0.5, n_modes, dtype=np.float64)
    dual = np.zeros(total, dtype=np.complex128)
    positive = frequencies >= 0.0

    iterations = 0
    converged = False
    for iterations in range(1, max_iterations + 1):  # noqa: B007 - final value is reported
        previous = modes_hat.copy()
        for index in range(n_modes):
            others = modes_hat.sum(axis=0) - modes_hat[index]
            numerator = spectrum - others + dual / 2.0
            modes_hat[index] = numerator / (1.0 + alpha * (frequencies - omega[index]) ** 2)
            power = np.abs(modes_hat[index][positive]) ** 2
            weight = float(power.sum())
            if weight > 0.0:
                omega[index] = float((frequencies[positive] * power).sum() / weight)
        if tau > 0.0:
            dual = dual + tau * (spectrum - modes_hat.sum(axis=0))
        change = float(np.sum(np.abs(modes_hat - previous) ** 2))
        scale = float(np.sum(np.abs(previous) ** 2))
        if scale > 0.0 and change / scale < tolerance:
            converged = True
            break

    order = np.argsort(-omega)  # highest frequency first, matching EMD's order
    start = length // 2
    modes = [
        np.asarray(np.real(np.fft.ifft(modes_hat[index]))[start : start + length], dtype=np.float64)
        for index in order
    ]
    residual = series - np.sum(modes, axis=0)
    return _finalize(
        series,
        modes,
        residual,
        method="vmd",
        iterations=iterations,
        max_iterations=max_iterations,
        converged=converged,
        reason="dual_tolerance" if converged else "max_iterations",
    )


def _child_stream(seed: int, label: str, index: int) -> np.random.Generator:
    """Return a named, reproducible child random stream.

    Deriving each realization from ``(seed, label, index)`` rather than from a
    single advancing generator means realization *i* is identical whether the
    ensemble runs in order, in parallel, or is resumed — a property a shared
    mutable generator cannot offer.
    """
    entropy = np.random.SeedSequence([seed, abs(hash(label)) % (2**32), index])
    return np.random.default_rng(entropy)


def _check_bounds(
    *, max_modes: int, max_sift_iterations: int, n_ensembles: int | None = None
) -> None:
    """Validate the shared iteration and size ceilings."""
    if not 1 <= max_modes <= MAX_MODES:
        raise DecompositionError(f"max_modes must be in [1, {MAX_MODES}], got {max_modes}")
    if not 1 <= max_sift_iterations <= MAX_SIFT_ITERATIONS:
        raise DecompositionError(
            f"max_sift_iterations must be in [1, {MAX_SIFT_ITERATIONS}], "
            f"got {max_sift_iterations}"
        )
    if n_ensembles is not None and not 1 <= n_ensembles <= MAX_ENSEMBLES:
        raise DecompositionError(f"n_ensembles must be in [1, {MAX_ENSEMBLES}], got {n_ensembles}")


def _finalize(
    series: FloatArray,
    modes: list[FloatArray],
    residual: FloatArray,
    *,
    method: str,
    iterations: int,
    max_iterations: int,
    converged: bool,
    reason: StoppingReason,
    seed: int | None = None,
    n_ensembles: int | None = None,
    noise_std: float | None = None,
) -> Decomposition:
    """Assemble a decomposition and measure its reconstruction error."""
    stacked = np.stack(modes, axis=0) if modes else np.zeros((0, series.size), dtype=np.float64)
    reconstruction = stacked.sum(axis=0) + residual
    error = float(np.max(np.abs(series - reconstruction))) if series.size else 0.0
    energy = float(np.sum(series**2))
    residual_fraction = float(np.sum(residual**2) / energy) if energy > 0.0 else 0.0
    report = DecompositionReport(
        method=method,
        n_modes=stacked.shape[0],
        iterations=min(iterations, max_iterations),
        max_iterations=max_iterations,
        converged=converged,
        stopping_reason=reason,
        residual_energy_fraction=residual_fraction,
        reconstruction_error=error,
        seed=seed,
        n_ensembles=n_ensembles,
        noise_std=noise_std,
    )
    return Decomposition(modes=stacked, residual=residual, report=report)
