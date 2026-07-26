"""Tests for adaptive decompositions and the representation contract (SF-S3-MR3).

Grouped by the property each defends: mathematical reference, reconstruction,
mode ordering, scale equivariance, seeded noise, degenerate input, mode mixing,
resource bounds and non-convergence reporting, and the shared comparison
contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_platform.config import RepresentationConfig, SpectralWindow
from quant_platform.features.decomposition import (
    MAX_ENSEMBLES,
    MAX_MODES,
    MIN_SERIES_LENGTH,
    Decomposition,
    DecompositionError,
    DecompositionReport,
    ceemdan,
    eemd,
    emd,
    vmd,
)
from quant_platform.features.representations import (
    REPRESENTATION_FAMILIES,
    _dominant_period,
    build_representation_descriptors,
    family_descriptor_names,
    mode_descriptors,
    shared_window_identity,
)

METHODS = ("emd", "eemd", "ceemdan", "vmd")


def _tone(length: int, period: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * np.arange(length, dtype=float) / period)


def _decompose(method: str, values: np.ndarray, **overrides):
    calls = {
        "emd": lambda: emd(values, max_modes=overrides.get("max_modes", 6)),
        "eemd": lambda: eemd(values, n_ensembles=8, noise_std=0.2, seed=3, max_modes=6),
        "ceemdan": lambda: ceemdan(values, n_ensembles=6, noise_std=0.2, seed=3, max_modes=6),
        "vmd": lambda: vmd(values, n_modes=3),
    }
    return calls[method]()


@pytest.fixture
def two_tone() -> np.ndarray:
    """A clean two-component signal with a linear trend."""
    length = 256
    return _tone(length, 8.0) + _tone(length, 32.0, 0.5) + 0.02 * np.arange(length, dtype=float)


@pytest.fixture
def intermittent() -> np.ndarray:
    """A low-frequency carrier with a short high-frequency burst.

    The canonical mode-mixing stress case: the burst exists in only part of the
    window, so sifting driven by whichever extrema happen to be present pulls
    the carrier into the highest-frequency mode.
    """
    length = 512
    signal = _tone(length, 40.0)
    signal[180:260] += _tone(length, 5.0)[180:260]
    return signal


# ---------------------------------------------------------------------------
# Mathematical reference
# ---------------------------------------------------------------------------


def test_vmd_recovers_known_component_periods() -> None:
    length = 512
    signal = _tone(length, 6.0) + _tone(length, 24.0, 0.8) + _tone(length, 96.0, 0.6)
    decomposition = vmd(signal, n_modes=3, alpha=2000.0)
    periods = sorted(_dominant_period(mode) for mode in decomposition.modes)
    # Period resolution degrades as period^2 / N, so a 96-bar component is
    # resolved an order of magnitude more coarsely than a 6-bar one. The
    # tolerance follows from that, not from convenience.
    for recovered, expected in zip(periods, [6.0, 24.0, 96.0], strict=True):
        assert recovered == pytest.approx(expected, rel=0.1)


def test_emd_separates_two_well_spaced_tones(two_tone: np.ndarray) -> None:
    decomposition = emd(two_tone, max_modes=6)
    assert decomposition.report.n_modes >= 2
    periods = [_dominant_period(mode) for mode in decomposition.modes[:2]]
    assert periods[0] == pytest.approx(8.0, rel=0.2)


def test_vmd_alpha_controls_bandwidth() -> None:
    """A larger bandwidth penalty must produce narrower (more tonal) modes."""
    signal = _tone(512, 10.0) + 0.4 * np.random.default_rng(2).normal(size=512)
    narrow = vmd(signal, n_modes=2, alpha=20_000.0)
    wide = vmd(signal, n_modes=2, alpha=50.0)

    def concentration(decomposition) -> float:
        mode = decomposition.modes[0]
        power = np.abs(np.fft.rfft(mode - mode.mean())) ** 2
        return float(power.max() / power.sum())

    assert concentration(narrow) > concentration(wide)


# ---------------------------------------------------------------------------
# Reconstruction — the governing invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_modes_plus_residual_reconstruct_the_input(method: str, two_tone: np.ndarray) -> None:
    decomposition = _decompose(method, two_tone)
    np.testing.assert_allclose(decomposition.reconstruction, two_tone, atol=1e-9)
    assert decomposition.report.reconstruction_error < 1e-9


@pytest.mark.parametrize("method", METHODS)
def test_report_records_residual_energy(method: str, two_tone: np.ndarray) -> None:
    report = _decompose(method, two_tone).report
    assert report.residual_energy_fraction >= 0.0
    assert report.method == method
    assert report.to_dict()["method"] == method


# ---------------------------------------------------------------------------
# Mode ordering and descriptors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["emd", "vmd"])
def test_deterministic_methods_order_modes_by_frequency(method: str, two_tone: np.ndarray) -> None:
    """EMD sifts highest-frequency first; VMD is explicitly sorted to match."""
    decomposition = _decompose(method, two_tone)
    periods = [_dominant_period(mode) for mode in decomposition.modes]
    finite = [value for value in periods if np.isfinite(value)]
    assert finite == sorted(finite), f"{method} modes are not frequency-ordered: {periods}"


def test_mode_index_is_not_a_timescale_for_noise_assisted_methods(
    two_tone: np.ndarray,
) -> None:
    """Documents a real limitation rather than asserting a guarantee we lack.

    Injecting noise at each CEEMDAN stage can leave a later mode with a higher
    dominant frequency than an earlier one, so the mode index is not a frequency
    rank. Every mode therefore carries a measured ``dominant_period``, and that
    is what a consumer must order and interpret by.
    """
    decomposition = ceemdan(two_tone, n_ensembles=6, noise_std=0.2, seed=3, max_modes=6)
    periods = [_dominant_period(mode) for mode in decomposition.modes]
    finite = [value for value in periods if np.isfinite(value)]
    assert finite != sorted(finite), (
        "fixture no longer demonstrates the ordering caveat; the documented "
        "limitation and this test must be revisited together"
    )
    columns = mode_descriptors(decomposition, n_reported=3)
    assert np.isfinite(columns["mode1_dominant_period"])


def test_mode_descriptors_are_fixed_width_and_scale_free(two_tone: np.ndarray) -> None:
    decomposition = vmd(two_tone, n_modes=3)
    scaled = vmd(two_tone * 7.0, n_modes=3)
    baseline = mode_descriptors(decomposition, n_reported=4)
    rescaled = mode_descriptors(scaled, n_reported=4)
    assert set(baseline) == set(rescaled)
    for name, value in baseline.items():
        if np.isfinite(value):
            assert rescaled[name] == pytest.approx(value, rel=1e-6, abs=1e-9), name


def test_absent_modes_are_missing_not_zero(two_tone: np.ndarray) -> None:
    """Reporting more modes than exist must not fabricate zero-energy modes."""
    decomposition = vmd(two_tone, n_modes=2)
    columns = mode_descriptors(decomposition, n_reported=5)
    assert np.isnan(columns["mode5_energy_fraction"])
    assert np.isfinite(columns["mode1_energy_fraction"])


def test_mode_descriptors_reject_an_invalid_width(two_tone: np.ndarray) -> None:
    with pytest.raises(DecompositionError, match="at least one"):
        mode_descriptors(vmd(two_tone, n_modes=2), n_reported=0)


# ---------------------------------------------------------------------------
# Scale equivariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [0.001, 5.0, 1000.0])
@pytest.mark.parametrize("method", ["emd", "vmd"])
def test_decomposition_is_scale_equivariant(
    method: str, factor: float, two_tone: np.ndarray
) -> None:
    """Scaling the input must scale the modes, not change the decomposition."""
    baseline = _decompose(method, two_tone)
    scaled = _decompose(method, two_tone * factor)
    assert scaled.report.n_modes == baseline.report.n_modes
    np.testing.assert_allclose(
        scaled.modes, baseline.modes * factor, rtol=1e-7, atol=1e-9 * abs(factor)
    )


# ---------------------------------------------------------------------------
# Seeded noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["eemd", "ceemdan"])
def test_noise_assisted_methods_are_reproducible(method: str, two_tone: np.ndarray) -> None:
    call = eemd if method == "eemd" else ceemdan
    first = call(two_tone, n_ensembles=6, noise_std=0.2, seed=11, max_modes=5)
    second = call(two_tone, n_ensembles=6, noise_std=0.2, seed=11, max_modes=5)
    np.testing.assert_array_equal(first.modes, second.modes)
    reseeded = call(two_tone, n_ensembles=6, noise_std=0.2, seed=12, max_modes=5)
    assert not np.array_equal(first.modes, reseeded.modes)
    assert first.report.seed == 11
    assert first.report.n_ensembles == 6
    assert first.report.noise_std == pytest.approx(0.2)


def test_zero_noise_ensemble_reduces_to_plain_emd(two_tone: np.ndarray) -> None:
    """Without noise, the ensemble is an average of identical decompositions."""
    plain = emd(two_tone, max_modes=5)
    ensemble = eemd(two_tone, n_ensembles=3, noise_std=0.0, seed=1, max_modes=5)
    np.testing.assert_allclose(ensemble.modes, plain.modes, atol=1e-9)


# ---------------------------------------------------------------------------
# Mode mixing — the reason the noise-assisted variants exist
# ---------------------------------------------------------------------------


def test_ceemdan_removes_the_carrier_leak_that_emd_suffers(intermittent: np.ndarray) -> None:
    """The classic mode-mixing demonstration, measured rather than asserted.

    On an intermittent signal, plain EMD pulls the low-frequency carrier into
    its *highest*-frequency mode, because sifting follows whichever extrema
    exist. Populating scale space with noise and averaging removes that leak.
    """
    frequencies = np.fft.rfftfreq(intermittent.size)
    carrier_band = (frequencies > 1.0 / 60.0) & (frequencies < 1.0 / 25.0)

    def carrier_leak(mode: np.ndarray) -> float:
        power = np.abs(np.fft.rfft(mode - mode.mean())) ** 2
        return float(power[carrier_band].sum() / power.sum())

    plain = carrier_leak(emd(intermittent, max_modes=6).modes[0])
    assisted = carrier_leak(
        ceemdan(intermittent, n_ensembles=20, noise_std=0.25, seed=3, max_modes=6).modes[0]
    )
    assert plain > 0.4, "the fixture must actually exhibit mode mixing under EMD"
    assert assisted < 0.05
    assert assisted < plain / 10.0


def test_ceemdan_mode_index_is_stable_under_perturbation(intermittent: np.ndarray) -> None:
    """CEEMDAN extracts from a shared residual, so the mode count is stable."""
    rng = np.random.default_rng(9)
    counts = {
        ceemdan(
            intermittent + 1e-3 * rng.normal(size=intermittent.size),
            n_ensembles=6,
            noise_std=0.25,
            seed=1,
            max_modes=8,
        ).report.n_modes
        for _ in range(3)
    }
    assert len(counts) == 1


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_short_series_are_refused(method: str) -> None:
    with pytest.raises(DecompositionError, match="at least"):
        _decompose(method, np.zeros(MIN_SERIES_LENGTH - 1))


@pytest.mark.parametrize("method", METHODS)
def test_non_finite_series_are_refused(method: str, two_tone: np.ndarray) -> None:
    corrupt = two_tone.copy()
    corrupt[10] = np.nan
    with pytest.raises(DecompositionError, match="non-finite"):
        _decompose(method, corrupt)


@pytest.mark.parametrize("method", METHODS)
def test_multidimensional_input_is_refused(method: str) -> None:
    with pytest.raises(DecompositionError, match="one-dimensional"):
        _decompose(method, np.zeros((32, 2)))


def test_constant_and_monotonic_series_terminate_immediately() -> None:
    constant = emd(np.full(64, 3.0), max_modes=6)
    assert constant.report.n_modes == 0
    assert constant.report.stopping_reason == "monotonic_residual"
    np.testing.assert_allclose(constant.reconstruction, np.full(64, 3.0))

    ramp = emd(np.arange(64, dtype=float), max_modes=6)
    assert ramp.report.n_modes == 0
    np.testing.assert_allclose(ramp.reconstruction, np.arange(64, dtype=float))


def test_vmd_handles_a_constant_series() -> None:
    decomposition = vmd(np.full(64, 2.5), n_modes=2)
    np.testing.assert_allclose(decomposition.reconstruction, np.full(64, 2.5), atol=1e-9)


# ---------------------------------------------------------------------------
# Resource bounds and non-convergence reporting
# ---------------------------------------------------------------------------


def test_exhausted_mode_budget_is_reported_as_unconverged(two_tone: np.ndarray) -> None:
    truncated = emd(two_tone, max_modes=1)
    assert truncated.report.stopping_reason == "max_modes"
    assert truncated.report.converged is False
    natural = emd(two_tone, max_modes=8)
    assert natural.report.stopping_reason == "monotonic_residual"
    assert natural.report.converged is True


def test_exhausted_vmd_budget_is_reported_as_unconverged(two_tone: np.ndarray) -> None:
    starved = vmd(two_tone, n_modes=3, max_iterations=1)
    assert starved.report.converged is False
    assert starved.report.stopping_reason == "max_iterations"
    assert starved.report.iterations == starved.report.max_iterations
    settled = vmd(two_tone, n_modes=3, max_iterations=500)
    assert settled.report.converged is True
    assert settled.report.stopping_reason == "dual_tolerance"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda x: emd(x, max_modes=MAX_MODES + 1), "max_modes"),
        (lambda x: emd(x, max_modes=0), "max_modes"),
        (lambda x: emd(x, max_sift_iterations=0), "max_sift_iterations"),
        (lambda x: emd(x, sd_tolerance=0.0), "sd_tolerance"),
        (lambda x: eemd(x, n_ensembles=MAX_ENSEMBLES + 1), "n_ensembles"),
        (lambda x: eemd(x, noise_std=-1.0), "noise_std"),
        (lambda x: eemd(x, seed=-1), "seed"),
        (lambda x: ceemdan(x, noise_std=float("nan")), "noise_std"),
        (lambda x: ceemdan(x, seed=-5), "seed"),
        (lambda x: vmd(x, n_modes=0), "n_modes"),
        (lambda x: vmd(x, alpha=0.0), "alpha"),
        (lambda x: vmd(x, tau=-1.0), "tau"),
        (lambda x: vmd(x, tolerance=0.0), "tolerance"),
        (lambda x: vmd(x, max_iterations=0), "max_iterations"),
    ],
)
def test_out_of_range_parameters_are_refused(call, message: str, two_tone: np.ndarray) -> None:
    with pytest.raises(DecompositionError, match=message):
        call(two_tone)


def test_report_invariants_fail_closed() -> None:
    fields = {
        "method": "emd",
        "n_modes": 2,
        "iterations": 5,
        "max_iterations": 10,
        "converged": True,
        "stopping_reason": "monotonic_residual",
        "residual_energy_fraction": 0.1,
        "reconstruction_error": 1e-12,
    }
    with pytest.raises(ValueError, match="non-negative"):
        DecompositionReport(**{**fields, "n_modes": -1})
    with pytest.raises(ValueError, match="cannot exceed"):
        DecompositionReport(**{**fields, "iterations": 99})
    with pytest.raises(ValueError, match="residual energy"):
        DecompositionReport(**{**fields, "residual_energy_fraction": float("inf")})
    with pytest.raises(ValueError, match="reconstruction error"):
        DecompositionReport(**{**fields, "reconstruction_error": -1.0})


def test_decomposition_rejects_mismatched_shapes() -> None:
    report = DecompositionReport(
        method="emd",
        n_modes=1,
        iterations=1,
        max_iterations=2,
        converged=True,
        stopping_reason="monotonic_residual",
        residual_energy_fraction=0.0,
        reconstruction_error=0.0,
    )
    with pytest.raises(ValueError, match="one-dimensional|residual length"):
        Decomposition(modes=np.zeros((1, 8)), residual=np.zeros(9), report=report)
    with pytest.raises(ValueError, match="n_modes, n_samples"):
        Decomposition(modes=np.zeros(8), residual=np.zeros(8), report=report)


# ---------------------------------------------------------------------------
# The shared comparison contract
# ---------------------------------------------------------------------------


@pytest.fixture
def series() -> np.ndarray:
    rng = np.random.default_rng(5)
    return _tone(300, 8.0) + 0.5 * _tone(300, 32.0) + 0.3 * rng.normal(size=300)


@pytest.fixture
def representation_config() -> RepresentationConfig:
    return RepresentationConfig(enabled=True, stride=8, n_reported_modes=3, max_modes=6)


@pytest.mark.parametrize("family", REPRESENTATION_FAMILIES)
def test_every_family_emits_its_declared_columns(
    family: str, series: np.ndarray, representation_config: RepresentationConfig
) -> None:
    columns = build_representation_descriptors(series, family, representation_config)
    assert tuple(sorted(columns)) == tuple(
        sorted(family_descriptor_names(family, representation_config))
    )
    for name, values in columns.items():
        assert values.shape == series.shape, name


@pytest.mark.parametrize("family", REPRESENTATION_FAMILIES)
def test_every_family_respects_the_same_warmup(
    family: str, series: np.ndarray, representation_config: RepresentationConfig
) -> None:
    """The arms must differ in representation only — including their warm-up."""
    columns = build_representation_descriptors(series, family, representation_config)
    warmup = representation_config.window.length - 1
    for name, values in columns.items():
        assert np.isnan(values[:warmup]).all(), f"{family}.{name} produced a warm-up value"


def test_stride_leaves_honest_gaps_not_forward_fills(
    series: np.ndarray, representation_config: RepresentationConfig
) -> None:
    columns = build_representation_descriptors(series, "emd", representation_config)
    values = columns["mode1_energy_fraction"]
    warmup = representation_config.window.length - 1
    evaluated = np.arange(warmup, series.size, representation_config.stride)
    skipped = np.setdiff1d(np.arange(warmup, series.size), evaluated)
    assert np.isfinite(values[evaluated]).any()
    assert np.isnan(values[skipped]).all()


def test_families_share_one_preprocessing_contract(
    representation_config: RepresentationConfig,
) -> None:
    identity = shared_window_identity(representation_config)
    assert identity["length"] == representation_config.window.length
    assert identity["detrend"] == representation_config.window.detrend
    assert identity["stride"] == representation_config.stride
    assert identity["frequency_unit"] == "cycles_per_bar"


def test_comparison_is_deterministic(
    series: np.ndarray, representation_config: RepresentationConfig
) -> None:
    for family in REPRESENTATION_FAMILIES:
        first = build_representation_descriptors(series, family, representation_config)
        second = build_representation_descriptors(series, family, representation_config)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])


def test_unknown_family_is_refused(
    series: np.ndarray, representation_config: RepresentationConfig
) -> None:
    with pytest.raises(ValueError, match="unknown representation family"):
        build_representation_descriptors(series, "fourier", representation_config)
    with pytest.raises(ValueError, match="unknown representation family"):
        family_descriptor_names("fourier", representation_config)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"families": ["raw", "raw"]}, "unique"),
        ({"n_reported_modes": 9, "max_modes": 4}, "cannot exceed"),
        (
            {
                "dwt_levels": 5,
                "window": SpectralWindow(length=48, segment_length=16, hop=8, n_fft=16),
            },
            "divisible",
        ),
    ],
)
def test_representation_config_rejects_incoherent_settings(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RepresentationConfig(enabled=True, **kwargs)


def test_windows_containing_missing_data_yield_missing_descriptors(
    representation_config: RepresentationConfig,
) -> None:
    values = _tone(300, 8.0)
    values[100] = np.nan
    columns = build_representation_descriptors(values, "emd", representation_config)
    energy = columns["mode1_energy_fraction"]
    covering = np.arange(100, 100 + representation_config.window.length)
    assert np.isnan(energy[covering]).all()
