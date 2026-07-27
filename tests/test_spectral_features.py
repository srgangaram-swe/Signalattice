"""Tests for the causal spectral transform and descriptor engine (SF-S3-MR1).

Organised by the property each group defends:

* **Reference** — the transforms reproduce independent references (scipy's
  Welch and STFT) and closed-form analytic results (Haar coefficients,
  Parseval's identity, the frequency of a known tone).
* **Causality** — mutating the future cannot change any already-computed value.
  This is the leakage-mutation test; everything else is secondary to it.
* **Property / metamorphic** — scale invariance, bar-time semantics under
  calendar gaps, and descriptor range invariants.
* **Numerical stability** — constant series, extreme amplitudes, degenerate
  windows, NaN propagation.
* **Deterministic replay** — identical inputs produce bit-identical outputs.
* **Integration** — registry and emitted columns agree exactly.
* **Malformed input and bounds** — every failure path is typed and fails closed.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from scipy import signal as sps

from quant_platform.config import (
    DescriptorConfig,
    FrequencyBand,
    SpectralConfig,
    SpectralWindow,
)
from quant_platform.features import spectral_descriptors as sd
from quant_platform.features import spectral_transforms as st
from quant_platform.features.spectral import (
    MAX_WINDOW_CELLS,
    SPECTRAL_PREFIX,
    build_spectral_features,
    channel_lookback,
    fit_training_normalizer,
    morlet_scales,
    spectral_column_names,
    spectral_feature_registry,
)

BENCHMARK = "SPY"


@pytest.fixture
def panel() -> pd.DataFrame:
    """A three-ticker synthetic panel with a benchmark, long enough to warm up."""
    rng = np.random.default_rng(20260726)
    dates = pd.bdate_range("2020-01-02", periods=420)
    frames = []
    for ticker in ("AAA", "BBB", BENCHMARK):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "return": rng.normal(0.0, 0.01, len(dates)),
                    "volume": rng.lognormal(15.0, 0.3, len(dates)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _tone(length: int, period: float, amplitude: float = 1.0) -> np.ndarray:
    """A pure sinusoid of the given period, in bars."""
    return amplitude * np.sin(2.0 * np.pi * np.arange(length, dtype=float) / period)


# ---------------------------------------------------------------------------
# Mathematical reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("detrend", ["none", "mean", "linear"])
def test_welch_matches_scipy_reference(detrend: str) -> None:
    """The from-scratch Welch estimate must equal an independent implementation."""
    rng = np.random.default_rng(7)
    values = rng.normal(size=512)
    length, segment, hop, n_fft = 128, 64, 32, 64
    windows = st.causal_windows(values, length)
    mine = st.welch_psd(
        windows,
        segment_length=segment,
        hop=hop,
        n_fft=n_fft,
        sampling_frequency=1.0,
        taper_name="hann",
        detrend=detrend,
    )
    scipy_detrend: str | bool = {"none": False, "mean": "constant", "linear": "linear"}[detrend]
    _, reference = sps.welch(
        values[-length:],
        fs=1.0,
        window=sps.get_window("hann", segment, fftbins=False),
        nperseg=segment,
        noverlap=segment - hop,
        nfft=n_fft,
        detrend=scipy_detrend,
        return_onesided=True,
        scaling="density",
        average="mean",
    )
    np.testing.assert_allclose(mine[-1], reference, rtol=1e-10, atol=1e-12)


def test_stft_slices_match_a_direct_segment_transform() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=256)
    windows = st.causal_windows(values, 64)
    magnitudes = st.stft(
        windows, segment_length=32, hop=16, n_fft=32, taper_name="hann", detrend="none"
    )
    tail = values[-32:] * st.taper("hann", 32)
    np.testing.assert_allclose(magnitudes[-1, -1], np.abs(np.fft.rfft(tail, n=32)), rtol=1e-12)


def test_haar_details_match_the_closed_form() -> None:
    """Level-1 Haar detail coefficients are ``(x[2k] - x[2k+1]) / sqrt(2)``."""
    values = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    windows = values.reshape(1, -1)
    energies = st.dwt_energies(windows, "haar", 1)
    expected_detail = ((values[0::2] - values[1::2]) / np.sqrt(2.0)) ** 2
    total = (values**2).sum()
    np.testing.assert_allclose(energies[0, 0], expected_detail.sum() / total, rtol=1e-12)


@pytest.mark.parametrize("wavelet", ["haar", "db2"])
@pytest.mark.parametrize("levels", [1, 2, 3])
def test_dwt_conserves_energy(wavelet: str, levels: int) -> None:
    """Parseval: an orthogonal transform moves energy, it does not create it."""
    rng = np.random.default_rng(3)
    values = rng.normal(size=400)
    windows = st.causal_windows(values, 64)
    energies = st.dwt_energies(windows, wavelet, levels)
    complete = energies[63:]
    np.testing.assert_allclose(complete.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("period", [4.0, 8.0, 16.0])
def test_dominant_frequency_recovers_a_known_tone(period: float) -> None:
    values = _tone(600, period)
    windows = st.causal_windows(values, 128)
    psd = st.welch_psd(
        windows,
        segment_length=128,
        hop=128,
        n_fft=256,
        sampling_frequency=1.0,
        taper_name="hann",
        detrend="mean",
    )
    frequencies = st.rfft_frequencies(256, 1.0)
    mass = sd.normalized_spectrum(psd)
    peak = sd.peak_frequency(mass, frequencies)[-1]
    assert peak == pytest.approx(1.0 / period, abs=frequencies[1])


def test_cwt_power_peaks_at_the_matching_scale() -> None:
    period = 16.0
    values = _tone(600, period)
    periods = np.array([4.0, 8.0, 16.0, 32.0])
    scales = 6.0 * periods / (2.0 * np.pi)
    power = st.cwt_power(st.causal_windows(values, 128), scales, 6.0)
    assert periods[int(np.argmax(power[300]))] == pytest.approx(period)


def test_causal_morlet_is_zero_mean_and_unit_energy() -> None:
    """Truncating the wavelet to the past breaks admissibility; we restore it."""
    filt = st.causal_morlet(64, scale=5.0)
    assert abs(complex(filt.sum())) < 1e-12
    assert float(np.abs(filt) ** 2 @ np.ones(64)) == pytest.approx(1.0, rel=1e-12)


def test_morlet_scale_frequency_map_is_exact() -> None:
    scales = np.array([2.0, 5.0])
    np.testing.assert_allclose(
        st.morlet_scale_frequencies(scales, 6.0), 6.0 / (2.0 * np.pi * scales), rtol=1e-15
    )


# ---------------------------------------------------------------------------
# Causality — the leakage-mutation test
# ---------------------------------------------------------------------------


def test_future_bars_cannot_change_past_features(panel: pd.DataFrame) -> None:
    """Rewrite the future arbitrarily; every earlier feature must be unchanged.

    This is the test that would fail if any window were centred, any transform
    reached forward, or any normalization were fit over the whole sample.
    """
    config = SpectralConfig(enabled=True, wavelet_channels=["return"])
    cutoff = 300
    baseline = build_spectral_features(panel, config, benchmark=BENCHMARK)

    mutated = panel.copy()
    future = mutated.groupby("ticker", sort=False).cumcount() >= cutoff
    rng = np.random.default_rng(999)
    mutated.loc[future, "return"] = rng.normal(0.0, 5.0, int(future.sum()))
    mutated.loc[future, "volume"] = 1.0
    perturbed = build_spectral_features(mutated, config, benchmark=BENCHMARK)

    past = (panel.groupby("ticker", sort=False).cumcount() < cutoff).to_numpy()
    pd.testing.assert_frame_equal(baseline.loc[past], perturbed.loc[past])


def test_warmup_bars_are_missing_not_zero(panel: pd.DataFrame) -> None:
    config = SpectralConfig(enabled=True, channels=["return"])
    features = build_spectral_features(panel, config, benchmark=BENCHMARK)
    first_ticker = panel["ticker"] == "AAA"
    warmup = features.loc[first_ticker].iloc[: config.window.length - 1]
    assert warmup.isna().all().all()
    assert features.loc[first_ticker].iloc[config.window.length - 1 :].notna().any().any()


def test_causal_windows_never_read_ahead() -> None:
    values = np.arange(10, dtype=float)
    windows = st.causal_windows(values, 4)
    assert np.isnan(windows[:3]).all()
    np.testing.assert_array_equal(windows[3], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(windows[9], [6.0, 7.0, 8.0, 9.0])


def test_causal_windows_are_not_a_shared_view() -> None:
    """A caller mutating the result must not corrupt overlapping windows."""
    values = np.arange(8, dtype=float)
    windows = st.causal_windows(values, 3)
    windows[5, 0] = -999.0
    np.testing.assert_array_equal(st.causal_windows(values, 3)[5], [3.0, 4.0, 5.0])


def test_tickers_do_not_leak_into_each_others_windows(panel: pd.DataFrame) -> None:
    config = SpectralConfig(enabled=True, channels=["return"])
    features = build_spectral_features(panel, config, benchmark=BENCHMARK)
    only_aaa = panel[panel["ticker"].isin(["AAA", BENCHMARK])].reset_index(drop=True)
    isolated = build_spectral_features(only_aaa, config, benchmark=BENCHMARK)
    left = features.loc[(panel["ticker"] == "AAA").to_numpy()].reset_index(drop=True)
    right = isolated.loc[(only_aaa["ticker"] == "AAA").to_numpy()].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


# ---------------------------------------------------------------------------
# Property and metamorphic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [1e-4, 0.5, 3.0, 1e4])
def test_every_descriptor_is_scale_free(factor: float) -> None:
    """The module's central claim: amplitude must not move a descriptor."""
    rng = np.random.default_rng(5)
    values = rng.normal(size=300) + _tone(300, 12.0, 0.5)
    config = DescriptorConfig()
    frequencies = st.rfft_frequencies(64, 1.0)

    def describe(scale: float) -> dict[str, np.ndarray]:
        windows = st.causal_windows(values * scale, 64)
        psd = st.welch_psd(
            windows,
            segment_length=32,
            hop=16,
            n_fft=64,
            sampling_frequency=1.0,
            taper_name="hann",
            detrend="mean",
        )
        return sd.describe(psd, frequencies, config)

    baseline = describe(1.0)
    scaled = describe(factor)
    for name, values_ in baseline.items():
        np.testing.assert_allclose(
            scaled[name][63:], values_[63:], rtol=1e-9, atol=1e-12, err_msg=name
        )


def test_calendar_gaps_do_not_change_bar_time_spectra(panel: pd.DataFrame) -> None:
    """Frequencies are cycles per *bar*; a missing calendar date is not a gap.

    The engine's stated assumption is that it works in bar time. Deleting
    calendar dates while keeping the same ordered observations must therefore
    leave every value untouched — if it did not, the frequency axis would
    silently mean something different for a holiday-heavy period.
    """
    config = SpectralConfig(enabled=True, channels=["return"])
    dense = build_spectral_features(panel, config, benchmark=BENCHMARK)
    irregular = panel.copy()
    # Stretch the calendar unevenly while keeping it strictly increasing, so the
    # observation order — the only thing bar time depends on — is untouched.
    position = irregular.groupby("ticker", sort=False).cumcount().to_numpy()
    elapsed = np.cumsum(1 + (np.arange(len(panel)) % 5))[position]
    irregular["date"] = pd.Timestamp("2020-01-02") + pd.to_timedelta(elapsed, unit="D")
    sparse = build_spectral_features(irregular, config, benchmark=BENCHMARK)
    pd.testing.assert_frame_equal(dense, sparse)


def test_descriptor_ranges_are_respected() -> None:
    rng = np.random.default_rng(13)
    windows = st.causal_windows(rng.normal(size=400), 64)
    psd = st.welch_psd(
        windows,
        segment_length=32,
        hop=16,
        n_fft=64,
        sampling_frequency=1.0,
        taper_name="hann",
        detrend="mean",
    )
    frequencies = st.rfft_frequencies(64, 1.0)
    columns = sd.describe(psd, frequencies, DescriptorConfig())
    complete = slice(63, None)
    for bounded in ("entropy", "flatness", "sparsity", "concentration"):
        values = columns[bounded][complete]
        assert np.nanmin(values) >= -1e-12, bounded
        assert np.nanmax(values) <= 1.0 + 1e-12, bounded
    assert np.nanmin(columns["centroid"][complete]) >= 0.0
    assert np.nanmax(columns["centroid"][complete]) <= 0.5 + 1e-12
    bands = columns["band_low"] + columns["band_mid"] + columns["band_high"]
    np.testing.assert_allclose(bands[complete], 1.0, rtol=1e-10)


def test_a_pure_tone_is_sparser_than_white_noise() -> None:
    rng = np.random.default_rng(17)
    frequencies = st.rfft_frequencies(64, 1.0)

    def sparsity(values: np.ndarray) -> float:
        psd = st.welch_psd(
            st.causal_windows(values, 64),
            segment_length=64,
            hop=64,
            n_fft=64,
            sampling_frequency=1.0,
            taper_name="hann",
            detrend="mean",
        )
        columns = sd.describe(psd, frequencies, DescriptorConfig())
        return float(columns["sparsity"][-1])

    assert sparsity(_tone(200, 8.0)) > sparsity(rng.normal(size=200))


def test_entropy_is_higher_for_noise_than_for_a_tone() -> None:
    rng = np.random.default_rng(19)
    frequencies = st.rfft_frequencies(64, 1.0)

    def entropy(values: np.ndarray) -> float:
        psd = st.welch_psd(
            st.causal_windows(values, 64),
            segment_length=64,
            hop=64,
            n_fft=64,
            sampling_frequency=1.0,
            taper_name="hann",
            detrend="mean",
        )
        return float(sd.describe(psd, frequencies, DescriptorConfig())["entropy"][-1])

    assert entropy(rng.normal(size=200)) > entropy(_tone(200, 8.0))


# ---------------------------------------------------------------------------
# Numerical stability and degenerate input
# ---------------------------------------------------------------------------


def test_a_constant_series_has_no_frequency_distribution() -> None:
    """Zero AC power means there is nothing to describe; NaN, not a fake zero."""
    windows = st.causal_windows(np.full(120, 4.2), 64)
    psd = st.welch_psd(
        windows,
        segment_length=32,
        hop=16,
        n_fft=64,
        sampling_frequency=1.0,
        taper_name="hann",
        detrend="mean",
    )
    columns = sd.describe(psd, st.rfft_frequencies(64, 1.0), DescriptorConfig())
    assert np.isnan(columns["centroid"][-1])
    assert np.isnan(columns["entropy"][-1])
    assert np.isnan(columns["sparsity"][-1])


def test_a_constant_window_yields_nan_wavelet_energy() -> None:
    energies = st.dwt_energies(np.zeros((1, 8)), "haar", 1)
    assert np.isnan(energies).all()


def test_nan_input_propagates_rather_than_being_imputed() -> None:
    values = np.arange(200, dtype=float)
    values[100] = np.nan
    windows = st.causal_windows(values, 32)
    psd = st.welch_psd(
        windows,
        segment_length=16,
        hop=8,
        n_fft=32,
        sampling_frequency=1.0,
        taper_name="hann",
        detrend="mean",
    )
    columns = sd.describe(psd, st.rfft_frequencies(32, 1.0), DescriptorConfig())
    # Every window covering the corrupted bar is unusable, and says so.
    assert np.isnan(columns["centroid"][100:132]).all()
    assert not np.isnan(columns["centroid"][140])


@pytest.mark.parametrize("amplitude", [1e-12, 1e12])
def test_extreme_amplitudes_stay_finite_and_identical(amplitude: float) -> None:
    rng = np.random.default_rng(23)
    values = rng.normal(size=200)
    frequencies = st.rfft_frequencies(64, 1.0)

    def descriptors(scale: float) -> np.ndarray:
        psd = st.welch_psd(
            st.causal_windows(values * scale, 64),
            segment_length=32,
            hop=16,
            n_fft=64,
            sampling_frequency=1.0,
            taper_name="hann",
            detrend="mean",
        )
        return sd.describe(psd, frequencies, DescriptorConfig())["entropy"]

    scaled = descriptors(amplitude)
    assert np.isfinite(scaled[63:]).all()
    np.testing.assert_allclose(scaled[63:], descriptors(1.0)[63:], rtol=1e-8)


def test_empty_band_reports_missing_rather_than_zero() -> None:
    mass = np.full((3, 5), 0.2)
    frequencies = np.linspace(0.0, 0.5, 5)
    band = FrequencyBand(name="narrow", low=0.31, high=0.32)
    assert np.isnan(sd.relative_band_power(mass, frequencies, band)).all()


def test_flux_requires_at_least_two_slices() -> None:
    assert np.isnan(sd.spectral_flux(np.ones((4, 1, 8)))).all()


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------


def test_repeated_builds_are_bit_identical(panel: pd.DataFrame) -> None:
    config = SpectralConfig(enabled=True, wavelet_channels=["return"])
    first = build_spectral_features(panel, config, benchmark=BENCHMARK)
    second = build_spectral_features(panel, config, benchmark=BENCHMARK)
    pd.testing.assert_frame_equal(first, second)


def test_row_order_does_not_change_results(panel: pd.DataFrame) -> None:
    """The engine sorts internally, so a shuffled panel must give the same answer."""
    config = SpectralConfig(enabled=True, channels=["return", "volume"])
    ordered = build_spectral_features(panel, config, benchmark=BENCHMARK)
    shuffled = panel.sample(frac=1.0, random_state=4)
    result = build_spectral_features(shuffled, config, benchmark=BENCHMARK)
    pd.testing.assert_frame_equal(ordered, result.loc[panel.index])


def test_implementation_hash_is_stable_within_a_process() -> None:
    from quant_platform.features.spectral import spectral_implementation_hash

    assert spectral_implementation_hash() == spectral_implementation_hash()
    assert len(spectral_implementation_hash()) == 64


# ---------------------------------------------------------------------------
# Registry and integration
# ---------------------------------------------------------------------------


def test_registry_describes_exactly_the_emitted_columns(panel: pd.DataFrame) -> None:
    config = SpectralConfig(enabled=True, wavelet_channels=["return"])
    features = build_spectral_features(panel, config, benchmark=BENCHMARK)
    registry = spectral_feature_registry(config)
    assert tuple(sorted(features.columns)) == registry.output_columns
    assert len(registry.identity) == 64


def test_registry_records_the_full_window_contract() -> None:
    config = SpectralConfig(enabled=True, channels=["return"])
    spec = spectral_feature_registry(config).specs[0]
    for key in (
        "length",
        "segment_length",
        "hop",
        "overlap",
        "n_segments",
        "n_fft",
        "padding",
        "sampling_frequency",
        "frequency_unit",
        "warmup_bars",
        "detrend",
        "taper",
    ):
        assert key in spec.parameters, key
    assert spec.parameters["frequency_unit"] == "cycles_per_bar"
    assert spec.normalization == "none"
    assert spec.fitted_state is None
    assert spec.family == "spectral"


def test_channel_lookback_covers_the_underlying_transform() -> None:
    config = SpectralConfig(enabled=True, volatility_window=21, beta_window=63)
    assert channel_lookback(config, "return") == config.window.length
    assert channel_lookback(config, "volatility") == config.window.length + 20
    assert channel_lookback(config, "residual") == config.window.length + 62
    specs = {spec.name: spec for spec in spectral_feature_registry(config).specs}
    volatility = specs[f"{SPECTRAL_PREFIX}volatility_centroid"]
    assert volatility.lookback_bars == config.window.length + 20
    assert volatility.warmup_bars >= volatility.lookback_bars


def test_residual_channel_is_flagged_as_cross_asset() -> None:
    config = SpectralConfig(enabled=True)
    specs = {spec.name: spec for spec in spectral_feature_registry(config).specs}
    assert specs[f"{SPECTRAL_PREFIX}residual_centroid"].leakage_risk == "medium"
    assert specs[f"{SPECTRAL_PREFIX}return_centroid"].leakage_risk == "low"


def test_column_names_match_the_built_frame(panel: pd.DataFrame) -> None:
    config = SpectralConfig(enabled=True, wavelet_channels=["return"])
    features = build_spectral_features(panel, config, benchmark=BENCHMARK)
    assert tuple(features.columns) == spectral_column_names(config)


def test_morlet_scales_track_the_configured_periods() -> None:
    config = SpectralConfig(enabled=True, cwt_periods=[8.0, 16.0])
    np.testing.assert_allclose(
        st.morlet_scale_frequencies(morlet_scales(config), config.morlet_omega0),
        [1.0 / 8.0, 1.0 / 16.0],
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Training-only normalization
# ---------------------------------------------------------------------------


def test_normalizer_fits_only_inside_the_training_interval(panel: pd.DataFrame) -> None:
    config = SpectralConfig(enabled=True, channels=["return"])
    features = build_spectral_features(panel, config, benchmark=BENCHMARK)
    aaa = (panel["ticker"] == "AAA").to_numpy()
    subset = features.loc[aaa].reset_index(drop=True)
    dates = panel.loc[aaa, "date"].reset_index(drop=True)
    fit_start, fit_end = dt.date(2020, 6, 1), dt.date(2020, 12, 31)

    mean, _std, state = fit_training_normalizer(subset, dates, fit_start=fit_start, fit_end=fit_end)
    window = subset.loc[((dates.dt.date >= fit_start) & (dates.dt.date <= fit_end)).to_numpy()]
    pd.testing.assert_series_equal(mean, window.mean())
    assert state.fit_start == fit_start
    assert state.fit_end == fit_end
    assert state.sample_count == len(window)
    assert len(state.state_sha256) == 64

    # Rewriting observations outside the interval must not move the fit.
    tampered = subset.copy()
    outside = (dates.dt.date > fit_end).to_numpy()
    tampered.loc[outside] = 1e6
    again, _, _ = fit_training_normalizer(tampered, dates, fit_start=fit_start, fit_end=fit_end)
    pd.testing.assert_series_equal(mean, again)


def test_normalizer_rejects_an_inverted_or_empty_interval(panel: pd.DataFrame) -> None:
    config = SpectralConfig(enabled=True, channels=["return"])
    features = build_spectral_features(panel, config, benchmark=BENCHMARK)
    dates = panel["date"]
    with pytest.raises(ValueError, match="on or after"):
        fit_training_normalizer(
            features, dates, fit_start=dt.date(2021, 1, 1), fit_end=dt.date(2020, 1, 1)
        )
    with pytest.raises(ValueError, match="no usable"):
        fit_training_normalizer(
            features, dates, fit_start=dt.date(1990, 1, 1), fit_end=dt.date(1990, 2, 1)
        )
    with pytest.raises(ValueError, match="same observations"):
        fit_training_normalizer(
            features, dates.iloc[:-1], fit_start=dt.date(2020, 1, 1), fit_end=dt.date(2021, 1, 1)
        )


# ---------------------------------------------------------------------------
# Malformed input, configuration, and resource bounds
# ---------------------------------------------------------------------------


def test_engine_is_opt_in(panel: pd.DataFrame) -> None:
    assert SpectralConfig().enabled is False
    with pytest.raises(ValueError, match="disabled"):
        build_spectral_features(panel, SpectralConfig(), benchmark=BENCHMARK)
    with pytest.raises(ValueError, match="disabled"):
        spectral_feature_registry(SpectralConfig())


def test_empty_or_incomplete_panels_are_rejected(panel: pd.DataFrame) -> None:
    config = SpectralConfig(enabled=True)
    with pytest.raises(ValueError, match="empty panel"):
        build_spectral_features(panel.iloc[:0], config, benchmark=BENCHMARK)
    with pytest.raises(ValueError, match="missing required columns"):
        build_spectral_features(panel.drop(columns=["volume"]), config, benchmark=BENCHMARK)


def test_missing_benchmark_fails_closed(panel: pd.DataFrame) -> None:
    config = SpectralConfig(enabled=True, channels=["residual"])
    with pytest.raises(ValueError, match="benchmark"):
        build_spectral_features(panel, config, benchmark="ABSENT")


def test_compute_ceiling_refuses_an_oversized_request(panel: pd.DataFrame) -> None:
    config = SpectralConfig(
        enabled=True, window=SpectralWindow(length=1024, segment_length=512, hop=256, n_fft=512)
    )
    oversized = pd.concat([panel] * 2, ignore_index=True)
    # Force the ceiling deterministically rather than by allocating a huge frame.
    import quant_platform.features.spectral as engine

    original = engine.MAX_WINDOW_CELLS
    try:
        engine.MAX_WINDOW_CELLS = 10
        with pytest.raises(ValueError, match="exceeds the"):
            build_spectral_features(oversized, config, benchmark=BENCHMARK)
    finally:
        engine.MAX_WINDOW_CELLS = original
    assert engine.MAX_WINDOW_CELLS == MAX_WINDOW_CELLS


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"segment_length": 128}, "segment_length must not exceed"),
        ({"hop": 64, "segment_length": 32}, "hop must not exceed"),
        ({"n_fft": 8, "segment_length": 32}, "n_fft must be at least"),
    ],
)
def test_window_geometry_is_validated(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SpectralWindow(length=64, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"channels": ["return", "return"]}, "unique"),
        ({"wavelet_channels": ["volume"], "channels": ["return"]}, "analysed channels"),
        ({"cwt_periods": [1.5]}, "Nyquist"),
        ({"cwt_periods": [8.0, 8.0]}, "unique"),
        (
            {
                "wavelet_channels": ["return"],
                "dwt_levels": 5,
                "window": SpectralWindow(length=48, segment_length=16, hop=8, n_fft=16),
            },
            "divisible",
        ),
    ],
)
def test_spectral_config_rejects_incoherent_settings(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SpectralConfig(enabled=True, **kwargs)


def test_frequency_band_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="strictly greater"):
        FrequencyBand(name="bad", low=0.4, high=0.2)


def test_duplicate_band_names_are_rejected() -> None:
    band = FrequencyBand(name="dup", low=0.0, high=0.2)
    with pytest.raises(ValueError, match="unique"):
        DescriptorConfig(bands=[band, band])


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: st.taper("gaussian", 8), "unknown taper"),
        (lambda: st.taper("hann", 0), "taper length"),
        (lambda: st.causal_windows(np.zeros(4), 0), "window length"),
        (lambda: st.detrend_segments(np.zeros((2, 4)), "quadratic"), "unknown detrend"),
        (lambda: st.dwt_energies(np.zeros((1, 8)), "coif1", 1), "unknown wavelet"),
        (lambda: st.dwt_energies(np.zeros((1, 12)), "haar", 3), "divisible"),
        (lambda: st.dwt_energies(np.zeros((1, 8)), "haar", 0), "levels must be"),
        (lambda: st.causal_morlet(1, 2.0), "morlet length"),
        (lambda: st.causal_morlet(8, 0.0), "morlet scale"),
        (lambda: st.segment_windows(np.zeros((2, 8)), 16, 4), "segment_length must be"),
        (lambda: st.segment_windows(np.zeros((2, 8)), 4, 8), "hop must be in"),
    ],
)
def test_transform_guards_fail_closed(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]


def test_welch_rejects_an_impossible_transform() -> None:
    windows = st.causal_windows(np.zeros(64), 32)
    with pytest.raises(ValueError, match="n_fft must be >="):
        st.welch_psd(windows, segment_length=32, hop=16, n_fft=16, sampling_frequency=1.0)
    with pytest.raises(ValueError, match="sampling_frequency"):
        st.welch_psd(windows, segment_length=32, hop=16, n_fft=32, sampling_frequency=0.0)


@pytest.mark.parametrize("name", ["boxcar", "hann", "hamming", "blackman"])
def test_every_taper_matches_the_scipy_reference(name: str) -> None:
    np.testing.assert_allclose(
        st.taper(name, 32), sps.get_window(name, 32, fftbins=False), rtol=1e-12, atol=1e-14
    )


def test_single_sample_taper_is_unity() -> None:
    np.testing.assert_array_equal(st.taper("hann", 1), np.ones(1))


def test_a_series_shorter_than_the_window_is_all_warmup() -> None:
    windows = st.causal_windows(np.arange(3, dtype=float), 8)
    assert windows.shape == (3, 8)
    assert np.isnan(windows).all()


def test_linear_detrend_of_a_single_sample_segment_is_zero() -> None:
    segments = np.array([[[5.0]]])
    np.testing.assert_array_equal(st.detrend_segments(segments, "linear"), np.zeros((1, 1, 1)))


def test_rounding_residue_is_collapsed_to_exact_zero() -> None:
    """A numerically constant window must not produce a spectrum from float noise."""
    constant = np.full((1, 16), 4.2)
    np.testing.assert_array_equal(st.detrend_segments(constant, "mean"), np.zeros((1, 16)))
    np.testing.assert_array_equal(st.detrend_segments(constant, "linear"), np.zeros((1, 16)))


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_disabled_engine_leaves_the_materialization_request_unchanged(
    app_config, synthetic_panel
) -> None:
    from quant_platform.features.integration import build_pipeline_materialization_request

    baseline = build_pipeline_materialization_request(app_config, synthetic_panel, {})
    assert not any(feature.name.startswith(SPECTRAL_PREFIX) for feature in baseline.features)


def test_enabled_engine_registers_its_columns_in_the_request(app_config, synthetic_panel) -> None:
    from quant_platform.features.integration import build_pipeline_materialization_request

    enabled = app_config.model_copy(deep=True)
    enabled.features.spectral = SpectralConfig(enabled=True, channels=["return"])
    request = build_pipeline_materialization_request(enabled, synthetic_panel, {})
    registered = {feature.name for feature in request.features}
    assert set(spectral_column_names(enabled.features.spectral)) <= registered
    # Enabling the family must change the request identity, so a materialization
    # made before the change can never be looked up and reused as if it matched.
    baseline = build_pipeline_materialization_request(app_config, synthetic_panel, {})
    assert request.identity != baseline.identity


def test_describe_rejects_a_mismatched_frequency_axis() -> None:
    with pytest.raises(ValueError, match="frequencies were supplied"):
        sd.describe(np.ones((2, 5)), np.linspace(0.0, 0.5, 4), DescriptorConfig())
    with pytest.raises(ValueError, match="two-dimensional"):
        sd.describe(np.ones(5), np.linspace(0.0, 0.5, 5), DescriptorConfig())
