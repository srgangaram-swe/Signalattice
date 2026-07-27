"""Tests for versioned time-frequency tensors and their store (SF-S3-MR2).

Grouped by the property each defends: deterministic identity, causality and
masking, train-only normalization, store integrity and containment, resource
bounds, and rendering metadata.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.config import SpectralWindow, TimeFrequencyConfig
from quant_platform.features.time_frequency import (
    LOG_POWER_FLOOR,
    TensorBoundsError,
    TensorIntegrityError,
    TimeFrequencyError,
    TimeFrequencyMetadata,
    TimeFrequencyTensor,
    build_time_frequency_tensor,
    digest_array,
    normalize_tensor,
)
from quant_platform.features.time_frequency_store import (
    MANIFEST_FILE,
    MASK_FILE,
    VALUES_FILE,
    TimeFrequencyStore,
)
from quant_platform.reporting.time_frequency_plots import (
    _caption,
    plot_channel_coverage,
    plot_time_frequency_sample,
)

BENCHMARK = "SPY"
FIT_START = dt.date(2020, 4, 1)
FIT_END = dt.date(2020, 10, 1)


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(20260726)
    dates = pd.bdate_range("2020-01-02", periods=300)
    frames = [
        pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "return": rng.normal(0.0, 0.01, len(dates)),
                "volume": rng.lognormal(15.0, 0.3, len(dates)),
            }
        )
        for ticker in ("AAA", BENCHMARK)
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def config() -> TimeFrequencyConfig:
    return TimeFrequencyConfig(enabled=True, channels=["return", "volume"])


def _build(panel: pd.DataFrame, config: TimeFrequencyConfig):
    return build_time_frequency_tensor(panel, config, benchmark=BENCHMARK)


# ---------------------------------------------------------------------------
# Deterministic identity and shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("representation", ["spectrogram", "scalogram"])
def test_identical_inputs_produce_identical_tensors(
    panel: pd.DataFrame, representation: str
) -> None:
    config = TimeFrequencyConfig(
        enabled=True, representation=representation, channels=["return", "volume"]
    )
    first, first_index = _build(panel, config)
    second, second_index = _build(panel, config)
    assert first.metadata == second.metadata
    assert first.metadata.identity == second.metadata.identity
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.mask, second.mask)
    pd.testing.assert_frame_equal(first_index, second_index)


@pytest.mark.parametrize(
    ("representation", "axis"),
    [("spectrogram", "cycles_per_bar"), ("scalogram", "period_bars")],
)
def test_shape_and_axis_match_the_representation(
    panel: pd.DataFrame, representation: str, axis: str
) -> None:
    config = TimeFrequencyConfig(
        enabled=True, representation=representation, channels=["return", "volume"]
    )
    tensor, index = _build(panel, config)
    metadata = tensor.metadata
    expected_freqs = (
        len(config.window.frequencies())
        if representation == "spectrogram"
        else len(config.scalogram_periods)
    )
    assert tensor.values.shape == (len(index), 2, expected_freqs, config.window.n_segments)
    assert metadata.shape == tensor.values.shape
    assert metadata.frequency_axis == axis
    assert list(metadata.frequency_values) == sorted(metadata.frequency_values)
    tensor.verify()


def test_configuration_change_changes_identity(panel: pd.DataFrame) -> None:
    baseline, _ = _build(panel, TimeFrequencyConfig(enabled=True, channels=["return"]))
    other, _ = _build(panel, TimeFrequencyConfig(enabled=True, channels=["volume"]))
    assert baseline.metadata.identity != other.metadata.identity


def test_digest_distinguishes_shape_and_dtype() -> None:
    values = np.arange(12, dtype=np.float64)
    assert digest_array(values) != digest_array(values.reshape(3, 4))
    assert digest_array(values) != digest_array(values.astype(np.float32))
    assert digest_array(values) == digest_array(values.copy())


# ---------------------------------------------------------------------------
# Causality, masking, alignment
# ---------------------------------------------------------------------------


def test_future_bars_cannot_change_earlier_samples(
    panel: pd.DataFrame, config: TimeFrequencyConfig
) -> None:
    baseline, index = _build(panel, config)
    cutoff = 220
    mutated = panel.copy()
    future = mutated.groupby("ticker", sort=False).cumcount() >= cutoff
    rng = np.random.default_rng(4)
    mutated.loc[future, "return"] = rng.normal(0.0, 5.0, int(future.sum()))
    mutated.loc[future, "volume"] = 1.0
    perturbed, _ = _build(mutated, config)

    ordered = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    past = (ordered.groupby("ticker", sort=False).cumcount() < cutoff).to_numpy()
    np.testing.assert_array_equal(baseline.values[past], perturbed.values[past])
    np.testing.assert_array_equal(baseline.mask[past], perturbed.mask[past])


def test_warmup_samples_are_masked_not_padded(
    panel: pd.DataFrame, config: TimeFrequencyConfig
) -> None:
    tensor, index = _build(panel, config)
    warmup = config.window.length - 1
    first_ticker = (index["ticker"] == "AAA").to_numpy()
    early = tensor.mask[first_ticker][:warmup]
    assert not early.any()
    assert np.isnan(tensor.values[first_ticker][:warmup]).all()
    assert tensor.mask[first_ticker][warmup:].any()


def test_missing_observations_are_masked_not_imputed(
    panel: pd.DataFrame, config: TimeFrequencyConfig
) -> None:
    gappy = panel.copy()
    gappy.loc[(gappy["ticker"] == "AAA") & (gappy.index % 300 == 150), "return"] = np.nan
    tensor, index = _build(gappy, config)
    affected = tensor.mask[(index["ticker"] == "AAA").to_numpy()]
    assert not affected.all(), "a NaN return must mask its covering windows"
    assert np.isnan(tensor.values[~tensor.mask]).all()


def test_channels_share_one_frequency_grid(panel: pd.DataFrame) -> None:
    config = TimeFrequencyConfig(
        enabled=True, channels=["return", "volatility", "volume", "residual"]
    )
    tensor, _ = _build(panel, config)
    assert tensor.values.shape[1] == 4
    assert tensor.metadata.channels == ("return", "volatility", "volume", "residual")
    # One grid for every channel is what makes the stack meaningful.
    assert len(tensor.metadata.frequency_values) == tensor.values.shape[2]


def test_index_aligns_with_samples(panel: pd.DataFrame, config: TimeFrequencyConfig) -> None:
    tensor, index = _build(panel, config)
    assert len(index) == tensor.metadata.n_samples
    ordered = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(index, ordered.loc[:, ["date", "ticker"]])


# ---------------------------------------------------------------------------
# Train-only normalization
# ---------------------------------------------------------------------------


def test_normalization_uses_only_the_training_interval(
    panel: pd.DataFrame, config: TimeFrequencyConfig
) -> None:
    tensor, index = _build(panel, config)
    normalized = normalize_tensor(tensor, index, fit_start=FIT_START, fit_end=FIT_END)
    state = normalized.metadata.fitted_state
    assert state is not None
    assert state.fit_start == FIT_START and state.fit_end == FIT_END
    assert normalized.metadata.normalization == "train_log_zscore"
    assert normalized.metadata.log_power_floor == LOG_POWER_FLOOR

    # Rewriting observations outside the interval must not move the statistics.
    stamps = pd.to_datetime(index["date"]).dt.date
    outside = (stamps > FIT_END).to_numpy()
    tampered_values = tensor.values.copy()
    tampered_values[outside] = np.where(
        np.isnan(tampered_values[outside]), np.nan, tampered_values[outside] * 1e6
    )
    tampered = TimeFrequencyTensor(
        values=tampered_values,
        mask=tensor.mask,
        metadata=tensor.metadata.model_copy(
            update={"values_sha256": digest_array(tampered_values)}
        ),
    )
    again = normalize_tensor(tampered, index, fit_start=FIT_START, fit_end=FIT_END)
    inside = (~outside) & normalized.mask.any(axis=1)
    np.testing.assert_allclose(again.values[inside], normalized.values[inside], rtol=1e-12)


def test_normalized_values_are_finite_where_observed(
    panel: pd.DataFrame, config: TimeFrequencyConfig
) -> None:
    tensor, index = _build(panel, config)
    normalized = normalize_tensor(tensor, index, fit_start=FIT_START, fit_end=FIT_END)
    assert np.isfinite(normalized.values[normalized.mask]).all()
    assert np.isnan(normalized.values[~normalized.mask]).all()
    normalized.verify()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fit_start": dt.date(2021, 1, 1), "fit_end": dt.date(2020, 1, 1)}, "on or after"),
        ({"fit_start": dt.date(1990, 1, 1), "fit_end": dt.date(1990, 2, 1)}, "no samples"),
        (
            {"fit_start": dt.date(2020, 1, 2), "fit_end": dt.date(2020, 1, 6)},
            "no observed samples",
        ),
    ],
)
def test_normalization_rejects_unusable_intervals(
    panel: pd.DataFrame, config: TimeFrequencyConfig, kwargs: dict, message: str
) -> None:
    tensor, index = _build(panel, config)
    with pytest.raises(ValueError, match=message):
        normalize_tensor(tensor, index, **kwargs)


def test_normalizing_twice_is_refused(panel: pd.DataFrame, config: TimeFrequencyConfig) -> None:
    tensor, index = _build(panel, config)
    once = normalize_tensor(tensor, index, fit_start=FIT_START, fit_end=FIT_END)
    with pytest.raises(ValueError, match="already normalized"):
        normalize_tensor(once, index, fit_start=FIT_START, fit_end=FIT_END)


def test_normalization_rejects_a_misaligned_index_or_floor(
    panel: pd.DataFrame, config: TimeFrequencyConfig
) -> None:
    tensor, index = _build(panel, config)
    with pytest.raises(ValueError, match="same samples"):
        normalize_tensor(tensor, index.iloc[:-1], fit_start=FIT_START, fit_end=FIT_END)
    with pytest.raises(ValueError, match="power floor"):
        normalize_tensor(tensor, index, fit_start=FIT_START, fit_end=FIT_END, floor=0.0)


# ---------------------------------------------------------------------------
# Store: integrity, containment, atomicity, bounds
# ---------------------------------------------------------------------------


@pytest.fixture
def stored(panel: pd.DataFrame, config: TimeFrequencyConfig, tmp_path: Path):
    tensor, index = _build(panel, config)
    normalized = normalize_tensor(tensor, index, fit_start=FIT_START, fit_end=FIT_END)
    store = TimeFrequencyStore(tmp_path / "tf")
    return store, normalized, store.write(normalized)


def test_store_round_trip_preserves_everything(stored) -> None:
    store, tensor, object_id = stored
    loaded = store.read(object_id)
    assert loaded.metadata == tensor.metadata
    np.testing.assert_array_equal(loaded.values, tensor.values)
    np.testing.assert_array_equal(loaded.mask, tensor.mask)
    assert store.exists(object_id)
    assert store.list_objects() == (object_id,)


def test_rewriting_the_same_tensor_is_idempotent(stored) -> None:
    store, tensor, object_id = stored
    assert store.write(tensor) == object_id
    assert store.list_objects() == (object_id,)


def test_corrupt_values_are_detected(stored) -> None:
    store, _tensor, object_id = stored
    path = store.object_path(object_id) / VALUES_FILE
    values = np.load(path, allow_pickle=False)
    values.flat[0] = 12345.0
    np.save(path, values, allow_pickle=False)
    with pytest.raises(TensorIntegrityError, match="values digest"):
        store.read(object_id)


def test_corrupt_mask_is_detected(stored) -> None:
    store, _tensor, object_id = stored
    path = store.object_path(object_id) / MASK_FILE
    mask = np.load(path, allow_pickle=False)
    mask.flat[0] = ~mask.flat[0]
    np.save(path, mask, allow_pickle=False)
    with pytest.raises(TensorIntegrityError, match="mask digest"):
        store.read(object_id)


def test_edited_manifest_is_detected(stored) -> None:
    store, _tensor, object_id = stored
    path = store.object_path(object_id) / MANIFEST_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["observed_fraction"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TensorIntegrityError, match="does not match object id"):
        store.read(object_id)


def test_unreadable_or_invalid_manifest_fails_closed(stored) -> None:
    store, _tensor, object_id = stored
    path = store.object_path(object_id) / MANIFEST_FILE
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(TimeFrequencyError, match="invalid tensor manifest"):
        store.read(object_id)


def test_missing_array_fails_closed(stored) -> None:
    store, _tensor, object_id = stored
    (store.object_path(object_id) / VALUES_FILE).unlink()
    with pytest.raises(TimeFrequencyError, match="missing tensor array"):
        store.read(object_id)


def test_wrong_dtype_array_is_refused(stored) -> None:
    store, _tensor, object_id = stored
    path = store.object_path(object_id) / VALUES_FILE
    np.save(path, np.zeros((2, 2), dtype=np.float32), allow_pickle=False)
    with pytest.raises(TensorIntegrityError, match="dtype"):
        store.read(object_id)


@pytest.mark.parametrize(
    "object_id",
    ["../escape", "not-a-digest", "A" * 64, "0" * 63, "", "/absolute/path"],
)
def test_object_ids_are_validated_before_touching_the_filesystem(
    tmp_path: Path, object_id: str
) -> None:
    store = TimeFrequencyStore(tmp_path / "tf")
    with pytest.raises(TimeFrequencyError, match="object id"):
        store.object_path(object_id)


def test_missing_object_is_reported(tmp_path: Path) -> None:
    store = TimeFrequencyStore(tmp_path / "tf")
    absent = "0" * 64
    assert not store.exists(absent)
    with pytest.raises(TimeFrequencyError, match="no time-frequency object"):
        store.read(absent)


def test_symlinked_root_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(TimeFrequencyError, match="symlink"):
        TimeFrequencyStore(link)


def test_store_rejects_an_oversized_object(stored) -> None:
    _store, tensor, _object_id = stored
    tiny = TimeFrequencyStore(Path(_store.root).parent / "tiny", max_object_bytes=1_000)
    with pytest.raises(TensorBoundsError, match="byte ceiling"):
        tiny.write(tensor)
    assert tiny.list_objects() == (), "a refused write must leave no partial object"


def test_store_rejects_an_invalid_ceiling(tmp_path: Path) -> None:
    with pytest.raises(TimeFrequencyError, match="must be positive"):
        TimeFrequencyStore(tmp_path / "tf", max_object_bytes=0)


def test_a_failed_write_leaves_no_staging_residue(stored) -> None:
    store, tensor, _object_id = stored
    broken = TimeFrequencyTensor(values=tensor.values, mask=tensor.mask, metadata=tensor.metadata)
    object.__setattr__(broken, "values", tensor.values * 2.0)
    with pytest.raises(TensorIntegrityError):
        store.write(broken)
    assert not any(store.staging_dir.iterdir())


# ---------------------------------------------------------------------------
# Resource bounds and configuration
# ---------------------------------------------------------------------------


def test_cell_ceiling_refuses_an_oversized_request(panel: pd.DataFrame) -> None:
    config = TimeFrequencyConfig(enabled=True, channels=["return"], max_tensor_cells=1_000)
    with pytest.raises(TensorBoundsError, match="exceeds the"):
        _build(panel, config)


def test_engine_is_opt_in(panel: pd.DataFrame) -> None:
    assert TimeFrequencyConfig().enabled is False
    with pytest.raises(ValueError, match="disabled"):
        _build(panel, TimeFrequencyConfig())


def test_malformed_panels_are_rejected(panel: pd.DataFrame, config: TimeFrequencyConfig) -> None:
    with pytest.raises(ValueError, match="empty panel"):
        _build(panel.iloc[:0], config)
    with pytest.raises(ValueError, match="missing required columns"):
        _build(panel.drop(columns=["volume"]), config)


def test_missing_benchmark_fails_closed(panel: pd.DataFrame) -> None:
    config = TimeFrequencyConfig(enabled=True, channels=["residual"])
    with pytest.raises(ValueError, match="benchmark"):
        build_time_frequency_tensor(panel, config, benchmark="ABSENT")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"channels": ["return", "return"]}, "unique"),
        ({"scalogram_periods": [8.0, 4.0]}, "sorted"),
        ({"scalogram_periods": [4.0, 4.0]}, "unique"),
        ({"scalogram_periods": [1.5, 8.0]}, "Nyquist"),
        ({"cache_dir": "/etc"}, "safe relative path"),
        ({"cache_dir": "../escape"}, "safe relative path"),
    ],
)
def test_configuration_rejects_incoherent_settings(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TimeFrequencyConfig(enabled=True, **kwargs)


# ---------------------------------------------------------------------------
# Metadata contract
# ---------------------------------------------------------------------------


def _metadata_fields(**overrides) -> dict:
    fields = {
        "representation": "spectrogram",
        "channels": ("return",),
        "frequency_axis": "cycles_per_bar",
        "frequency_values": (0.0, 0.25, 0.5),
        "window_parameters": {"length": 8},
        "n_samples": 4,
        "n_time": 2,
        "coverage_start": dt.date(2020, 1, 1),
        "coverage_end": dt.date(2020, 2, 1),
        "tickers": ("AAA",),
        "normalization": "none",
        "observed_fraction": 0.5,
        "values_sha256": "a" * 64,
        "mask_sha256": "b" * 64,
    }
    fields.update(overrides)
    return fields


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"frequency_values": (0.5, 0.25)}, "ascending"),
        ({"frequency_values": (0.0, float("nan"))}, "finite"),
        ({"channels": ("a", "a")}, "unique"),
        ({"coverage_end": dt.date(2019, 1, 1)}, "on or after"),
        ({"values_sha256": "zz"}, "digest"),
        ({"normalization": "train_log_zscore"}, "requires a recorded fitted state"),
        ({"representation": "scalogram"}, "expressed in periods"),
    ],
)
def test_metadata_invariants_fail_closed(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TimeFrequencyMetadata(**_metadata_fields(**overrides))


def test_tensor_rejects_a_shape_mismatch() -> None:
    metadata = TimeFrequencyMetadata(**_metadata_fields())
    with pytest.raises(TimeFrequencyError, match="does not match metadata"):
        TimeFrequencyTensor(
            values=np.zeros((1, 1, 1, 1)),
            mask=np.ones((1, 1), dtype=bool),
            metadata=metadata,
        )
    with pytest.raises(TimeFrequencyError, match="mask shape"):
        TimeFrequencyTensor(
            values=np.zeros(metadata.shape),
            mask=np.ones((2, 2), dtype=bool),
            metadata=metadata,
        )


def test_verify_detects_in_memory_tampering(
    panel: pd.DataFrame, config: TimeFrequencyConfig
) -> None:
    tensor, _ = _build(panel, config)
    tensor.verify()
    tampered = tensor.values.copy()
    tampered.flat[0] = 1.0
    broken = TimeFrequencyTensor(values=tampered, mask=tensor.mask, metadata=tensor.metadata)
    with pytest.raises(TensorIntegrityError, match="values"):
        broken.verify()


# ---------------------------------------------------------------------------
# Rendering metadata
# ---------------------------------------------------------------------------


def test_caption_records_window_normalization_and_provenance(stored) -> None:
    _store, tensor, _object_id = stored
    caption = _caption(tensor, ticker="AAA", observation="2020-12-31")
    for fragment in ("length=", "segment=", "hop=", "taper=", "detrend=", "train_log_zscore"):
        assert fragment in caption
    assert str(FIT_START) in caption and str(FIT_END) in caption
    assert "not evidence of predictability" in caption


def test_rendering_writes_an_inspectable_figure(stored, tmp_path: Path) -> None:
    _store, tensor, _object_id = stored
    sample = int(np.flatnonzero(tensor.mask.all(axis=1))[-1])
    path = plot_time_frequency_sample(
        tensor,
        tmp_path / "surface.png",
        sample=sample,
        channel="return",
        ticker="AAA",
        observation="2020-12-31",
    )
    assert path.is_file() and path.stat().st_size > 1_000
    coverage = plot_channel_coverage(tensor, tmp_path / "coverage.png")
    assert coverage.is_file() and coverage.stat().st_size > 1_000


def test_rendering_refuses_a_masked_or_unknown_selection(stored, tmp_path: Path) -> None:
    _store, tensor, _object_id = stored
    with pytest.raises(TimeFrequencyError, match="unknown channel"):
        plot_time_frequency_sample(
            tensor, tmp_path / "x.png", sample=0, channel="nope", ticker="AAA", observation="d"
        )
    with pytest.raises(TimeFrequencyError, match="out of range"):
        plot_time_frequency_sample(
            tensor,
            tmp_path / "x.png",
            sample=tensor.metadata.n_samples,
            channel="return",
            ticker="AAA",
            observation="d",
        )
    masked = int(np.flatnonzero(~tensor.mask.any(axis=1))[0])
    with pytest.raises(TimeFrequencyError, match="masked"):
        plot_time_frequency_sample(
            tensor,
            tmp_path / "x.png",
            sample=masked,
            channel="return",
            ticker="AAA",
            observation="d",
        )


def test_scalogram_window_geometry_is_recorded(panel: pd.DataFrame) -> None:
    window = SpectralWindow(length=64, segment_length=32, hop=16, n_fft=64)
    config = TimeFrequencyConfig(
        enabled=True, representation="scalogram", channels=["return"], window=window
    )
    tensor, _ = _build(panel, config)
    parameters = tensor.metadata.window_parameters
    assert parameters["representation"] == "scalogram"
    assert parameters["length"] == 64
    assert parameters["n_segments"] == window.n_segments
    assert parameters["morlet_omega0"] == config.morlet_omega0
