"""Causality and shape-contract tests for optional temporal models."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from quant_platform.models.torch_lstm import (
    TemporalConvEstimator,
    build_panel_sequences,
)


def _interleaved_panel():
    # Intentionally neither date-sorted nor ticker-contiguous.
    dates = np.array(["2025-01-03", "2025-01-02", "2025-01-01", "2025-01-03", "2025-01-01"])
    tickers = np.array(["AAA", "BBB", "AAA", "BBB", "BBB"])
    features = np.array([[3.0], [20.0], [1.0], [30.0], [10.0]], dtype=np.float32)
    targets = np.arange(len(dates))
    return features, dates, tickers, targets


def test_panel_sequences_are_chronological_and_ticker_isolated():
    features, dates, tickers, targets = _interleaved_panel()
    batch = build_panel_sequences(
        features,
        dates,
        tickers,
        targets=targets,
        sequence_length=3,
    )

    assert batch.X.shape == (5, 3, 1)
    assert (
        batch.metadata.dates.tolist()
        == np.array(
            ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-03"],
            dtype="datetime64[ns]",
        ).tolist()
    )
    assert batch.y.tolist() == targets[batch.metadata.row_indices].tolist()

    for sample_index, source_row in enumerate(batch.metadata.row_indices):
        history = batch.metadata.history_row_indices[sample_index]
        history = history[history >= 0]
        assert history[-1] == source_row
        assert set(tickers[history]) == {tickers[source_row]}
        assert np.all(
            dates[history].astype("datetime64[D]") <= dates[source_row].astype("datetime64[D]")
        )
        assert np.array_equal(
            batch.X[sample_index, batch.metadata.valid_mask[sample_index], 0],
            features[history, 0],
        )


def test_panel_sequences_include_only_prior_ticker_history_and_left_pad():
    features, dates, tickers, _ = _interleaved_panel()
    batch = build_panel_sequences(features, dates, tickers, sequence_length=3, pad_value=-99.0)

    aaa_final = np.flatnonzero(
        (batch.metadata.tickers == "AAA") & (batch.metadata.dates == np.datetime64("2025-01-03"))
    )[0]
    assert batch.X[aaa_final, :, 0].tolist() == [-99.0, 1.0, 3.0]
    assert batch.metadata.lengths[aaa_final] == 2
    assert batch.metadata.history_row_indices[aaa_final].tolist() == [-1, 2, 0]
    assert batch.metadata.valid_mask[aaa_final].tolist() == [False, True, True]

    complete = build_panel_sequences(
        features,
        dates,
        tickers,
        sequence_length=3,
        min_history=3,
    )
    assert len(complete) == 1
    assert complete.metadata.tickers.tolist() == ["BBB"]
    assert complete.X[0, :, 0].tolist() == [10.0, 20.0, 30.0]


def test_past_sequence_is_invariant_to_future_rows_and_values():
    base_features = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    base_dates = np.array(["2025-01-01", "2025-01-02", "2025-01-03"])
    base_tickers = np.array(["AAA", "AAA", "AAA"])
    base = build_panel_sequences(base_features, base_dates, base_tickers, sequence_length=3)

    extended = build_panel_sequences(
        np.array([[1.0], [2.0], [9_999.0], [8_888.0]], dtype=np.float32),
        np.array(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
        np.array(["AAA", "AAA", "AAA", "AAA"]),
        sequence_length=3,
    )

    base_second = np.flatnonzero(base.metadata.dates == np.datetime64("2025-01-02"))[0]
    extended_second = np.flatnonzero(extended.metadata.dates == np.datetime64("2025-01-02"))[0]
    assert np.array_equal(base.X[base_second], extended.X[extended_second])
    assert np.array_equal(
        base.metadata.valid_mask[base_second], extended.metadata.valid_mask[extended_second]
    )


def test_panel_sequence_key_and_shape_validation_does_not_require_torch():
    with pytest.raises(ValueError, match="unique"):
        build_panel_sequences(
            np.ones((2, 1)),
            ["2025-01-01", "2025-01-01"],
            ["AAA", "AAA"],
        )
    with pytest.raises(ValueError, match=r"\[row, feature\]"):
        build_panel_sequences(np.ones(3), ["2025-01-01"] * 3, ["AAA"] * 3)


def test_temporal_estimator_rejects_2d_arrays_before_training():
    estimator = TemporalConvEstimator(epochs=1)
    with pytest.raises(ValueError, match=r"\[sample, time, feature\]"):
        estimator.fit(np.ones((8, 3), dtype=np.float32), np.arange(8) % 2)


def test_validation_tail_keeps_each_date_in_exactly_one_partition():
    estimator = TemporalConvEstimator(validation_fraction=0.25)
    dates = np.repeat(np.arange(4), 3)

    train_indices, validation_indices = estimator._split_indices(len(dates), dates)

    assert set(dates[train_indices]).isdisjoint(dates[validation_indices])
    assert dates[train_indices].max() < dates[validation_indices].min()


def test_tiny_tcn_training_prediction_and_attribution_shapes():
    pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    features = rng.normal(size=(32, 5, 2)).astype(np.float32)
    targets = (features[:, -1, 0] > 0.0).astype(int)
    estimator = TemporalConvEstimator(
        channels=4,
        num_blocks=1,
        epochs=2,
        batch_size=16,
        validation_fraction=0.2,
        patience=2,
        inference_batch_size=8,
        seed=11,
    )

    estimator.fit(features, targets, sample_dates=np.arange(len(features)))
    probabilities = estimator.predict_proba(features[:4])

    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert estimator.predict(features[:4]).shape == (4,)
    assert estimator.input_attributions(features[:2]).shape == (2, 5, 2)
    assert 1 <= estimator.n_epochs_ <= 2

    restored = pickle.loads(pickle.dumps(estimator))
    assert np.allclose(restored.predict_proba(features[:4]), probabilities)
