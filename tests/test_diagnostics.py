"""Tests for probabilistic forecast and economic-separation diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.models.diagnostics import (
    brier_decomposition,
    brier_skill_score,
    calibration_table,
    date_block_bootstrap_ci,
    expected_calibration_error,
    prediction_decile_return_table,
    proper_scoring_rules,
    selective_prediction_frontier,
)


def test_calibration_table_and_ece_are_exact_for_perfect_probabilities() -> None:
    y = np.array([0, 0, 1, 1])
    probability = y.astype(float)

    table = calibration_table(y, probability, n_bins=5)

    assert list(table.columns) == [
        "bin",
        "bin_lower",
        "bin_upper",
        "count",
        "fraction",
        "mean_probability",
        "observed_rate",
        "absolute_gap",
        "weighted_absolute_gap",
    ]
    assert len(table) == 5
    assert table["count"].sum() == 4
    assert table["fraction"].sum() == pytest.approx(1.0)
    assert expected_calibration_error(y, probability, n_bins=5) == pytest.approx(0.0)
    assert table.attrs["expected_calibration_error"] == pytest.approx(0.0)


def test_brier_decomposition_matches_grouped_forecasts() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.1, 0.9, 0.9])

    decomposition = brier_decomposition(y, probability, n_bins=2, strategy="uniform")

    assert decomposition["brier_score"] == pytest.approx(0.01)
    assert decomposition["reliability"] == pytest.approx(0.01)
    assert decomposition["resolution"] == pytest.approx(0.25)
    assert decomposition["uncertainty"] == pytest.approx(0.25)
    assert decomposition["decomposition_residual"] == pytest.approx(0.0, abs=1e-12)


def test_proper_scores_and_brier_skill_reward_informative_forecast() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])

    scores = proper_scoring_rules(y, probability, climatology=0.5)

    assert scores["log_loss"] > 0.0
    assert scores["brier_score"] == pytest.approx(0.025)
    assert scores["brier_skill_score"] == pytest.approx(0.9)
    assert brier_skill_score(y, probability, climatology=0.5) == pytest.approx(0.9)


def test_brier_skill_is_undefined_for_perfectly_constant_reference() -> None:
    y = np.zeros(8)
    assert np.isnan(brier_skill_score(y, np.zeros(8), climatology=0.0))


def test_selective_frontier_improves_when_uncertain_errors_are_abstained() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    probability = np.array([0.99, 0.01, 0.90, 0.10, 0.49, 0.51, 0.48, 0.52])

    frontier = selective_prediction_frontier(
        y,
        probability,
        coverage_levels=[0.5, 1.0],
    )

    assert frontier.loc[0, "coverage"] == pytest.approx(0.5)
    assert frontier.loc[0, "accuracy"] == pytest.approx(1.0)
    assert frontier.loc[1, "accuracy"] == pytest.approx(0.5)
    assert frontier.loc[0, "confidence_threshold"] > frontier.loc[1, "confidence_threshold"]


def test_prediction_deciles_show_monotone_return_separation() -> None:
    probability = np.linspace(0.005, 0.995, 100)
    forward_returns = probability - 0.5

    table = prediction_decile_return_table(probability, forward_returns)

    assert len(table) == 10
    assert (table["count"] == 10).all()
    assert table["mean_probability"].is_monotonic_increasing
    assert table["mean_return"].is_monotonic_increasing
    expected_spread = table.iloc[-1]["mean_return"] - table.iloc[0]["mean_return"]
    assert table.attrs["top_minus_bottom_return"] == pytest.approx(expected_spread)


def test_prediction_deciles_remain_balanced_under_ties() -> None:
    table = prediction_decile_return_table(
        np.full(20, 0.5),
        np.linspace(-0.01, 0.01, 20),
        n_bins=5,
    )
    assert list(table["count"]) == [4, 4, 4, 4, 4]


def test_date_block_bootstrap_is_deterministic_and_panel_aware() -> None:
    dates = np.repeat(pd.bdate_range("2025-01-02", periods=20), 3)
    values = np.repeat(np.linspace(-0.02, 0.03, 20), 3)

    first = date_block_bootstrap_ci(
        values,
        dates,
        block_size=4,
        n_bootstrap=250,
        random_state=77,
    )
    second = date_block_bootstrap_ci(
        values,
        dates,
        block_size=4,
        n_bootstrap=250,
        random_state=77,
    )

    assert first == second
    assert first["estimate"] == pytest.approx(values.mean())
    assert first["lower"] < first["estimate"] < first["upper"]
    assert first["standard_error"] > 0.0
    assert first["n_dates"] == 20


def test_constant_date_block_bootstrap_has_zero_width_interval() -> None:
    dates = np.repeat(pd.bdate_range("2025-01-02", periods=8), 2)
    result = date_block_bootstrap_ci(
        np.full(16, 0.02),
        dates,
        block_size=2,
        n_bootstrap=50,
        random_state=4,
    )
    assert result["lower"] == pytest.approx(0.02)
    assert result["upper"] == pytest.approx(0.02)
    assert result["standard_error"] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    ("y", "probability", "message"),
    [
        ([0, 1], [0.2], "equal length"),
        ([0, 2], [0.2, 0.8], "binary"),
        ([0, 1], [0.2, 1.2], r"\[0, 1\]"),
    ],
)
def test_invalid_binary_forecasts_fail_clearly(
    y: list[float],
    probability: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calibration_table(y, probability)


def test_bootstrap_rejects_unsorted_dates() -> None:
    with pytest.raises(ValueError, match="sorted"):
        date_block_bootstrap_ci(
            [1.0, 2.0, 3.0],
            pd.to_datetime(["2025-01-02", "2025-01-01", "2025-01-03"]),
            block_size=1,
            n_bootstrap=10,
        )
