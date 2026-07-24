"""Fast smoke tests for the model-to-execution diagnostic figure suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_platform.reporting.diagnostic_plots import generate_diagnostic_figures


@dataclass
class FakeTrainResult:
    predictions: pd.DataFrame
    fold_metrics: list[dict[str, float]]
    extra: dict[str, Any]


@dataclass
class FakeBacktestResult:
    gross_returns: pd.Series
    returns: pd.Series
    costs: pd.Series
    exposures: pd.DataFrame


def _train_result() -> FakeTrainResult:
    scores = np.array(
        [
            0.08,
            0.88,
            0.18,
            0.78,
            0.28,
            0.68,
            0.38,
            0.58,
            0.42,
            0.62,
            0.32,
            0.72,
            0.22,
            0.82,
            0.12,
            0.92,
            0.45,
            0.55,
        ]
    )
    y_true = (scores + np.tile([0.08, -0.08], len(scores) // 2) >= 0.5).astype(int)
    predictions = pd.DataFrame(
        {
            "y_true": y_true,
            "score": scores,
            "forward_return": (scores - 0.5) / 20.0,
            "fold": np.repeat([0, 1, 2], 6),
        }
    )
    calibration = pd.DataFrame(
        {
            "mean_probability": [0.15, 0.38, 0.62, 0.85],
            "observed_rate": [0.10, 0.33, 0.67, 0.90],
            "count": [4, 5, 5, 4],
        }
    )
    selective = pd.DataFrame(
        {
            "coverage": [0.25, 0.50, 0.75, 1.00],
            "accuracy": [1.00, 0.94, 0.89, 0.83],
            "n_selected": [4, 9, 13, 18],
        }
    )
    deciles = pd.DataFrame(
        {
            "quantile": [1, 2, 3, 4, 5],
            "mean_return": [-0.020, -0.008, 0.001, 0.009, 0.021],
            "standard_error": [0.004, 0.003, 0.003, 0.004, 0.005],
            "count": [4, 4, 4, 3, 3],
        }
    )
    return FakeTrainResult(
        predictions=predictions,
        fold_metrics=[
            {
                "fold": 0,
                "roc_auc": 0.64,
                "average_precision": 0.66,
                "balanced_accuracy": 0.61,
            },
            {
                "fold": 1,
                "roc_auc": 0.59,
                "average_precision": 0.62,
                "balanced_accuracy": 0.57,
            },
            {
                "fold": 2,
                "roc_auc": 0.67,
                "average_precision": 0.69,
                "balanced_accuracy": 0.63,
            },
        ],
        extra={
            "calibration_table": calibration,
            "selective_prediction_frontier": selective,
            "prediction_deciles": deciles,
            "fold_feature_importances": pd.DataFrame(
                {
                    "momentum": [0.31, 0.28, 0.33],
                    "volatility": [0.19, 0.22, 0.18],
                    "liquidity": [0.11, 0.09, 0.12],
                },
                index=[0, 1, 2],
            ),
            "fold_ensemble_weights": pd.DataFrame(
                {
                    "fold": [0, 1, 2],
                    "logistic": [0.45, 0.40, 0.42],
                    "hist_gradient_boosting": [0.55, 0.60, 0.58],
                }
            ),
            "ensemble_weights": {
                "logistic": 0.41,
                "hist_gradient_boosting": 0.59,
            },
        },
    )


def _backtest_result() -> FakeBacktestResult:
    dates = pd.bdate_range("2025-01-02", periods=12)
    gross = pd.Series(
        [0.004, -0.002, 0.003, 0.001, -0.001, 0.005, -0.003, 0.002, 0.001, 0.003, -0.002, 0.004],
        index=dates,
    )
    costs = pd.Series([0.0004, 0.0002] * 6, index=dates)
    exposures = pd.DataFrame(
        {
            "gross_exposure": np.linspace(0.8, 1.0, len(dates)),
            "net_exposure": np.linspace(-0.05, 0.05, len(dates)),
            "long_exposure": np.linspace(0.40, 0.53, len(dates)),
            "short_exposure": -np.linspace(0.40, 0.47, len(dates)),
            "n_long": [3, 4] * 6,
            "n_short": [3, 3, 4, 4] * 3,
        },
        index=dates,
    )
    return FakeBacktestResult(
        gross_returns=gross,
        returns=gross - costs,
        costs=costs,
        exposures=exposures,
    )


def _decision_analysis() -> dict[str, pd.DataFrame]:
    return {
        "cost_sensitivity": pd.DataFrame(
            {
                "total_one_way_cost_bps": [0.0, 5.0, 10.0, 20.0],
                "net_total_return": [0.08, 0.05, 0.02, -0.04],
                "sharpe": [1.1, 0.8, 0.4, -0.3],
            }
        ),
        "delay_sensitivity": pd.DataFrame(
            {
                "additional_delay_bars": [0, 1, 2, 3],
                "net_total_return": [0.08, 0.05, 0.01, -0.02],
                "sharpe": [1.1, 0.7, 0.2, -0.2],
            }
        ),
        "capacity": pd.DataFrame(
            {
                "aum": [1e6, 1e7, 1e8],
                "median_participation_rate": [0.001, 0.01, 0.10],
                "p95_participation_rate": [0.003, 0.03, 0.30],
                "max_participation_rate": [0.006, 0.06, 0.60],
                "participation_limit": [0.10, 0.10, 0.10],
            }
        ),
        "inference_latency": pd.DataFrame(
            {
                "batch_size": [1, 16, 64],
                "p50_latency_ms": [1.0, 2.2, 4.8],
                "p95_latency_ms": [1.4, 2.8, 5.7],
                "p99_latency_ms": [1.6, 3.1, 6.2],
                "throughput_rows_per_second": [900, 6_000, 11_000],
            }
        ),
    }


def test_generate_diagnostic_figures_emits_complete_available_suite(tmp_path) -> None:
    figures = generate_diagnostic_figures(
        _train_result(),
        _backtest_result(),
        _decision_analysis(),
        tmp_path,
    )

    expected = {
        "reliability",
        "score_distribution",
        "precision_recall",
        "selective_coverage",
        "prediction_deciles",
        "fold_stability",
        "feature_stability",
        "ensemble_weights",
        "cost_frontier",
        "delay_decay",
        "capacity_participation",
        "inference_performance",
        "implementation_drag",
        "exposure_history",
    }
    assert set(figures) == expected
    assert all(path.parent == tmp_path for path in figures.values())
    assert all(path.suffix == ".png" and path.stat().st_size > 0 for path in figures.values())
    assert not plt.get_fignums()


def test_generate_diagnostic_figures_skips_unavailable_inputs(tmp_path) -> None:
    empty_train = FakeTrainResult(pd.DataFrame(), [], {})
    empty_backtest = FakeBacktestResult(
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        pd.Series(dtype=float),
        pd.DataFrame(),
    )

    figures = generate_diagnostic_figures(empty_train, empty_backtest, {}, tmp_path)

    assert figures == {}
    assert not plt.get_fignums()
