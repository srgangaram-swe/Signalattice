"""Tests for chronology-safe calibrated heterogeneous ensembling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from quant_platform.models.ensemble import ChronologicalCalibratedEnsemble


def _classification_sample(
    *,
    n_dates: int = 60,
    rows_per_date: int = 3,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    n_rows = n_dates * rows_per_date
    X = pd.DataFrame(
        rng.normal(size=(n_rows, 3)),
        columns=["momentum", "volatility", "carry"],
    )
    latent = (
        1.4 * X["momentum"]
        - 0.7 * X["volatility"]
        + rng.normal(
            scale=0.6,
            size=n_rows,
        )
    )
    y = (latent > 0.0).to_numpy(dtype=int)
    dates = np.repeat(pd.bdate_range("2024-01-02", periods=n_dates), rows_per_date)
    return X, y, dates


def _ensemble(*, method: str = "sigmoid") -> ChronologicalCalibratedEnsemble:
    return ChronologicalCalibratedEnsemble(
        {
            "linear": LogisticRegression(max_iter=500, random_state=7),
            "tree": DecisionTreeClassifier(max_depth=2, min_samples_leaf=8, random_state=7),
        },
        calibration_fraction=0.25,
        calibration_method=method,
    )


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_ensemble_produces_normalized_probabilities_and_weights(method: str) -> None:
    X, y, dates = _classification_sample()
    model = _ensemble(method=method).fit(X, y, dates=dates)

    probabilities = model.predict_proba(X.tail(11))
    candidates = model.candidate_probabilities(X.tail(11))

    assert probabilities.shape == (11, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    assert set(candidates) == {"linear", "tree"}
    assert all(candidate.shape == (11,) for candidate in candidates.values())
    assert set(model.predict(X.tail(11))).issubset({0, 1})
    assert sum(model.weights.values()) == pytest.approx(1.0)
    assert all(weight >= 0.0 for weight in model.weights.values())
    assert 1.0 <= model.effective_ensemble_size_ <= 2.0
    assert method in model.effective_identity


def test_panel_split_never_divides_a_cross_section() -> None:
    X, y, dates = _classification_sample(n_dates=20, rows_per_date=4)
    model = _ensemble().fit(X, y, dates=dates)

    fit_dates = set(dates[model.fit_indices_])
    calibration_dates = set(dates[model.calibration_indices_])

    assert fit_dates.isdisjoint(calibration_dates)
    assert max(fit_dates) < min(calibration_dates)
    assert len(calibration_dates) == 5
    assert model.n_fit_samples_ == 60
    assert model.n_calibration_samples_ == 20
    calibrator_dates = set(dates[model.calibration_fit_indices_])
    weighting_dates = set(dates[model.weighting_indices_])
    assert calibrator_dates.isdisjoint(weighting_dates)
    assert max(calibrator_dates) < min(weighting_dates)
    assert fit_dates.isdisjoint(calibrator_dates | weighting_dates)


def test_estimators_are_cloned_and_loss_weights_are_order_consistent() -> None:
    X, y, dates = _classification_sample()
    original = LogisticRegression(max_iter=500, random_state=7)
    model = ChronologicalCalibratedEnsemble(
        {
            "linear": original,
            "stump": DecisionTreeClassifier(max_depth=1, random_state=7),
        },
        calibration_fraction=0.3,
    ).fit(X, y, dates=dates)

    assert not hasattr(original, "coef_")
    best = min(model.calibration_log_losses_, key=model.calibration_log_losses_.get)
    assert model.weights_[best] == max(model.weights_.values())


def test_decision_function_candidate_is_supported() -> None:
    X, y, dates = _classification_sample()
    model = ChronologicalCalibratedEnsemble(
        {"linear_svc": LinearSVC(random_state=11)},
        calibration_fraction=0.2,
    ).fit(X, y, dates=dates)

    probability = model.predict_proba(X.iloc[:7])[:, 1]
    assert np.all((probability > 0.0) & (probability < 1.0))
    assert model.weights == {"linear_svc": pytest.approx(1.0)}


def test_single_class_calibration_window_falls_back_safely() -> None:
    X, _, dates = _classification_sample(n_dates=40, rows_per_date=1)
    y = np.tile([0, 1], 20)
    y[-10:] = 1
    model = _ensemble().fit(X, y, dates=dates)

    assert all(
        status.endswith("identity:single_calibration_class")
        for status in model.calibration_status_.values()
    )
    assert np.all(np.isfinite(model.predict_proba(X)[:, 1]))
    assert sum(model.weights_.values()) == pytest.approx(1.0)


def test_single_class_base_window_uses_smoothed_constant_candidate() -> None:
    X, _, dates = _classification_sample(n_dates=20, rows_per_date=2)
    y = np.zeros(len(X), dtype=int)
    y[-10::2] = 1
    model = ChronologicalCalibratedEnsemble(
        {"linear": LogisticRegression()},
        calibration_fraction=0.25,
        calibration_method="isotonic",
    ).fit(X, y, dates=dates)

    assert model.calibration_status_["linear"].startswith("constant:single_fit_class")
    assert np.all(np.isfinite(model.predict_proba(X)))


def test_unsorted_dates_and_feature_reordering_are_rejected() -> None:
    X, y, dates = _classification_sample()
    swapped_dates = dates.to_numpy(copy=True)
    swapped_dates[[0, -1]] = swapped_dates[[-1, 0]]
    with pytest.raises(ValueError, match="sorted"):
        _ensemble().fit(X, y, dates=swapped_dates)

    model = _ensemble().fit(X, y, dates=dates)
    with pytest.raises(ValueError, match="feature names"):
        model.predict_proba(X[["carry", "volatility", "momentum"]])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"estimators": {}}, "non-empty mapping"),
        (
            {"estimators": {"model": LogisticRegression()}, "calibration_fraction": 1.0},
            "strictly between",
        ),
        (
            {"estimators": {"model": LogisticRegression()}, "calibration_method": "beta"},
            "sigmoid",
        ),
        (
            {"estimators": {"model": LogisticRegression()}, "weighting_fraction": 1.0},
            "weighting_fraction",
        ),
    ],
)
def test_invalid_configuration_fails_clearly(kwargs: dict[str, object], message: str) -> None:
    X, y, dates = _classification_sample(n_dates=10)
    with pytest.raises(ValueError, match=message):
        ChronologicalCalibratedEnsemble(**kwargs).fit(X, y, dates=dates)
