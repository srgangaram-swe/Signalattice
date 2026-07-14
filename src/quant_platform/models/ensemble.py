"""Chronology-safe probability calibration and heterogeneous ensembling.

The final portion of each training sample is split chronologically again: an
earlier sub-window fits probability calibrators and a later sub-window chooses
ensemble weights. If ``dates`` are supplied, every boundary is made between
unique dates so a cross-section is never divided. Callers remain responsible
for fitting the class independently inside each outer walk-forward fold.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

CalibrationMethod = Literal["sigmoid", "isotonic"]


class _Calibrator(Protocol):
    """Structural type shared by fitted calibration strategies."""

    def predict(self, probabilities: np.ndarray) -> np.ndarray: ...


class _IdentityCalibrator:
    """Return clipped probabilities unchanged."""

    def __init__(self, probability_clip: float) -> None:
        self.probability_clip = probability_clip

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.clip(probabilities, self.probability_clip, 1.0 - self.probability_clip),
            dtype=float,
        )


class _SigmoidCalibrator:
    """Platt-style sigmoid fit to the log odds of held-out probabilities."""

    def __init__(self, probability_clip: float) -> None:
        self.probability_clip = probability_clip
        self.model = LogisticRegression(max_iter=1_000, solver="lbfgs")

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> _SigmoidCalibrator:
        self.model.fit(self._log_odds(probabilities), y)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = self.model.predict_proba(self._log_odds(probabilities))[:, 1]
        return np.asarray(
            np.clip(calibrated, self.probability_clip, 1.0 - self.probability_clip),
            dtype=float,
        )

    def _log_odds(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(
            np.asarray(probabilities, dtype=float),
            self.probability_clip,
            1.0 - self.probability_clip,
        )
        return np.asarray(np.log(clipped / (1.0 - clipped)).reshape(-1, 1), dtype=float)


class _IsotonicCalibrator:
    """Monotone non-parametric calibrator with safe out-of-range behavior."""

    def __init__(self, probability_clip: float) -> None:
        self.probability_clip = probability_clip
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> _IsotonicCalibrator:
        self.model.fit(probabilities, y)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = self.model.predict(np.asarray(probabilities, dtype=float))
        return np.asarray(
            np.clip(calibrated, self.probability_clip, 1.0 - self.probability_clip),
            dtype=float,
        )


@dataclass(frozen=True)
class _ConstantProbabilityModel:
    """Fallback for a base-fit window containing only one target class."""

    positive_probability: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = np.full(len(X), self.positive_probability, dtype=float)
        return np.column_stack((1.0 - positive, positive))

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1])


class ChronologicalCalibratedEnsemble(ClassifierMixin, BaseEstimator):
    """Calibrate and combine heterogeneous binary classifiers chronologically.

    Parameters
    ----------
    estimators:
        Named, unfitted scikit-learn-compatible binary classifiers.  Each
        estimator is cloned before fitting, so the supplied objects are not
        mutated.  Models without ``predict_proba`` may expose
        ``decision_function``; a sigmoid transform is then used to obtain a
        candidate probability.
    calibration_fraction:
        Fraction of trailing rows, or trailing unique dates when ``dates`` are
        passed to :meth:`fit`, reserved exclusively for calibration and model
        weighting.
    calibration_method:
        ``"sigmoid"`` (Platt scaling) or ``"isotonic"``.  Isotonic calibration
        is best reserved for comparatively large calibration windows.
    weighting_fraction:
        Fraction of the trailing holdout reserved for model weighting *after*
        the calibrators are fit. Keeping this window independent avoids rating
        candidates on the observations used to fit their calibrators.
    weight_temperature:
        Positive temperature for exponentially weighting candidates by their
        held-out log loss.  Smaller values concentrate weight more strongly on
        the best calibration-window model.
    probability_clip:
        Numerical probability bound used for log loss and returned forecasts.

    Notes
    -----
    The input must already be in chronological order.  This estimator performs
    an *inner* chronological holdout and must still be fitted separately within
    each outer walk-forward fold.  Reusing a calibration window that overlaps
    evaluation or backtest dates leaks future information.
    """

    def __init__(
        self,
        estimators: Mapping[str, Any],
        *,
        calibration_fraction: float = 0.2,
        calibration_method: CalibrationMethod = "sigmoid",
        weighting_fraction: float = 0.5,
        weight_temperature: float = 0.25,
        probability_clip: float = 1e-6,
    ) -> None:
        self.estimators = estimators
        self.calibration_fraction = calibration_fraction
        self.calibration_method = calibration_method
        self.weighting_fraction = weighting_fraction
        self.weight_temperature = weight_temperature
        self.probability_clip = probability_clip

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        dates: Any | None = None,
    ) -> ChronologicalCalibratedEnsemble:
        """Fit base models on the past and calibrators on trailing observations.

        ``dates`` is strongly recommended for panel data.  It must be aligned
        one-for-one with ``X`` and nondecreasing.  All rows for a date are kept
        on the same side of the chronological split.
        """

        self._validate_parameters()
        feature_names = self._feature_names(X)
        X_checked, y_checked = check_X_y(X, y, dtype=float, ensure_min_samples=2)
        y_binary = self._validate_binary_target(y_checked)
        fit_indices, calibration_indices = self._chronological_indices(len(y_binary), dates)
        calibration_fit_indices, weighting_indices = self._calibration_weighting_indices(
            calibration_indices,
            dates,
        )

        self.n_features_in_ = X_checked.shape[1]
        if feature_names is not None:
            self.feature_names_in_ = feature_names
        self.classes_ = np.array([0, 1])
        self.fit_indices_ = fit_indices
        self.calibration_indices_ = calibration_indices
        self.calibration_fit_indices_ = calibration_fit_indices
        self.weighting_indices_ = weighting_indices
        self.n_fit_samples_ = len(fit_indices)
        self.n_calibration_samples_ = len(calibration_indices)
        self.n_calibration_fit_samples_ = len(calibration_fit_indices)
        self.n_weighting_samples_ = len(weighting_indices)

        X_fit = X_checked[fit_indices]
        y_fit = y_binary[fit_indices]
        X_calibration = X_checked[calibration_fit_indices]
        y_calibration = y_binary[calibration_fit_indices]
        X_weighting = X_checked[weighting_indices]
        y_weighting = y_binary[weighting_indices]

        fitted_estimators: dict[str, Any] = {}
        calibrators: dict[str, Any] = {}
        statuses: dict[str, str] = {}
        losses: dict[str, float] = {}
        calibration_probabilities: dict[str, np.ndarray] = {}

        for name, estimator in self.estimators.items():
            model, fit_status = self._fit_candidate(estimator, X_fit, y_fit, name)
            raw_probability = self._positive_probability(model, X_calibration)
            calibrator, calibration_status = self._fit_calibrator(raw_probability, y_calibration)
            weighting_probability = calibrator.predict(
                self._positive_probability(model, X_weighting)
            )

            fitted_estimators[name] = model
            calibrators[name] = calibrator
            statuses[name] = f"{fit_status};{calibration_status}"
            calibration_probabilities[name] = weighting_probability
            losses[name] = float(log_loss(y_weighting, weighting_probability, labels=[0, 1]))

        self.estimators_ = fitted_estimators
        self.calibrators_ = calibrators
        self.calibration_status_ = statuses
        self.calibration_log_losses_ = losses
        self.calibration_probabilities_ = calibration_probabilities
        self.weights_ = self._loss_weights(losses)
        self.effective_ensemble_size_ = float(
            1.0 / sum(weight**2 for weight in self.weights_.values())
        )
        self.effective_model_identity_ = self._identity()
        return self

    def candidate_probabilities(self, X: Any) -> dict[str, np.ndarray]:
        """Return each candidate's calibrated positive-class probabilities."""

        check_is_fitted(self, ("estimators_", "calibrators_", "weights_"))
        X_checked = self._validate_prediction_features(X)
        return {
            name: self.calibrators_[name].predict(self._positive_probability(model, X_checked))
            for name, model in self.estimators_.items()
        }

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return weighted probabilities with columns ordered as ``[0, 1]``."""

        probabilities = self.candidate_probabilities(X)
        positive = sum(self.weights_[name] * candidate for name, candidate in probabilities.items())
        positive = np.clip(positive, self.probability_clip, 1.0 - self.probability_clip)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: Any) -> np.ndarray:
        """Return binary class forecasts using a 0.5 decision threshold."""

        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def weights(self) -> dict[str, float]:
        """A defensive copy of fitted, normalized ensemble weights."""

        check_is_fitted(self, "weights_")
        return dict(self.weights_)

    @property
    def effective_identity(self) -> str:
        """Stable reporting identity including method and fitted model weights."""

        check_is_fitted(self, "effective_model_identity_")
        return self.effective_model_identity_

    def _validate_parameters(self) -> None:
        if not isinstance(self.estimators, Mapping) or not self.estimators:
            raise ValueError("estimators must be a non-empty mapping of unique names to models")
        if any(not isinstance(name, str) or not name.strip() for name in self.estimators):
            raise ValueError("every estimator name must be a non-empty string")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be strictly between 0 and 1")
        if not 0.0 < self.weighting_fraction < 1.0:
            raise ValueError("weighting_fraction must be strictly between 0 and 1")
        if self.calibration_method not in {"sigmoid", "isotonic"}:
            raise ValueError("calibration_method must be 'sigmoid' or 'isotonic'")
        if self.weight_temperature <= 0.0 or not np.isfinite(self.weight_temperature):
            raise ValueError("weight_temperature must be a finite positive number")
        if not 0.0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must be strictly between 0 and 0.5")

    @staticmethod
    def _feature_names(X: Any) -> np.ndarray | None:
        columns = getattr(X, "columns", None)
        if columns is None or not all(isinstance(column, str) for column in columns):
            return None
        return np.asarray(columns, dtype=object)

    @staticmethod
    def _validate_binary_target(y: np.ndarray) -> np.ndarray:
        values = np.asarray(y)
        if not np.all(np.isin(values, (0, 1))):
            raise ValueError("ChronologicalCalibratedEnsemble requires a binary 0/1 target")
        return values.astype(int)

    def _chronological_indices(
        self, n_samples: int, dates: Any | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if dates is None:
            n_calibration = int(np.ceil(n_samples * self.calibration_fraction))
            n_calibration = min(max(n_calibration, 1), n_samples - 1)
            split = n_samples - n_calibration
            return np.arange(split), np.arange(split, n_samples)

        date_array = np.asarray(dates)
        if date_array.ndim != 1 or len(date_array) != n_samples:
            raise ValueError("dates must be one-dimensional and aligned with X")
        date_index = pd.Index(date_array)
        if date_index.hasnans:
            raise ValueError("dates cannot contain missing values")
        if not date_index.is_monotonic_increasing:
            raise ValueError("dates must be sorted in nondecreasing chronological order")

        group_codes, unique_dates = pd.factorize(date_index, sort=False)
        n_dates = len(unique_dates)
        if n_dates < 2:
            raise ValueError("at least two unique dates are required for calibration")
        n_calibration_dates = int(np.ceil(n_dates * self.calibration_fraction))
        n_calibration_dates = min(max(n_calibration_dates, 1), n_dates - 1)
        split_group = n_dates - n_calibration_dates
        fit_indices = np.flatnonzero(group_codes < split_group)
        calibration_indices = np.flatnonzero(group_codes >= split_group)
        self.fit_dates_ = np.asarray(unique_dates[:split_group])
        self.calibration_dates_ = np.asarray(unique_dates[split_group:])
        self.calibration_start_ = unique_dates[split_group]
        return fit_indices, calibration_indices

    def _calibration_weighting_indices(
        self,
        holdout_indices: np.ndarray,
        dates: Any | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split the trailing holdout into calibrator-fit and weighting tails."""
        if len(holdout_indices) < 2:
            # This path supports tiny API smoke tests. Production configuration
            # enforces many unique calibration dates, yielding disjoint windows.
            return holdout_indices, holdout_indices

        if dates is None:
            n_weighting = min(
                len(holdout_indices) - 1,
                max(1, int(np.ceil(len(holdout_indices) * self.weighting_fraction))),
            )
            return holdout_indices[:-n_weighting], holdout_indices[-n_weighting:]

        holdout_dates = pd.Index(np.asarray(dates)[holdout_indices])
        group_codes, unique_dates = pd.factorize(holdout_dates, sort=False)
        if len(unique_dates) < 2:
            return holdout_indices, holdout_indices
        n_weighting_dates = min(
            len(unique_dates) - 1,
            max(1, int(np.ceil(len(unique_dates) * self.weighting_fraction))),
        )
        split_group = len(unique_dates) - n_weighting_dates
        calibration_fit = holdout_indices[group_codes < split_group]
        weighting = holdout_indices[group_codes >= split_group]
        self.calibrator_fit_dates_ = np.asarray(unique_dates[:split_group])
        self.weighting_dates_ = np.asarray(unique_dates[split_group:])
        return calibration_fit, weighting

    def _fit_candidate(
        self,
        estimator: Any,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        name: str,
    ) -> tuple[Any, str]:
        if len(np.unique(y_fit)) < 2:
            # Laplace smoothing avoids exact zero/one forecasts and remains
            # well-defined even when the earliest market regime is one-sided.
            probability = float((y_fit.sum() + 1.0) / (len(y_fit) + 2.0))
            return _ConstantProbabilityModel(probability), "constant:single_fit_class"
        try:
            model = clone(estimator)
            model.fit(X_fit, y_fit)
        except Exception as exc:  # pragma: no cover - message path depends on backend
            raise ValueError(f"candidate '{name}' failed during fit: {exc}") from exc
        return model, "fitted"

    def _fit_calibrator(
        self,
        probability: np.ndarray,
        y_calibration: np.ndarray,
    ) -> tuple[_Calibrator, str]:
        if len(np.unique(y_calibration)) < 2:
            return _IdentityCalibrator(self.probability_clip), "identity:single_calibration_class"

        if self.calibration_method == "sigmoid":
            calibrator: _SigmoidCalibrator | _IsotonicCalibrator = _SigmoidCalibrator(
                self.probability_clip
            )
        else:
            calibrator = _IsotonicCalibrator(self.probability_clip)
        calibrator.fit(probability, y_calibration)
        return calibrator, self.calibration_method

    def _positive_probability(self, model: Any, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            output = np.asarray(model.predict_proba(X), dtype=float)
            if output.ndim == 1:
                probability = output
            elif output.ndim == 2 and output.shape[1] == 1:
                classes_array = np.asarray(getattr(model, "classes_", [1]))
                probability = (
                    output[:, 0]
                    if len(classes_array) and classes_array[0] == 1
                    else np.zeros(len(X))
                )
            elif output.ndim == 2 and output.shape[1] >= 2:
                classes_list = list(getattr(model, "classes_", range(output.shape[1])))
                column = classes_list.index(1) if 1 in classes_list else output.shape[1] - 1
                probability = output[:, column]
            else:
                raise ValueError("predict_proba must return one probability per input row")
        elif hasattr(model, "decision_function"):
            decision = np.asarray(model.decision_function(X), dtype=float)
            if decision.ndim == 2:
                decision_classes = list(getattr(model, "classes_", range(decision.shape[1])))
                column = (
                    decision_classes.index(1) if 1 in decision_classes else decision.shape[1] - 1
                )
                decision = decision[:, column]
            probability = expit(decision.reshape(-1))
        elif hasattr(model, "predict"):
            probability = np.asarray(model.predict(X), dtype=float).reshape(-1)
        else:
            raise TypeError(
                "each candidate must implement predict_proba, decision_function, or predict"
            )

        probability = np.asarray(probability, dtype=float).reshape(-1)
        if len(probability) != len(X) or not np.all(np.isfinite(probability)):
            raise ValueError("candidate produced invalid or non-finite probabilities")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("candidate probabilities must lie in [0, 1]")
        return np.asarray(
            np.clip(probability, self.probability_clip, 1.0 - self.probability_clip),
            dtype=float,
        )

    def _loss_weights(self, losses: Mapping[str, float]) -> dict[str, float]:
        loss_values = np.asarray(list(losses.values()), dtype=float)
        if not np.all(np.isfinite(loss_values)):
            return {name: 1.0 / len(losses) for name in losses}
        relative_loss = loss_values - loss_values.min()
        raw_weights = np.exp(-relative_loss / self.weight_temperature)
        total = raw_weights.sum()
        if not np.isfinite(total) or total <= 0.0:
            raw_weights = np.ones_like(raw_weights)
            total = raw_weights.sum()
        return {
            name: float(weight / total) for name, weight in zip(losses, raw_weights, strict=True)
        }

    def _identity(self) -> str:
        components = ",".join(f"{name}={weight:.6f}" for name, weight in self.weights_.items())
        return f"chronological-{self.calibration_method}-ensemble[{components}]"

    def _validate_prediction_features(self, X: Any) -> np.ndarray:
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            columns = np.asarray(X.columns, dtype=object)
            if not np.array_equal(columns, self.feature_names_in_):
                raise ValueError("prediction feature names and order must match fitted features")
        X_checked = check_array(X, dtype=float)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_checked.shape[1]} features; expected {self.n_features_in_}")
        return np.asarray(X_checked, dtype=float)


# Concise alias for callers that do not need the chronology qualifier in every
# type annotation.  The full class name remains the canonical reporting name.
CalibratedEnsembleClassifier = ChronologicalCalibratedEnsemble


__all__ = [
    "CalibratedEnsembleClassifier",
    "CalibrationMethod",
    "ChronologicalCalibratedEnsemble",
]
