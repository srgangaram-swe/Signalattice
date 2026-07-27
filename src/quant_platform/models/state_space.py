"""Causal linear-Gaussian state-space baselines and interval diagnostics.

The filters in this module are deliberately small, interpretable references.
At index ``t`` each one-step forecast is formed from the posterior at ``t - 1``
before observation ``t`` is incorporated.  This ordering is the central
chronology invariant and makes prefix-mutation tests meaningful.

All returned arrays are defensive, read-only copies.  The functions perform no
I/O, parameter fitting, or global-state mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LocalLevelConfig:
    """Validated parameters for a scalar random-walk local-level model.

    The model is ``level[t] = level[t-1] + eta[t]`` and
    ``observation[t] = level[t] + epsilon[t]``.  ``process_variance`` and
    ``observation_variance`` are variances in squared observation units.
    """

    process_variance: float
    observation_variance: float
    initial_mean: float = 0.0
    initial_variance: float = 1.0
    confidence_level: float = 0.95
    variance_floor: float = 1e-12

    def __post_init__(self) -> None:
        _validate_positive(self.process_variance, "process_variance", allow_zero=True)
        _validate_positive(self.observation_variance, "observation_variance")
        _validate_finite(self.initial_mean, "initial_mean")
        _validate_positive(self.initial_variance, "initial_variance")
        _validate_probability(self.confidence_level, "confidence_level")
        _validate_positive(self.variance_floor, "variance_floor")


@dataclass(frozen=True)
class DynamicLinearConfig:
    """Parameters for a random-walk dynamic linear regression.

    ``state_variance`` is the isotropic variance added to every coefficient
    between observations.  An explicit tuple may initialize the coefficient
    mean; otherwise the filter begins at zero.
    """

    observation_variance: float
    state_variance: float
    initial_state: tuple[float, ...] | None = None
    initial_variance: float = 1.0
    confidence_level: float = 0.95
    variance_floor: float = 1e-12

    def __post_init__(self) -> None:
        _validate_positive(self.observation_variance, "observation_variance")
        _validate_positive(self.state_variance, "state_variance", allow_zero=True)
        _validate_positive(self.initial_variance, "initial_variance")
        _validate_probability(self.confidence_level, "confidence_level")
        _validate_positive(self.variance_floor, "variance_floor")
        if self.initial_state is not None:
            if not self.initial_state:
                raise ValueError("initial_state must not be empty")
            if not np.all(np.isfinite(np.asarray(self.initial_state, dtype=float))):
                raise ValueError("initial_state must contain only finite values")


@dataclass(frozen=True)
class LocalLevelResult:
    """Immutable scalar filter output, with one value per observation."""

    forecast_mean: FloatArray
    forecast_variance: FloatArray
    innovation: FloatArray
    filtered_mean: FloatArray
    filtered_variance: FloatArray
    interval_lower: FloatArray
    interval_upper: FloatArray
    confidence_level: float
    log_likelihood: float


@dataclass(frozen=True)
class DynamicLinearResult:
    """Immutable dynamic-regression output.

    ``filtered_state`` has shape ``(n_observations, n_features)`` and
    ``filtered_covariance`` has shape
    ``(n_observations, n_features, n_features)``.  Other arrays contain one
    scalar per observation.
    """

    forecast_mean: FloatArray
    forecast_variance: FloatArray
    innovation: FloatArray
    filtered_state: FloatArray
    filtered_covariance: FloatArray
    interval_lower: FloatArray
    interval_upper: FloatArray
    confidence_level: float
    log_likelihood: float


@dataclass(frozen=True)
class GaussianIntervalDiagnostics:
    """Out-of-sample quality summary for fixed Gaussian forecasts."""

    n_observations: int
    confidence_level: float
    empirical_coverage: float
    mean_interval_width: float
    standardized_innovation_mean: float
    standardized_innovation_std: float
    gaussian_negative_log_score: float


def local_level_filter(observations: Any, config: LocalLevelConfig) -> LocalLevelResult:
    """Run a causal scalar Kalman filter in ``O(n)`` time and memory.

    The forecast at each index is formed before that index's observation
    update.  Parameters must be selected on an earlier training interval; this
    function intentionally does not estimate them from ``observations``.
    """

    values = _finite_vector(observations, "observations")
    n_observations = len(values)
    forecast_mean = np.empty(n_observations, dtype=float)
    forecast_variance = np.empty(n_observations, dtype=float)
    innovation = np.empty(n_observations, dtype=float)
    filtered_mean = np.empty(n_observations, dtype=float)
    filtered_variance = np.empty(n_observations, dtype=float)

    posterior_mean = config.initial_mean
    posterior_variance = config.initial_variance
    log_likelihood = 0.0

    for index, observation in enumerate(values):
        prior_mean = posterior_mean
        prior_variance = max(
            posterior_variance + config.process_variance,
            config.variance_floor,
        )
        predictive_variance = prior_variance + config.observation_variance
        residual = observation - prior_mean
        gain = prior_variance / predictive_variance

        posterior_mean = prior_mean + gain * residual
        # The Joseph form preserves non-negativity better than ``(1-K)P``.
        posterior_variance = max(
            (1.0 - gain) ** 2 * prior_variance + gain**2 * config.observation_variance,
            config.variance_floor,
        )

        forecast_mean[index] = prior_mean
        forecast_variance[index] = predictive_variance
        innovation[index] = residual
        filtered_mean[index] = posterior_mean
        filtered_variance[index] = posterior_variance
        log_likelihood += _gaussian_log_likelihood(residual, predictive_variance)

    lower, upper = _intervals(
        forecast_mean,
        forecast_variance,
        confidence_level=config.confidence_level,
    )
    return LocalLevelResult(
        forecast_mean=_readonly(forecast_mean),
        forecast_variance=_readonly(forecast_variance),
        innovation=_readonly(innovation),
        filtered_mean=_readonly(filtered_mean),
        filtered_variance=_readonly(filtered_variance),
        interval_lower=lower,
        interval_upper=upper,
        confidence_level=config.confidence_level,
        log_likelihood=float(log_likelihood),
    )


def dynamic_linear_filter(
    observations: Any,
    design: Any,
    config: DynamicLinearConfig,
) -> DynamicLinearResult:
    """Run a random-walk coefficient Kalman filter.

    ``design[t]`` is the declared information vector available when forecasting
    ``observations[t]``.  Runtime is ``O(n * p^3)`` under dense covariance
    algebra and memory is ``O(n * p^2)`` because every posterior covariance is
    retained for auditability.
    """

    values = _finite_vector(observations, "observations")
    matrix = _finite_matrix(design, "design")
    if len(values) != matrix.shape[0]:
        raise ValueError("observations and design must have the same number of rows")
    n_observations, n_features = matrix.shape
    if config.initial_state is not None and len(config.initial_state) != n_features:
        raise ValueError("initial_state length must equal the number of design columns")

    posterior_state = (
        np.zeros(n_features, dtype=float)
        if config.initial_state is None
        else np.asarray(config.initial_state, dtype=float).copy()
    )
    posterior_covariance = np.eye(n_features, dtype=float) * config.initial_variance
    process_covariance = np.eye(n_features, dtype=float) * config.state_variance
    identity = np.eye(n_features, dtype=float)

    forecast_mean = np.empty(n_observations, dtype=float)
    forecast_variance = np.empty(n_observations, dtype=float)
    innovation = np.empty(n_observations, dtype=float)
    filtered_state = np.empty((n_observations, n_features), dtype=float)
    filtered_covariance = np.empty(
        (n_observations, n_features, n_features),
        dtype=float,
    )
    log_likelihood = 0.0

    for index, (observation, feature_row) in enumerate(zip(values, matrix, strict=True)):
        prior_state = posterior_state
        prior_covariance = posterior_covariance + process_covariance
        predicted_mean = float(feature_row @ prior_state)
        predicted_variance = max(
            float(feature_row @ prior_covariance @ feature_row) + config.observation_variance,
            config.variance_floor,
        )
        residual = observation - predicted_mean
        gain = (prior_covariance @ feature_row) / predicted_variance
        posterior_state = prior_state + gain * residual

        residual_operator = identity - np.outer(gain, feature_row)
        posterior_covariance = (
            residual_operator @ prior_covariance @ residual_operator.T
            + config.observation_variance * np.outer(gain, gain)
        )
        posterior_covariance = _stabilize_covariance(
            posterior_covariance,
            variance_floor=config.variance_floor,
        )

        forecast_mean[index] = predicted_mean
        forecast_variance[index] = predicted_variance
        innovation[index] = residual
        filtered_state[index] = posterior_state
        filtered_covariance[index] = posterior_covariance
        log_likelihood += _gaussian_log_likelihood(residual, predicted_variance)

    lower, upper = _intervals(
        forecast_mean,
        forecast_variance,
        confidence_level=config.confidence_level,
    )
    return DynamicLinearResult(
        forecast_mean=_readonly(forecast_mean),
        forecast_variance=_readonly(forecast_variance),
        innovation=_readonly(innovation),
        filtered_state=_readonly(filtered_state),
        filtered_covariance=_readonly(filtered_covariance),
        interval_lower=lower,
        interval_upper=upper,
        confidence_level=config.confidence_level,
        log_likelihood=float(log_likelihood),
    )


def gaussian_interval_diagnostics(
    observations: Any,
    forecast_mean: Any,
    forecast_variance: Any,
    *,
    confidence_level: float = 0.95,
) -> GaussianIntervalDiagnostics:
    """Evaluate fixed one-step Gaussian forecasts without fitting parameters.

    Callers must pass genuinely out-of-sample forecasts.  The function is a
    pure diagnostic: it does not recalibrate intervals or use the observations
    to modify the supplied means or variances.
    """

    values = _finite_vector(observations, "observations", min_length=2)
    means = _finite_vector(forecast_mean, "forecast_mean", min_length=2)
    variances = _finite_vector(forecast_variance, "forecast_variance", min_length=2)
    if not (len(values) == len(means) == len(variances)):
        raise ValueError("observations, forecast_mean, and forecast_variance must align")
    if np.any(variances <= 0.0):
        raise ValueError("forecast_variance must be strictly positive")
    _validate_probability(confidence_level, "confidence_level")

    lower, upper = _intervals(means, variances, confidence_level=confidence_level)
    residual = values - means
    standardized = residual / np.sqrt(variances)
    negative_log_score = 0.5 * (np.log(2.0 * np.pi * variances) + np.square(residual) / variances)
    return GaussianIntervalDiagnostics(
        n_observations=len(values),
        confidence_level=confidence_level,
        empirical_coverage=float(np.mean((values >= lower) & (values <= upper))),
        mean_interval_width=float(np.mean(upper - lower)),
        standardized_innovation_mean=float(np.mean(standardized)),
        standardized_innovation_std=float(np.std(standardized, ddof=1)),
        gaussian_negative_log_score=float(np.mean(negative_log_score)),
    )


def _intervals(
    mean: FloatArray,
    variance: FloatArray,
    *,
    confidence_level: float,
) -> tuple[FloatArray, FloatArray]:
    critical_value = float(ndtri((1.0 + confidence_level) / 2.0))
    half_width = critical_value * np.sqrt(variance)
    return _readonly(mean - half_width), _readonly(mean + half_width)


def _stabilize_covariance(covariance: FloatArray, *, variance_floor: float) -> FloatArray:
    symmetric = (covariance + covariance.T) / 2.0
    minimum_eigenvalue = float(np.linalg.eigvalsh(symmetric)[0])
    tolerance = 100.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(symmetric, 2)))
    if minimum_eigenvalue < -tolerance:
        raise FloatingPointError("Kalman covariance lost positive-semidefinite structure")
    if minimum_eigenvalue < variance_floor:
        symmetric = symmetric + np.eye(len(symmetric)) * (variance_floor - minimum_eigenvalue)
    return symmetric


def _gaussian_log_likelihood(innovation: float, variance: float) -> float:
    return float(-0.5 * (np.log(2.0 * np.pi * variance) + innovation**2 / variance))


def _finite_vector(value: Any, name: str, *, min_length: int = 1) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) < min_length:
        raise ValueError(f"{name} must contain at least {min_length} observations")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=float)


def _finite_matrix(value: Any, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one column")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=float)


def _readonly(value: FloatArray) -> FloatArray:
    output = np.array(value, dtype=float, copy=True)
    output.setflags(write=False)
    return output


def _validate_finite(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    _validate_finite(value, name)
    invalid = value < 0.0 if allow_zero else value <= 0.0
    if invalid:
        qualifier = "non-negative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be {qualifier}")


def _validate_probability(value: float, name: str) -> None:
    _validate_finite(value, name)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
