"""Deterministic linear-shrinkage covariance estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ShrinkageCovarianceResult:
    """Immutable covariance estimate and numerical-conditioning evidence."""

    covariance: FloatArray
    sample_covariance: FloatArray
    target: FloatArray
    location: FloatArray
    shrinkage: float
    sample_condition_number: float
    condition_number: float
    minimum_eigenvalue: float
    n_observations: int
    n_features: int


def shrinkage_covariance(
    observations: Any,
    *,
    shrinkage: float | None = None,
    variance_floor: float = 1e-12,
) -> ShrinkageCovarianceResult:
    """Estimate covariance toward a scaled-identity target.

    When ``shrinkage`` is omitted, the Oracle Approximating Shrinkage (OAS)
    closed-form intensity is used.  Supplying a fixed value in ``[0, 1]`` is
    useful when selection occurred on an earlier training interval.  Runtime is
    ``O(n * p^2 + p^3)`` and memory is ``O(n * p + p^2)``.
    """

    matrix = _finite_matrix(observations, "observations")
    if matrix.shape[0] < 2:
        raise ValueError("observations must contain at least two rows")
    _validate_positive(variance_floor, "variance_floor")
    if shrinkage is not None and (not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0):
        raise ValueError("shrinkage must be finite and lie in [0, 1]")

    n_observations, n_features = matrix.shape
    location = np.mean(matrix, axis=0)
    centered = matrix - location
    # OAS uses the maximum-likelihood (1/n) covariance, not the unbiased
    # sample covariance.  The convention is explicit for reproducibility.
    sample = centered.T @ centered / n_observations
    sample = (sample + sample.T) / 2.0
    mean_variance = float(np.trace(sample) / n_features)
    target_scale = max(mean_variance, variance_floor)
    target = np.eye(n_features, dtype=float) * target_scale

    intensity = _oas_intensity(sample, n_observations) if shrinkage is None else shrinkage
    covariance = (1.0 - intensity) * sample + intensity * target
    covariance = (covariance + covariance.T) / 2.0
    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance)[0])
    if minimum_eigenvalue < variance_floor:
        covariance += np.eye(n_features) * (variance_floor - minimum_eigenvalue)
        minimum_eigenvalue = variance_floor

    return ShrinkageCovarianceResult(
        covariance=_readonly(covariance),
        sample_covariance=_readonly(sample),
        target=_readonly(target),
        location=_readonly(location),
        shrinkage=float(intensity),
        sample_condition_number=float(np.linalg.cond(sample)),
        condition_number=float(np.linalg.cond(covariance)),
        minimum_eigenvalue=float(minimum_eigenvalue),
        n_observations=n_observations,
        n_features=n_features,
    )


def _oas_intensity(sample_covariance: FloatArray, n_observations: int) -> float:
    n_features = len(sample_covariance)
    mean_variance = float(np.trace(sample_covariance) / n_features)
    mean_squared_entry = float(np.mean(np.square(sample_covariance)))
    numerator = mean_squared_entry + mean_variance**2
    denominator = (n_observations + 1.0) * (mean_squared_entry - mean_variance**2 / n_features)
    if denominator <= np.finfo(float).eps:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _finite_matrix(value: Any, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a two-dimensional matrix with at least one column")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=float)


def _readonly(value: FloatArray) -> FloatArray:
    output = np.array(value, dtype=float, copy=True)
    output.setflags(write=False)
    return output


def _validate_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
