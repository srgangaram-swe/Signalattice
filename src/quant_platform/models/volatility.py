"""Causal conditional-variance baselines.

These recursions forecast variance one observation ahead.  At index ``t`` the
returned variance was fixed before residual ``t`` was consumed; the update
using that residual becomes the forecast at ``t + 1``.  An initial variance is
required so the implementation never estimates initialization from the
evaluation interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
VarianceMethod = Literal["ewma", "garch11"]


@dataclass(frozen=True)
class ConditionalVarianceResult:
    """Immutable one-step conditional-variance forecast sequence."""

    method: VarianceMethod
    variance: FloatArray
    volatility: FloatArray
    persistence: float
    unconditional_variance: float | None


def ewma_variance(
    residuals: Any,
    *,
    decay: float = 0.94,
    initial_variance: float,
    variance_floor: float = 1e-12,
) -> ConditionalVarianceResult:
    """Return RiskMetrics-style EWMA variance forecasts in ``O(n)`` time."""

    values = _finite_vector(residuals, "residuals")
    _validate_unit_interval(decay, "decay")
    _validate_positive(initial_variance, "initial_variance")
    _validate_positive(variance_floor, "variance_floor")

    forecast = np.empty(len(values), dtype=float)
    next_variance = max(float(initial_variance), variance_floor)
    for index, residual in enumerate(values):
        forecast[index] = next_variance
        next_variance = max(
            decay * next_variance + (1.0 - decay) * residual**2,
            variance_floor,
        )
    return ConditionalVarianceResult(
        method="ewma",
        variance=_readonly(forecast),
        volatility=_readonly(np.sqrt(forecast)),
        persistence=float(decay),
        unconditional_variance=None,
    )


def garch11_variance(
    residuals: Any,
    *,
    omega: float,
    alpha: float,
    beta: float,
    initial_variance: float,
    variance_floor: float = 1e-12,
) -> ConditionalVarianceResult:
    """Return fixed-parameter GARCH(1,1) one-step variance forecasts.

    Parameters are validated but not estimated.  ``alpha + beta < 1`` enforces
    weak covariance stationarity and a finite unconditional variance.
    """

    values = _finite_vector(residuals, "residuals")
    _validate_positive(omega, "omega")
    _validate_unit_interval(alpha, "alpha", allow_zero=True)
    _validate_unit_interval(beta, "beta", allow_zero=True)
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be strictly less than one")
    _validate_positive(initial_variance, "initial_variance")
    _validate_positive(variance_floor, "variance_floor")

    forecast = np.empty(len(values), dtype=float)
    next_variance = max(float(initial_variance), variance_floor)
    for index, residual in enumerate(values):
        forecast[index] = next_variance
        next_variance = max(
            omega + alpha * residual**2 + beta * next_variance,
            variance_floor,
        )
    persistence = alpha + beta
    return ConditionalVarianceResult(
        method="garch11",
        variance=_readonly(forecast),
        volatility=_readonly(np.sqrt(forecast)),
        persistence=float(persistence),
        unconditional_variance=float(omega / (1.0 - persistence)),
    )


def _finite_vector(value: Any, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or len(array) < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
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


def _validate_unit_interval(value: float, name: str, *, allow_zero: bool = False) -> None:
    valid_lower = value >= 0.0 if allow_zero else value > 0.0
    if not np.isfinite(value) or not valid_lower or value >= 1.0:
        interval = "[0, 1)" if allow_zero else "(0, 1)"
        raise ValueError(f"{name} must be finite and lie in {interval}")
