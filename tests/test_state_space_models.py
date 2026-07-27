"""Invariant and simulation tests for state-space and risk baselines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from sklearn.covariance import oas

from quant_platform.models import (
    DynamicLinearConfig,
    LocalLevelConfig,
    dynamic_linear_filter,
    ewma_variance,
    garch11_variance,
    gaussian_interval_diagnostics,
    local_level_filter,
    shrinkage_covariance,
)


def test_local_level_matches_independent_scalar_recursion() -> None:
    observations = np.array([2.0, -0.5])
    config = LocalLevelConfig(
        process_variance=0.1,
        observation_variance=0.5,
        initial_mean=0.0,
        initial_variance=1.0,
    )

    result = local_level_filter(observations, config)

    first_prior_variance = 1.0 + 0.1
    first_predictive_variance = first_prior_variance + 0.5
    first_gain = first_prior_variance / first_predictive_variance
    first_posterior_mean = first_gain * observations[0]
    first_posterior_variance = (1.0 - first_gain) ** 2 * first_prior_variance + first_gain**2 * 0.5
    second_prior_variance = first_posterior_variance + 0.1

    assert result.forecast_mean[0] == pytest.approx(0.0)
    assert result.forecast_variance[0] == pytest.approx(first_predictive_variance)
    assert result.filtered_mean[0] == pytest.approx(first_posterior_mean)
    assert result.filtered_variance[0] == pytest.approx(first_posterior_variance)
    assert result.forecast_mean[1] == pytest.approx(first_posterior_mean)
    assert result.forecast_variance[1] == pytest.approx(second_prior_variance + 0.5)


def test_dynamic_one_feature_matches_local_level_reference() -> None:
    observations = np.array([0.2, 0.1, -0.3, 0.5, 0.7])
    local = local_level_filter(
        observations,
        LocalLevelConfig(
            process_variance=0.02,
            observation_variance=0.1,
            initial_mean=0.4,
            initial_variance=0.7,
        ),
    )
    dynamic = dynamic_linear_filter(
        observations,
        np.ones((len(observations), 1)),
        DynamicLinearConfig(
            state_variance=0.02,
            observation_variance=0.1,
            initial_state=(0.4,),
            initial_variance=0.7,
        ),
    )

    np.testing.assert_allclose(dynamic.forecast_mean, local.forecast_mean, atol=1e-12)
    np.testing.assert_allclose(dynamic.forecast_variance, local.forecast_variance, atol=1e-12)
    np.testing.assert_allclose(dynamic.filtered_state[:, 0], local.filtered_mean, atol=1e-12)
    np.testing.assert_allclose(
        dynamic.filtered_covariance[:, 0, 0],
        local.filtered_variance,
        atol=1e-12,
    )


def test_state_space_outputs_are_prefix_causal_and_immutable() -> None:
    rng = np.random.default_rng(20260726)
    observations = rng.normal(size=120)
    design = np.column_stack((np.ones(120), rng.normal(size=120)))
    local_config = LocalLevelConfig(process_variance=0.03, observation_variance=0.2)
    dynamic_config = DynamicLinearConfig(
        state_variance=0.01,
        observation_variance=0.2,
        initial_state=(0.0, 0.0),
    )
    local = local_level_filter(observations, local_config)
    dynamic = dynamic_linear_filter(observations, design, dynamic_config)

    mutated_observations = observations.copy()
    mutated_design = design.copy()
    mutated_observations[80:] += 10_000.0
    mutated_design[80:, 1] *= -500.0
    mutated_local = local_level_filter(mutated_observations, local_config)
    mutated_dynamic = dynamic_linear_filter(
        mutated_observations,
        mutated_design,
        dynamic_config,
    )

    np.testing.assert_array_equal(local.forecast_mean[:80], mutated_local.forecast_mean[:80])
    np.testing.assert_array_equal(local.filtered_mean[:79], mutated_local.filtered_mean[:79])
    np.testing.assert_array_equal(dynamic.forecast_mean[:80], mutated_dynamic.forecast_mean[:80])
    np.testing.assert_array_equal(
        dynamic.filtered_state[:79],
        mutated_dynamic.filtered_state[:79],
    )
    assert not local.forecast_mean.flags.writeable
    assert not dynamic.filtered_covariance.flags.writeable
    with pytest.raises(ValueError):
        local.forecast_mean[0] = 1.0


def test_local_level_recovers_simulated_state_better_than_constant_reference() -> None:
    rng = np.random.default_rng(743)
    n_observations = 1_500
    process_variance = 0.015
    observation_variance = 0.08
    state = np.empty(n_observations)
    state[0] = 0.0
    for index in range(1, n_observations):
        state[index] = state[index - 1] + rng.normal(scale=np.sqrt(process_variance))
    observations = state + rng.normal(scale=np.sqrt(observation_variance), size=n_observations)

    result = local_level_filter(
        observations,
        LocalLevelConfig(
            process_variance=process_variance,
            observation_variance=observation_variance,
            initial_mean=0.0,
            initial_variance=1.0,
        ),
    )
    warmup = 100
    filter_mse = np.mean(np.square(result.filtered_mean[warmup:] - state[warmup:]))
    constant_mse = np.mean(np.square(state[warmup:] - state[:warmup].mean()))

    assert filter_mse < 0.25 * constant_mse
    assert result.log_likelihood < 0.0


def test_dynamic_linear_filter_tracks_time_varying_coefficients() -> None:
    rng = np.random.default_rng(1801)
    n_observations = 1_200
    design = np.column_stack((np.ones(n_observations), rng.normal(size=n_observations)))
    coefficients = np.empty((n_observations, 2))
    coefficients[0] = (0.2, -0.4)
    for index in range(1, n_observations):
        coefficients[index] = coefficients[index - 1] + rng.normal(scale=0.015, size=2)
    observations = np.sum(design * coefficients, axis=1) + rng.normal(
        scale=0.1,
        size=n_observations,
    )

    result = dynamic_linear_filter(
        observations,
        design,
        DynamicLinearConfig(
            state_variance=0.015**2,
            observation_variance=0.1**2,
            initial_state=(0.0, 0.0),
            initial_variance=1.0,
        ),
    )
    warmup = 100
    filter_mse = np.mean(np.square(result.filtered_state[warmup:] - coefficients[warmup:]))
    zero_state_mse = np.mean(np.square(coefficients[warmup:]))

    assert filter_mse < 0.2 * zero_state_mse
    assert np.all(np.linalg.eigvalsh(result.filtered_covariance) > 0.0)


def test_gaussian_interval_diagnostics_are_calibrated_on_known_simulation() -> None:
    rng = np.random.default_rng(909)
    variance = np.linspace(0.2, 1.2, 10_000)
    mean = np.sin(np.linspace(0.0, 4.0 * np.pi, len(variance)))
    observations = mean + rng.normal(size=len(variance)) * np.sqrt(variance)

    diagnostics = gaussian_interval_diagnostics(
        observations,
        mean,
        variance,
        confidence_level=0.95,
    )

    assert diagnostics.n_observations == len(variance)
    assert diagnostics.empirical_coverage == pytest.approx(0.95, abs=0.01)
    assert diagnostics.standardized_innovation_mean == pytest.approx(0.0, abs=0.03)
    assert diagnostics.standardized_innovation_std == pytest.approx(1.0, abs=0.03)
    assert diagnostics.mean_interval_width > 0.0
    assert np.isfinite(diagnostics.gaussian_negative_log_score)


def test_ewma_matches_hand_calculated_one_step_recursion() -> None:
    residuals = np.array([2.0, -1.0, 0.5])

    result = ewma_variance(residuals, decay=0.8, initial_variance=1.5)

    expected = np.array(
        [
            1.5,
            0.8 * 1.5 + 0.2 * 2.0**2,
            0.8 * (0.8 * 1.5 + 0.2 * 2.0**2) + 0.2 * (-1.0) ** 2,
        ]
    )
    np.testing.assert_allclose(result.variance, expected)
    np.testing.assert_allclose(result.volatility, np.sqrt(expected))
    assert result.method == "ewma"
    assert result.unconditional_variance is None


def test_garch_matches_simulation_variance_and_beats_constant_reference() -> None:
    rng = np.random.default_rng(4004)
    omega, alpha, beta = 0.03, 0.12, 0.82
    unconditional_variance = omega / (1.0 - alpha - beta)
    n_observations = 5_000
    true_variance = np.empty(n_observations)
    residuals = np.empty(n_observations)
    next_variance = unconditional_variance
    for index in range(n_observations):
        true_variance[index] = next_variance
        residuals[index] = rng.normal(scale=np.sqrt(next_variance))
        next_variance = omega + alpha * residuals[index] ** 2 + beta * next_variance

    result = garch11_variance(
        residuals,
        omega=omega,
        alpha=alpha,
        beta=beta,
        initial_variance=unconditional_variance,
    )
    constant_error = np.mean(np.square(true_variance - unconditional_variance))
    garch_error = np.mean(np.square(result.variance - true_variance))

    np.testing.assert_allclose(result.variance, true_variance, rtol=1e-13, atol=1e-13)
    assert garch_error < constant_error
    assert result.persistence == pytest.approx(alpha + beta)
    assert result.unconditional_variance == pytest.approx(unconditional_variance)


def test_conditional_variance_forecasts_are_prefix_causal() -> None:
    rng = np.random.default_rng(77)
    residuals = rng.normal(size=200)
    altered = residuals.copy()
    altered[120:] *= 1_000.0

    ewma = ewma_variance(residuals, decay=0.96, initial_variance=1.0)
    altered_ewma = ewma_variance(altered, decay=0.96, initial_variance=1.0)
    garch = garch11_variance(
        residuals,
        omega=0.03,
        alpha=0.1,
        beta=0.85,
        initial_variance=0.6,
    )
    altered_garch = garch11_variance(
        altered,
        omega=0.03,
        alpha=0.1,
        beta=0.85,
        initial_variance=0.6,
    )

    np.testing.assert_array_equal(ewma.variance[:121], altered_ewma.variance[:121])
    np.testing.assert_array_equal(garch.variance[:121], altered_garch.variance[:121])
    assert not ewma.variance.flags.writeable
    assert np.all(garch.variance > 0.0)


def test_oas_shrinkage_matches_independent_sklearn_reference() -> None:
    rng = np.random.default_rng(192)
    observations = rng.multivariate_normal(
        mean=np.array([0.0, 0.5, -0.2]),
        cov=np.array(
            [
                [1.0, 0.4, 0.1],
                [0.4, 0.8, -0.1],
                [0.1, -0.1, 0.5],
            ]
        ),
        size=500,
    )

    result = shrinkage_covariance(observations)
    reference_covariance, reference_shrinkage = oas(
        observations,
        assume_centered=False,
    )

    np.testing.assert_allclose(result.covariance, reference_covariance, rtol=1e-12, atol=1e-12)
    assert result.shrinkage == pytest.approx(reference_shrinkage, rel=1e-12)
    np.testing.assert_allclose(result.location, observations.mean(axis=0))


def test_shrinkage_covariance_conditions_rank_deficient_input() -> None:
    rng = np.random.default_rng(81)
    first = rng.normal(size=40)
    observations = np.column_stack((first, first, 2.0 * first, rng.normal(size=40)))

    result = shrinkage_covariance(observations)

    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(result.covariance) > 0.0)
    assert result.condition_number < result.sample_condition_number
    assert 0.0 <= result.shrinkage <= 1.0
    assert result.n_observations == 40
    assert result.n_features == 4
    assert not result.covariance.flags.writeable


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"process_variance": -0.1, "observation_variance": 1.0}, "process_variance"),
        ({"process_variance": 0.1, "observation_variance": 0.0}, "observation_variance"),
        (
            {
                "process_variance": 0.1,
                "observation_variance": 1.0,
                "confidence_level": 1.0,
            },
            "confidence_level",
        ),
    ],
)
def test_local_level_config_rejects_invalid_parameters(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LocalLevelConfig(**kwargs)


@pytest.mark.parametrize(
    "observations",
    [
        np.array([]),
        np.array([[1.0]]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_local_level_rejects_malformed_observations(observations: np.ndarray) -> None:
    with pytest.raises(ValueError):
        local_level_filter(
            observations,
            LocalLevelConfig(process_variance=0.1, observation_variance=0.2),
        )


def test_dynamic_linear_rejects_shape_and_state_contract_violations() -> None:
    config = DynamicLinearConfig(
        state_variance=0.1,
        observation_variance=0.2,
        initial_state=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="same number of rows"):
        dynamic_linear_filter([1.0, 2.0], [[1.0, 0.0]], config)
    with pytest.raises(ValueError, match="initial_state length"):
        dynamic_linear_filter([1.0], [[1.0]], config)
    with pytest.raises(ValueError, match="finite"):
        dynamic_linear_filter([1.0], [[1.0, np.nan]], config)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: ewma_variance([0.1], decay=1.0, initial_variance=1.0),
        lambda: ewma_variance([0.1], decay=0.9, initial_variance=0.0),
        lambda: garch11_variance(
            [0.1],
            omega=0.1,
            alpha=0.6,
            beta=0.4,
            initial_variance=1.0,
        ),
        lambda: garch11_variance(
            [np.nan],
            omega=0.1,
            alpha=0.1,
            beta=0.8,
            initial_variance=1.0,
        ),
    ],
)
def test_conditional_variance_rejects_invalid_contracts(operation: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        operation()


def test_interval_diagnostics_reject_invalid_evaluation_arrays() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        gaussian_interval_diagnostics([1.0], [0.0], [1.0])
    with pytest.raises(ValueError, match="align"):
        gaussian_interval_diagnostics([1.0, 2.0], [0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        gaussian_interval_diagnostics([1.0, 2.0], [0.0, 0.0], [1.0, 0.0])


@pytest.mark.parametrize(
    "observations",
    [
        np.array([1.0, 2.0]),
        np.array([[1.0], [np.nan]]),
        np.empty((1, 2)),
    ],
)
def test_shrinkage_covariance_rejects_malformed_input(observations: np.ndarray) -> None:
    with pytest.raises(ValueError):
        shrinkage_covariance(observations)


def test_shrinkage_covariance_rejects_invalid_intensity() -> None:
    observations = np.eye(3)
    with pytest.raises(ValueError, match="shrinkage"):
        shrinkage_covariance(observations, shrinkage=1.1)
