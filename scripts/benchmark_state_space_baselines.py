"""Benchmark SF-S3-MR5 numerical baselines on deterministic simulations.

The workload measures single-process laptop behavior and controlled simulation
recovery.  It is not market evidence, a distributed-system benchmark, an alpha
claim, or a trading-readiness artifact.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from quant_platform.models import (
    DynamicLinearConfig,
    LocalLevelConfig,
    dynamic_linear_filter,
    garch11_variance,
    gaussian_interval_diagnostics,
    local_level_filter,
    shrinkage_covariance,
)

SEED = 20260726
REPETITIONS = 7
N_OBSERVATIONS = 5_000
N_COVARIANCE_FEATURES = 24


def _timed[T](operation: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def _timing_summary(samples: list[dict[str, float]], metric: str) -> dict[str, float]:
    values = sorted(sample[metric] for sample in samples)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95_nearest_rank": values[-1],
        "max": max(values),
    }


def _workload() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    process_variance = 0.012
    observation_variance = 0.08
    latent_level = np.empty(N_OBSERVATIONS)
    latent_level[0] = 0.0
    for index in range(1, N_OBSERVATIONS):
        latent_level[index] = latent_level[index - 1] + rng.normal(scale=np.sqrt(process_variance))
    level_observations = latent_level + rng.normal(
        scale=np.sqrt(observation_variance),
        size=N_OBSERVATIONS,
    )

    design = np.column_stack(
        (
            np.ones(N_OBSERVATIONS),
            rng.normal(size=N_OBSERVATIONS),
            rng.normal(size=N_OBSERVATIONS),
            rng.normal(size=N_OBSERVATIONS),
        )
    )
    coefficients = np.empty((N_OBSERVATIONS, design.shape[1]))
    coefficients[0] = np.array([0.1, -0.25, 0.4, 0.15])
    coefficient_scale = 0.01
    for index in range(1, N_OBSERVATIONS):
        coefficients[index] = coefficients[index - 1] + rng.normal(
            scale=coefficient_scale,
            size=design.shape[1],
        )
    regression_observations = np.sum(design * coefficients, axis=1) + rng.normal(
        scale=0.12,
        size=N_OBSERVATIONS,
    )

    omega, alpha, beta = 0.025, 0.11, 0.84
    unconditional_variance = omega / (1.0 - alpha - beta)
    true_variance = np.empty(N_OBSERVATIONS)
    residuals = np.empty(N_OBSERVATIONS)
    next_variance = unconditional_variance
    for index in range(N_OBSERVATIONS):
        true_variance[index] = next_variance
        residuals[index] = rng.normal(scale=np.sqrt(next_variance))
        next_variance = omega + alpha * residuals[index] ** 2 + beta * next_variance

    factors = rng.normal(size=(N_OBSERVATIONS, 3))
    loadings = rng.normal(size=(3, N_COVARIANCE_FEATURES))
    covariance_observations = factors @ loadings + rng.normal(
        scale=1e-4,
        size=(N_OBSERVATIONS, N_COVARIANCE_FEATURES),
    )
    return {
        "latent_level": latent_level,
        "level_observations": level_observations,
        "design": design,
        "coefficients": coefficients,
        "regression_observations": regression_observations,
        "residuals": residuals,
        "true_variance": true_variance,
        "covariance_observations": covariance_observations,
        "process_variance": process_variance,
        "observation_variance": observation_variance,
        "coefficient_variance": coefficient_scale**2,
        "regression_observation_variance": 0.12**2,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "unconditional_variance": unconditional_variance,
    }


def run_benchmark() -> dict[str, object]:
    """Return timing, simulation-recovery, and conditioning evidence."""

    workload = _workload()
    samples: list[dict[str, float]] = []
    local_result = None
    dynamic_result = None
    garch_result = None
    covariance_result = None

    tracemalloc.start()
    for _ in range(REPETITIONS):
        local_result, local_seconds = _timed(
            lambda: local_level_filter(
                workload["level_observations"],
                LocalLevelConfig(
                    process_variance=workload["process_variance"],
                    observation_variance=workload["observation_variance"],
                    initial_variance=1.0,
                ),
            )
        )
        dynamic_result, dynamic_seconds = _timed(
            lambda: dynamic_linear_filter(
                workload["regression_observations"],
                workload["design"],
                DynamicLinearConfig(
                    state_variance=workload["coefficient_variance"],
                    observation_variance=workload["regression_observation_variance"],
                    initial_state=(0.0, 0.0, 0.0, 0.0),
                    initial_variance=1.0,
                ),
            )
        )
        garch_result, garch_seconds = _timed(
            lambda: garch11_variance(
                workload["residuals"],
                omega=workload["omega"],
                alpha=workload["alpha"],
                beta=workload["beta"],
                initial_variance=workload["unconditional_variance"],
            )
        )
        covariance_result, covariance_seconds = _timed(
            lambda: shrinkage_covariance(workload["covariance_observations"])
        )
        samples.append(
            {
                "local_level_seconds": local_seconds,
                "dynamic_linear_seconds": dynamic_seconds,
                "garch11_seconds": garch_seconds,
                "shrinkage_covariance_seconds": covariance_seconds,
            }
        )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert local_result is not None
    assert dynamic_result is not None
    assert garch_result is not None
    assert covariance_result is not None
    warmup = 200
    level_mse = float(
        np.mean(
            np.square(
                local_result.filtered_mean[warmup:]
                - cast(np.ndarray, workload["latent_level"])[warmup:]
            )
        )
    )
    level_naive_mse = float(
        np.mean(
            np.square(
                cast(np.ndarray, workload["latent_level"])[warmup:]
                - cast(np.ndarray, workload["latent_level"])[:warmup].mean()
            )
        )
    )
    coefficient_mse = float(
        np.mean(
            np.square(
                dynamic_result.filtered_state[warmup:]
                - cast(np.ndarray, workload["coefficients"])[warmup:]
            )
        )
    )
    coefficient_naive_mse = float(
        np.mean(np.square(cast(np.ndarray, workload["coefficients"])[warmup:]))
    )
    garch_mse = float(
        np.mean(np.square(garch_result.variance - cast(np.ndarray, workload["true_variance"])))
    )
    garch_naive_mse = float(
        np.mean(
            np.square(
                cast(np.ndarray, workload["true_variance"])
                - float(workload["unconditional_variance"])
            )
        )
    )
    interval = gaussian_interval_diagnostics(
        workload["level_observations"],
        local_result.forecast_mean,
        local_result.forecast_variance,
    )
    timing = {metric: _timing_summary(samples, metric) for metric in samples[0]}
    return {
        "benchmark": "state_space_volatility_baselines",
        "schema_version": "1.0.0",
        "seed": SEED,
        "repetitions": REPETITIONS,
        "n_observations": N_OBSERVATIONS,
        "n_dynamic_features": int(cast(np.ndarray, workload["design"]).shape[1]),
        "n_covariance_features": N_COVARIANCE_FEATURES,
        "timing_seconds": timing,
        "raw_samples": samples,
        "throughput_observations_per_second": {
            metric.removesuffix("_seconds"): N_OBSERVATIONS / summary["median"]
            for metric, summary in timing.items()
        },
        "peak_tracemalloc_bytes": peak_bytes,
        "simulation_evidence": {
            "local_level_filtered_mse": level_mse,
            "local_level_constant_mse": level_naive_mse,
            "local_level_mse_ratio": level_mse / level_naive_mse,
            "dynamic_state_filtered_mse": coefficient_mse,
            "dynamic_state_zero_mse": coefficient_naive_mse,
            "dynamic_state_mse_ratio": coefficient_mse / coefficient_naive_mse,
            "garch11_variance_mse": garch_mse,
            "garch11_constant_variance_mse": garch_naive_mse,
            "garch11_mse_ratio": max(
                garch_mse / garch_naive_mse,
                np.finfo(float).eps,
            ),
            "gaussian_interval_nominal_coverage": interval.confidence_level,
            "gaussian_interval_empirical_coverage": interval.empirical_coverage,
        },
        "conditioning_evidence": {
            "sample_condition_number": covariance_result.sample_condition_number,
            "shrinkage_condition_number": covariance_result.condition_number,
            "condition_number_ratio": (
                covariance_result.condition_number / covariance_result.sample_condition_number
            ),
            "shrinkage_intensity": covariance_result.shrinkage,
            "minimum_eigenvalue": covariance_result.minimum_eigenvalue,
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            package: version(package) for package in ("numpy", "scipy", "scikit-learn", "seaborn")
        },
        "evidence": "deterministic synthetic implementation-recovery workload",
        "limitations": [
            "single local process on one laptop",
            "synthetic linear-Gaussian and fixed-parameter GARCH processes",
            "no parameter-estimation, market-data, transaction-cost, or execution evidence",
            "not evidence of alpha, paper-trading readiness, or live-trading readiness",
            "timings are descriptive and are not a CI pass/fail threshold",
        ],
    }


def _plot(result: dict[str, object], destination: Path) -> None:
    raw_samples = pd.DataFrame(cast(list[dict[str, float]], result["raw_samples"]))
    labels = {
        "local_level_seconds": "Local level",
        "dynamic_linear_seconds": "Dynamic linear",
        "garch11_seconds": "GARCH(1,1)",
        "shrinkage_covariance_seconds": "OAS covariance",
    }
    latency = raw_samples.rename(columns=labels).melt(
        var_name="baseline",
        value_name="seconds",
    )
    simulation = cast(dict[str, float], result["simulation_evidence"])
    conditioning = cast(dict[str, float], result["conditioning_evidence"])
    relative = pd.DataFrame(
        {
            "evidence": [
                "Local-level MSE",
                "Dynamic-state MSE",
                "GARCH variance MSE",
                "Cov. condition no.",
            ],
            "ratio_to_naive_or_sample": [
                simulation["local_level_mse_ratio"],
                simulation["dynamic_state_mse_ratio"],
                simulation["garch11_mse_ratio"],
                conditioning["condition_number_ratio"],
            ],
        }
    )

    sns.set_theme(context="talk", style="whitegrid", palette="colorblind")
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    sns.boxplot(
        data=latency,
        x="baseline",
        y="seconds",
        hue="baseline",
        palette="colorblind",
        legend=False,
        showfliers=False,
        ax=axes[0],
    )
    sns.stripplot(
        data=latency,
        x="baseline",
        y="seconds",
        color="black",
        alpha=0.65,
        jitter=0.08,
        size=5,
        ax=axes[0],
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Bounded single-process runtime")
    axes[0].set_xlabel("Baseline")
    axes[0].set_ylabel("Wall time (seconds, log scale)")
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(
        data=relative,
        x="evidence",
        y="ratio_to_naive_or_sample",
        hue="evidence",
        palette="colorblind",
        legend=False,
        ax=axes[1],
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Reference")
    axes[1].set_yscale("log")
    axes[1].set_title("Controlled recovery and conditioning")
    axes[1].set_xlabel("Diagnostic")
    axes[1].set_ylabel("Error/condition ratio (lower is better, log scale)")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].legend(loc="upper right")

    figure.suptitle("SF-S3-MR5 state-space and risk baselines — synthetic evidence")
    figure.text(
        0.01,
        0.01,
        (
            f"seed={result['seed']}; n={result['n_observations']:,}; "
            f"{result['repetitions']} timing repetitions; {result['platform']}. "
            "Implementation-recovery evidence only—not market edge or trading readiness."
        ),
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-plot", type=Path)
    arguments = parser.parse_args()
    result = run_benchmark()
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output_json is None:
        print(serialized, end="")
    else:
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(serialized, encoding="utf-8")
    if arguments.output_plot is not None:
        _plot(result, arguments.output_plot)


if __name__ == "__main__":
    main()
