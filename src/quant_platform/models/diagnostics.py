"""Probability-quality, uncertainty, and forecast-economic diagnostics.

All functions operate on genuinely out-of-sample probabilities.  They do not
create a time-series split themselves: passing in-sample forecasts will make
calibration, skill, and return-separation evidence optimistically biased.  A
climatology estimated from training data should be passed explicitly to
:func:`brier_skill_score` when it is used for model selection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

BinningStrategy = Literal["uniform", "quantile"]


def calibration_table(
    y_true: Any,
    probabilities: Any,
    *,
    n_bins: int = 10,
    strategy: BinningStrategy = "uniform",
) -> pd.DataFrame:
    """Return a reliability table whose weighted absolute gap is the ECE.

    Empty uniform-width bins are retained with a zero weight and ``NaN``
    conditional statistics, which keeps axes stable across folds.  Quantile
    binning can contain fewer than ``n_bins`` when forecast probabilities are
    tied.
    """

    y, probability = _validate_binary_forecasts(y_true, probabilities)
    edges = _bin_edges(probability, n_bins=n_bins, strategy=strategy)
    bin_index = np.searchsorted(edges[1:-1], probability, side="right")

    rows: list[dict[str, float | int]] = []
    n_observations = len(y)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        in_bin = bin_index == index
        count = int(in_bin.sum())
        fraction = count / n_observations
        if count:
            mean_probability = float(probability[in_bin].mean())
            observed_rate = float(y[in_bin].mean())
            absolute_gap = abs(mean_probability - observed_rate)
        else:
            mean_probability = float("nan")
            observed_rate = float("nan")
            absolute_gap = float("nan")
        rows.append(
            {
                "bin": index + 1,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "fraction": float(fraction),
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_gap": absolute_gap,
                "weighted_absolute_gap": (float(fraction * absolute_gap) if count else 0.0),
            }
        )

    table = pd.DataFrame(rows)
    table.attrs["expected_calibration_error"] = float(table["weighted_absolute_gap"].sum())
    table.attrs["strategy"] = strategy
    table.attrs["n_observations"] = n_observations
    return table


def expected_calibration_error(
    y_true: Any,
    probabilities: Any,
    *,
    n_bins: int = 10,
    strategy: BinningStrategy = "uniform",
) -> float:
    """Compute expected calibration error (ECE) as a weighted absolute gap."""

    table = calibration_table(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy=strategy,
    )
    return float(table.attrs["expected_calibration_error"])


def brier_decomposition(
    y_true: Any,
    probabilities: Any,
    *,
    n_bins: int = 10,
    strategy: BinningStrategy = "quantile",
) -> dict[str, float]:
    """Decompose the Brier score into reliability, resolution, and uncertainty.

    The Murphy decomposition is estimated from probability bins.  Its residual
    is reported because a finite binning approximation need not reproduce the
    observation-level Brier score exactly.  Better forecasts have lower
    reliability error, higher resolution, and a lower Brier score.
    """

    y, probability = _validate_binary_forecasts(y_true, probabilities)
    table = calibration_table(y, probability, n_bins=n_bins, strategy=strategy)
    populated = table.loc[table["count"] > 0]
    climatology = float(y.mean())
    reliability = float(
        (
            populated["fraction"]
            * (populated["mean_probability"] - populated["observed_rate"]) ** 2
        ).sum()
    )
    resolution = float(
        (populated["fraction"] * (populated["observed_rate"] - climatology) ** 2).sum()
    )
    uncertainty = float(climatology * (1.0 - climatology))
    score = float(np.mean((probability - y) ** 2))
    decomposed_score = reliability - resolution + uncertainty
    return {
        "brier_score": score,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "decomposed_brier_score": float(decomposed_score),
        "decomposition_residual": float(score - decomposed_score),
    }


def brier_skill_score(
    y_true: Any,
    probabilities: Any,
    *,
    climatology: float | None = None,
) -> float:
    """Return Brier skill relative to a constant climatology forecast.

    A value above zero beats climatology, one is perfect, and a value below
    zero is worse.  If ``climatology`` is omitted, the event rate of ``y_true``
    is used; that descriptive default must not be used for prospective model
    selection because it observes the evaluation labels.  Skill is undefined
    (``NaN``) when the reference Brier score is zero.
    """

    y, probability = _validate_binary_forecasts(y_true, probabilities)
    reference_probability = float(y.mean()) if climatology is None else float(climatology)
    if not np.isfinite(reference_probability) or not 0.0 <= reference_probability <= 1.0:
        raise ValueError("climatology must be a finite probability in [0, 1]")

    score = float(np.mean((probability - y) ** 2))
    reference_score = float(np.mean((reference_probability - y) ** 2))
    if reference_score <= np.finfo(float).eps:
        return float("nan")
    return float(1.0 - score / reference_score)


def proper_scoring_rules(
    y_true: Any,
    probabilities: Any,
    *,
    climatology: float | None = None,
    probability_clip: float = 1e-12,
) -> dict[str, float]:
    """Return binary log loss, Brier score, and Brier skill in one pass."""

    y, probability = _validate_binary_forecasts(y_true, probabilities)
    if not 0.0 < probability_clip < 0.5:
        raise ValueError("probability_clip must be strictly between 0 and 0.5")
    clipped = np.clip(probability, probability_clip, 1.0 - probability_clip)
    log_score = -np.mean(y * np.log(clipped) + (1 - y) * np.log(1.0 - clipped))
    return {
        "log_loss": float(log_score),
        "brier_score": float(np.mean((probability - y) ** 2)),
        "brier_skill_score": brier_skill_score(
            y,
            probability,
            climatology=climatology,
        ),
    }


def selective_prediction_frontier(
    y_true: Any,
    probabilities: Any,
    *,
    coverage_levels: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Measure accuracy when acting only on the most confident forecasts.

    Rows are ranked by ``2 * abs(probability - 0.5)``.  This is a diagnostic
    frontier, not a tuned trading rule: coverage levels selected after viewing
    the test set need a separate validation period before deployment.
    """

    y, probability = _validate_binary_forecasts(y_true, probabilities)
    if coverage_levels is None:
        coverage = np.linspace(1.0, 0.1, 10)
    else:
        coverage = np.asarray(list(coverage_levels), dtype=float)
    if coverage.ndim != 1 or not len(coverage):
        raise ValueError("coverage_levels must contain at least one value")
    if not np.all(np.isfinite(coverage)) or np.any((coverage <= 0.0) | (coverage > 1.0)):
        raise ValueError("coverage_levels must lie in (0, 1]")

    confidence = 2.0 * np.abs(probability - 0.5)
    predicted = (probability >= 0.5).astype(int)
    correct = predicted == y
    order = np.argsort(-confidence, kind="stable")
    n_observations = len(y)

    rows: list[dict[str, float | int]] = []
    for requested_coverage in coverage:
        n_selected = min(
            n_observations,
            max(1, int(np.ceil(requested_coverage * n_observations))),
        )
        selected = order[:n_selected]
        accuracy = float(correct[selected].mean())
        rows.append(
            {
                "requested_coverage": float(requested_coverage),
                "coverage": float(n_selected / n_observations),
                "n_selected": n_selected,
                "confidence_threshold": float(confidence[selected].min()),
                "mean_confidence": float(confidence[selected].mean()),
                "accuracy": accuracy,
                "selective_risk": float(1.0 - accuracy),
            }
        )
    return pd.DataFrame(rows)


def prediction_decile_return_table(
    probabilities: Any,
    forward_returns: Any,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize realized forward returns from low to high forecast quantiles.

    Equal probabilities are assigned deterministically in input order so that
    every requested quantile is populated when there are enough observations.
    The top-minus-bottom mean-return spread is available in
    ``table.attrs["top_minus_bottom_return"]``.
    """

    probability = _as_finite_vector(probabilities, "probabilities")
    returns = _as_finite_vector(forward_returns, "forward_returns")
    if len(probability) != len(returns):
        raise ValueError("probabilities and forward_returns must have equal length")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2")
    if n_bins > len(probability):
        raise ValueError("n_bins cannot exceed the number of observations")

    # ``method='first'`` provides a total, stable ordering under ties.  qcut on
    # that ordering creates balanced groups even for discretized classifiers.
    ranks = pd.Series(probability).rank(method="first")
    quantile = pd.qcut(ranks, q=n_bins, labels=False).to_numpy(dtype=int) + 1
    frame = pd.DataFrame(
        {
            "quantile": quantile,
            "probability": probability,
            "forward_return": returns,
        }
    )
    universe_return = float(returns.mean())

    rows: list[dict[str, float | int]] = []
    for _, group in frame.groupby("quantile", sort=True):
        count = len(group)
        bin_number = int(group["quantile"].to_numpy(dtype=int)[0])
        mean_return = float(group["forward_return"].mean())
        return_std = float(group["forward_return"].std(ddof=1)) if count > 1 else 0.0
        standard_error = return_std / np.sqrt(count)
        rows.append(
            {
                "quantile": bin_number,
                "count": count,
                "probability_min": float(group["probability"].min()),
                "probability_max": float(group["probability"].max()),
                "mean_probability": float(group["probability"].mean()),
                "mean_return": mean_return,
                "median_return": float(group["forward_return"].median()),
                "hit_rate": float((group["forward_return"] > 0.0).mean()),
                "return_std": return_std,
                "standard_error": float(standard_error),
                "return_t_stat": (
                    float(mean_return / standard_error) if standard_error > 0.0 else float("nan")
                ),
                "excess_return_vs_universe": float(mean_return - universe_return),
            }
        )

    table = pd.DataFrame(rows)
    table.attrs["top_minus_bottom_return"] = float(
        table.iloc[-1]["mean_return"] - table.iloc[0]["mean_return"]
    )
    table.attrs["n_observations"] = len(probability)
    return table


def date_block_bootstrap_ci(
    values: Any,
    dates: Any,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1_000,
    block_size: int = 5,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, float | int]:
    """Estimate a percentile CI using a circular moving-date-block bootstrap.

    Every row of a selected date is resampled together, preserving a panel's
    contemporaneous cross-section.  Adjacent unique dates are sampled in
    blocks, retaining short-horizon serial dependence.  ``values`` can be a
    vector of returns, correctness indicators, or per-observation scoring-rule
    contributions; ``statistic`` must reduce a resample to one finite scalar.
    Dates must be nondecreasing and aligned with ``values``.
    """

    vector = _as_finite_vector(values, "values")
    date_array = np.asarray(dates)
    if date_array.ndim != 1 or len(date_array) != len(vector):
        raise ValueError("dates must be one-dimensional and aligned with values")
    date_index = pd.Index(date_array)
    if date_index.hasnans:
        raise ValueError("dates cannot contain missing values")
    if not date_index.is_monotonic_increasing:
        raise ValueError("dates must be sorted in nondecreasing chronological order")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least 2")

    group_codes, unique_dates = pd.factorize(date_index, sort=False)
    n_dates = len(unique_dates)
    if not isinstance(block_size, int) or not 1 <= block_size <= n_dates:
        raise ValueError("block_size must be an integer between 1 and the number of dates")

    estimate = float(statistic(vector))
    if not np.isfinite(estimate):
        raise ValueError("statistic must return a finite scalar")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    rows_by_date = [np.flatnonzero(group_codes == code) for code in range(n_dates)]
    n_blocks = int(np.ceil(n_dates / block_size))
    bootstrap_statistics = np.empty(n_bootstrap, dtype=float)

    for bootstrap_index in range(n_bootstrap):
        starts = rng.integers(0, n_dates, size=n_blocks)
        sampled_dates = np.concatenate(
            [(start + np.arange(block_size)) % n_dates for start in starts]
        )[:n_dates]
        sampled_rows = np.concatenate([rows_by_date[index] for index in sampled_dates])
        bootstrap_statistics[bootstrap_index] = float(statistic(vector[sampled_rows]))

    if not np.all(np.isfinite(bootstrap_statistics)):
        raise ValueError("statistic returned a non-finite bootstrap value")
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        bootstrap_statistics,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "standard_error": float(bootstrap_statistics.std(ddof=1)),
        "confidence_level": float(confidence_level),
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
        "n_dates": n_dates,
    }


def _validate_binary_forecasts(
    y_true: Any,
    probabilities: Any,
) -> tuple[np.ndarray, np.ndarray]:
    y = _as_finite_vector(y_true, "y_true")
    probability = _as_finite_vector(probabilities, "probabilities")
    if len(y) != len(probability):
        raise ValueError("y_true and probabilities must have equal length")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y_true must contain only binary 0/1 values")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return y.astype(int), probability


def _as_finite_vector(values: Any, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector):
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _bin_edges(
    probability: np.ndarray,
    *,
    n_bins: int,
    strategy: BinningStrategy,
) -> np.ndarray:
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy != "quantile":
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    quantiles = np.quantile(probability, np.linspace(0.0, 1.0, n_bins + 1))
    # Always span the legal probability support; unique removes zero-width
    # bins caused by tied or constant predictions.
    return np.unique(np.concatenate(([0.0], quantiles[1:-1], [1.0])))


__all__ = [
    "BinningStrategy",
    "brier_decomposition",
    "brier_skill_score",
    "calibration_table",
    "date_block_bootstrap_ci",
    "expected_calibration_error",
    "prediction_decile_return_table",
    "proper_scoring_rules",
    "selective_prediction_frontier",
]
