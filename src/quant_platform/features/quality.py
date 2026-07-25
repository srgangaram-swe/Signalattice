"""Machine-readable feature quality and distribution-drift evidence."""

from __future__ import annotations

import math
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import ks_2samp

from quant_platform.data.schema import DATE_COL, TICKER_COL


class _Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FeatureQualityPolicy(_Evidence):
    """Versioned mandatory thresholds for a materialization."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    max_missing_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    max_business_day_gap: int = Field(default=5, ge=1, le=366)
    max_staleness_days: int = Field(default=7, ge=0, le=3_650)
    min_drift_samples: int = Field(default=100, ge=20, le=10_000_000)
    max_ks_statistic: float = Field(default=0.35, gt=0.0, le=1.0)
    max_psi: float = Field(default=0.25, gt=0.0, le=100.0)
    psi_bins: int = Field(default=10, ge=4, le=100)


class QualityCheck(_Evidence):
    """One measured fact and its pass/fail decision."""

    code: str
    status: Literal["pass", "warn", "fail"]
    observed: float | int | str | list[str]
    threshold: float | int | str | None = None
    detail: str


class FeatureQualityReport(_Evidence):
    """Complete deterministic quality result for one candidate frame."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["pass", "fail"]
    rows: int = Field(ge=0)
    tickers: tuple[str, ...]
    date_min: date | None
    date_max: date | None
    checks: tuple[QualityCheck, ...]

    @property
    def failed_codes(self) -> tuple[str, ...]:
        """Return stable machine-readable identifiers for mandatory failures."""
        return tuple(check.code for check in self.checks if check.status == "fail")


class DriftMetric(_Evidence):
    """Two-sample drift evidence for one numeric feature."""

    feature: str
    reference_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    ks_statistic: float | None
    psi: float | None
    status: Literal["pass", "fail"]
    reason: str


class DriftReport(_Evidence):
    """Aggregate drift decision across all declared feature columns."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    status: Literal["pass", "fail"]
    metrics: tuple[DriftMetric, ...]


def evaluate_feature_quality(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    requested_tickers: tuple[str, ...],
    expected_end: date,
    policy: FeatureQualityPolicy,
) -> FeatureQualityReport:
    """Evaluate schema, key, coverage, finiteness, gap, and freshness SLAs."""
    checks: list[QualityCheck] = []
    required = (DATE_COL, TICKER_COL, *feature_columns)
    missing_columns = sorted(set(required).difference(frame.columns))
    checks.append(
        QualityCheck(
            code="schema.required_columns",
            status="fail" if missing_columns else "pass",
            observed=missing_columns,
            threshold="none missing",
            detail="Required key and feature columns are present.",
        )
    )
    if missing_columns or frame.empty:
        if frame.empty:
            checks.append(
                QualityCheck(
                    code="rows.non_empty",
                    status="fail",
                    observed=0,
                    threshold=1,
                    detail="Materializations must contain at least one row.",
                )
            )
        return FeatureQualityReport(
            status="fail",
            rows=int(len(frame)),
            tickers=(),
            date_min=None,
            date_max=None,
            checks=tuple(checks),
        )

    dates = pd.to_datetime(frame[DATE_COL], errors="coerce")
    invalid_dates = int(dates.isna().sum())
    checks.append(
        QualityCheck(
            code="dates.valid",
            status="fail" if invalid_dates else "pass",
            observed=invalid_dates,
            threshold=0,
            detail="Every date must parse unambiguously.",
        )
    )
    tickers = tuple(sorted(frame[TICKER_COL].astype(str).unique()))
    missing_tickers = sorted(set(requested_tickers).difference(tickers))
    checks.append(
        QualityCheck(
            code="universe.complete",
            status="fail" if missing_tickers else "pass",
            observed=missing_tickers,
            threshold="none missing",
            detail="Every requested ticker must be represented.",
        )
    )
    duplicate_count = int(frame.duplicated([DATE_COL, TICKER_COL], keep=False).sum())
    checks.append(
        QualityCheck(
            code="keys.unique",
            status="fail" if duplicate_count else "pass",
            observed=duplicate_count,
            threshold=0,
            detail="The (date, ticker) key must be unique.",
        )
    )

    feature_frame = frame.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    nonfinite = int((~np.isfinite(feature_frame.to_numpy(dtype=float))).sum())
    total_values = int(feature_frame.size)
    missing_fraction = nonfinite / total_values if total_values else 1.0
    checks.append(
        QualityCheck(
            code="values.missing_or_nonfinite_fraction",
            status="fail" if missing_fraction > policy.max_missing_fraction else "pass",
            observed=float(missing_fraction),
            threshold=policy.max_missing_fraction,
            detail="Feature values must satisfy the declared missingness/finiteness SLA.",
        )
    )

    largest_gap = 0
    if invalid_dates == 0:
        for _ticker, group in frame.assign(_date=dates).groupby(TICKER_COL, sort=True):
            ordered = pd.DatetimeIndex(group["_date"].sort_values().unique())
            if len(ordered) > 1:
                gaps = np.busday_count(
                    ordered[:-1].date.astype("datetime64[D]"),
                    ordered[1:].date.astype("datetime64[D]"),
                )
                largest_gap = max(largest_gap, int(gaps.max(initial=0)))
    checks.append(
        QualityCheck(
            code="dates.max_business_day_gap",
            status="fail" if largest_gap > policy.max_business_day_gap else "pass",
            observed=largest_gap,
            threshold=policy.max_business_day_gap,
            detail="Per-ticker observation gaps must remain within the configured bound.",
        )
    )

    date_min = dates.min().date() if invalid_dates < len(dates) else None
    date_max = dates.max().date() if invalid_dates < len(dates) else None
    staleness = (expected_end - date_max).days if date_max is not None else math.inf
    checks.append(
        QualityCheck(
            code="dates.staleness_days",
            status="fail" if staleness > policy.max_staleness_days else "pass",
            observed=int(staleness) if math.isfinite(staleness) else "unknown",
            threshold=policy.max_staleness_days,
            detail="Latest evidence must reach the declared expected endpoint.",
        )
    )
    status: Literal["pass", "fail"] = (
        "fail" if any(check.status == "fail" for check in checks) else "pass"
    )
    return FeatureQualityReport(
        status=status,
        rows=int(len(frame)),
        tickers=tickers,
        date_min=date_min,
        date_max=date_max,
        checks=tuple(checks),
    )


def evaluate_distribution_drift(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    policy: FeatureQualityPolicy,
) -> DriftReport:
    """Evaluate two-sample KS and population-stability-index evidence.

    Quantile bins are fitted exclusively on the reference sample.  Insufficient
    evidence is a failure rather than a silent pass.
    """
    metrics: list[DriftMetric] = []
    for feature in feature_columns:
        reference_values = _finite_values(reference, feature)
        candidate_values = _finite_values(candidate, feature)
        if (
            len(reference_values) < policy.min_drift_samples
            or len(candidate_values) < policy.min_drift_samples
        ):
            metrics.append(
                DriftMetric(
                    feature=feature,
                    reference_count=len(reference_values),
                    candidate_count=len(candidate_values),
                    ks_statistic=None,
                    psi=None,
                    status="fail",
                    reason="insufficient_samples",
                )
            )
            continue
        ks = float(ks_2samp(reference_values, candidate_values, method="auto").statistic)
        psi = _population_stability_index(
            reference_values,
            candidate_values,
            bins=policy.psi_bins,
        )
        failed = ks > policy.max_ks_statistic or psi > policy.max_psi
        metrics.append(
            DriftMetric(
                feature=feature,
                reference_count=len(reference_values),
                candidate_count=len(candidate_values),
                ks_statistic=ks,
                psi=psi,
                status="fail" if failed else "pass",
                reason="threshold_exceeded" if failed else "within_threshold",
            )
        )
    return DriftReport(
        status="fail" if any(metric.status == "fail" for metric in metrics) else "pass",
        metrics=tuple(metrics),
    )


def _finite_values(frame: pd.DataFrame, feature: str) -> np.ndarray:
    if feature not in frame:
        return np.array([], dtype=float)
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _population_stability_index(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    bins: int,
) -> float:
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0 if np.allclose(reference, candidate) else float("inf")
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts = np.histogram(reference, bins=edges)[0].astype(float)
    candidate_counts = np.histogram(candidate, bins=edges)[0].astype(float)
    epsilon = 1e-8
    reference_share = np.maximum(reference_counts / reference_counts.sum(), epsilon)
    candidate_share = np.maximum(candidate_counts / candidate_counts.sum(), epsilon)
    return float(
        np.sum((candidate_share - reference_share) * np.log(candidate_share / reference_share))
    )
