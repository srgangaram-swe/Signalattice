"""Benchmark immutable feature publication and verified DuckDB/Parquet reads.

The workload is deterministic, synthetic, local, and single-process.  It
measures engineering mechanics only; it is not evidence of market-data scale,
distributed serving, model quality, or trading performance.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from datetime import UTC, date, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from quant_platform.features.registry import FeatureRegistry, FeatureSpec, canonical_json
from quant_platform.features.store import (
    DatasetLineage,
    FeatureMaterializationRequest,
    FeatureOutputContract,
    FeatureStore,
    RuntimeIdentity,
)
from quant_platform.utils import hash_dataframe

REPETITIONS = 7
EVIDENCE_TIME = datetime(2026, 7, 25, tzinfo=UTC)


def _frame(*, ticker_count: int = 32, date_count: int = 1_500) -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", periods=date_count)
    rows = [
        {
            "date": timestamp,
            "ticker": f"S{ticker_index:03d}",
            "close": 50.0 + ticker_index + date_index * 0.01,
            "f_return_5d": (date_index % 23 - 11) / 1_000.0,
            "f_volume_z": (ticker_index - 15.5) / 8.0 + (date_index % 7) / 10.0,
        }
        for ticker_index in range(ticker_count)
        for date_index, timestamp in enumerate(dates)
    ]
    return pd.DataFrame(rows)


def _spec(name: str, family: str, input_columns: tuple[str, ...]) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        family=family,
        input_columns=input_columns,
        parameters={"window": 5},
        lookback_bars=5,
        warmup_bars=5,
        normalization="rolling",
        missing_policy="fail",
        sampling_frequency="1d",
        leakage_risk="low",
        implementation_sha256="1" * 64,
    )


def _request(frame: pd.DataFrame) -> FeatureMaterializationRequest:
    dates = pd.to_datetime(frame["date"])
    tickers = tuple(sorted(frame["ticker"].unique()))
    registry = FeatureRegistry(
        (
            _spec("f_return_5d", "returns", ("close",)),
            _spec("f_volume_z", "liquidity", ("close",)),
        )
    )
    lineage = DatasetLineage(
        dataset_sha256=hash_dataframe(frame, length=64),
        source="synthetic_benchmark",
        source_revision="feature-store-benchmark-v1",
        request_sha256="2" * 64,
        schema_version="synthetic-1.0.0",
        retrieved_at=EVIDENCE_TIME,
        coverage_start=dates.min().date(),
        coverage_end=dates.max().date(),
        requested_tickers=tickers,
        returned_tickers=tickers,
        observations_redistributable=True,
        historical_revisions_complete=True,
        universe_membership_point_in_time=True,
        corporate_actions_complete=True,
    )
    return FeatureMaterializationRequest(
        lineage=lineage,
        features=registry.specs,
        output_contract=FeatureOutputContract(
            benchmark="S000",
            price_field="close",
        ),
        application_start=dates.min().date(),
        application_end=dates.max().date(),
        expected_end=dates.max().date(),
        partition_by="year",
        code_commit="3" * 40,
        runtime=RuntimeIdentity.capture(),
        evidence_time=EVIDENCE_TIME,
    )


def _timed[T](operation: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    value = operation()
    return value, time.perf_counter() - start


def _summary(samples: list[dict[str, float]], metric: str) -> dict[str, float]:
    values = sorted(sample[metric] for sample in samples)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95_nearest_rank": values[-1],
        "max": max(values),
    }


def _plot(result: dict[str, object], destination: Path) -> None:
    samples = pd.DataFrame(cast(list[dict[str, float]], result["raw_samples"]))
    long = samples.melt(
        var_name="operation",
        value_name="seconds",
    )
    labels = {
        "materialize_seconds": "Cold materialize",
        "cache_hit_seconds": "Verified cache hit",
        "full_read_seconds": "Verified full read",
        "filtered_read_seconds": "Predicate read",
    }
    long["operation"] = long["operation"].map(labels)
    order = list(labels.values())
    sns.set_theme(context="talk", style="whitegrid", palette="colorblind")
    figure, axis = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=long,
        x="operation",
        y="seconds",
        order=order,
        hue="operation",
        palette="colorblind",
        legend=False,
        showfliers=False,
        ax=axis,
    )
    sns.stripplot(
        data=long,
        x="operation",
        y="seconds",
        order=order,
        color="black",
        alpha=0.65,
        jitter=0.08,
        size=5,
        ax=axis,
    )
    axis.set_yscale("log")
    axis.set_title("Signalattice feature-store latency — deterministic synthetic workload")
    axis.set_xlabel("Verified operation")
    axis.set_ylabel("Wall time (seconds, logarithmic scale)")
    axis.tick_params(axis="x", rotation=12)
    axis.set_ylim(
        bottom=float(long["seconds"].min()) / 2.0,
        top=float(long["seconds"].max()) * 1.5,
    )
    for index, operation in enumerate(order):
        median = float(long.loc[long["operation"] == operation, "seconds"].median())
        axis.annotate(
            f"median {median:.3f} s",
            xy=(index, median),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )
    figure.text(
        0.01,
        0.01,
        (
            f"n={result['repetitions']} per operation; {result['rows']:,} rows, "
            f"{result['tickers']} tickers; {result['platform']}. "
            "Single-process laptop evidence; warm filesystem state is uncontrolled."
        ),
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_benchmark() -> dict[str, object]:
    frame = _frame()
    request = _request(frame)
    samples: list[dict[str, float]] = []
    object_bytes = 0
    predicate_rows = 0
    tracemalloc.start()
    for _ in range(REPETITIONS):
        with tempfile.TemporaryDirectory(prefix="signalattice-feature-store-") as temporary:
            store = FeatureStore(Path(temporary) / "store")
            manifest, materialize_seconds = _timed(partial(store.materialize, request, frame))
            _, cache_hit_seconds = _timed(partial(store.materialize, request, frame))
            full, full_read_seconds = _timed(partial(store.read, manifest.object_id))
            predicate, filtered_read_seconds = _timed(
                partial(
                    store.read,
                    manifest.object_id,
                    start=date(2022, 1, 1),
                    end=date(2022, 12, 31),
                    tickers=("S007",),
                    columns=("date", "ticker", "f_return_5d"),
                )
            )
            assert len(full) == len(frame)
            predicate_rows = len(predicate)
            object_dir = store.objects_dir / manifest.object_id
            object_bytes = sum(
                path.stat().st_size for path in object_dir.rglob("*") if path.is_file()
            )
            samples.append(
                {
                    "materialize_seconds": materialize_seconds,
                    "cache_hit_seconds": cache_hit_seconds,
                    "full_read_seconds": full_read_seconds,
                    "filtered_read_seconds": filtered_read_seconds,
                }
            )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timing = {metric: _summary(samples, metric) for metric in samples[0]}
    return {
        "benchmark": "content_addressed_feature_store",
        "schema_version": "1.0.0",
        "repetitions": REPETITIONS,
        "rows": len(frame),
        "tickers": int(frame["ticker"].nunique()),
        "columns": len(frame.columns),
        "partitions": len(pd.to_datetime(frame["date"]).dt.year.unique()),
        "predicate_rows": predicate_rows,
        "timing_seconds": timing,
        "raw_samples": samples,
        "median_materialized_rows_per_second": (
            len(frame) / timing["materialize_seconds"]["median"]
        ),
        "object_bytes": object_bytes,
        "peak_tracemalloc_bytes": peak_bytes,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            package: version(package)
            for package in ("duckdb", "numpy", "pandas", "pyarrow", "seaborn")
        },
        "evidence": "deterministic synthetic engineering workload",
        "limitations": [
            "single local process",
            "synthetic deterministic data",
            "warm filesystem state is not controlled",
            "no provider network, distributed catalog, or concurrent process writers",
            "not evidence of model quality, market edge, or trading performance",
        ],
    }


def write_example_manifest(destination: Path) -> None:
    """Write non-reconstructive synthetic lineage evidence without observations."""
    frame = _frame()
    request = _request(frame)
    with tempfile.TemporaryDirectory(prefix="signalattice-feature-example-") as temporary:
        manifest = FeatureStore(Path(temporary) / "store").materialize(request, frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json(manifest.model_dump(mode="json")) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument("--output-example-manifest", type=Path)
    args = parser.parse_args()
    result = run_benchmark()
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_bytes(canonical_json(result) + b"\n")
    if args.output_plot is not None:
        _plot(result, args.output_plot)
    if args.output_example_manifest is not None:
        write_example_manifest(args.output_example_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
