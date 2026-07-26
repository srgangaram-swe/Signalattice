"""Adversarial tests for feature identity, storage, quality, and recovery."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from quant_platform.features.backfill import (
    BackfillError,
    BackfillInterrupted,
    BackfillOrchestrator,
    BackfillPartition,
    BackfillPlan,
)
from quant_platform.features.quality import (
    FeatureQualityPolicy,
    evaluate_distribution_drift,
)
from quant_platform.features.registry import (
    FeatureRegistry,
    FeatureSpec,
    FittedTransformState,
)
from quant_platform.features.store import (
    DatasetLineage,
    FeatureMaterializationManifest,
    FeatureMaterializationRequest,
    FeatureOutputContract,
    FeatureQualityGateError,
    FeatureStore,
    FeatureStoreError,
    FeatureStoreIntegrityError,
    RuntimeIdentity,
)

FIXED_TIME = datetime(2026, 7, 25, tzinfo=UTC)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _spec(
    name: str = "f_signal",
    *,
    normalization: str = "none",
    fitted_state: FittedTransformState | None = None,
) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        family="test",
        input_columns=("close",),
        parameters={"window": 2},
        lookback_bars=2,
        warmup_bars=2,
        normalization=normalization,
        missing_policy="fail",
        sampling_frequency="1d",
        leakage_risk="low",
        implementation_sha256="1" * 64,
        fitted_state=fitted_state,
    )


def _lineage(
    *,
    requested: tuple[str, ...] = ("AAA", "BBB"),
    returned: tuple[str, ...] = ("AAA", "BBB"),
) -> DatasetLineage:
    return DatasetLineage(
        dataset_sha256="2" * 64,
        source="synthetic_test",
        source_revision="fixture-v1",
        request_sha256="3" * 64,
        schema_version="1.1.0",
        retrieved_at=FIXED_TIME,
        coverage_start=date(2024, 1, 2),
        coverage_end=date(2024, 3, 29),
        requested_tickers=requested,
        returned_tickers=returned,
        observations_redistributable=True,
        historical_revisions_complete=True,
        universe_membership_point_in_time=True,
        corporate_actions_complete=True,
    )


def _request(
    *,
    lineage: DatasetLineage | None = None,
    start: date = date(2024, 1, 2),
    end: date = date(2024, 3, 29),
    partition_by: str = "month",
    policy: FeatureQualityPolicy | None = None,
) -> FeatureMaterializationRequest:
    return FeatureMaterializationRequest(
        lineage=lineage or _lineage(),
        features=(_spec(),),
        output_contract=FeatureOutputContract(
            benchmark="AAA",
            price_field="close",
        ),
        application_start=start,
        application_end=end,
        expected_end=end,
        partition_by=partition_by,
        code_commit="4" * 40,
        runtime=RuntimeIdentity(
            python="3.13.0",
            implementation="CPython",
            operating_system="test",
            machine="test",
            dependencies={
                "duckdb": "1.5.5",
                "numpy": "2.0.0",
                "pandas": "3.0.0",
                "pyarrow": "24.0.0",
                "pydantic": "2.13.0",
                "scipy": "1.18.0",
            },
        ),
        quality_policy=policy or FeatureQualityPolicy(max_business_day_gap=5),
        evidence_time=FIXED_TIME,
    )


def _frame(
    *,
    start: str = "2024-01-02",
    end: str = "2024-03-29",
    tickers: tuple[str, ...] = ("AAA", "BBB"),
    offset: float = 0.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    rows = [
        {
            "date": timestamp,
            "ticker": ticker,
            "close": 100.0 + index + offset,
            "f_signal": float(index) / 100.0 + ticker_index + offset,
        }
        for ticker_index, ticker in enumerate(tickers)
        for index, timestamp in enumerate(dates)
    ]
    return pd.DataFrame(rows)


def test_registry_is_order_independent_and_rejects_invalid_definitions():
    first = FeatureRegistry((_spec("f_z"), _spec("f_a")))
    second = FeatureRegistry((_spec("f_a"), _spec("f_z")))
    assert first.identity == second.identity
    assert first.output_columns == ("f_a", "f_z")

    with pytest.raises(ValueError, match="unique"):
        FeatureRegistry((_spec(), _spec()))
    with pytest.raises(ValidationError, match="finite"):
        _spec().model_copy(update={"parameters": {"window": float("nan")}})
        FeatureSpec.model_validate(
            {
                **_spec().model_dump(),
                "parameters": {"window": float("nan")},
            }
        )


def test_committed_example_manifest_is_strict_and_non_observational():
    path = REPOSITORY_ROOT / "docs/examples/feature_store_manifest.json"
    manifest = FeatureMaterializationManifest.model_validate_json(path.read_bytes())
    assert manifest.schema_version == "1.0.0"
    assert manifest.quality.status == "pass"
    serialized = path.read_text(encoding="utf-8")
    assert '"f_return_5d"' in serialized
    assert '"close":' not in serialized


def test_train_fitted_state_must_precede_application_interval():
    state = FittedTransformState(
        method="standard_scaler",
        state_sha256="5" * 64,
        fit_start=date(2024, 1, 2),
        fit_end=date(2024, 2, 1),
        sample_count=100,
    )
    feature = _spec(normalization="train_fitted", fitted_state=state)
    payload = _request().model_dump()
    payload["features"] = (feature,)
    payload["application_start"] = date(2024, 2, 1)
    with pytest.raises(ValidationError, match="overlaps"):
        FeatureMaterializationRequest.model_validate(payload)


def test_output_and_label_semantics_change_request_identity():
    unlabeled = _request()
    payload = unlabeled.model_dump()
    payload["output_contract"] = FeatureOutputContract(
        benchmark="AAA",
        price_field="close",
        forward_horizon=1,
        target_columns=("target_forward_return", "target_direction"),
    )
    one_day = FeatureMaterializationRequest.model_validate(payload)
    payload["output_contract"] = FeatureOutputContract(
        benchmark="AAA",
        price_field="close",
        forward_horizon=5,
        target_columns=("target_forward_return", "target_direction"),
    )
    five_day = FeatureMaterializationRequest.model_validate(payload)

    assert unlabeled.identity != one_day.identity
    assert one_day.identity != five_day.identity


def test_store_roundtrip_cache_hit_and_predicate_projection(tmp_path):
    store = FeatureStore(tmp_path / "store", max_query_rows=1_000)
    request = _request()
    frame = _frame()

    first = store.materialize(request, frame)
    second = store.materialize(request, frame.sample(frac=1.0, random_state=7))

    assert first == second
    assert store.lookup(request) == first
    filtered = store.read(
        first.object_id,
        start=date(2024, 2, 1),
        end=date(2024, 2, 29),
        tickers=("BBB",),
        columns=("date", "ticker", "f_signal"),
    )
    assert filtered["ticker"].unique().tolist() == ["BBB"]
    assert filtered["date"].min().date() == date(2024, 2, 1)
    assert filtered["date"].max().date() == date(2024, 2, 29)
    assert filtered.equals(filtered.sort_values(["date", "ticker"]).reset_index(drop=True))


def test_equivalent_request_with_different_content_fails(tmp_path):
    store = FeatureStore(tmp_path / "store")
    request = _request()
    store.materialize(request, _frame())

    with pytest.raises(FeatureStoreIntegrityError, match="different content"):
        store.materialize(request, _frame(offset=0.5))


def test_quality_failure_is_recorded_without_observations(tmp_path):
    store = FeatureStore(tmp_path / "store")
    request = _request()
    incomplete = _frame(tickers=("AAA",))

    with pytest.raises(FeatureQualityGateError) as captured:
        store.materialize(request, incomplete)

    assert "universe.complete" in captured.value.quality.failed_codes
    failure = json.loads(
        (store.failures_dir / f"{request.identity}.json").read_text(encoding="utf-8")
    )
    assert failure["quality"]["status"] == "fail"
    assert "f_signal" not in json.dumps(failure)


def test_tamper_undeclared_file_and_unsafe_queries_fail_closed(tmp_path):
    store = FeatureStore(tmp_path / "store", max_query_rows=10)
    manifest = store.materialize(_request(), _frame())
    object_dir = store.objects_dir / manifest.object_id

    with pytest.raises(FeatureStoreError, match="unknown requested"):
        store.read(manifest.object_id, columns=("date; DROP TABLE x",))
    with pytest.raises(FeatureStoreError, match="exceeds max_query_rows"):
        store.read(manifest.object_id)

    extra = object_dir / "unexpected.txt"
    extra.write_text("unexpected", encoding="utf-8")
    with pytest.raises(FeatureStoreIntegrityError, match="undeclared"):
        store.validate(manifest.object_id)
    extra.unlink()

    manifest_path = object_dir / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    tampered_manifest = json.loads(original_manifest)
    tampered_manifest["quality"]["checks"][0]["detail"] = "tampered quality claim"
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(FeatureStoreIntegrityError, match="does not bind"):
        store.validate(manifest.object_id)
    manifest_path.write_bytes(original_manifest)

    partition = object_dir / manifest.partitions[0].relative_path
    with partition.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(FeatureStoreIntegrityError, match="size mismatch"):
        store.validate(manifest.object_id)


def test_catalog_failure_is_recoverable_without_rewriting_object(tmp_path, monkeypatch):
    store = FeatureStore(tmp_path / "store")
    request = _request()
    original_publish = store._publish_catalog
    calls = 0

    def fail_once(manifest):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected catalog failure")
        original_publish(manifest)

    monkeypatch.setattr(store, "_publish_catalog", fail_once)
    with pytest.raises(OSError, match="injected"):
        store.materialize(request, _frame())
    object_directories = list(store.objects_dir.iterdir())
    assert len(object_directories) == 1

    recovered = store.materialize(request, _frame())
    assert recovered.object_id == object_directories[0].name
    assert store.lookup(request) == recovered


def test_drift_checks_pass_fail_and_reject_insufficient_evidence():
    reference = _frame()
    policy = FeatureQualityPolicy(
        min_drift_samples=20,
        max_ks_statistic=0.20,
        max_psi=0.20,
    )
    same = evaluate_distribution_drift(
        reference,
        reference.copy(),
        feature_columns=("f_signal",),
        policy=policy,
    )
    shifted = evaluate_distribution_drift(
        reference,
        _frame(offset=10.0),
        feature_columns=("f_signal",),
        policy=policy,
    )
    insufficient = evaluate_distribution_drift(
        reference.head(5),
        reference.head(5),
        feature_columns=("f_signal",),
        policy=policy,
    )
    assert same.status == "pass"
    assert shifted.status == "fail"
    assert shifted.metrics[0].reason == "threshold_exceeded"
    assert insufficient.status == "fail"
    assert insufficient.metrics[0].reason == "insufficient_samples"


def test_backfill_resumes_after_durable_interruption_and_is_idempotent(tmp_path):
    store = FeatureStore(tmp_path / "store")
    orchestrator = BackfillOrchestrator(store)
    request = _request()
    plan = BackfillPlan.create(request, max_workers=1, max_attempts=2)
    full = _frame()
    calls: list[str] = []

    def loader(partition: BackfillPartition) -> pd.DataFrame:
        calls.append(partition.key)
        dates = pd.to_datetime(full["date"])
        return full.loc[
            (dates.dt.date >= partition.start) & (dates.dt.date <= partition.end)
        ].copy()

    with pytest.raises(BackfillInterrupted):
        orchestrator.run(plan, loader, _test_interrupt_after=1)
    interrupted = orchestrator.status(plan.identity)
    assert interrupted["status"] == "interrupted"
    assert sum(item["status"] == "completed" for item in interrupted["partitions"]) == 1

    result = orchestrator.run(plan, loader)
    assert result.status == "published"
    assert result.reused_partitions == 1
    assert result.computed_partitions == 2
    assert len(store.read(result.object_id)) == len(full)

    repeated = orchestrator.run(plan, loader)
    assert repeated.object_id == result.object_id
    assert repeated.reused_partitions == 3
    assert repeated.computed_partitions == 0
    assert calls.count(plan.partitions[0].key) == 1


def test_backfill_accepts_empty_warmup_partition_but_rejects_empty_assembly(tmp_path):
    store = FeatureStore(tmp_path / "store")
    orchestrator = BackfillOrchestrator(store)
    request = _request()
    plan = BackfillPlan.create(request, max_workers=1, max_attempts=2)
    full = _frame()

    def leading_warmup_loader(partition: BackfillPartition) -> pd.DataFrame:
        if partition == plan.partitions[0]:
            return full.iloc[0:0].copy()
        dates = pd.to_datetime(full["date"]).dt.date
        return full.loc[(dates >= partition.start) & (dates <= partition.end)].copy()

    result = orchestrator.run(plan, leading_warmup_loader)
    status = orchestrator.status(plan.identity)
    expected = full.loc[pd.to_datetime(full["date"]).dt.date >= plan.partitions[1].start]

    assert status["partitions"][0]["status"] == "completed"
    assert status["partitions"][0]["rows"] == 0
    assert len(store.read(result.object_id)) == len(expected)

    empty_store = FeatureStore(tmp_path / "empty-store")
    empty_orchestrator = BackfillOrchestrator(empty_store)
    with pytest.raises(BackfillError, match="no usable feature rows"):
        empty_orchestrator.run(
            plan,
            lambda _partition: full.iloc[0:0].copy(),
        )


def test_backfill_records_redacted_bounded_failure(tmp_path):
    store = FeatureStore(tmp_path / "store")
    orchestrator = BackfillOrchestrator(store)
    plan = BackfillPlan.create(_request(), max_workers=1, max_attempts=1)

    def loader(_partition: BackfillPartition) -> pd.DataFrame:
        raise RuntimeError("api_key=should-not-survive " + ("x" * 2_000))

    with pytest.raises(FeatureStoreError):
        orchestrator.run(plan, loader)
    status = orchestrator.status(plan.identity)
    assert status["status"] == "failed"
    assert "should-not-survive" not in str(status)
    assert len(str(status["failure_message"])) <= 1_000
