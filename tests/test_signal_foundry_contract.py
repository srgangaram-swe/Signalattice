"""Producer/consumer contract tests for local Signal Foundry bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.data.signal_foundry_contract import (
    SignalFoundryContractError,
    export_signal_foundry_bundle,
    load_signal_foundry_bundle,
    validate_signal_foundry_bundle,
)


def _contract_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03"])
    effective = pd.to_datetime(dates, utc=True) + pd.Timedelta(hours=21)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["SPY"] * 3,
            "open": [99.0, 100.0, 101.0],
            "high": [101.0, 102.0, 103.0],
            "low": [98.0, 99.0, 100.0],
            "close": [100.0, 101.0, 102.0],
            "adj_close": [99.5, 100.5, 101.5],
            "volume": [1_000_000.0, 1_100_000.0, 1_200_000.0],
            "effective_at": effective,
            "available_at": effective + pd.Timedelta(hours=8),
            "observed_at": pd.Timestamp("2026-07-23T00:00:00Z"),
            "provider_updated_at": pd.Timestamp("2026-07-20T00:00:00Z"),
            "instrument_id": ["SPY"] * 3,
            "currency": ["USD"] * 3,
            "exchange_calendar": ["XNYS"] * 3,
            "adjustment_state": ["provider_adjusted_close_unadjusted_ohlc"] * 3,
            "source": ["nasdaq_data_link"] * 3,
            "source_table": ["SHARADAR/SEP"] * 3,
        }
    )


def _source_manifest() -> dict[str, object]:
    return {
        "provider": "nasdaq_data_link",
        "request": {"table": "SHARADAR/SEP"},
        "request_hash": "a" * 64,
        "snapshot_hash": "b" * 64,
        "retrieved_at": "2026-07-23T00:00:00Z",
        "adjustment_state": ["provider_adjusted_close_unadjusted_ohlc"],
        "availability_policy": {"available_at": "effective_at plus 8 hours"},
        "point_in_time_limits": {
            "historical_revisions_complete": False,
            "universe_membership_point_in_time": False,
        },
        "contains_api_key": False,
    }


def test_export_is_deterministic_partitioned_and_round_trips(tmp_path):
    first = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "bundles",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
    )
    second = export_signal_foundry_bundle(
        _contract_panel().sample(frac=1.0, random_state=7),
        tmp_path / "bundles",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
    )

    assert first == second
    assert len(list(first.glob("prices/year=*/part-00000.parquet"))) == 2
    manifest = validate_signal_foundry_bundle(first)
    assert manifest["bundle_id"] == first.name
    assert manifest["rows"] == 3
    loaded = load_signal_foundry_bundle(first)
    assert list(loaded["date"]) == sorted(loaded["date"].tolist())


def test_as_of_reader_excludes_future_available_rows(tmp_path):
    bundle = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "bundles",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
    )

    before_second = load_signal_foundry_bundle(bundle, as_of="2024-01-02T22:00:00Z")
    after_second = load_signal_foundry_bundle(bundle, as_of="2024-01-03T06:00:00Z")

    assert list(before_second["date"].dt.strftime("%Y-%m-%d")) == ["2023-12-29"]
    assert list(after_second["date"].dt.strftime("%Y-%m-%d")) == [
        "2023-12-29",
        "2024-01-02",
    ]


def test_future_mutation_cannot_change_earlier_as_of_view(tmp_path):
    original = _contract_panel()
    mutated = original.copy()
    mutated.loc[mutated["date"].eq(pd.Timestamp("2024-01-03")), "close"] = 102.5
    original_bundle = export_signal_foundry_bundle(
        original,
        tmp_path / "original",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
    )
    changed_manifest = _source_manifest()
    changed_manifest["snapshot_hash"] = "d" * 64
    mutated_bundle = export_signal_foundry_bundle(
        mutated,
        tmp_path / "mutated",
        source_manifest=changed_manifest,
        producer_git_sha="c" * 40,
    )

    cutoff = "2024-01-03T04:59:59Z"
    pd.testing.assert_frame_equal(
        load_signal_foundry_bundle(original_bundle, as_of=cutoff),
        load_signal_foundry_bundle(mutated_bundle, as_of=cutoff),
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda frame: frame.drop(columns=["available_at"]), "missing required"),
        (
            lambda frame: frame.assign(available_at=frame["effective_at"] - pd.Timedelta(hours=1)),
            "precedes",
        ),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate"),
        (lambda frame: frame.assign(volume=-1.0), "non-negative"),
        (lambda frame: frame.assign(high=frame["low"] - 1.0), "OHLC bounds"),
    ],
)
def test_invalid_contract_frames_fail_closed(tmp_path, mutation, match):
    with pytest.raises(SignalFoundryContractError, match=match):
        export_signal_foundry_bundle(
            mutation(_contract_panel()),
            tmp_path / "bundles",
            source_manifest=_source_manifest(),
            producer_git_sha="c" * 40,
        )


def test_corrupt_file_and_path_traversal_are_rejected(tmp_path):
    bundle = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "bundles",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
    )
    data_file = next(bundle.glob("prices/year=*/part-00000.parquet"))
    data_file.write_bytes(b"corrupt")
    with pytest.raises(SignalFoundryContractError, match="hash mismatch"):
        validate_signal_foundry_bundle(bundle)

    clean = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "other",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
    )
    manifest_path = clean / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["path"] = "../escape.parquet"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SignalFoundryContractError, match="unsafe bundle path"):
        validate_signal_foundry_bundle(clean)


def test_manifest_must_attest_secret_absence(tmp_path):
    manifest = _source_manifest()
    manifest["contains_api_key"] = True
    with pytest.raises(SignalFoundryContractError, match="contains_api_key=false"):
        export_signal_foundry_bundle(
            _contract_panel(),
            tmp_path / "bundles",
            source_manifest=manifest,
            producer_git_sha="c" * 40,
        )


def test_license_and_point_in_time_metadata_are_identity_bound(tmp_path):
    bundle = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "bundles",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["license"]["observations_redistributable"] = True
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(SignalFoundryContractError, match="semantic identity mismatch"):
        validate_signal_foundry_bundle(bundle)


def test_committed_consumer_fixture_is_valid_and_redistributable():
    fixture_root = Path(__file__).parent / "fixtures/signal_foundry_v1"
    pointer = json.loads((fixture_root / "current.json").read_text())
    bundle = fixture_root / pointer["bundle_id"]

    manifest = validate_signal_foundry_bundle(bundle)
    frame = load_signal_foundry_bundle(bundle, as_of="2024-01-04T06:00:00Z")

    assert manifest["schema_version"] == "1.0.0"
    assert manifest["license"]["observations_redistributable"] is True
    assert set(frame["ticker"]) == {"AAA", "SPY"}
