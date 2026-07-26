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
    load_signal_foundry_bundle_view,
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
            "corporate_actions_complete": False,
        },
        "contains_api_key": False,
        "observations_redistributable": False,
    }


def _universe_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "membership_id": ["sp500-spy", "sp500-spy"],
            "universe_id": ["SP500", "SP500"],
            "instrument_id": ["SPY", "SPY"],
            "ticker": ["SPY", "SPY"],
            "effective_at": pd.to_datetime(
                ["2023-01-01T00:00:00Z", "2023-01-01T00:00:00Z"], utc=True
            ),
            "available_at": pd.to_datetime(
                ["2023-01-01T01:00:00Z", "2024-02-01T01:00:00Z"], utc=True
            ),
            "observed_at": pd.to_datetime(
                ["2023-01-01T02:00:00Z", "2024-02-01T02:00:00Z"], utc=True
            ),
            "provider_updated_at": pd.to_datetime(
                ["2023-01-01T01:30:00Z", "2024-02-01T01:30:00Z"], utc=True
            ),
            "is_member": [True, False],
            "reason": ["initial inclusion", "provider correction"],
            "source": ["synthetic", "synthetic"],
            "source_table": ["TEST/UNIVERSE", "TEST/UNIVERSE"],
        }
    )


def _corporate_actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": ["spy-split", "spy-split"],
            "instrument_id": ["SPY", "SPY"],
            "ticker": ["SPY", "SPY"],
            "action_type": ["split", "split"],
            "effective_at": pd.to_datetime(
                ["2024-02-15T14:30:00Z", "2024-02-15T14:30:00Z"], utc=True
            ),
            "available_at": pd.to_datetime(
                ["2024-02-01T12:00:00Z", "2024-02-10T12:00:00Z"], utc=True
            ),
            "observed_at": pd.to_datetime(
                ["2024-02-01T13:00:00Z", "2024-02-10T13:00:00Z"], utc=True
            ),
            "provider_updated_at": pd.to_datetime(
                ["2024-02-01T12:30:00Z", "2024-02-10T12:30:00Z"], utc=True
            ),
            "cash_amount": [None, None],
            "split_ratio": [2.0, 3.0],
            "currency": ["USD", "USD"],
            "old_ticker": ["", ""],
            "new_ticker": ["", ""],
            "adjustment_state": ["announced", "corrected"],
            "source": ["synthetic", "synthetic"],
            "source_table": ["TEST/ACTIONS", "TEST/ACTIONS"],
        }
    )


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


def test_semantic_identity_and_parquet_bytes_are_deterministic(tmp_path):
    first = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "first",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
        universe=_universe_records(),
        corporate_actions=_corporate_actions(),
    )
    second = export_signal_foundry_bundle(
        _contract_panel().sample(frac=1.0, random_state=19),
        tmp_path / "second",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
        universe=_universe_records().sample(frac=1.0, random_state=23),
        corporate_actions=_corporate_actions().sample(frac=1.0, random_state=29),
    )

    assert first.name == second.name
    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files

    changed = _corporate_actions()
    changed.loc[1, "split_ratio"] = 4.0
    changed_bundle = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "changed",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
        universe=_universe_records(),
        corporate_actions=changed,
    )
    assert changed_bundle.name != first.name


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


def test_bitemporal_record_families_hide_future_events_and_revisions(tmp_path):
    bundle = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "bundles",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
        universe=_universe_records(),
        corporate_actions=_corporate_actions(),
    )

    early = load_signal_foundry_bundle_view(bundle, as_of="2024-01-15T00:00:00Z")
    after_correction = load_signal_foundry_bundle_view(bundle, as_of="2024-02-12T00:00:00Z")
    after_effective = load_signal_foundry_bundle_view(bundle, as_of="2024-02-16T00:00:00Z")

    assert early.universe["is_member"].tolist() == [True]
    assert early.corporate_actions.empty
    assert after_correction.universe["is_member"].tolist() == [False]
    assert after_correction.corporate_actions.empty
    assert after_effective.corporate_actions["split_ratio"].tolist() == [3.0]


def test_future_auxiliary_mutations_cannot_change_an_earlier_view(tmp_path):
    original_actions = _corporate_actions()
    mutated_actions = original_actions.copy()
    mutated_actions.loc[1, "split_ratio"] = 4.0
    original_universe = _universe_records()
    mutated_universe = original_universe.copy()
    mutated_universe.loc[1, "reason"] = "later corrected exclusion"
    original = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "original",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
        universe=original_universe,
        corporate_actions=original_actions,
    )
    mutated = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "mutated",
        source_manifest={**_source_manifest(), "snapshot_hash": "d" * 64},
        producer_git_sha="c" * 40,
        universe=mutated_universe,
        corporate_actions=mutated_actions,
    )

    cutoff = "2024-01-15T00:00:00Z"
    first = load_signal_foundry_bundle_view(original, as_of=cutoff)
    second = load_signal_foundry_bundle_view(mutated, as_of=cutoff)
    pd.testing.assert_frame_equal(first.prices, second.prices)
    pd.testing.assert_frame_equal(first.universe, second.universe)
    pd.testing.assert_frame_equal(first.corporate_actions, second.corporate_actions)


@pytest.mark.parametrize(
    ("records", "argument", "match"),
    [
        (
            lambda: _universe_records().assign(available_at=pd.Timestamp("2024-01-01T00:00:00")),
            "universe",
            "timezone-aware",
        ),
        (
            lambda: pd.concat([_universe_records(), _universe_records().iloc[[0]]]),
            "universe",
            "duplicate",
        ),
        (
            lambda: _corporate_actions().assign(split_ratio=-1.0),
            "corporate_actions",
            "positive",
        ),
        (
            lambda: _corporate_actions().assign(action_type="unsupported"),
            "corporate_actions",
            "unsupported",
        ),
    ],
)
def test_invalid_bitemporal_records_fail_closed(tmp_path, records, argument, match):
    kwargs = {argument: records()}
    with pytest.raises(SignalFoundryContractError, match=match):
        export_signal_foundry_bundle(
            _contract_panel(),
            tmp_path / "bundles",
            source_manifest=_source_manifest(),
            producer_git_sha="c" * 40,
            **kwargs,
        )


def test_naive_price_and_as_of_timestamps_fail_closed(tmp_path):
    panel = _contract_panel()
    panel["available_at"] = panel["available_at"].dt.tz_localize(None)
    with pytest.raises(SignalFoundryContractError, match="timezone-ambiguous"):
        export_signal_foundry_bundle(
            panel,
            tmp_path / "naive",
            source_manifest=_source_manifest(),
            producer_git_sha="c" * 40,
        )

    bundle = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "valid",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
    )
    with pytest.raises(SignalFoundryContractError, match="explicit timezone"):
        load_signal_foundry_bundle_view(bundle, as_of="2024-01-01")


def test_all_missing_optional_provider_revision_time_is_preserved(tmp_path):
    panel = _contract_panel().assign(provider_updated_at=pd.NaT)
    bundle = export_signal_foundry_bundle(
        panel,
        tmp_path / "valid",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
    )

    loaded = load_signal_foundry_bundle(bundle)

    assert loaded["provider_updated_at"].isna().all()
    assert str(loaded["provider_updated_at"].dtype) == "datetime64[ns, UTC]"


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


def test_auxiliary_corruption_missing_files_and_unsupported_versions_fail_closed(tmp_path):
    bundle = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "corrupt",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
        universe=_universe_records(),
        corporate_actions=_corporate_actions(),
    )
    action_file = bundle / "corporate_actions/part-00000.parquet"
    action_file.write_bytes(action_file.read_bytes()[:64])
    with pytest.raises(SignalFoundryContractError, match="hash mismatch"):
        validate_signal_foundry_bundle(bundle)

    missing = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "missing",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
        universe=_universe_records(),
    )
    (missing / "universe/part-00000.parquet").unlink()
    with pytest.raises(SignalFoundryContractError, match="missing"):
        validate_signal_foundry_bundle(missing)

    unsupported = export_signal_foundry_bundle(
        _contract_panel(),
        tmp_path / "unsupported",
        source_manifest=_source_manifest(),
        producer_git_sha="c" * 40,
    )
    manifest_path = unsupported / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "2.0.0"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SignalFoundryContractError, match="unsupported schema version"):
        validate_signal_foundry_bundle(unsupported)


def test_stale_publication_state_fails_without_overwriting(tmp_path):
    root = tmp_path / "bundles"
    staging = root / ".publishing"
    staging.mkdir(parents=True)
    marker = staging / "operator-inspection-required"
    marker.write_text("preserve")

    with pytest.raises(SignalFoundryContractError, match="stale publication staging"):
        export_signal_foundry_bundle(
            _contract_panel(),
            root,
            source_manifest=_source_manifest(),
            producer_git_sha="c" * 40,
        )

    assert marker.read_text() == "preserve"


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
    view = load_signal_foundry_bundle_view(bundle, as_of="2024-01-04T06:00:00Z")

    assert manifest["schema_version"] == "1.1.0"
    assert manifest["license"]["observations_redistributable"] is True
    assert set(view.prices["ticker"]) == {"AAA", "SPY"}
    assert set(view.universe["ticker"]) == {"AAA", "SPY"}
    assert view.corporate_actions["action_type"].tolist() == ["cash_dividend"]


def test_legacy_v1_0_fixture_remains_compatible():
    fixture = (
        Path(__file__).parent
        / "fixtures/signal_foundry_v1"
        / "68cb928cde713032f18fe065bbfb835dc25fd014b8dedfb20d79bafb3b0fefea"
    )

    manifest = validate_signal_foundry_bundle(fixture)
    view = load_signal_foundry_bundle_view(fixture, as_of="2024-01-04T06:00:00Z")

    assert manifest["schema_version"] == "1.0.0"
    assert len(view.prices) == 10
    assert view.universe.empty
    assert view.corporate_actions.empty
