"""Offline tests for the bounded Nasdaq Data Link provider boundary."""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pytest

from quant_platform.config import NasdaqDataLinkConfig
from quant_platform.data.nasdaq_data_link import (
    HttpResponse,
    NasdaqDataLinkClient,
    RequestBudget,
    UrllibTransport,
)
from quant_platform.data.sources import DataSourceError

SECRET = "secret-that-must-never-be-persisted"
COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "closeadj",
    "lastupdated",
]


def _payload(rows: list[list[object]], cursor: str | None = None) -> bytes:
    return json.dumps(
        {
            "datatable": {
                "columns": [{"name": name, "type": "text"} for name in COLUMNS],
                "data": rows,
            },
            "meta": {"next_cursor_id": cursor},
        }
    ).encode()


def _row(ticker: str, date: str, close: float) -> list[object]:
    return [
        ticker,
        date,
        close - 1.0,
        close + 1.0,
        close - 2.0,
        close,
        1_000_000.0,
        close - 0.5,
        "2026-07-20",
    ]


def _time_series_payload(
    ticker: str,
    rows: list[list[object]],
    *,
    unadjusted: bool = False,
) -> bytes:
    dataset_code = f"{ticker}_UADJ" if unadjusted else ticker
    return json.dumps(
        {
            "dataset_data": {
                "database_code": "XDUS",
                "dataset_code": dataset_code,
                "column_names": [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Adjustment Factor",
                    "Adjustment Type",
                ],
                "data": rows,
                "frequency": "daily",
                "oldest_available_date": "2007-01-02",
                "newest_available_date": "2018-11-30",
                "refreshed_at": "2018-12-01T03:00:00.000Z",
                "premium": True,
            }
        }
    ).encode()


class QueueTransport:
    """Return predetermined responses and retain only call counts for assertions."""

    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def get(
        self,
        url: str,
        *,
        timeout: float,
        headers: Mapping[str, str],
    ) -> HttpResponse:
        assert SECRET not in url
        assert headers["X-Api-Token"] == SECRET
        assert timeout > 0
        self.calls += 1
        if not self.responses:
            raise AssertionError("unexpected network request")
        return self.responses.pop(0)


class NoNetworkTransport:
    def get(
        self,
        url: str,
        *,
        timeout: float,
        headers: Mapping[str, str],
    ) -> HttpResponse:
        raise AssertionError("cache replay attempted network access")


def _response(
    status: int,
    body: bytes,
    headers: Mapping[str, str] | None = None,
) -> HttpResponse:
    return HttpResponse(status=status, body=body, headers=headers or {})


def _config(tmp_path: Path, **overrides: object) -> NasdaqDataLinkConfig:
    values: dict[str, object] = {
        "cache_dir": str(Path("cache")),
        "requests_per_minute": 300,
        "max_requests": 10,
        "retry_backoff_seconds": 0.01,
    }
    values.update(overrides)
    return NasdaqDataLinkConfig.model_validate(values)


def test_paginated_fetch_is_hashed_redacted_and_cache_replayable(tmp_path):
    transport = QueueTransport(
        [
            _response(200, _payload([_row("SPY", "2024-01-02", 100.0)], "cursor-2")),
            _response(200, _payload([_row("SPY", "2024-01-03", 101.0)])),
        ]
    )
    config = _config(tmp_path)
    client = NasdaqDataLinkClient(
        config,
        transport=transport,
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
        now=lambda: datetime(2026, 7, 23, tzinfo=UTC),
    )

    first = client.fetch(["SPY"], "2024-01-01", "2024-01-31", base_dir=tmp_path)

    assert transport.calls == 2
    assert list(first.panel["ticker"]) == ["SPY", "SPY"]
    assert first.manifest["request_budget"]["used"] == 2
    assert first.manifest["contains_api_key"] is False
    assert SECRET not in json.dumps(first.manifest)
    assert SECRET not in first.snapshot_dir.joinpath("source_manifest.json").read_text()

    cached_config = _config(tmp_path, cache_mode="cache_only")
    cached = NasdaqDataLinkClient(
        cached_config,
        transport=NoNetworkTransport(),
        secret_resolver=lambda: None,
    ).fetch(["SPY"], "2024-01-01", "2024-01-31", base_dir=tmp_path)

    assert cached.manifest["snapshot_hash"] == first.manifest["snapshot_hash"]
    assert cached.panel.equals(first.panel)


def test_tables_api_preserves_standard_adjusted_close_column(tmp_path):
    columns = [*COLUMNS[:-2], "adj_close", "lastupdated"]
    row = _row("AAPL", "2018-03-27", 100.0)
    row[-2] = 75.0
    body = json.dumps(
        {
            "datatable": {
                "columns": [{"name": name, "type": "text"} for name in columns],
                "data": [row],
            },
            "meta": {"next_cursor_id": None},
        }
    ).encode()
    result = NasdaqDataLinkClient(
        _config(tmp_path, table="WIKI/PRICES"),
        transport=QueueTransport([_response(200, body)]),
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
        now=lambda: datetime(2026, 7, 24, tzinfo=UTC),
    ).fetch(["AAPL"], "2018-03-27", "2018-03-27", base_dir=tmp_path)

    assert result.panel.loc[0, "close"] == 100.0
    assert result.panel.loc[0, "adj_close"] == 75.0
    assert result.panel.loc[0, "adjustment_state"] == ("provider_adjusted_close_unadjusted_ohlc")


def test_time_series_fetch_is_bounded_normalised_and_cache_replayable(tmp_path):
    rows = [
        ["2018-11-29", 35.0, 36.0, 34.0, 35.5, 1000.0, 1.0, 0],
        ["2018-11-30", 35.5, 37.0, 35.0, 36.5, 1200.0, 1.0, 0],
    ]
    config = _config(
        tmp_path,
        api_kind="time_series",
        table="XDUS",
        currency="EUR",
        exchange_calendar="XDUS",
        market_close_utc_hour=16,
        availability_lag_hours=12,
        max_requests=1,
    )
    client = NasdaqDataLinkClient(
        config,
        transport=QueueTransport([_response(200, _time_series_payload("BSX", rows))]),
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
        now=lambda: datetime(2026, 7, 24, tzinfo=UTC),
    )

    first = client.fetch(["BSX"], "2018-11-29", "2018-11-30", base_dir=tmp_path)

    assert list(first.panel["ticker"]) == ["BSX", "BSX"]
    assert list(first.panel["close"]) == [35.5, 36.5]
    assert list(first.panel["adj_close"]) == [35.5, 36.5]
    assert set(first.panel["source_table"]) == {"XDUS/BSX"}
    assert set(first.panel["currency"]) == {"EUR"}
    assert set(first.panel["exchange_calendar"]) == {"XDUS"}
    assert set(first.panel["adjustment_state"]) == {"provider_adjusted_ohlcv"}
    assert first.panel["effective_at"].dt.hour.unique().tolist() == [16]
    assert first.panel["available_at"].dt.hour.unique().tolist() == [4]
    assert first.manifest["request_budget"]["used"] == 1
    assert first.manifest["provider_metadata"] == [
        {
            "database_code": "XDUS",
            "dataset_code": "BSX",
            "frequency": "daily",
            "oldest_available_date": "2007-01-02",
            "newest_available_date": "2018-11-30",
            "refreshed_at": "2018-12-01T03:00:00.000Z",
            "premium": True,
        }
    ]
    assert SECRET not in json.dumps(first.manifest)

    cached = NasdaqDataLinkClient(
        _config(
            tmp_path,
            api_kind="time_series",
            table="XDUS",
            currency="EUR",
            exchange_calendar="XDUS",
            market_close_utc_hour=16,
            availability_lag_hours=12,
            max_requests=1,
            cache_mode="cache_only",
        ),
        transport=NoNetworkTransport(),
        secret_resolver=lambda: None,
    ).fetch(["BSX"], "2018-11-29", "2018-11-30", base_dir=tmp_path)

    assert cached.panel.equals(first.panel)
    assert cached.manifest == first.manifest


def test_time_series_unadjusted_code_and_state_are_explicit(tmp_path):
    rows = [["2018-11-30", 35.5, 37.0, 35.0, 36.5, 1200.0, None, 0]]
    config = _config(
        tmp_path,
        api_kind="time_series",
        table="XDUS",
        adjustment="unadjusted",
        max_requests=1,
    )
    result = NasdaqDataLinkClient(
        config,
        transport=QueueTransport(
            [_response(200, _time_series_payload("BSX", rows, unadjusted=True))]
        ),
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
        now=lambda: datetime(2026, 7, 24, tzinfo=UTC),
    ).fetch(["BSX"], "2018-11-30", "2018-11-30", base_dir=tmp_path)

    assert set(result.panel["source_table"]) == {"XDUS/BSX_UADJ"}
    assert set(result.panel["adjustment_state"]) == {"provider_unadjusted_ohlcv"}


def test_time_series_response_identity_mismatch_fails_before_snapshot(tmp_path):
    rows = [["2018-11-30", 35.5, 37.0, 35.0, 36.5, 1200.0, 1.0, 0]]
    client = NasdaqDataLinkClient(
        _config(tmp_path, api_kind="time_series", table="XDUS", max_requests=1),
        transport=QueueTransport([_response(200, _time_series_payload("SAP", rows))]),
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(DataSourceError, match="identity mismatch"):
        client.fetch(["BMW"], "2018-11-30", "2018-11-30", base_dir=tmp_path)

    assert not list(tmp_path.rglob("source_manifest.json"))


def test_transient_rate_limit_honors_retry_after_and_remains_bounded(tmp_path):
    waits: list[float] = []
    transport = QueueTransport(
        [
            _response(
                429,
                b'{"quandl_error":{"message":"slow down"}}',
                {"retry-after": "2.5"},
            ),
            _response(200, _payload([_row("SPY", "2024-01-02", 100.0)])),
        ]
    )
    client = NasdaqDataLinkClient(
        _config(tmp_path, max_retries=1),
        transport=transport,
        secret_resolver=lambda: SECRET,
        sleep=waits.append,
        jitter=lambda: 0.0,
        now=lambda: datetime(2026, 7, 23, tzinfo=UTC),
    )

    result = client.fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path)

    assert len(result.panel) == 1
    assert transport.calls == 2
    assert any(wait >= 2.5 for wait in waits)


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_error_never_exposes_key(tmp_path, status):
    transport = QueueTransport([_response(status, f'{{"request":"api_key={SECRET}"}}'.encode())])
    client = NasdaqDataLinkClient(
        _config(tmp_path),
        transport=transport,
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(DataSourceError) as raised:
        client.fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path)

    assert SECRET not in str(raised.value)
    cache_text = "".join(
        path.read_text(errors="ignore") for path in tmp_path.rglob("*") if path.is_file()
    )
    assert SECRET not in cache_text


def test_success_payload_echoing_secret_is_rejected_before_persistence(tmp_path):
    payload = json.loads(_payload([_row("SPY", "2024-01-02", 100.0)]))
    payload["credential_echo"] = SECRET
    client = NasdaqDataLinkClient(
        _config(tmp_path),
        transport=QueueTransport([_response(200, json.dumps(payload).encode())]),
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(DataSourceError, match="rejected before persistence") as raised:
        client.fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path)

    assert SECRET not in str(raised.value)
    persisted = b"".join(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    assert SECRET.encode() not in persisted


def test_provider_error_message_redacts_secret(tmp_path):
    body = json.dumps(
        {"quandl_error": {"message": f"request failed for api_key={SECRET}"}}
    ).encode()
    client = NasdaqDataLinkClient(
        _config(tmp_path, max_retries=0),
        transport=QueueTransport([_response(404, body)]),
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(DataSourceError, match=r"api_key=\[REDACTED\]") as raised:
        client.fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path)

    assert SECRET not in str(raised.value)


def test_non_retryable_not_found_is_terminal(tmp_path):
    transport = QueueTransport([_response(404, b'{"quandl_error":{"message":"table not found"}}')])
    client = NasdaqDataLinkClient(
        _config(tmp_path, max_retries=4),
        transport=transport,
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(DataSourceError, match="HTTP 404"):
        client.fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path)
    assert transport.calls == 1


def test_server_error_retries_then_succeeds(tmp_path):
    transport = QueueTransport(
        [
            _response(503, b'{"quandl_error":{"message":"temporarily unavailable"}}'),
            _response(200, _payload([_row("SPY", "2024-01-02", 100.0)])),
        ]
    )
    client = NasdaqDataLinkClient(
        _config(tmp_path, max_retries=1),
        transport=transport,
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
        jitter=lambda: 0.0,
        now=lambda: datetime(2026, 7, 23, tzinfo=UTC),
    )
    assert len(client.fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path).panel) == 1
    assert transport.calls == 2


def test_interrupted_pagination_resumes_from_verified_staging_page(tmp_path):
    first_transport = QueueTransport(
        [
            _response(200, _payload([_row("SPY", "2024-01-02", 100.0)], "cursor-2")),
            _response(503, b'{"quandl_error":{"message":"offline"}}'),
        ]
    )
    with pytest.raises(DataSourceError, match="HTTP 503"):
        NasdaqDataLinkClient(
            _config(tmp_path, max_retries=0),
            transport=first_transport,
            secret_resolver=lambda: SECRET,
            sleep=lambda _seconds: None,
            now=lambda: datetime(2026, 7, 23, tzinfo=UTC),
        ).fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path)

    resumed_transport = QueueTransport(
        [_response(200, _payload([_row("SPY", "2024-01-03", 101.0)]))]
    )
    result = NasdaqDataLinkClient(
        _config(tmp_path),
        transport=resumed_transport,
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
        now=lambda: datetime(2026, 7, 23, tzinfo=UTC),
    ).fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path)

    assert resumed_transport.calls == 1
    assert list(result.panel["date"].dt.strftime("%Y-%m-%d")) == [
        "2024-01-02",
        "2024-01-03",
    ]


def test_missing_secret_fails_before_network(tmp_path):
    client = NasdaqDataLinkClient(
        _config(tmp_path),
        transport=NoNetworkTransport(),
        secret_resolver=lambda: None,
    )
    with pytest.raises(DataSourceError, match="Keychain"):
        client.fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path)


def test_default_transport_redacts_timeout_url(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise urllib.error.URLError(f"timeout for api_key={SECRET}")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    with pytest.raises(DataSourceError) as raised:
        UrllibTransport().get(
            "https://data.nasdaq.com/api/v3/datatables/X/Y.json",
            timeout=1.0,
            headers={"X-Api-Token": SECRET},
        )
    assert SECRET not in str(raised.value)


def test_default_transport_bounds_response_bytes():
    transport = UrllibTransport(max_response_bytes=4)

    with pytest.raises(DataSourceError, match="transport safety bound"):
        transport._read_bounded(BytesIO(b"12345"))

    assert transport._read_bounded(BytesIO(b"1234")) == b"1234"
    with pytest.raises(ValueError, match="positive"):
        UrllibTransport(max_response_bytes=0)


def test_provider_rows_outside_requested_dates_fail_closed(tmp_path):
    client = NasdaqDataLinkClient(
        _config(tmp_path),
        transport=QueueTransport([_response(200, _payload([_row("SPY", "2024-02-01", 100.0)]))]),
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
        now=lambda: datetime(2026, 7, 23, tzinfo=UTC),
    )

    with pytest.raises(DataSourceError, match="outside the requested dates"):
        client.fetch(["SPY"], "2024-01-01", "2024-01-31", base_dir=tmp_path)


@pytest.mark.parametrize(
    "body,match",
    [
        (b"not json", "malformed JSON"),
        (b'{"datatable":{"columns":[],"data":[[1]]},"meta":{}}', "malformed row"),
        (
            b'{"datatable":{"columns":[{"name":"ticker"},{"name":"date"}],'
            b'"data":[["SPY","2024-01-02"]]},"meta":{}}',
            "missing required columns",
        ),
        (
            _payload([_row("SPY", "2024-01-02", float("nan"))]),
            "missing/non-finite",
        ),
        (
            _payload(
                [
                    _row("SPY", "2024-01-02", 100.0),
                    _row("SPY", "2024-01-02", 100.0),
                ]
            ),
            "duplicate",
        ),
    ],
)
def test_malformed_provider_data_fails_closed(tmp_path, body, match):
    client = NasdaqDataLinkClient(
        _config(tmp_path),
        transport=QueueTransport([_response(200, body)]),
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
        now=lambda: datetime(2026, 7, 23, tzinfo=UTC),
    )
    with pytest.raises(DataSourceError, match=match):
        client.fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path)


def test_cache_corruption_is_detected_without_network(tmp_path):
    config = _config(tmp_path)
    result = NasdaqDataLinkClient(
        config,
        transport=QueueTransport([_response(200, _payload([_row("SPY", "2024-01-02", 100.0)]))]),
        secret_resolver=lambda: SECRET,
        sleep=lambda _seconds: None,
        now=lambda: datetime(2026, 7, 23, tzinfo=UTC),
    ).fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path)
    result.snapshot_dir.joinpath("page-00000.json").write_text("{}")

    with pytest.raises(DataSourceError, match="hash mismatch"):
        NasdaqDataLinkClient(
            _config(tmp_path, cache_mode="cache_only"),
            transport=NoNetworkTransport(),
            secret_resolver=lambda: None,
        ).fetch(["SPY"], "2024-01-01", None, base_dir=tmp_path)


def test_request_budget_never_exceeds_cap():
    budget = RequestBudget(
        requests_per_minute=60,
        max_requests=1,
        clock=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    budget.before_request()
    with pytest.raises(DataSourceError, match="exhausted"):
        budget.before_request()
    assert budget.used == 1
