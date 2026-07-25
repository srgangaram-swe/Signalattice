"""Bounded, auditable Nasdaq Data Link table and time-series ingestion.

The adapter deliberately keeps network mechanics outside the research pipeline:

* credentials are resolved at runtime and never enter configuration or manifests;
* a hard request budget and minimum request interval bound provider usage;
* retries are limited to transient responses and honor ``Retry-After``;
* response pages are staged, hashed, and atomically promoted into immutable
  content-addressed snapshots; and
* transport, clock, sleep, randomness, and secret resolution are injectable so
  the complete failure surface can be tested without a network or credential.

The canonical mapping targets daily OHLCV Tables API products with the
SHARADAR/SEP vocabulary and daily Time-Series API products with the standard
Date/Open/High/Low/Close/Volume vocabulary. Other products fail closed when
required columns are absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from quant_platform.config import NasdaqDataLinkConfig
from quant_platform.data.schema import DATE_COL, OHLCV_COLUMNS, TICKER_COL
from quant_platform.data.sources import DataSourceError
from quant_platform.utils import ensure_dir, resolve_path

API_KEY_ENV = "NASDAQ_DATA_LINK_API_KEY"
TABLES_API_ROOT = "https://data.nasdaq.com/api/v3/datatables"
TIME_SERIES_API_ROOT = "https://data.nasdaq.com/api/v3/datasets"
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
TERMINAL_AUTH_STATUS = frozenset({401, 403})


@dataclass(frozen=True)
class HttpResponse:
    """Transport response without a credential-bearing request URL."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class HttpTransport(Protocol):
    """Minimal injectable HTTP transport."""

    def get(self, url: str, *, timeout: float) -> HttpResponse:
        """Return one HTTP response without logging ``url``."""


class UrllibTransport:
    """Standard-library HTTPS transport with bounded, sanitized responses."""

    def __init__(self, *, max_response_bytes: int = MAX_RESPONSE_BYTES) -> None:
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.max_response_bytes = max_response_bytes

    def _read_bounded(self, stream: Any) -> bytes:
        body = stream.read(self.max_response_bytes + 1)
        if not isinstance(body, bytes):
            raise DataSourceError("Nasdaq Data Link transport returned a non-bytes response")
        if len(body) > self.max_response_bytes:
            raise DataSourceError(
                "Nasdaq Data Link response exceeded the configured transport safety bound "
                f"({self.max_response_bytes} bytes)"
            )
        return body

    def get(self, url: str, *, timeout: float) -> HttpResponse:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Signalattice/0.2 (+https://github.com/srgangaram-swe/Signalattice)",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return HttpResponse(
                    status=int(response.status),
                    headers={key.lower(): value for key, value in response.headers.items()},
                    body=self._read_bounded(response),
                )
        except urllib.error.HTTPError as exc:
            # Do not stringify HTTPError: its URL contains the API key.
            return HttpResponse(
                status=int(exc.code),
                headers={key.lower(): value for key, value in exc.headers.items()},
                body=self._read_bounded(exc),
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DataSourceError(
                f"Nasdaq Data Link transport failed ({type(exc).__name__}); "
                "the request URL and credential were redacted"
            ) from None


@dataclass
class RequestBudget:
    """Hard request cap plus deterministic minimum-interval rate limit."""

    requests_per_minute: int
    max_requests: int
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    used: int = 0
    _last_request_at: float | None = None

    def before_request(self) -> None:
        """Wait for the rate window and consume exactly one request token."""
        if self.used >= self.max_requests:
            raise DataSourceError(
                f"Nasdaq Data Link request budget exhausted ({self.used}/{self.max_requests})"
            )
        now = self.clock()
        if self._last_request_at is not None:
            minimum_interval = 60.0 / float(self.requests_per_minute)
            remaining = minimum_interval - (now - self._last_request_at)
            if remaining > 0:
                self.sleep(remaining)
                now = self.clock()
        self.used += 1
        self._last_request_at = now


@dataclass(frozen=True)
class FetchResult:
    """Canonical panel plus redacted source provenance."""

    panel: pd.DataFrame
    manifest: dict[str, Any]
    snapshot_dir: Path


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    payload = _canonical_json(value) + b"\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _safe_error_message(response: HttpResponse, *, api_key: str) -> str:
    """Extract a bounded provider message that cannot contain a request URL."""
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "provider returned a non-JSON error body"
    message = payload.get("quandl_error", {}).get("message") if isinstance(payload, dict) else None
    if not isinstance(message, str):
        return "provider returned an unspecified error"
    sanitized = message.replace(api_key, "[REDACTED]").replace("\r", " ").replace("\n", " ")
    return sanitized[:300]


class NasdaqDataLinkClient:
    """Fetch and cache one configured Nasdaq Data Link product request."""

    def __init__(
        self,
        config: NasdaqDataLinkConfig,
        *,
        transport: HttpTransport | None = None,
        secret_resolver: Callable[[], str | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibTransport()
        self.secret_resolver = secret_resolver or (lambda: os.getenv(API_KEY_ENV))
        self.sleep = sleep
        self.jitter = jitter
        self.now = now or (lambda: datetime.now(UTC))
        self.budget = RequestBudget(
            requests_per_minute=config.requests_per_minute,
            max_requests=config.max_requests,
            clock=clock,
            sleep=sleep,
        )

    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str | None,
        *,
        base_dir: str | Path | None = None,
    ) -> FetchResult:
        """Return a verified cached or freshly downloaded canonical panel."""
        request = {
            "provider": "nasdaq_data_link",
            "api_kind": self.config.api_kind,
            "table": self.config.table,
            "adjustment": self.config.adjustment,
            "currency": self.config.currency,
            "exchange_calendar": self.config.exchange_calendar,
            "market_close_utc_hour": self.config.market_close_utc_hour,
            "tickers": sorted(set(tickers)),
            "start": start,
            "end": end,
            "page_size": self.config.page_size,
        }
        request_hash = _sha256_bytes(_canonical_json(request))
        cache_root = ensure_dir(resolve_path(self.config.cache_dir, base_dir))
        request_dir = ensure_dir(cache_root / "requests" / request_hash)

        if self.config.cache_mode in {"prefer_cache", "cache_only"}:
            cached = self._load_latest(request_dir, request)
            if cached is not None:
                return cached
            if self.config.cache_mode == "cache_only":
                raise DataSourceError(
                    "Nasdaq Data Link cache-only mode found no complete verified snapshot "
                    f"for request {request_hash[:12]}"
                )

        api_key = self.secret_resolver()
        if not api_key or not api_key.strip():
            raise DataSourceError(
                f"{API_KEY_ENV} is not available; store it in macOS Keychain and expose it "
                "only to the bounded ingestion process"
            )

        staging = request_dir / "staging"
        ensure_dir(staging)
        lock = request_dir / ".fetch.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise DataSourceError(
                f"another Nasdaq Data Link fetch owns request {request_hash[:12]}"
            ) from exc
        else:
            os.close(descriptor)

        try:
            return self._fetch_network(
                request=request,
                request_hash=request_hash,
                request_dir=request_dir,
                staging=staging,
                api_key=api_key.strip(),
            )
        finally:
            lock.unlink(missing_ok=True)

    def _fetch_network(
        self,
        *,
        request: dict[str, Any],
        request_hash: str,
        request_dir: Path,
        staging: Path,
        api_key: str,
    ) -> FetchResult:
        page_number = 0
        page_records: list[dict[str, Any]] = []
        page_hashes: list[str] = []
        columns: list[str] | None = None
        provider_metadata: list[dict[str, Any]] = []

        if self.config.api_kind == "tables":
            cursor: str | None = None
            while True:
                page_path = staging / f"page-{page_number:05d}.json"
                if page_path.exists():
                    payload = self._read_page(page_path)
                else:
                    payload = self._request_table_page(
                        request=request,
                        cursor=cursor,
                        api_key=api_key,
                    )
                    _atomic_json(page_path, payload)

                page_columns, rows, next_cursor = self._parse_table_page(payload)
                if columns is None:
                    columns = page_columns
                elif columns != page_columns:
                    raise DataSourceError("Nasdaq Data Link schema changed between response pages")
                page_records.extend(dict(zip(page_columns, row, strict=True)) for row in rows)
                page_hashes.append(_sha256_file(page_path))
                page_number += 1
                if next_cursor is None:
                    break
                cursor = next_cursor
        else:
            for ticker in request["tickers"]:
                page_path = staging / f"page-{page_number:05d}.json"
                if page_path.exists():
                    payload = self._read_page(page_path)
                else:
                    payload = self._request_time_series(
                        request=request,
                        ticker=ticker,
                        api_key=api_key,
                    )
                    _atomic_json(page_path, payload)
                page_columns, rows, metadata = self._parse_time_series(payload)
                self._validate_time_series_identity(metadata, ticker=ticker)
                if columns is None:
                    columns = page_columns
                elif columns != page_columns:
                    raise DataSourceError(
                        "Nasdaq Data Link schema changed between time-series responses"
                    )
                page_records.extend(dict(zip(page_columns, row, strict=True)) for row in rows)
                provider_metadata.append(metadata)
                page_hashes.append(_sha256_file(page_path))
                page_number += 1

        retrieved_at = self.now().astimezone(UTC).isoformat().replace("+00:00", "Z")
        snapshot_hash = _sha256_bytes(_canonical_json(page_hashes))
        snapshots_dir = ensure_dir(request_dir / "snapshots")
        snapshot_dir = snapshots_dir / snapshot_hash
        if snapshot_dir.exists():
            shutil.rmtree(staging)
            manifest_path = snapshot_dir / "source_manifest.json"
            if not manifest_path.exists():
                raise DataSourceError(
                    "Nasdaq Data Link immutable snapshot exists without a source manifest"
                )
            manifest = self._read_page(manifest_path)
            _atomic_json(
                request_dir / "latest.json",
                {
                    "request_hash": request_hash,
                    "snapshot_hash": snapshot_hash,
                    "manifest_sha256": _sha256_file(manifest_path),
                },
            )
            panel = self._normalise(
                page_records,
                retrieved_at=str(manifest.get("retrieved_at")),
            )
            self._validate_panel_scope(panel, request=request)
            return FetchResult(panel=panel, manifest=manifest, snapshot_dir=snapshot_dir)
        staging.replace(snapshot_dir)

        panel = self._normalise(page_records, retrieved_at=retrieved_at)
        self._validate_panel_scope(panel, request=request)
        manifest = self._manifest(
            request=request,
            request_hash=request_hash,
            snapshot_hash=snapshot_hash,
            retrieved_at=retrieved_at,
            panel=panel,
            page_hashes=page_hashes,
            provider_metadata=provider_metadata,
        )
        _atomic_json(snapshot_dir / "source_manifest.json", manifest)
        _atomic_json(
            request_dir / "latest.json",
            {
                "request_hash": request_hash,
                "snapshot_hash": snapshot_hash,
                "manifest_sha256": _sha256_file(snapshot_dir / "source_manifest.json"),
            },
        )
        return FetchResult(panel=panel, manifest=manifest, snapshot_dir=snapshot_dir)

    def _request_table_page(
        self,
        *,
        request: dict[str, Any],
        cursor: str | None,
        api_key: str,
    ) -> dict[str, Any]:
        params: dict[str, str | int] = {
            "api_key": api_key,
            "ticker": ",".join(request["tickers"]),
            "date.gte": request["start"],
            "qopts.per_page": self.config.page_size,
        }
        if request["end"] is not None:
            params["date.lte"] = request["end"]
        if cursor is not None:
            params["qopts.cursor_id"] = cursor
        encoded_table = "/".join(
            urllib.parse.quote(part, safe="") for part in self.config.table.split("/")
        )
        url = f"{TABLES_API_ROOT}/{encoded_table}.json?{urllib.parse.urlencode(params)}"
        return self._request_json(url=url, api_key=api_key)

    def _request_time_series(
        self,
        *,
        request: dict[str, Any],
        ticker: str,
        api_key: str,
    ) -> dict[str, Any]:
        suffix = "" if self.config.adjustment == "adjusted" else "_UADJ"
        dataset_code = f"{ticker}{suffix}"
        encoded_database = urllib.parse.quote(self.config.table, safe="")
        encoded_dataset = urllib.parse.quote(dataset_code, safe="")
        params: dict[str, str] = {
            "api_key": api_key,
            "start_date": request["start"],
            "order": "asc",
        }
        if request["end"] is not None:
            params["end_date"] = request["end"]
        url = (
            f"{TIME_SERIES_API_ROOT}/{encoded_database}/{encoded_dataset}.json?"
            f"{urllib.parse.urlencode(params)}"
        )
        return self._request_json(url=url, api_key=api_key)

    def _request_json(self, *, url: str, api_key: str) -> dict[str, Any]:
        """Perform one redacted, budgeted JSON request with bounded retries."""
        for retry in range(self.config.max_retries + 1):
            self.budget.before_request()
            response = self.transport.get(url, timeout=self.config.timeout_seconds)
            if response.status == 200:
                try:
                    payload = json.loads(response.body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise DataSourceError(
                        "Nasdaq Data Link returned malformed JSON; credential and URL redacted"
                    ) from exc
                if not isinstance(payload, dict):
                    raise DataSourceError("Nasdaq Data Link returned a non-object JSON payload")
                if api_key.encode("utf-8") in _canonical_json(payload):
                    raise DataSourceError(
                        "Nasdaq Data Link response unexpectedly echoed the credential; "
                        "the response was rejected before persistence"
                    )
                return payload
            if response.status in TERMINAL_AUTH_STATUS:
                raise DataSourceError(
                    f"Nasdaq Data Link authentication/entitlement failed (HTTP {response.status}); "
                    "credential and URL redacted"
                )
            if response.status not in RETRYABLE_STATUS or retry >= self.config.max_retries:
                raise DataSourceError(
                    f"Nasdaq Data Link request failed (HTTP {response.status}): "
                    f"{_safe_error_message(response, api_key=api_key)}"
                )
            retry_after = response.headers.get("retry-after")
            try:
                provider_wait = float(retry_after) if retry_after is not None else 0.0
            except ValueError:
                provider_wait = 0.0
            exponential = self.config.retry_backoff_seconds * (2**retry)
            self.sleep(max(provider_wait, exponential + self.jitter() * exponential))
        raise AssertionError("retry loop must return or raise")

    @staticmethod
    def _read_page(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DataSourceError(f"cached Nasdaq Data Link page is corrupt: {path.name}") from exc
        if not isinstance(value, dict):
            raise DataSourceError(f"cached Nasdaq Data Link page is not an object: {path.name}")
        return value

    @staticmethod
    def _parse_table_page(
        payload: dict[str, Any],
    ) -> tuple[list[str], list[list[Any]], str | None]:
        datatable = payload.get("datatable")
        meta = payload.get("meta", {})
        if not isinstance(datatable, dict) or not isinstance(meta, dict):
            raise DataSourceError("Nasdaq Data Link payload lacks datatable/meta objects")
        raw_columns = datatable.get("columns")
        rows = datatable.get("data")
        if not isinstance(raw_columns, list) or not isinstance(rows, list):
            raise DataSourceError("Nasdaq Data Link payload lacks column or row arrays")
        columns: list[str] = []
        for column in raw_columns:
            if not isinstance(column, dict) or not isinstance(column.get("name"), str):
                raise DataSourceError("Nasdaq Data Link payload contains malformed column metadata")
            columns.append(column["name"].lower())
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                raise DataSourceError("Nasdaq Data Link payload contains a malformed row")
        cursor = meta.get("next_cursor_id")
        if cursor in (None, ""):
            return columns, rows, None
        if not isinstance(cursor, str) or len(cursor) > 512:
            raise DataSourceError("Nasdaq Data Link returned an invalid pagination cursor")
        return columns, rows, cursor

    @staticmethod
    def _parse_time_series(
        payload: dict[str, Any],
    ) -> tuple[list[str], list[list[Any]], dict[str, Any]]:
        dataset = payload.get("dataset")
        if not isinstance(dataset, dict):
            raise DataSourceError("Nasdaq Data Link payload lacks a dataset object")
        raw_columns = dataset.get("column_names")
        rows = dataset.get("data")
        database_code = dataset.get("database_code")
        dataset_code = dataset.get("dataset_code")
        if (
            not isinstance(raw_columns, list)
            or not isinstance(rows, list)
            or not isinstance(database_code, str)
            or not isinstance(dataset_code, str)
        ):
            raise DataSourceError("Nasdaq Data Link time-series metadata is malformed")
        columns: list[str] = []
        for column in raw_columns:
            if not isinstance(column, str):
                raise DataSourceError(
                    "Nasdaq Data Link time-series contains malformed column metadata"
                )
            normalized = column.strip().lower().replace(" ", "_")
            if not normalized or not normalized.replace("_", "").isalnum():
                raise DataSourceError(
                    "Nasdaq Data Link time-series contains an invalid column name"
                )
            columns.append(normalized)
        if len(columns) != len(set(columns)):
            raise DataSourceError("Nasdaq Data Link time-series contains duplicate columns")
        ticker = dataset_code.removesuffix("_UADJ").upper()
        source_code = f"{database_code.upper()}/{dataset_code.upper()}"
        refreshed_at = dataset.get("refreshed_at")
        enriched_columns = [*columns, TICKER_COL, "_source_code", "_provider_updated_at"]
        enriched_rows: list[list[Any]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                raise DataSourceError("Nasdaq Data Link time-series contains a malformed row")
            enriched_rows.append([*row, ticker, source_code, refreshed_at])
        metadata = {
            "database_code": database_code.upper(),
            "dataset_code": dataset_code.upper(),
            "frequency": dataset.get("frequency"),
            "oldest_available_date": dataset.get("oldest_available_date"),
            "newest_available_date": dataset.get("newest_available_date"),
            "refreshed_at": refreshed_at,
            "premium": dataset.get("premium"),
        }
        return enriched_columns, enriched_rows, metadata

    def _validate_time_series_identity(
        self,
        metadata: Mapping[str, Any],
        *,
        ticker: str,
    ) -> None:
        """Reject a valid-looking response for a different provider series."""
        suffix = "" if self.config.adjustment == "adjusted" else "_UADJ"
        expected_dataset = f"{ticker}{suffix}".upper()
        if (
            metadata.get("database_code") != self.config.table
            or metadata.get("dataset_code") != expected_dataset
        ):
            raise DataSourceError("Nasdaq Data Link time-series response identity mismatch")

    def _normalise(self, records: list[dict[str, Any]], *, retrieved_at: str) -> pd.DataFrame:
        if not records:
            raise DataSourceError("Nasdaq Data Link returned no rows")
        raw = pd.DataFrame.from_records(records)
        required = {TICKER_COL, DATE_COL, "open", "high", "low", "close", "volume"}
        missing = sorted(required.difference(raw.columns))
        if missing:
            raise DataSourceError(f"Nasdaq Data Link table is missing required columns: {missing}")
        adjusted = "closeadj" if "closeadj" in raw.columns else "close"
        frame = raw.copy()
        frame["adj_close"] = raw[adjusted]
        if frame.columns.duplicated().any():
            frame = frame.loc[:, ~frame.columns.duplicated(keep="first")]
        frame[TICKER_COL] = frame[TICKER_COL].astype(str).str.upper()
        frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="raise").dt.tz_localize(None)
        for column in OHLCV_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        numeric = frame.loc[:, OHLCV_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise DataSourceError("Nasdaq Data Link OHLCV data contains missing/non-finite values")
        if (frame.loc[:, ["open", "high", "low", "close", "adj_close"]] <= 0).any().any():
            raise DataSourceError("Nasdaq Data Link data contains non-positive prices")
        if (frame["volume"] < 0).any():
            raise DataSourceError("Nasdaq Data Link data contains negative volume")
        if (
            (frame["high"] < frame["low"])
            | (frame["open"] > frame["high"])
            | (frame["open"] < frame["low"])
            | (frame["close"] > frame["high"])
            | (frame["close"] < frame["low"])
        ).any():
            raise DataSourceError("Nasdaq Data Link data violates OHLC bounds")
        if frame.duplicated([DATE_COL, TICKER_COL]).any():
            raise DataSourceError("Nasdaq Data Link data contains duplicate (date, ticker) rows")

        effective = pd.to_datetime(frame[DATE_COL], utc=True) + pd.Timedelta(
            hours=self.config.market_close_utc_hour
        )
        frame["effective_at"] = effective
        frame["available_at"] = effective + pd.Timedelta(hours=self.config.availability_lag_hours)
        frame["observed_at"] = pd.Timestamp(retrieved_at)
        if "_provider_updated_at" in raw.columns:
            frame["provider_updated_at"] = pd.to_datetime(
                raw["_provider_updated_at"], errors="coerce", utc=True
            )
        elif "lastupdated" in raw.columns:
            frame["provider_updated_at"] = pd.to_datetime(
                raw["lastupdated"], errors="coerce", utc=True
            )
        else:
            frame["provider_updated_at"] = pd.NaT
        frame["source"] = "nasdaq_data_link"
        frame["source_table"] = (
            raw["_source_code"].astype(str) if "_source_code" in raw.columns else self.config.table
        )
        frame["instrument_id"] = frame[TICKER_COL]
        frame["currency"] = self.config.currency
        frame["exchange_calendar"] = self.config.exchange_calendar
        if self.config.api_kind == "time_series":
            frame["adjustment_state"] = f"provider_{self.config.adjustment}_ohlcv"
        else:
            frame["adjustment_state"] = (
                "provider_adjusted_close_unadjusted_ohlc"
                if adjusted == "closeadj"
                else "provider_unadjusted"
            )
        columns = [
            DATE_COL,
            TICKER_COL,
            *OHLCV_COLUMNS,
            "effective_at",
            "available_at",
            "observed_at",
            "provider_updated_at",
            "source",
            "source_table",
            "instrument_id",
            "currency",
            "exchange_calendar",
            "adjustment_state",
        ]
        return frame[columns].sort_values([TICKER_COL, DATE_COL]).reset_index(drop=True)

    @staticmethod
    def _validate_panel_scope(panel: pd.DataFrame, *, request: Mapping[str, Any]) -> None:
        """Reject provider observations outside the requested identity/date scope."""
        requested_tickers = set(request["tickers"])
        returned_tickers = set(panel[TICKER_COL].astype(str))
        unexpected = sorted(returned_tickers.difference(requested_tickers))
        if unexpected:
            raise DataSourceError(
                f"Nasdaq Data Link returned unrequested ticker identities: {unexpected}"
            )
        start = pd.Timestamp(str(request["start"]))
        end = pd.Timestamp(str(request["end"])) if request["end"] is not None else None
        if (panel[DATE_COL] < start).any() or (end is not None and (panel[DATE_COL] > end).any()):
            raise DataSourceError(
                "Nasdaq Data Link returned observations outside the requested dates"
            )

    def _manifest(
        self,
        *,
        request: dict[str, Any],
        request_hash: str,
        snapshot_hash: str,
        retrieved_at: str,
        panel: pd.DataFrame,
        page_hashes: list[str],
        provider_metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "manifest_version": 2,
            "provider": "nasdaq_data_link",
            "request": request,
            "request_hash": request_hash,
            "snapshot_hash": snapshot_hash,
            "retrieved_at": retrieved_at,
            "page_count": len(page_hashes),
            "page_sha256": page_hashes,
            "provider_metadata": provider_metadata,
            "request_budget": {
                "used": self.budget.used,
                "maximum": self.budget.max_requests,
                "requests_per_minute": self.budget.requests_per_minute,
            },
            "row_count": int(len(panel)),
            "tickers": sorted(panel[TICKER_COL].unique().tolist()),
            "date_min": str(panel[DATE_COL].min().date()),
            "date_max": str(panel[DATE_COL].max().date()),
            "adjustment_state": sorted(panel["adjustment_state"].unique().tolist()),
            "availability_policy": {
                "effective_at": (
                    "market date at configured "
                    f"{self.config.market_close_utc_hour:02d}:00 UTC close"
                ),
                "available_at": (
                    f"effective_at plus {self.config.availability_lag_hours} hours; "
                    "policy assumption, not a provider publication timestamp"
                ),
            },
            "point_in_time_limits": {
                "historical_revisions_complete": False,
                "universe_membership_point_in_time": False,
                "corporate_actions_complete": False,
            },
            "contains_api_key": False,
            "licensed_observations_must_remain_local": True,
        }

    def _load_latest(self, request_dir: Path, request: dict[str, Any]) -> FetchResult | None:
        pointer_path = request_dir / "latest.json"
        if not pointer_path.exists():
            return None
        pointer = self._read_page(pointer_path)
        snapshot_hash = pointer.get("snapshot_hash")
        if not isinstance(snapshot_hash, str) or len(snapshot_hash) != 64:
            raise DataSourceError("Nasdaq Data Link cache pointer has an invalid snapshot hash")
        snapshot_dir = request_dir / "snapshots" / snapshot_hash
        manifest_path = snapshot_dir / "source_manifest.json"
        if not manifest_path.exists():
            raise DataSourceError("Nasdaq Data Link cache pointer references a missing manifest")
        if _sha256_file(manifest_path) != pointer.get("manifest_sha256"):
            raise DataSourceError("Nasdaq Data Link cached manifest hash mismatch")
        manifest = self._read_page(manifest_path)
        if manifest.get("request") != request or manifest.get("snapshot_hash") != snapshot_hash:
            raise DataSourceError("Nasdaq Data Link cached manifest/request mismatch")
        page_hashes = manifest.get("page_sha256")
        if not isinstance(page_hashes, list) or not page_hashes:
            raise DataSourceError("Nasdaq Data Link cached manifest has no pages")
        if self.config.api_kind == "time_series" and len(page_hashes) != len(request["tickers"]):
            raise DataSourceError(
                "Nasdaq Data Link cached time-series page count does not match the request"
            )
        records: list[dict[str, Any]] = []
        columns: list[str] | None = None
        for page_number, expected_hash in enumerate(page_hashes):
            page_path = snapshot_dir / f"page-{page_number:05d}.json"
            if not isinstance(expected_hash, str) or _sha256_file(page_path) != expected_hash:
                raise DataSourceError(
                    f"Nasdaq Data Link cached page hash mismatch: {page_path.name}"
                )
            payload = self._read_page(page_path)
            if self.config.api_kind == "tables":
                page_columns, rows, _ = self._parse_table_page(payload)
            else:
                page_columns, rows, metadata = self._parse_time_series(payload)
                self._validate_time_series_identity(
                    metadata,
                    ticker=str(request["tickers"][page_number]),
                )
            if columns is None:
                columns = page_columns
            elif columns != page_columns:
                raise DataSourceError("Nasdaq Data Link cached page schema mismatch")
            records.extend(dict(zip(page_columns, row, strict=True)) for row in rows)
        panel = self._normalise(records, retrieved_at=str(manifest["retrieved_at"]))
        self._validate_panel_scope(panel, request=request)
        if int(manifest.get("row_count", -1)) != len(panel):
            raise DataSourceError("Nasdaq Data Link cached row-count mismatch")
        return FetchResult(panel=panel, manifest=manifest, snapshot_dir=snapshot_dir)


def fetch_nasdaq_data_link(
    config: NasdaqDataLinkConfig,
    tickers: list[str],
    start: str,
    end: str | None,
    *,
    base_dir: str | Path | None = None,
    client: NasdaqDataLinkClient | None = None,
) -> FetchResult:
    """Fetch a Nasdaq Data Link product through the bounded client."""
    active_client = client or NasdaqDataLinkClient(config)
    return active_client.fetch(tickers, start, end, base_dir=base_dir)
