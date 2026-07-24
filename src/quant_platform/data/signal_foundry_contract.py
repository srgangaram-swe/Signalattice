"""Versioned, fail-closed dataset exchange contract for Signal Foundry.

Signalattice is the producer and AlphaForge is the consumer. The repository
boundary is files and a canonical manifest—not a Python import dependency.
Licensed observations remain local; committed tests use redistribution-safe
synthetic fixtures with the identical schema and validator.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_platform.data.schema import DATE_COL, OHLCV_COLUMNS, TICKER_COL
from quant_platform.data.validation import validate_price_panel
from quant_platform.utils import ensure_dir, git_commit_hash

CONTRACT_NAME = "signal-foundry-market-data"
SCHEMA_VERSION = "1.0.0"
MANIFEST_NAME = "manifest.json"
TEMPORAL_COLUMNS: tuple[str, ...] = (
    "effective_at",
    "available_at",
    "observed_at",
    "provider_updated_at",
)
IDENTITY_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "currency",
    "exchange_calendar",
    "adjustment_state",
    "source",
    "source_table",
)
CONTRACT_COLUMNS: tuple[str, ...] = (
    DATE_COL,
    TICKER_COL,
    *OHLCV_COLUMNS,
    *TEMPORAL_COLUMNS,
    *IDENTITY_COLUMNS,
)
SEMANTIC_MANIFEST_FIELDS: tuple[str, ...] = (
    "contract",
    "schema_version",
    "source_snapshot_hash",
    "source_manifest_sha256",
    "producer_git_sha",
    "files",
    "columns",
    "rows",
    "date_min",
    "date_max",
    "tickers",
    "temporal_contract",
    "point_in_time_limits",
    "license",
    "source_provenance",
)


class SignalFoundryContractError(ValueError):
    """Raised when a bundle cannot be safely published or consumed."""


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
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    temporary.replace(path)


def _coerce_contract_frame(panel: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(CONTRACT_COLUMNS).difference(panel.columns))
    if missing:
        raise SignalFoundryContractError(f"contract frame is missing required columns: {missing}")
    frame: pd.DataFrame = panel.loc[:, list(CONTRACT_COLUMNS)].copy()
    frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="raise").dt.tz_localize(None)
    for column in TEMPORAL_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    if frame[["effective_at", "available_at", "observed_at"]].isna().any().any():
        raise SignalFoundryContractError(
            "effective_at, available_at, and observed_at must be valid UTC timestamps"
        )
    if (frame["available_at"] < frame["effective_at"]).any():
        raise SignalFoundryContractError("available_at precedes effective_at")
    if (frame["observed_at"] < frame["effective_at"]).any():
        raise SignalFoundryContractError("observed_at precedes effective_at")
    if frame.duplicated([DATE_COL, TICKER_COL]).any():
        raise SignalFoundryContractError("contract frame has duplicate (date, ticker) keys")
    numeric = frame.loc[:, OHLCV_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise SignalFoundryContractError("contract OHLCV values must be finite")
    if (frame["volume"] < 0).any():
        raise SignalFoundryContractError("contract volume must be non-negative")
    if (
        (frame["high"] < frame["low"])
        | (frame["open"] > frame["high"])
        | (frame["open"] < frame["low"])
        | (frame["close"] > frame["high"])
        | (frame["close"] < frame["low"])
    ).any():
        raise SignalFoundryContractError("contract data violates OHLC bounds")
    validate_price_panel(frame, min_observations=1, raise_on_error=True)
    text_columns = [TICKER_COL, *IDENTITY_COLUMNS]
    for column in text_columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise SignalFoundryContractError(f"contract column {column!r} contains empty values")
        frame[column] = frame[column].astype(str)
    frame = frame.sort_values([DATE_COL, TICKER_COL], kind="stable").reset_index(drop=True)
    return frame


def _safe_relative_file(bundle_dir: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SignalFoundryContractError(f"unsafe bundle path: {relative!r}")
    resolved = (bundle_dir / relative_path).resolve()
    root = bundle_dir.resolve()
    if resolved != root and root not in resolved.parents:
        raise SignalFoundryContractError(f"bundle path escapes root: {relative!r}")
    return resolved


def export_signal_foundry_bundle(
    panel: pd.DataFrame,
    output_root: str | Path,
    *,
    source_manifest: Mapping[str, Any],
    producer_git_sha: str | None = None,
) -> Path:
    """Publish a deterministic, content-addressed Signal Foundry bundle.

    Parameters
    ----------
    panel:
        Validated canonical market observations with contract temporal and
        identity columns.
    output_root:
        Parent directory for immutable ``<bundle_id>/`` directories.
    source_manifest:
        Redacted provider provenance. It must explicitly state that no API key
        is present and identify the source snapshot.
    producer_git_sha:
        Full producer commit. Defaults to the current checkout when available.

    Returns
    -------
    pathlib.Path
        The immutable bundle directory.
    """
    if source_manifest.get("contains_api_key") is not False:
        raise SignalFoundryContractError(
            "source manifest must explicitly attest contains_api_key=false"
        )
    if not isinstance(source_manifest.get("snapshot_hash"), str):
        raise SignalFoundryContractError("source manifest lacks snapshot_hash")
    retrieved_at = source_manifest.get("retrieved_at")
    if not isinstance(retrieved_at, str):
        raise SignalFoundryContractError("source manifest has invalid retrieved_at")
    try:
        created_at = pd.Timestamp(retrieved_at).tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise SignalFoundryContractError("source manifest has invalid retrieved_at") from exc

    frame = _coerce_contract_frame(panel)
    root = ensure_dir(output_root)
    staging = root / ".publishing"
    if staging.exists():
        raise SignalFoundryContractError(
            f"stale publication staging exists at {staging}; inspect it before retrying"
        )
    ensure_dir(staging / "prices")

    files: list[dict[str, Any]] = []
    destination: Path | None = None
    destination_created = False
    try:
        years = frame[DATE_COL].dt.year.astype(int)
        for year in sorted(years.unique().tolist()):
            partition = frame.loc[years.eq(year)].reset_index(drop=True)
            relative = f"prices/year={year}/part-00000.parquet"
            path = _safe_relative_file(staging, relative)
            path.parent.mkdir(parents=True, exist_ok=False)
            table = pa.Table.from_pandas(partition, preserve_index=False)
            pq.write_table(
                table,
                path,
                compression="zstd",
                compression_level=9,
                version="2.6",
                data_page_version="2.0",
                use_dictionary=True,
                write_statistics=True,
            )
            files.append(
                {
                    "path": relative,
                    "sha256": _sha256_file(path),
                    "rows": int(len(partition)),
                    "year": year,
                }
            )

        source_manifest_hash = _sha256_bytes(_canonical_json(dict(source_manifest)))
        temporal_contract = {
            "effective_at": "economic event time",
            "available_at": "earliest permitted research decision time",
            "observed_at": "provider retrieval time",
            "provider_updated_at": "provider revision metadata when available",
            "as_of_rule": "available_at <= decision timestamp",
        }
        point_in_time_limits = source_manifest.get("point_in_time_limits", {})
        redistributable = bool(source_manifest.get("observations_redistributable", False))
        license_policy = {
            "observations_redistributable": redistributable,
            "bundle_must_remain_local": not redistributable,
            "public_evidence_must_be_aggregate_or_synthetic": not redistributable,
        }
        source_provenance = {
            "provider": source_manifest.get("provider"),
            "table": source_manifest.get("request", {}).get("table"),
            "request_hash": source_manifest.get("request_hash"),
            "snapshot_hash": source_manifest.get("snapshot_hash"),
            "retrieved_at": source_manifest.get("retrieved_at"),
            "adjustment_state": source_manifest.get("adjustment_state"),
            "availability_policy": source_manifest.get("availability_policy"),
        }
        semantic: dict[str, Any] = {
            "contract": CONTRACT_NAME,
            "schema_version": SCHEMA_VERSION,
            "source_snapshot_hash": source_manifest["snapshot_hash"],
            "source_manifest_sha256": source_manifest_hash,
            "producer_git_sha": producer_git_sha or git_commit_hash(short=False),
            "files": files,
            "columns": list(CONTRACT_COLUMNS),
            "rows": int(len(frame)),
            "date_min": str(frame[DATE_COL].min().date()),
            "date_max": str(frame[DATE_COL].max().date()),
            "tickers": sorted(frame[TICKER_COL].unique().tolist()),
            "temporal_contract": temporal_contract,
            "point_in_time_limits": point_in_time_limits,
            "license": license_policy,
            "source_provenance": source_provenance,
        }
        bundle_id = _sha256_bytes(_canonical_json(semantic))
        manifest = {
            **semantic,
            "manifest_version": 1,
            "bundle_id": bundle_id,
            "created_at": created_at.isoformat().replace("+00:00", "Z"),
        }
        _atomic_json(staging / MANIFEST_NAME, manifest)
        destination = root / bundle_id
        if destination.exists():
            validate_signal_foundry_bundle(destination)
            shutil.rmtree(staging)
            return destination
        staging.replace(destination)
        destination_created = True
        validate_signal_foundry_bundle(destination)
        return destination
    except Exception:
        # A failed publication is never a valid-looking bundle. Preserve no
        # partial licensed output after the failure path completes.
        if staging.exists():
            shutil.rmtree(staging)
        if destination_created and destination is not None and destination.exists():
            shutil.rmtree(destination)
        raise


def read_signal_foundry_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    """Read and structurally validate a bundle manifest."""
    root = Path(bundle_dir)
    path = root / MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SignalFoundryContractError(f"cannot read canonical manifest at {path}") from exc
    if not isinstance(value, dict):
        raise SignalFoundryContractError("bundle manifest must be a JSON object")
    if value.get("contract") != CONTRACT_NAME:
        raise SignalFoundryContractError("unsupported bundle contract name")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise SignalFoundryContractError(
            f"unsupported schema version {value.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    bundle_id = value.get("bundle_id")
    if not isinstance(bundle_id, str) or len(bundle_id) != 64 or root.name != bundle_id:
        raise SignalFoundryContractError("bundle identity does not match its directory")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise SignalFoundryContractError("bundle manifest has no data files")
    if value.get("columns") != list(CONTRACT_COLUMNS):
        raise SignalFoundryContractError("bundle manifest declares an unsupported column schema")
    return value


def validate_signal_foundry_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify identity, hashes, schema, temporal rules, and row counts."""
    root = Path(bundle_dir)
    manifest = read_signal_foundry_manifest(root)
    frames: list[pd.DataFrame] = []
    total_rows = 0
    seen_paths: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise SignalFoundryContractError("bundle file entry must be an object")
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        expected_rows = entry.get("rows")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise SignalFoundryContractError("bundle file entry is incomplete")
        if relative in seen_paths:
            raise SignalFoundryContractError(f"duplicate bundle data path: {relative}")
        seen_paths.add(relative)
        path = _safe_relative_file(root, relative)
        if not path.is_file():
            raise SignalFoundryContractError(f"bundle data file is missing: {relative}")
        if _sha256_file(path) != expected_hash:
            raise SignalFoundryContractError(f"bundle data hash mismatch: {relative}")
        frame = pd.read_parquet(path)
        if list(frame.columns) != list(CONTRACT_COLUMNS):
            raise SignalFoundryContractError(f"bundle schema mismatch: {relative}")
        if not isinstance(expected_rows, int) or len(frame) != expected_rows:
            raise SignalFoundryContractError(f"bundle row-count mismatch: {relative}")
        total_rows += len(frame)
        frames.append(frame)
    if total_rows != manifest.get("rows"):
        raise SignalFoundryContractError("bundle aggregate row-count mismatch")

    frame = _coerce_contract_frame(pd.concat(frames, ignore_index=True))
    try:
        semantic = {key: manifest[key] for key in SEMANTIC_MANIFEST_FIELDS}
    except KeyError as exc:
        raise SignalFoundryContractError(
            f"bundle manifest is missing identity field: {exc.args[0]}"
        ) from exc
    if _sha256_bytes(_canonical_json(semantic)) != manifest["bundle_id"]:
        raise SignalFoundryContractError("bundle semantic identity mismatch")
    if str(frame[DATE_COL].min().date()) != manifest["date_min"]:
        raise SignalFoundryContractError("bundle minimum date mismatch")
    if str(frame[DATE_COL].max().date()) != manifest["date_max"]:
        raise SignalFoundryContractError("bundle maximum date mismatch")
    if sorted(frame[TICKER_COL].unique().tolist()) != manifest["tickers"]:
        raise SignalFoundryContractError("bundle ticker universe mismatch")
    return manifest


def load_signal_foundry_bundle(
    bundle_dir: str | Path,
    *,
    as_of: str | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load a verified bundle, optionally applying the point-in-time rule."""
    root = Path(bundle_dir)
    manifest = validate_signal_foundry_bundle(root)
    frames = [
        pd.read_parquet(_safe_relative_file(root, entry["path"])) for entry in manifest["files"]
    ]
    frame = _coerce_contract_frame(pd.concat(frames, ignore_index=True))
    if as_of is not None:
        timestamp = pd.Timestamp(as_of)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(UTC)
        else:
            timestamp = timestamp.tz_convert(UTC)
        frame = frame.loc[frame["available_at"].le(timestamp)]
    return frame.sort_values([DATE_COL, TICKER_COL], kind="stable").reset_index(drop=True)
