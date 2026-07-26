"""Immutable content-addressed feature materialization over DuckDB and Parquet.

The DuckDB file is a transactional local catalog, not the source of truth for
feature values.  Values live in immutable, hash-verified Parquet objects.  A
reader validates the canonical manifest and every declared file before it
constructs a parameterized DuckDB query.

Concurrency is deliberately bounded to a single writer process at a time via
an advisory file lock.  DuckDB supports concurrent writers within one process,
but stable cross-process read/write coordination requires a server/catalog
architecture outside this local-first contract.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Literal

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.features.quality import (
    DriftReport,
    FeatureQualityPolicy,
    FeatureQualityReport,
    evaluate_distribution_drift,
    evaluate_feature_quality,
)
from quant_platform.features.registry import (
    SHA256_RE,
    FeatureRegistry,
    FeatureSpec,
    canonical_json,
    semantic_hash,
)
from quant_platform.utils import git_commit_hash, hash_dataframe

try:
    import fcntl
except ImportError:  # pragma: no cover - supported CI/runtime is POSIX
    fcntl = None  # type: ignore[assignment]

MANIFEST_SCHEMA_VERSION = "1.0.0"
CATALOG_SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class FeatureStoreError(RuntimeError):
    """Base error for store contract, integrity, and publication failures."""


class FeatureStoreIntegrityError(FeatureStoreError):
    """Raised when persisted evidence does not match its manifest."""


class FeatureQualityGateError(FeatureStoreError):
    """Raised when mandatory quality or drift evidence fails."""

    def __init__(
        self,
        message: str,
        *,
        quality: FeatureQualityReport,
        drift: DriftReport | None,
    ) -> None:
        super().__init__(message)
        self.quality = quality
        self.drift = drift


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DatasetLineage(_Contract):
    """Non-observational source identity required for feature provenance."""

    dataset_sha256: str
    source: str = Field(min_length=1, max_length=120)
    source_revision: str = Field(min_length=1, max_length=256)
    request_sha256: str
    schema_version: str = Field(min_length=1, max_length=32)
    retrieved_at: datetime
    coverage_start: date
    coverage_end: date
    requested_tickers: tuple[str, ...] = Field(min_length=1, max_length=10_000)
    returned_tickers: tuple[str, ...] = Field(min_length=1, max_length=10_000)
    observations_redistributable: bool
    historical_revisions_complete: bool
    universe_membership_point_in_time: bool
    corporate_actions_complete: bool

    @field_validator("dataset_sha256", "request_sha256")
    @classmethod
    def _full_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("lineage hashes must be lowercase full SHA-256 digests")
        return value

    @field_validator("requested_tickers", "returned_tickers")
    @classmethod
    def _canonical_tickers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted({value.strip().upper() for value in values if value.strip()}))
        if not normalized:
            raise ValueError("ticker collections must be non-empty")
        if any(not value.replace(".", "").replace("-", "").isalnum() for value in normalized):
            raise ValueError("ticker values contain unsupported characters")
        return normalized

    @model_validator(mode="after")
    def _consistent_lineage(self) -> DatasetLineage:
        if self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone-aware")
        if self.coverage_end < self.coverage_start:
            raise ValueError("coverage_end must be on or after coverage_start")
        if not set(self.returned_tickers).issubset(self.requested_tickers):
            raise ValueError("returned_tickers must be a subset of requested_tickers")
        return self


class RuntimeIdentity(_Contract):
    """Allowlisted runtime and dependency identity."""

    python: str
    implementation: str
    operating_system: str
    machine: str
    dependencies: dict[str, str]

    @classmethod
    def capture(cls) -> RuntimeIdentity:
        dependencies: dict[str, str] = {}
        for package in ("duckdb", "numpy", "pandas", "pyarrow", "pydantic", "scipy"):
            try:
                dependencies[package] = version(package)
            except PackageNotFoundError:  # pragma: no cover - core deps are installed
                dependencies[package] = "not-installed"
        return cls(
            python=platform.python_version(),
            implementation=platform.python_implementation(),
            operating_system=platform.system(),
            machine=platform.machine(),
            dependencies=dependencies,
        )


class FeatureOutputContract(_Contract):
    """Non-feature outputs and market context stored beside feature columns."""

    benchmark: str = Field(min_length=1, max_length=64)
    price_field: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    forward_horizon: int | None = Field(default=None, ge=1, le=1_000_000)
    target_columns: tuple[str, ...] = Field(default_factory=tuple, max_length=100)

    @field_validator("benchmark")
    @classmethod
    def _benchmark_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized.replace(".", "").replace("-", "").isalnum():
            raise ValueError("benchmark contains unsupported characters")
        return normalized

    @field_validator("target_columns")
    @classmethod
    def _target_identifiers(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("target columns must be unique")
        if any(not SAFE_IDENTIFIER_RE.fullmatch(value) for value in values):
            raise ValueError("target columns must be safe identifiers")
        return values

    @model_validator(mode="after")
    def _label_contract(self) -> FeatureOutputContract:
        if (self.forward_horizon is None) != (not self.target_columns):
            raise ValueError(
                "forward_horizon and target_columns must either both be present or both be absent"
            )
        return self


class FeatureMaterializationRequest(_Contract):
    """All semantic inputs that may alter a stored feature matrix."""

    contract_version: Literal["1.0.0"] = "1.0.0"
    lineage: DatasetLineage
    features: tuple[FeatureSpec, ...] = Field(min_length=1, max_length=10_000)
    output_contract: FeatureOutputContract
    application_start: date
    application_end: date
    expected_end: date
    partition_by: Literal["year", "month"] = "year"
    code_commit: str
    runtime: RuntimeIdentity
    quality_policy: FeatureQualityPolicy = Field(default_factory=FeatureQualityPolicy)
    evidence_time: datetime

    @field_validator("code_commit")
    @classmethod
    def _commit_identity(cls, value: str) -> str:
        if value != "unavailable" and not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("code_commit must be a full Git SHA or 'unavailable'")
        return value

    @model_validator(mode="after")
    def _validate_request(self) -> FeatureMaterializationRequest:
        if self.evidence_time.tzinfo is None:
            raise ValueError("evidence_time must be timezone-aware")
        if self.application_end < self.application_start:
            raise ValueError("application_end must be on or after application_start")
        if self.expected_end < self.application_end:
            raise ValueError("expected_end cannot precede application_end")
        names = [feature.name for feature in self.features]
        if len(set(names)) != len(names):
            raise ValueError("feature output names must be unique")
        for feature in self.features:
            state = feature.fitted_state
            if state is not None and state.fit_end >= self.application_start:
                raise ValueError(
                    f"feature {feature.name!r} fit interval overlaps its application interval"
                )
        return self

    @property
    def identity(self) -> str:
        """Return the deterministic semantic request identity."""
        return semantic_hash(self.model_dump(mode="json"))

    @classmethod
    def create(
        cls,
        *,
        lineage: DatasetLineage,
        registry: FeatureRegistry,
        output_contract: FeatureOutputContract,
        application_start: date,
        application_end: date,
        expected_end: date | None = None,
        partition_by: Literal["year", "month"] = "year",
        quality_policy: FeatureQualityPolicy | None = None,
        evidence_time: datetime,
    ) -> FeatureMaterializationRequest:
        """Create a request using the current repository/runtime identities."""
        return cls(
            lineage=lineage,
            features=registry.specs,
            output_contract=output_contract,
            application_start=application_start,
            application_end=application_end,
            expected_end=expected_end or application_end,
            partition_by=partition_by,
            code_commit=git_commit_hash(short=False) or "unavailable",
            runtime=RuntimeIdentity.capture(),
            quality_policy=quality_policy or FeatureQualityPolicy(),
            evidence_time=evidence_time.astimezone(UTC),
        )


class PartitionEvidence(_Contract):
    """Integrity and coverage record for one Parquet partition."""

    relative_path: str
    sha256: str
    size_bytes: int = Field(ge=1)
    rows: int = Field(ge=1)
    date_min: date
    date_max: date

    @field_validator("sha256")
    @classmethod
    def _hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("partition hash must be a lowercase full SHA-256 digest")
        return value

    @field_validator("relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".parquet":
            raise ValueError("partition path must be a contained relative Parquet path")
        return value


class FeatureMaterializationManifest(_Contract):
    """Canonical manifest that binds lineage, schema, quality, and files."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    object_id: str
    request_id: str
    content_sha256: str
    request: FeatureMaterializationRequest
    columns: tuple[str, ...]
    arrow_schema: str
    rows: int = Field(ge=1)
    partitions: tuple[PartitionEvidence, ...] = Field(min_length=1)
    quality: FeatureQualityReport
    drift: DriftReport | None

    @field_validator("object_id", "request_id", "content_sha256")
    @classmethod
    def _hashes(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("manifest identities must be lowercase full SHA-256 digests")
        return value

    @model_validator(mode="after")
    def _consistent_manifest(self) -> FeatureMaterializationManifest:
        if self.request.identity != self.request_id:
            raise ValueError("request_id does not match the canonical request")
        if self.quality.status != "pass":
            raise ValueError("a published manifest cannot contain failed quality evidence")
        if self.drift is not None and self.drift.status != "pass":
            raise ValueError("a published manifest cannot contain failed drift evidence")
        if sum(partition.rows for partition in self.partitions) != self.rows:
            raise ValueError("partition row counts do not equal manifest rows")
        return self


class FeatureStore:
    """Local immutable feature store with a transactional DuckDB catalog."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_query_rows: int = 5_000_000,
        max_query_columns: int = 1_000,
    ) -> None:
        if max_query_rows < 1 or max_query_columns < 1:
            raise ValueError("query bounds must be positive")
        candidate = Path(root).expanduser()
        if candidate.exists() and candidate.is_symlink():
            raise FeatureStoreError("feature-store root cannot be a symlink")
        self.root = candidate.resolve()
        self.objects_dir = self.root / "objects"
        self.staging_dir = self.root / ".staging"
        self.failures_dir = self.root / "failures"
        self.catalog_path = self.root / "catalog.duckdb"
        self.lock_path = self.root / ".writer.lock"
        self.max_query_rows = max_query_rows
        self.max_query_columns = max_query_columns
        for directory in (self.root, self.objects_dir, self.staging_dir, self.failures_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._initialize_catalog()

    def lookup(
        self, request: FeatureMaterializationRequest
    ) -> FeatureMaterializationManifest | None:
        """Return and fully verify the existing object for a semantic request."""
        with self._connect(read_only=True) as connection:
            row = connection.execute(
                """
                SELECT object_id
                FROM feature_materializations
                WHERE request_id = ? AND status = 'published'
                """,
                [request.identity],
            ).fetchone()
        if row is None:
            return None
        manifest = self.validate(str(row[0]))
        if manifest.request != request:
            raise FeatureStoreIntegrityError("catalog request does not match stored manifest")
        return manifest

    def materialize(
        self,
        request: FeatureMaterializationRequest,
        frame: pd.DataFrame,
        *,
        drift_reference: pd.DataFrame | None = None,
    ) -> FeatureMaterializationManifest:
        """Validate and atomically publish one immutable feature object.

        Repeating an equivalent request with equivalent content is an
        idempotent verified cache hit.  Equivalent semantics with different
        feature content fail because they demonstrate nondeterminism.
        """
        registry = FeatureRegistry(request.features)
        ordered = _canonicalize_frame(frame)
        quality = evaluate_feature_quality(
            ordered,
            feature_columns=registry.output_columns,
            requested_tickers=request.lineage.requested_tickers,
            expected_end=request.expected_end,
            policy=request.quality_policy,
        )
        drift = (
            evaluate_distribution_drift(
                drift_reference,
                ordered,
                feature_columns=registry.output_columns,
                policy=request.quality_policy,
            )
            if drift_reference is not None
            else None
        )
        if quality.status == "fail" or (drift is not None and drift.status == "fail"):
            self._record_failure(request, quality, drift)
            raise FeatureQualityGateError(
                "feature materialization failed mandatory quality evidence",
                quality=quality,
                drift=drift,
            )
        content_sha256 = hash_dataframe(ordered, length=64)

        with self._exclusive_writer():
            existing = self.lookup(request)
            if existing is not None:
                if existing.content_sha256 != content_sha256:
                    raise FeatureStoreIntegrityError(
                        "equivalent request produced different content; deterministic cache refused"
                    )
                return existing

            stage = self.staging_dir / f"{request.identity}.{uuid.uuid4().hex}"
            stage.mkdir(mode=0o700)
            try:
                partitions = self._write_partitions(stage, ordered, request.partition_by)
                columns = tuple(str(column) for column in ordered.columns)
                arrow_schema = str(pa.Schema.from_pandas(ordered, preserve_index=False))
                identity_payload = {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "request_id": request.identity,
                    "content_sha256": content_sha256,
                    "request": request.model_dump(mode="json"),
                    "columns": columns,
                    "arrow_schema": arrow_schema,
                    "rows": len(ordered),
                    "partitions": [partition.model_dump(mode="json") for partition in partitions],
                    "quality": quality.model_dump(mode="json"),
                    "drift": drift.model_dump(mode="json") if drift is not None else None,
                }
                object_id = semantic_hash(identity_payload)
                manifest = FeatureMaterializationManifest(
                    object_id=object_id,
                    request_id=request.identity,
                    content_sha256=content_sha256,
                    request=request,
                    columns=columns,
                    arrow_schema=arrow_schema,
                    rows=len(ordered),
                    partitions=partitions,
                    quality=quality,
                    drift=drift,
                )
                _atomic_write_json(stage / "manifest.json", manifest.model_dump(mode="json"))
                final = self.objects_dir / object_id
                if final.exists():
                    recovered = self.validate(object_id)
                    if recovered != manifest:
                        raise FeatureStoreIntegrityError(
                            "existing object identity does not match candidate manifest"
                        )
                    shutil.rmtree(stage)
                else:
                    os.replace(stage, final)
                    _fsync_directory(self.objects_dir)
                self._publish_catalog(manifest)
                return self.validate(object_id)
            except Exception:
                if stage.exists():
                    shutil.rmtree(stage)
                raise

    def validate(self, object_id: str) -> FeatureMaterializationManifest:
        """Verify manifest, path containment, file inventory, hashes, and schema."""
        if not SHA256_RE.fullmatch(object_id):
            raise FeatureStoreIntegrityError("object_id must be a lowercase full SHA-256 digest")
        object_dir = self.objects_dir / object_id
        if object_dir.is_symlink() or not object_dir.is_dir():
            raise FeatureStoreIntegrityError("materialization object directory is unavailable")
        manifest_path = object_dir / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise FeatureStoreIntegrityError("materialization manifest is unavailable")
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise FeatureStoreIntegrityError("materialization manifest exceeds the size bound")
        try:
            manifest = FeatureMaterializationManifest.model_validate_json(
                manifest_path.read_bytes()
            )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise FeatureStoreIntegrityError("materialization manifest is invalid") from exc
        if manifest.object_id != object_id:
            raise FeatureStoreIntegrityError("manifest object_id does not match its directory")
        expected_object_id = semantic_hash(_manifest_identity_payload(manifest))
        if expected_object_id != object_id:
            raise FeatureStoreIntegrityError("object_id does not bind the canonical manifest")
        expected_files = {"manifest.json"}
        total_rows = 0
        observed_schema: pa.Schema | None = None
        for partition in manifest.partitions:
            path = _contained_path(object_dir, partition.relative_path)
            expected_files.add(partition.relative_path)
            if path.is_symlink() or not path.is_file():
                raise FeatureStoreIntegrityError(
                    f"partition is missing or a symlink: {partition.relative_path}"
                )
            if path.stat().st_size != partition.size_bytes:
                raise FeatureStoreIntegrityError(
                    f"partition size mismatch: {partition.relative_path}"
                )
            if _sha256_file(path) != partition.sha256:
                raise FeatureStoreIntegrityError(
                    f"partition hash mismatch: {partition.relative_path}"
                )
            metadata = pq.read_metadata(path)
            if metadata.num_rows != partition.rows:
                raise FeatureStoreIntegrityError(
                    f"partition row-count mismatch: {partition.relative_path}"
                )
            schema = pq.read_schema(path)
            if observed_schema is None:
                observed_schema = schema
            elif schema != observed_schema:
                raise FeatureStoreIntegrityError("Parquet partition schemas differ")
            total_rows += metadata.num_rows
        actual_files = {
            path.relative_to(object_dir).as_posix()
            for path in object_dir.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise FeatureStoreIntegrityError("materialization contains undeclared or missing files")
        if total_rows != manifest.rows:
            raise FeatureStoreIntegrityError("verified partition rows do not match manifest")
        if observed_schema is None or str(observed_schema) != manifest.arrow_schema:
            raise FeatureStoreIntegrityError("verified Parquet schema does not match manifest")
        return manifest

    def read(
        self,
        object_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
        tickers: tuple[str, ...] | None = None,
        columns: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Return a bounded deterministic projection using pushed-down filters."""
        manifest = self.validate(object_id)
        selected = columns or manifest.columns
        if not selected or len(selected) > self.max_query_columns:
            raise FeatureStoreError("requested column count is empty or exceeds the bound")
        unknown = sorted(set(selected).difference(manifest.columns))
        if unknown:
            raise FeatureStoreError(f"unknown requested columns: {unknown}")
        if any(not SAFE_IDENTIFIER_RE.fullmatch(column) for column in selected):
            raise FeatureStoreError("manifest contains an unsafe SQL identifier")
        if start is not None and end is not None and end < start:
            raise FeatureStoreError("end must be on or after start")
        normalized_tickers: tuple[str, ...] | None = None
        if tickers is not None:
            normalized_tickers = tuple(sorted({ticker.strip().upper() for ticker in tickers}))
            if not normalized_tickers or len(normalized_tickers) > 10_000:
                raise FeatureStoreError("ticker filter is empty or exceeds the bound")

        object_dir = self.objects_dir / object_id
        paths = [
            str(_contained_path(object_dir, partition.relative_path))
            for partition in manifest.partitions
        ]
        projection = ", ".join(f'"{column}"' for column in selected)
        clauses: list[str] = []
        parameters: list[object] = [paths]
        if start is not None:
            clauses.append(f'"{DATE_COL}" >= ?')
            parameters.append(start)
        if end is not None:
            clauses.append(f'"{DATE_COL}" <= ?')
            parameters.append(end)
        if normalized_tickers is not None:
            placeholders = ", ".join("?" for _ in normalized_tickers)
            clauses.append(f'"{TICKER_COL}" IN ({placeholders})')
            parameters.extend(normalized_tickers)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT {projection} FROM read_parquet(?)"
            f'{where} ORDER BY "{DATE_COL}", "{TICKER_COL}" LIMIT ?'
        )
        parameters.append(self.max_query_rows + 1)
        with duckdb.connect(":memory:") as connection:
            result = connection.execute(query, parameters).fetch_df()
        if len(result) > self.max_query_rows:
            raise FeatureStoreError("query result exceeds max_query_rows")
        return result

    def _write_partitions(
        self,
        stage: Path,
        frame: pd.DataFrame,
        partition_by: Literal["year", "month"],
    ) -> tuple[PartitionEvidence, ...]:
        dates = pd.to_datetime(frame[DATE_COL])
        keys = dates.dt.strftime("%Y" if partition_by == "year" else "%Y-%m")
        records: list[PartitionEvidence] = []
        for key, partition in frame.assign(_partition=keys).groupby("_partition", sort=True):
            clean = partition.drop(columns="_partition").reset_index(drop=True)
            directory_name = f"{partition_by}={key}"
            relative = PurePosixPath(directory_name) / "part-00000.parquet"
            path = _contained_path(stage, relative.as_posix())
            path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(clean, preserve_index=False)
            pq.write_table(
                table,
                path,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
                row_group_size=min(250_000, max(1, len(clean))),
            )
            _fsync_file(path)
            partition_dates = pd.to_datetime(clean[DATE_COL])
            records.append(
                PartitionEvidence(
                    relative_path=relative.as_posix(),
                    sha256=_sha256_file(path),
                    size_bytes=path.stat().st_size,
                    rows=len(clean),
                    date_min=partition_dates.min().date(),
                    date_max=partition_dates.max().date(),
                )
            )
        if not records:
            raise FeatureStoreError("cannot publish a materialization without partitions")
        return tuple(records)

    def _record_failure(
        self,
        request: FeatureMaterializationRequest,
        quality: FeatureQualityReport,
        drift: DriftReport | None,
    ) -> None:
        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "request_id": request.identity,
            "evidence_time": request.evidence_time.isoformat(),
            "quality": quality.model_dump(mode="json"),
            "drift": drift.model_dump(mode="json") if drift is not None else None,
        }
        _atomic_write_json(self.failures_dir / f"{request.identity}.json", payload)

    def _publish_catalog(self, manifest: FeatureMaterializationManifest) -> None:
        relative_manifest = (Path("objects") / manifest.object_id / "manifest.json").as_posix()
        with self._connect() as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                existing = connection.execute(
                    "SELECT object_id FROM feature_materializations WHERE request_id = ?",
                    [manifest.request_id],
                ).fetchone()
                if existing is not None and str(existing[0]) != manifest.object_id:
                    raise FeatureStoreIntegrityError(
                        "catalog already binds request to a different object"
                    )
                connection.execute(
                    """
                    INSERT INTO feature_materializations
                    (request_id, object_id, status, manifest_path, evidence_time, rows)
                    VALUES (?, ?, 'published', ?, ?, ?)
                    ON CONFLICT (request_id) DO NOTHING
                    """,
                    [
                        manifest.request_id,
                        manifest.object_id,
                        relative_manifest,
                        manifest.request.evidence_time,
                        manifest.rows,
                    ],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def _initialize_catalog(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS feature_store_metadata (
                    schema_version INTEGER PRIMARY KEY
                )
                """)
            connection.execute(
                """
                INSERT INTO feature_store_metadata VALUES (?)
                ON CONFLICT (schema_version) DO NOTHING
                """,
                [CATALOG_SCHEMA_VERSION],
            )
            versions = connection.execute(
                "SELECT schema_version FROM feature_store_metadata ORDER BY schema_version"
            ).fetchall()
            if versions != [(CATALOG_SCHEMA_VERSION,)]:
                raise FeatureStoreIntegrityError("unsupported DuckDB catalog schema")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS feature_materializations (
                    request_id VARCHAR PRIMARY KEY,
                    object_id VARCHAR NOT NULL UNIQUE,
                    status VARCHAR NOT NULL CHECK (status IN ('published')),
                    manifest_path VARCHAR NOT NULL,
                    evidence_time TIMESTAMPTZ NOT NULL,
                    rows BIGINT NOT NULL CHECK (rows > 0)
                )
                """)

    @contextmanager
    def _connect(self, *, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
        connection = duckdb.connect(str(self.catalog_path), read_only=read_only)
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _exclusive_writer(self) -> Iterator[None]:
        if fcntl is None:
            raise FeatureStoreError("single-writer locking requires a POSIX runtime")
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL not in frame or TICKER_COL not in frame:
        return frame.copy()
    result = frame.copy()
    result[DATE_COL] = pd.to_datetime(result[DATE_COL], errors="coerce")
    result[TICKER_COL] = result[TICKER_COL].astype(str).str.upper()
    return result.sort_values([DATE_COL, TICKER_COL], kind="mergesort").reset_index(drop=True)


def _manifest_identity_payload(
    manifest: FeatureMaterializationManifest,
) -> dict[str, object]:
    """Return every manifest field except the self-referential object ID."""
    payload = manifest.model_dump(mode="json")
    payload.pop("object_id")
    return payload


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _contained_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts:
        raise FeatureStoreIntegrityError("artifact path escapes the object root")
    candidate = root.joinpath(*parts.parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise FeatureStoreIntegrityError("artifact path escapes the object root") from exc
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if sys.platform == "win32":  # pragma: no cover - Windows lacks directory fsync
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
