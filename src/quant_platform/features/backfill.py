"""Persistent bounded orchestration for resumable feature-store backfills."""

from __future__ import annotations

import calendar
import hashlib
import os
import re
import sys
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.features.registry import SHA256_RE, canonical_json, semantic_hash
from quant_platform.features.store import (
    FeatureMaterializationManifest,
    FeatureMaterializationRequest,
    FeatureStore,
    FeatureStoreError,
)
from quant_platform.utils import hash_dataframe

try:
    import fcntl
except ImportError:  # pragma: no cover - supported CI/runtime is POSIX
    fcntl = None  # type: ignore[assignment]

MAX_FAILURE_CHARS = 1_000
_CREDENTIAL_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)([^\s,;]+)")


class BackfillError(FeatureStoreError):
    """Base error for plan, state, recovery, or worker failures."""


class BackfillInterrupted(BackfillError):
    """Explicit test-boundary interruption after durable partition completion."""


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BackfillPartition(_Contract):
    """One deterministic, inclusive application-date partition."""

    key: str = Field(pattern=r"^[0-9]{4}(?:-[0-9]{2})?$")
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> BackfillPartition:
        if self.end < self.start:
            raise ValueError("partition end must be on or after start")
        return self


class BackfillPlan(_Contract):
    """Semantic job identity and bounded execution policy."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    request: FeatureMaterializationRequest
    partitions: tuple[BackfillPartition, ...] = Field(min_length=1, max_length=10_000)
    max_workers: int = Field(default=2, ge=1, le=32)
    max_attempts: int = Field(default=2, ge=1, le=10)
    max_rows_per_partition: int = Field(default=5_000_000, ge=1, le=100_000_000)

    @property
    def identity(self) -> str:
        return semantic_hash(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def _coverage(self) -> BackfillPlan:
        ordered = tuple(sorted(self.partitions, key=lambda partition: partition.start))
        if ordered != self.partitions:
            raise ValueError("partitions must be ordered by start date")
        if ordered[0].start != self.request.application_start:
            raise ValueError("first partition must start at application_start")
        if ordered[-1].end != self.request.application_end:
            raise ValueError("last partition must end at application_end")
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if (current.start - previous.end).days != 1:
                raise ValueError("partitions must be contiguous and non-overlapping")
        return self

    @classmethod
    def create(
        cls,
        request: FeatureMaterializationRequest,
        *,
        max_workers: int = 2,
        max_attempts: int = 2,
        max_rows_per_partition: int = 5_000_000,
    ) -> BackfillPlan:
        return cls(
            request=request,
            partitions=_date_partitions(
                request.application_start,
                request.application_end,
                request.partition_by,
            ),
            max_workers=max_workers,
            max_attempts=max_attempts,
            max_rows_per_partition=max_rows_per_partition,
        )


class BackfillResult(_Contract):
    """Terminal published job result."""

    plan_id: str
    object_id: str
    status: Literal["published"] = "published"
    reused_partitions: int = Field(ge=0)
    computed_partitions: int = Field(ge=0)
    manifest: FeatureMaterializationManifest


class _Checkpoint(_Contract):
    partition_key: str
    relative_path: str
    sha256: str
    content_sha256: str
    rows: int = Field(ge=1)


PartitionLoader = Callable[[BackfillPartition], pd.DataFrame]


class BackfillOrchestrator:
    """Single-process, bounded and resumable feature partition orchestrator."""

    def __init__(self, store: FeatureStore) -> None:
        self.store = store
        self.root = store.root / "backfills"
        self.checkpoint_root = self.root / "checkpoints"
        self.catalog_path = self.root / "backfills.duckdb"
        self.lock_path = self.root / ".writer.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self._initialize_catalog()

    def run(
        self,
        plan: BackfillPlan,
        loader: PartitionLoader,
        *,
        drift_reference: pd.DataFrame | None = None,
        _test_interrupt_after: int | None = None,
    ) -> BackfillResult:
        """Run or resume a plan and atomically publish its final object.

        ``_test_interrupt_after`` is an explicit test-only fault boundary.  It
        raises after that many newly completed checkpoints have been durably
        recorded, allowing exact recovery behavior to be exercised without
        process signals or wall-clock sleeps.
        """
        if _test_interrupt_after is not None and _test_interrupt_after < 1:
            raise ValueError("_test_interrupt_after must be positive")
        with self._exclusive_writer():
            existing_object = self._ensure_job(plan)
            if existing_object is not None:
                manifest = self.store.validate(existing_object)
                return BackfillResult(
                    plan_id=plan.identity,
                    object_id=existing_object,
                    reused_partitions=len(plan.partitions),
                    computed_partitions=0,
                    manifest=manifest,
                )
            checkpoints, pending = self._recover_checkpoints(plan)
            reused = len(checkpoints)
            if pending:
                self._transition_job(plan.identity, {"planned", "interrupted", "failed"}, "running")
                completed_now = self._execute_pending(
                    plan,
                    pending,
                    loader,
                    interrupt_after=_test_interrupt_after,
                )
                checkpoints.update(completed_now)
            if len(checkpoints) != len(plan.partitions):
                self._transition_job(plan.identity, {"running"}, "failed")
                raise BackfillError("backfill did not produce every planned partition")
            self._transition_job(plan.identity, {"running", "planned"}, "assembling")
            assembled = self._assemble(plan, checkpoints)
            try:
                manifest = self.store.materialize(
                    plan.request,
                    assembled,
                    drift_reference=drift_reference,
                )
            except Exception as exc:
                self._fail_job(plan.identity, exc)
                raise
            self._publish_job(plan.identity, manifest.object_id)
            return BackfillResult(
                plan_id=plan.identity,
                object_id=manifest.object_id,
                reused_partitions=reused,
                computed_partitions=len(plan.partitions) - reused,
                manifest=manifest,
            )

    def status(self, plan_id: str) -> dict[str, object]:
        """Return bounded machine-readable job and partition state."""
        if not SHA256_RE.fullmatch(plan_id):
            raise BackfillError("plan_id must be a full lowercase SHA-256 digest")
        with self._connect(read_only=True) as connection:
            job = connection.execute(
                """
                SELECT status, object_id, failure_code, failure_message
                FROM backfill_jobs WHERE plan_id = ?
                """,
                [plan_id],
            ).fetchone()
            if job is None:
                raise BackfillError("unknown backfill plan")
            partitions = connection.execute(
                """
                SELECT partition_key, status, attempts, rows, content_sha256
                FROM backfill_partitions
                WHERE plan_id = ?
                ORDER BY partition_key
                """,
                [plan_id],
            ).fetchall()
        return {
            "plan_id": plan_id,
            "status": job[0],
            "object_id": job[1],
            "failure_code": job[2],
            "failure_message": job[3],
            "partitions": [
                {
                    "key": row[0],
                    "status": row[1],
                    "attempts": row[2],
                    "rows": row[3],
                    "content_sha256": row[4],
                }
                for row in partitions
            ],
        }

    def _execute_pending(
        self,
        plan: BackfillPlan,
        pending: tuple[BackfillPartition, ...],
        loader: PartitionLoader,
        *,
        interrupt_after: int | None,
    ) -> dict[str, _Checkpoint]:
        futures: dict[Future[_Checkpoint], BackfillPartition] = {}
        for partition in pending:
            attempts = self._mark_partition_running(plan, partition)
            if attempts > plan.max_attempts:
                error = BackfillError(
                    f"partition {partition.key} exceeded max_attempts={plan.max_attempts}"
                )
                self._fail_job(plan.identity, error)
                raise error
        with ThreadPoolExecutor(
            max_workers=min(plan.max_workers, len(pending)),
            thread_name_prefix="signalattice-backfill",
        ) as executor:
            for partition in pending:
                future = executor.submit(self._compute_checkpoint, plan, partition, loader)
                futures[future] = partition
            completed: dict[str, _Checkpoint] = {}
            for future in as_completed(futures):
                partition = futures[future]
                try:
                    checkpoint = future.result()
                except Exception as exc:
                    self._mark_partition_failed(plan.identity, partition.key, exc)
                    for outstanding in futures:
                        outstanding.cancel()
                    self._fail_job(plan.identity, exc)
                    raise BackfillError(
                        f"partition {partition.key} failed: {_redacted_failure(exc)}"
                    ) from exc
                self._mark_partition_completed(plan.identity, checkpoint)
                completed[partition.key] = checkpoint
                if interrupt_after is not None and len(completed) >= interrupt_after:
                    for outstanding in futures:
                        outstanding.cancel()
                    self._transition_job(plan.identity, {"running"}, "interrupted")
                    raise BackfillInterrupted(
                        f"injected interruption after {len(completed)} durable partitions"
                    )
        return completed

    def _compute_checkpoint(
        self,
        plan: BackfillPlan,
        partition: BackfillPartition,
        loader: PartitionLoader,
    ) -> _Checkpoint:
        frame = loader(partition)
        if not isinstance(frame, pd.DataFrame):
            raise BackfillError("partition loader must return a pandas DataFrame")
        if len(frame) < 1 or len(frame) > plan.max_rows_per_partition:
            raise BackfillError(f"partition rows must be in [1, {plan.max_rows_per_partition}]")
        if DATE_COL not in frame or TICKER_COL not in frame:
            raise BackfillError("partition is missing date/ticker keys")
        dates = pd.to_datetime(frame[DATE_COL], errors="coerce")
        if dates.isna().any():
            raise BackfillError("partition contains invalid dates")
        if dates.min().date() < partition.start or dates.max().date() > partition.end:
            raise BackfillError("partition loader returned rows outside the planned interval")
        ordered = frame.copy()
        ordered[DATE_COL] = dates
        ordered[TICKER_COL] = ordered[TICKER_COL].astype(str).str.upper()
        ordered = ordered.sort_values([DATE_COL, TICKER_COL], kind="mergesort").reset_index(
            drop=True
        )
        relative = (Path(plan.identity) / f"{partition.key}.parquet").as_posix()
        final = _contained_checkpoint(self.checkpoint_root, relative)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
        try:
            pq.write_table(
                pa.Table.from_pandas(ordered, preserve_index=False),
                temporary,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
                row_group_size=min(250_000, len(ordered)),
            )
            _fsync_file(temporary)
            os.replace(temporary, final)
            _fsync_directory(final.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
        return _Checkpoint(
            partition_key=partition.key,
            relative_path=relative,
            sha256=_sha256_file(final),
            content_sha256=hash_dataframe(ordered, length=64),
            rows=len(ordered),
        )

    def _assemble(
        self,
        plan: BackfillPlan,
        checkpoints: dict[str, _Checkpoint],
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for partition in plan.partitions:
            checkpoint = checkpoints[partition.key]
            path = self._verify_checkpoint(checkpoint)
            frames.append(pd.read_parquet(path))
        result = pd.concat(frames, ignore_index=True)
        result[DATE_COL] = pd.to_datetime(result[DATE_COL])
        result = result.sort_values([DATE_COL, TICKER_COL], kind="mergesort").reset_index(drop=True)
        if result.duplicated([DATE_COL, TICKER_COL]).any():
            raise BackfillError("assembled backfill contains duplicate keys")
        return result

    def _ensure_job(self, plan: BackfillPlan) -> str | None:
        payload = canonical_json(plan.model_dump(mode="json")).decode("utf-8")
        with self._connect() as connection:
            connection.execute("BEGIN TRANSACTION")
            try:
                row = connection.execute(
                    "SELECT plan_json, status, object_id FROM backfill_jobs WHERE plan_id = ?",
                    [plan.identity],
                ).fetchone()
                if row is not None:
                    if row[0] != payload:
                        raise BackfillError("stored plan payload does not match plan identity")
                    connection.execute("COMMIT")
                    return str(row[2]) if row[1] == "published" else None
                connection.execute(
                    """
                    INSERT INTO backfill_jobs
                    (plan_id, request_id, plan_json, status, created_at, updated_at)
                    VALUES (?, ?, ?, 'planned', ?, ?)
                    """,
                    [
                        plan.identity,
                        plan.request.identity,
                        payload,
                        plan.request.evidence_time,
                        datetime.now(UTC),
                    ],
                )
                for partition in plan.partitions:
                    connection.execute(
                        """
                        INSERT INTO backfill_partitions
                        (plan_id, partition_key, start_date, end_date, status, attempts)
                        VALUES (?, ?, ?, ?, 'planned', 0)
                        """,
                        [
                            plan.identity,
                            partition.key,
                            partition.start,
                            partition.end,
                        ],
                    )
                connection.execute("COMMIT")
                return None
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def _recover_checkpoints(
        self, plan: BackfillPlan
    ) -> tuple[dict[str, _Checkpoint], tuple[BackfillPartition, ...]]:
        with self._connect(read_only=True) as connection:
            rows = connection.execute(
                """
                SELECT partition_key, status, attempts, relative_path, sha256,
                       content_sha256, rows
                FROM backfill_partitions
                WHERE plan_id = ?
                ORDER BY partition_key
                """,
                [plan.identity],
            ).fetchall()
        by_key = {partition.key: partition for partition in plan.partitions}
        if {str(row[0]) for row in rows} != set(by_key):
            raise BackfillError("stored partitions do not match the plan")
        checkpoints: dict[str, _Checkpoint] = {}
        pending: list[BackfillPartition] = []
        for row in rows:
            key, status, attempts, relative, sha256, content_sha256, count = row
            if status == "completed":
                checkpoint = _Checkpoint(
                    partition_key=str(key),
                    relative_path=str(relative),
                    sha256=str(sha256),
                    content_sha256=str(content_sha256),
                    rows=int(count),
                )
                self._verify_checkpoint(checkpoint)
                checkpoints[str(key)] = checkpoint
            elif status in {"planned", "running", "failed"} and int(attempts) < plan.max_attempts:
                pending.append(by_key[str(key)])
            else:
                raise BackfillError(
                    f"partition {key} cannot resume from status={status!r}, attempts={attempts}"
                )
        return checkpoints, tuple(sorted(pending, key=lambda partition: partition.start))

    def _verify_checkpoint(self, checkpoint: _Checkpoint) -> Path:
        path = _contained_checkpoint(self.checkpoint_root, checkpoint.relative_path)
        if path.is_symlink() or not path.is_file():
            raise BackfillError(f"checkpoint is missing or a symlink: {checkpoint.partition_key}")
        if _sha256_file(path) != checkpoint.sha256:
            raise BackfillError(f"checkpoint hash mismatch: {checkpoint.partition_key}")
        if pq.read_metadata(path).num_rows != checkpoint.rows:
            raise BackfillError(f"checkpoint row-count mismatch: {checkpoint.partition_key}")
        frame = pd.read_parquet(path)
        if hash_dataframe(frame, length=64) != checkpoint.content_sha256:
            raise BackfillError(f"checkpoint content mismatch: {checkpoint.partition_key}")
        return path

    def _mark_partition_running(self, plan: BackfillPlan, partition: BackfillPartition) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT status, attempts FROM backfill_partitions
                WHERE plan_id = ? AND partition_key = ?
                """,
                [plan.identity, partition.key],
            ).fetchone()
            if row is None or row[0] not in {"planned", "running", "failed"}:
                raise BackfillError(f"invalid partition claim state for {partition.key}")
            attempts = int(row[1]) + 1
            connection.execute(
                """
                UPDATE backfill_partitions
                SET status = 'running', attempts = ?, started_at = ?,
                    failure_code = NULL, failure_message = NULL
                WHERE plan_id = ? AND partition_key = ?
                """,
                [attempts, datetime.now(UTC), plan.identity, partition.key],
            )
        return attempts

    def _mark_partition_completed(self, plan_id: str, checkpoint: _Checkpoint) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE backfill_partitions
                SET status = 'completed', completed_at = ?, relative_path = ?,
                    sha256 = ?, content_sha256 = ?, rows = ?,
                    failure_code = NULL, failure_message = NULL
                WHERE plan_id = ? AND partition_key = ? AND status = 'running'
                """,
                [
                    datetime.now(UTC),
                    checkpoint.relative_path,
                    checkpoint.sha256,
                    checkpoint.content_sha256,
                    checkpoint.rows,
                    plan_id,
                    checkpoint.partition_key,
                ],
            )

    def _mark_partition_failed(self, plan_id: str, key: str, error: Exception) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE backfill_partitions
                SET status = 'failed', completed_at = ?,
                    failure_code = ?, failure_message = ?
                WHERE plan_id = ? AND partition_key = ?
                """,
                [
                    datetime.now(UTC),
                    type(error).__name__,
                    _redacted_failure(error),
                    plan_id,
                    key,
                ],
            )

    def _transition_job(
        self,
        plan_id: str,
        allowed: set[str],
        target: Literal["running", "assembling", "interrupted", "failed"],
    ) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM backfill_jobs WHERE plan_id = ?",
                [plan_id],
            ).fetchone()
            if row is None or str(row[0]) not in allowed:
                raise BackfillError(
                    f"invalid job transition {row[0] if row else None!r} -> {target!r}"
                )
            connection.execute(
                """
                UPDATE backfill_jobs
                SET status = ?, updated_at = ?
                WHERE plan_id = ?
                """,
                [target, datetime.now(UTC), plan_id],
            )

    def _fail_job(self, plan_id: str, error: Exception) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE backfill_jobs
                SET status = 'failed', updated_at = ?,
                    failure_code = ?, failure_message = ?
                WHERE plan_id = ?
                """,
                [
                    datetime.now(UTC),
                    type(error).__name__,
                    _redacted_failure(error),
                    plan_id,
                ],
            )

    def _publish_job(self, plan_id: str, object_id: str) -> None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM backfill_jobs WHERE plan_id = ?",
                [plan_id],
            ).fetchone()
            if row is None or row[0] != "assembling":
                raise BackfillError("only an assembling job may publish")
            connection.execute(
                """
                UPDATE backfill_jobs
                SET status = 'published', object_id = ?, updated_at = ?,
                    failure_code = NULL, failure_message = NULL
                WHERE plan_id = ?
                """,
                [object_id, datetime.now(UTC), plan_id],
            )

    def _initialize_catalog(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS backfill_jobs (
                    plan_id VARCHAR PRIMARY KEY,
                    request_id VARCHAR NOT NULL,
                    plan_json VARCHAR NOT NULL,
                    status VARCHAR NOT NULL CHECK (
                        status IN ('planned', 'running', 'assembling',
                                   'interrupted', 'failed', 'published')
                    ),
                    object_id VARCHAR,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    failure_code VARCHAR,
                    failure_message VARCHAR
                )
                """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS backfill_partitions (
                    plan_id VARCHAR NOT NULL,
                    partition_key VARCHAR NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    status VARCHAR NOT NULL CHECK (
                        status IN ('planned', 'running', 'completed', 'failed')
                    ),
                    attempts INTEGER NOT NULL CHECK (attempts >= 0),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    relative_path VARCHAR,
                    sha256 VARCHAR,
                    content_sha256 VARCHAR,
                    rows BIGINT,
                    failure_code VARCHAR,
                    failure_message VARCHAR,
                    PRIMARY KEY (plan_id, partition_key)
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
            raise BackfillError("backfill locking requires a POSIX runtime")
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _date_partitions(
    start: date,
    end: date,
    partition_by: Literal["year", "month"],
) -> tuple[BackfillPartition, ...]:
    partitions: list[BackfillPartition] = []
    cursor = start
    while cursor <= end:
        if partition_by == "year":
            boundary = date(cursor.year, 12, 31)
            key = f"{cursor.year:04d}"
        else:
            boundary = date(
                cursor.year,
                cursor.month,
                calendar.monthrange(cursor.year, cursor.month)[1],
            )
            key = f"{cursor.year:04d}-{cursor.month:02d}"
        partition_end = min(boundary, end)
        partitions.append(BackfillPartition(key=key, start=cursor, end=partition_end))
        cursor = partition_end.fromordinal(partition_end.toordinal() + 1)
    return tuple(partitions)


def _contained_checkpoint(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".parquet":
        raise BackfillError("checkpoint path is unsafe")
    candidate = root / path
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise BackfillError("checkpoint path escapes its root") from exc
    return candidate


def _redacted_failure(error: Exception) -> str:
    message = _CREDENTIAL_RE.sub(r"\1\2[REDACTED]", str(error).replace("\n", " "))
    return message[:MAX_FAILURE_CHARS]


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
    if sys.platform == "win32":  # pragma: no cover
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
