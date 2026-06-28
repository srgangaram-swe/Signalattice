"""Experiment tracker implementations and the public :func:`get_tracker` factory."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_platform.config import TrackingConfig
from quant_platform.logging_utils import get_logger
from quant_platform.utils import ensure_dir, git_commit_hash, resolve_path

logger = get_logger(__name__)


@dataclass
class RunContext:
    """Accumulates everything logged during a single experiment run."""

    run_id: str
    experiment: str
    name: str
    started_at: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    git_commit: str | None = None
    data_hash: str | None = None
    tickers: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    status: str = "running"

    # -- logging helpers (return self for chaining) --------------------------
    def log_params(self, params: dict[str, Any]) -> RunContext:
        self.params.update(_flatten(params))
        return self

    def log_metrics(self, metrics: dict[str, Any]) -> RunContext:
        for k, v in metrics.items():
            try:
                self.metrics[k] = float(v)
            except (TypeError, ValueError):
                self.tags[k] = v
        return self

    def log_tags(self, tags: dict[str, Any]) -> RunContext:
        self.tags.update(tags)
        return self

    def log_artifact(self, path: str | Path) -> RunContext:
        self.artifacts.append(str(path))
        return self

    def set_dataset(self, *, data_hash: str, tickers: list[str], features: list[str]) -> RunContext:
        self.data_hash = data_hash
        self.tickers = list(tickers)
        self.features = list(features)
        return self

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "status": self.status,
            "git_commit": self.git_commit,
            "data_hash": self.data_hash,
            "tickers": self.tickers,
            "features": self.features,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "artifacts": self.artifacts,
        }


class ExperimentTracker:
    """Base tracker interface (also serves as the no-op tracker)."""

    backend = "none"

    def __init__(self, config: TrackingConfig, *, base_dir: str | None = None) -> None:
        self.config = config
        self.base_dir = base_dir

    @contextmanager
    def run(self, name: str) -> Iterator[RunContext]:
        ctx = RunContext(
            run_id=uuid.uuid4().hex[:12],
            experiment=self.config.experiment_name,
            name=name,
            started_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git_commit_hash(),
        )
        logger.info("[%s] started run '%s' (id=%s)", self.backend, name, ctx.run_id)
        try:
            yield ctx
            ctx.status = "completed"
        except Exception:
            ctx.status = "failed"
            raise
        finally:
            self._persist(ctx)
            logger.info("[%s] finished run '%s' status=%s", self.backend, name, ctx.status)

    def _persist(self, ctx: RunContext) -> None:  # pragma: no cover - no-op base
        pass

    def list_runs(self) -> list[dict[str, Any]]:
        return []


class JSONTracker(ExperimentTracker):
    """Persist each run as a JSON document under ``json_dir``."""

    backend = "json"

    def _dir(self) -> Path:
        return ensure_dir(resolve_path(self.config.json_dir, self.base_dir))

    def _persist(self, ctx: RunContext) -> None:
        path = self._dir() / f"{ctx.started_at[:10]}_{ctx.name}_{ctx.run_id}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(ctx.to_record(), fh, indent=2, default=str)
        logger.debug("Wrote experiment JSON to %s", path)

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for p in sorted(self._dir().glob("*.json")):
            with p.open(encoding="utf-8") as fh:
                runs.append(json.load(fh))
        return runs


class SQLiteTracker(ExperimentTracker):
    """Persist runs into a single SQLite database (default backend)."""

    backend = "sqlite"

    def _db_path(self) -> Path:
        path = resolve_path(self.config.db_path, self.base_dir)
        ensure_dir(path.parent)
        return path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path())
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment TEXT,
                name TEXT,
                started_at TEXT,
                ended_at TEXT,
                status TEXT,
                git_commit TEXT,
                data_hash TEXT,
                tickers TEXT,
                features TEXT,
                params TEXT,
                metrics TEXT,
                tags TEXT,
                artifacts TEXT
            )
            """)
        return conn

    def _persist(self, ctx: RunContext) -> None:
        rec = ctx.to_record()
        conn = self._connect()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, experiment, name, started_at, ended_at, status, git_commit,
                 data_hash, tickers, features, params, metrics, tags, artifacts)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rec["run_id"],
                    rec["experiment"],
                    rec["name"],
                    rec["started_at"],
                    rec["ended_at"],
                    rec["status"],
                    rec["git_commit"],
                    rec["data_hash"],
                    json.dumps(rec["tickers"]),
                    json.dumps(rec["features"]),
                    json.dumps(rec["params"], default=str),
                    json.dumps(rec["metrics"], default=str),
                    json.dumps(rec["tags"], default=str),
                    json.dumps(rec["artifacts"], default=str),
                ),
            )
        conn.close()
        logger.debug("Inserted run %s into %s", ctx.run_id, self._db_path())

    def list_runs(self) -> list[dict[str, Any]]:
        conn = self._connect()
        cur = conn.execute("SELECT * FROM runs ORDER BY started_at DESC")
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
        conn.close()
        for r in rows:
            for jcol in ("tickers", "features", "params", "metrics", "tags", "artifacts"):
                if r.get(jcol):
                    r[jcol] = json.loads(r[jcol])
        return rows


class MLflowTracker(ExperimentTracker):
    """Optional MLflow-backed tracker."""

    backend = "mlflow"

    @contextmanager
    def run(self, name: str) -> Iterator[RunContext]:  # pragma: no cover - optional dep
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflow backend requested but mlflow is not installed "
                "(`pip install '.[mlflow]'`)."
            ) from exc
        if self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        ctx = RunContext(
            run_id=uuid.uuid4().hex[:12],
            experiment=self.config.experiment_name,
            name=name,
            started_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git_commit_hash(),
        )
        with mlflow.start_run(run_name=name):
            try:
                yield ctx
                ctx.status = "completed"
            except Exception:
                ctx.status = "failed"
                raise
            finally:
                mlflow.log_params(ctx.params)
                mlflow.log_metrics(dict(ctx.metrics))
                mlflow.set_tags(
                    {
                        **ctx.tags,
                        "git_commit": ctx.git_commit or "",
                        "data_hash": ctx.data_hash or "",
                    }
                )
                for art in ctx.artifacts:
                    if Path(art).exists():
                        mlflow.log_artifact(art)


def get_tracker(config: TrackingConfig, *, base_dir: str | None = None) -> ExperimentTracker:
    """Factory returning the configured tracker backend."""
    backend = config.backend
    if backend == "sqlite":
        return SQLiteTracker(config, base_dir=base_dir)
    if backend == "json":
        return JSONTracker(config, base_dir=base_dir)
    if backend == "mlflow":
        return MLflowTracker(config, base_dir=base_dir)
    return ExperimentTracker(config, base_dir=base_dir)


def _flatten(d: dict[str, Any], parent: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dict into dotted keys (for tabular param storage)."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            items.update(_flatten(v, key, sep))
        elif isinstance(v, (list, tuple)):
            items[key] = json.dumps(list(v), default=str)
        else:
            items[key] = v
    return items
