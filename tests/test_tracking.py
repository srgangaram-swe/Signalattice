"""Tests for experiment tracking backends."""

from __future__ import annotations

from quant_platform.config import TrackingConfig
from quant_platform.tracking import get_tracker


def test_sqlite_tracker_roundtrip(tmp_path):
    cfg = TrackingConfig(
        backend="sqlite",
        experiment_name="t",
        db_path=str(tmp_path / "exp.sqlite"),
    )
    tracker = get_tracker(cfg, base_dir=str(tmp_path))
    with tracker.run("run1") as ctx:
        ctx.log_params({"model": {"type": "rf", "depth": 5}})
        ctx.log_metrics({"sharpe": 1.23, "cagr": 0.1})
        ctx.set_dataset(data_hash="abc123", tickers=["SPY"], features=["f_x"])
    runs = tracker.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r["status"] == "completed"
    assert r["metrics"]["sharpe"] == 1.23
    assert r["data_hash"] == "abc123"
    assert r["params"]["model.type"] == "rf"  # flattened


def test_json_tracker_writes_file(tmp_path):
    cfg = TrackingConfig(backend="json", experiment_name="t", json_dir=str(tmp_path / "runs"))
    tracker = get_tracker(cfg, base_dir=str(tmp_path))
    with tracker.run("r") as ctx:
        ctx.log_metrics({"x": 1.0})
    files = list((tmp_path / "runs").glob("*.json"))
    assert len(files) == 1


def test_none_tracker_is_noop(tmp_path):
    cfg = TrackingConfig(backend="none")
    tracker = get_tracker(cfg, base_dir=str(tmp_path))
    with tracker.run("r") as ctx:
        ctx.log_metrics({"x": 1.0})
    assert tracker.list_runs() == []


def test_tracker_records_failure(tmp_path):
    cfg = TrackingConfig(backend="sqlite", db_path=str(tmp_path / "exp.sqlite"))
    tracker = get_tracker(cfg, base_dir=str(tmp_path))
    try:
        with tracker.run("boom"):
            raise RuntimeError("kaboom")
    except RuntimeError:
        pass
    runs = tracker.list_runs()
    assert runs[0]["status"] == "failed"
