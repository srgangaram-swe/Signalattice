"""End-to-end pipeline and CLI smoke tests (fully offline / synthetic)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from quant_platform.cli import app
from quant_platform.config import AppConfig
from quant_platform.data.signal_foundry_contract import load_signal_foundry_bundle
from quant_platform.pipeline import Pipeline

runner = CliRunner()


@pytest.fixture
def small_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"name": "itest", "seed": 7},
            "data": {
                "source": "synthetic",
                "tickers": ["SPY", "AAA", "BBB", "CCC", "DDD", "EEE"],
                "benchmark": "SPY",
                "start": "2018-01-01",
                "synthetic": {"n_days": 800, "start": "2018-01-01"},
                "min_observations": 100,
            },
            "model": {
                "task": "classification",
                "type": "logistic",
                "cv": {"n_splits": 2, "test_size": 60, "min_train_size": 200, "embargo": 2},
            },
            "backtest": {"strategy": "long_short", "signal": "model", "top_quantile": 0.34},
            "tracking": {"backend": "json"},
        }
    )


def test_full_pipeline_produces_artifacts(tmp_path, small_config):
    # Point all outputs inside tmp_path.
    small_config.data.raw_dir = str(tmp_path / "data/raw")
    small_config.data.processed_dir = str(tmp_path / "data/processed")
    small_config.report.output_dir = str(tmp_path / "reports")
    small_config.report.figures_dir = str(tmp_path / "reports/figures")
    small_config.tracking.json_dir = str(tmp_path / "experiments/runs")

    pipe = Pipeline(small_config, base_dir=str(tmp_path))
    art = pipe.run_full()

    assert art.report_path is not None and Path(art.report_path).exists()
    assert art.backtest is not None
    assert len(art.figures) >= 5
    for p in art.figures.values():
        assert Path(p).exists()
    # Experiment recorded.
    runs = list((tmp_path / "experiments/runs").glob("*.json"))
    assert len(runs) == 1


def test_baseline_signal_pipeline(tmp_path, small_config):
    small_config.backtest.signal = "momentum"
    small_config.data.raw_dir = str(tmp_path / "data/raw")
    small_config.data.processed_dir = str(tmp_path / "data/processed")
    small_config.report.output_dir = str(tmp_path / "reports")
    small_config.report.figures_dir = str(tmp_path / "reports/figures")

    pipe = Pipeline(small_config, base_dir=str(tmp_path))
    bt = pipe.backtest()
    assert bt.equity_curve.iloc[-1] > 0
    assert pipe.art.train_result is None  # no model trained for baseline


def test_feature_cache_is_keyed_by_feature_contract(tmp_path, small_config):
    small_config.data.raw_dir = str(tmp_path / "data/raw")
    small_config.data.processed_dir = str(tmp_path / "data/processed")
    pipe = Pipeline(small_config, base_dir=str(tmp_path))
    first = pipe.build_features()
    first_manifest = json.loads(pipe.features_manifest_path.read_text(encoding="utf-8"))

    small_config.features.rsi_window = 10
    pipe.art.features = None
    second = pipe.build_features()
    second_manifest = json.loads(pipe.features_manifest_path.read_text(encoding="utf-8"))

    assert first_manifest["fingerprint"] != second_manifest["fingerprint"]
    assert "f_rsi_14" in first
    assert "f_rsi_10" in second


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "signalattice" in result.stdout


def test_cli_run_full_pipeline(tmp_path):
    cfg = AppConfig.model_validate(
        {
            "project": {"name": "cli_itest", "seed": 7},
            "data": {
                "source": "synthetic",
                "tickers": ["SPY", "AAA", "BBB", "CCC", "DDD"],
                "benchmark": "SPY",
                "synthetic": {"n_days": 700},
                "min_observations": 100,
                "raw_dir": str(tmp_path / "data/raw"),
                "processed_dir": str(tmp_path / "data/processed"),
            },
            "model": {"type": "logistic"},
            "backtest": {"signal": "momentum"},
            "tracking": {"backend": "none"},
            "report": {
                "output_dir": str(tmp_path / "reports"),
                "figures_dir": str(tmp_path / "reports/figures"),
            },
        }
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)
    # Run from tmp_path so relative model/experiment paths land in the sandbox.
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["run-full-pipeline", "--config", str(cfg_path)])
    finally:
        os.chdir(cwd)
    assert result.exit_code == 0, result.stdout
    assert "SUMMARY" in result.stdout


def test_cli_exports_and_validates_signal_foundry_bundle(tmp_path, monkeypatch):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    effective = pd.to_datetime(dates, utc=True) + pd.Timedelta(hours=21)
    panel = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["SPY", "SPY"],
            "open": [99.0, 100.0],
            "high": [101.0, 102.0],
            "low": [98.0, 99.0],
            "close": [100.0, 101.0],
            "adj_close": [99.5, 100.5],
            "volume": [1_000_000.0, 1_100_000.0],
            "effective_at": effective,
            "available_at": effective + pd.Timedelta(hours=8),
            "observed_at": pd.Timestamp("2026-07-23T00:00:00Z"),
            "provider_updated_at": pd.Timestamp("2026-07-20T00:00:00Z"),
            "instrument_id": ["SPY", "SPY"],
            "currency": ["USD", "USD"],
            "exchange_calendar": ["XNYS", "XNYS"],
            "adjustment_state": ["synthetic_fixture", "synthetic_fixture"],
            "source": ["nasdaq_data_link", "nasdaq_data_link"],
            "source_table": ["SHARADAR/SEP", "SHARADAR/SEP"],
        }
    )
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    source_manifest = {
        "provider": "nasdaq_data_link",
        "request": {"table": "SHARADAR/SEP"},
        "request_hash": "a" * 64,
        "snapshot_hash": "b" * 64,
        "retrieved_at": "2026-07-23T00:00:00Z",
        "contains_api_key": False,
        "observations_redistributable": True,
        "point_in_time_limits": {
            "historical_revisions_complete": False,
            "universe_membership_point_in_time": False,
            "corporate_actions_complete": False,
        },
    }
    (processed_dir / "panel_metadata.json").write_text(
        json.dumps({"source": "nasdaq_data_link", "source_manifest": source_manifest})
    )
    config = AppConfig.model_validate(
        {
            "data": {
                "source": "nasdaq_data_link",
                "tickers": ["SPY"],
                "benchmark": "SPY",
                "processed_dir": str(processed_dir),
                "min_observations": 1,
            }
        }
    )
    config_path = tmp_path / "config.yaml"
    config.to_yaml(config_path)
    monkeypatch.setattr(Pipeline, "ingest", lambda self, force=False: panel)
    output_root = tmp_path / "bundles"

    exported = runner.invoke(
        app,
        [
            "export-signal-foundry-bundle",
            "--config",
            str(config_path),
            "--output",
            str(output_root),
        ],
    )

    assert exported.exit_code == 0, exported.stdout
    bundle = next(path for path in output_root.iterdir() if path.is_dir())
    assert len(load_signal_foundry_bundle(bundle)) == 2
    validated = runner.invoke(app, ["validate-signal-foundry-bundle", str(bundle)])
    assert validated.exit_code == 0, validated.stdout
    assert "Verified" in validated.stdout
