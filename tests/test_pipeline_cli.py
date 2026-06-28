"""End-to-end pipeline and CLI smoke tests (fully offline / synthetic)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_platform.cli import app
from quant_platform.config import AppConfig
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


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "quant-research-data-platform" in result.stdout


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
    import os

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["run-full-pipeline", "--config", str(cfg_path)])
    finally:
        os.chdir(cwd)
    assert result.exit_code == 0, result.stdout
    assert "SUMMARY" in result.stdout
