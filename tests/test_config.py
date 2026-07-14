"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from quant_platform.config import AppConfig, load_config


def test_default_config_is_valid():
    cfg = AppConfig()
    assert cfg.project.seed == 42
    assert cfg.data.benchmark in cfg.data.tickers


def test_tickers_uppercased_and_deduplicated():
    cfg = AppConfig.model_validate(
        {"data": {"tickers": ["spy", "SPY", "aapl"], "benchmark": "spy"}}
    )
    assert cfg.data.tickers == ["SPY", "AAPL"]
    assert cfg.data.benchmark == "SPY"


def test_benchmark_added_to_tickers_if_missing():
    cfg = AppConfig.model_validate({"data": {"tickers": ["AAPL", "MSFT"], "benchmark": "SPY"}})
    assert "SPY" in cfg.data.tickers


def test_invalid_top_quantile_rejected():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"backtest": {"top_quantile": 0.9}})


def test_invalid_var_confidence_rejected():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"risk": {"var_confidence": 1.5}})


def test_embargo_must_cover_forward_horizon():
    with pytest.raises(ValidationError, match="embargo"):
        AppConfig.model_validate({"model": {"forward_horizon": 5, "cv": {"embargo": 4}}})


def test_task_and_target_must_be_consistent():
    with pytest.raises(ValidationError, match="forward_return"):
        AppConfig.model_validate({"model": {"task": "regression", "target": "direction"}})


def test_close_signal_execution_lag_is_conservative():
    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        AppConfig.model_validate({"backtest": {"execution_lag": 1}})


def test_readiness_aum_must_be_positive():
    with pytest.raises(ValidationError, match="greater than 0"):
        AppConfig.model_validate({"evaluation": {"readiness_aum_usd": 0}})


def test_capacity_aum_grid_must_be_positive():
    with pytest.raises(ValidationError, match="aum_grid_usd"):
        AppConfig.model_validate({"evaluation": {"aum_grid_usd": [0, 1_000_000]}})


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"data": {"not_a_real_key": 1}})


def test_from_yaml_roundtrip(tmp_path):
    cfg = AppConfig()
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    loaded = load_config(p)
    assert loaded.project.name == cfg.project.name
    assert loaded.data.tickers == cfg.data.tickers


def test_env_override_seed(tmp_path, monkeypatch):
    p = tmp_path / "cfg.yaml"
    with p.open("w") as fh:
        yaml.safe_dump({"project": {"seed": 1}}, fh)
    monkeypatch.setenv("QRDP_SEED", "123")
    cfg = AppConfig.from_yaml(p)
    assert cfg.project.seed == 123


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        AppConfig.from_yaml("does/not/exist.yaml")
