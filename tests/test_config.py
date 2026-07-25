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


def test_nasdaq_data_link_config_is_strict_and_has_no_secret_field():
    cfg = AppConfig.model_validate(
        {
            "data": {
                "source": "nasdaq_data_link",
                "nasdaq_data_link": {
                    "table": "sharadar/sep",
                    "requests_per_minute": 12,
                    "max_requests": 20,
                },
            }
        }
    )
    assert cfg.data.nasdaq_data_link.table == "SHARADAR/SEP"
    assert "api_key" not in cfg.data.nasdaq_data_link.model_dump()


def test_nasdaq_time_series_config_has_explicit_market_semantics():
    cfg = AppConfig.model_validate(
        {
            "data": {
                "source": "nasdaq_data_link",
                "nasdaq_data_link": {
                    "api_kind": "time_series",
                    "table": "xdus",
                    "adjustment": "unadjusted",
                    "currency": "eur",
                    "exchange_calendar": "xdus",
                    "market_close_utc_hour": 16,
                },
            }
        }
    )

    assert cfg.data.nasdaq_data_link.table == "XDUS"
    assert cfg.data.nasdaq_data_link.currency == "EUR"
    assert cfg.data.nasdaq_data_link.exchange_calendar == "XDUS"
    assert cfg.data.nasdaq_data_link.adjustment == "unadjusted"


@pytest.mark.parametrize(
    ("path", "expected_tickers", "expected_benchmark"),
    [
        ("configs/nasdaq_smoke.yaml", ["AAPL"], "AAPL"),
        (
            "configs/nasdaq_data_link.yaml",
            ["AAPL", "MSFT", "NVDA", "JPM", "XOM"],
            "AAPL",
        ),
    ],
)
def test_committed_sep_profiles_use_equity_universes(path, expected_tickers, expected_benchmark):
    cfg = load_config(path)

    assert cfg.data.source == "nasdaq_data_link"
    assert cfg.data.nasdaq_data_link.table == "SHARADAR/SEP"
    assert cfg.data.tickers == expected_tickers
    assert cfg.data.benchmark == expected_benchmark
    assert cfg.data.allow_synthetic_fallback is False


def test_committed_xdus_profile_is_bounded_engineering_data():
    cfg = load_config("configs/nasdaq_xdus_sample.yaml")

    assert cfg.data.source == "nasdaq_data_link"
    assert cfg.data.nasdaq_data_link.api_kind == "time_series"
    assert cfg.data.nasdaq_data_link.table == "XDUS"
    assert cfg.data.nasdaq_data_link.currency == "EUR"
    assert cfg.data.nasdaq_data_link.exchange_calendar == "XDUS"
    assert cfg.data.nasdaq_data_link.max_requests == len(cfg.data.tickers) == 5
    assert cfg.data.nasdaq_data_link.max_retries == 0
    assert cfg.data.end == "2018-11-30"
    assert cfg.data.allow_synthetic_fallback is False


@pytest.mark.parametrize(
    "nasdaq_config",
    [
        {"table": "not-a-table"},
        {"api_kind": "time_series", "table": "SHARADAR/SEP"},
        {"api_kind": "tables", "table": "XDUS"},
        {"api_kind": "tables", "adjustment": "unadjusted"},
        {"currency": "US"},
        {"exchange_calendar": "TOO-LONG"},
        {"cache_dir": "../escape"},
        {"requests_per_minute": 0},
        {"max_requests": 0},
        {"page_size": 10_001},
        {"cache_mode": "fallback_to_network"},
        {"api_key": "must-not-be-configurable"},
    ],
)
def test_invalid_nasdaq_data_link_config_is_rejected(nasdaq_config):
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"data": {"nasdaq_data_link": nasdaq_config}})


def test_invalid_data_date_range_is_rejected():
    with pytest.raises(ValidationError, match="on or after"):
        AppConfig.model_validate({"data": {"start": "2025-01-01", "end": "2024-01-01"}})
