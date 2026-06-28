"""Typed, validated configuration for the platform.

The entire pipeline is config-driven. A single YAML file is parsed into a tree
of :class:`pydantic.BaseModel` objects so that:

* invalid configs fail fast with clear error messages,
* defaults are explicit and documented,
* downstream code gets autocompletion and type-checking, and
* runs are fully reproducible from a committed config file.

Environment variables (prefixed ``QRDP_``) can override a small set of runtime
settings; see :class:`AppConfig.from_yaml`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectConfig(_Base):
    """Top-level run metadata."""

    name: str = "example"
    seed: int = 42
    description: str = ""


class SyntheticConfig(_Base):
    """Parameters for the synthetic data fallback generator."""

    n_days: int = 2500
    start: str = "2015-01-01"
    annual_drift: float = 0.08
    annual_vol: float = 0.20
    market_beta_mean: float = 1.0
    market_vol: float = 0.16


class DataConfig(_Base):
    """Market-data ingestion configuration."""

    source: Literal["yfinance", "stooq", "synthetic", "auto"] = "auto"
    tickers: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "AAPL"])
    benchmark: str = "SPY"
    start: str = "2015-01-01"
    end: str | None = None
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    price_field: Literal["adj_close", "close"] = "adj_close"
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    min_observations: int = 252
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)

    @field_validator("tickers")
    @classmethod
    def _upper_unique(cls, v: list[str]) -> list[str]:
        seen: list[str] = []
        for t in v:
            tt = t.strip().upper()
            if tt and tt not in seen:
                seen.append(tt)
        if not seen:
            raise ValueError("`data.tickers` must contain at least one ticker")
        return seen

    @model_validator(mode="after")
    def _benchmark_present(self) -> DataConfig:
        bench = self.benchmark.strip().upper()
        object.__setattr__(self, "benchmark", bench)
        if bench not in self.tickers:
            # Ensure benchmark is always ingested.
            object.__setattr__(self, "tickers", [*self.tickers, bench])
        return self


class MACDConfig(_Base):
    fast: int = 12
    slow: int = 26
    signal: int = 9


class BollingerConfig(_Base):
    window: int = 20
    n_std: float = 2.0


class FeatureConfig(_Base):
    """Feature-engineering configuration."""

    vol_windows: list[int] = Field(default_factory=lambda: [5, 21, 63])
    ma_windows: list[int] = Field(default_factory=lambda: [10, 20, 50, 200])
    ema_windows: list[int] = Field(default_factory=lambda: [12, 26])
    sharpe_window: int = 63
    rsi_window: int = 14
    macd: MACDConfig = Field(default_factory=MACDConfig)
    bollinger: BollingerConfig = Field(default_factory=BollingerConfig)
    momentum_windows: list[int] = Field(default_factory=lambda: [21, 63, 126, 252])
    beta_window: int = 63
    realized_vol_window: int = 21
    drawdown_window: int = 252
    cross_sectional: bool = True
    dropna: bool = True


class CVConfig(_Base):
    """Time-series cross-validation configuration."""

    scheme: Literal["walk_forward", "expanding"] = "walk_forward"
    n_splits: int = 5
    test_size: int = 252
    min_train_size: int = 504
    embargo: int = 5


class ModelConfig(_Base):
    """Modelling configuration."""

    task: Literal["classification", "regression"] = "classification"
    target: Literal["direction", "forward_return"] = "direction"
    forward_horizon: int = 1
    type: Literal[
        "logistic",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "ridge",
        "lstm",
    ] = "random_forest"
    params: dict[str, Any] = Field(default_factory=dict)
    standardize: bool = True
    cv: CVConfig = Field(default_factory=CVConfig)
    feature_blocklist: list[str] = Field(default_factory=list)


class BacktestConfig(_Base):
    """Backtesting configuration."""

    strategy: Literal["long_only", "long_short"] = "long_short"
    signal: Literal["model", "momentum", "ma_crossover"] = "model"
    top_quantile: float = 0.3
    cost_bps: float = 1.0
    slippage_bps: float = 0.5
    initial_capital: float = 1_000_000.0
    max_leverage: float = 1.0
    position_sizing: Literal["equal_weight", "vol_target", "rank"] = "equal_weight"
    vol_target_annual: float = 0.10
    max_position_weight: float = 0.25
    rebalance_threshold: float = 0.0
    long_threshold: float = 0.55
    short_threshold: float = 0.45

    @field_validator("top_quantile")
    @classmethod
    def _check_quantile(cls, v: float) -> float:
        if not 0 < v <= 0.5:
            raise ValueError("`backtest.top_quantile` must be in (0, 0.5]")
        return v


class RiskConfig(_Base):
    """Risk-analytics configuration."""

    var_confidence: float = 0.95
    trading_days: int = 252
    stress_scenarios: dict[str, float] = Field(
        default_factory=lambda: {
            "equity_-10pct": -0.10,
            "equity_-20pct": -0.20,
            "vol_spike_2x": 2.0,
        }
    )

    @field_validator("var_confidence")
    @classmethod
    def _check_conf(cls, v: float) -> float:
        if not 0.5 < v < 1.0:
            raise ValueError("`risk.var_confidence` must be in (0.5, 1.0)")
        return v


class TrackingConfig(_Base):
    """Experiment-tracking configuration."""

    backend: Literal["sqlite", "json", "mlflow", "none"] = "sqlite"
    experiment_name: str = "default"
    db_path: str = "experiments/experiments.sqlite"
    json_dir: str = "experiments/runs"
    mlflow_tracking_uri: str | None = None


class ReportConfig(_Base):
    """Reporting configuration."""

    output_dir: str = "reports"
    figures_dir: str = "reports/figures"
    format: Literal["markdown", "html"] = "markdown"
    title: str = "Quant Research Platform — Run Report"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class AppConfig(_Base):
    """Root configuration object for an end-to-end run."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    # ----- constructors -----------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | os.PathLike, *, apply_env: bool = True) -> AppConfig:
        """Load and validate configuration from a YAML file.

        Selected environment variables override file values when ``apply_env``
        is true:

        * ``QRDP_SEED`` -> ``project.seed``
        * ``QRDP_TRACKING_BACKEND`` -> ``tracking.backend``
        * ``MLFLOW_TRACKING_URI`` -> ``tracking.mlflow_tracking_uri``
        * ``QRDP_DATA_DIR`` -> ``data.raw_dir`` / ``data.processed_dir``
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
        if apply_env:
            raw = cls._apply_env_overrides(raw)
        return cls.model_validate(raw)

    @staticmethod
    def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
        raw = dict(raw)
        seed = os.getenv("QRDP_SEED")
        if seed:
            raw.setdefault("project", {})
            raw["project"]["seed"] = int(seed)
        backend = os.getenv("QRDP_TRACKING_BACKEND")
        if backend:
            raw.setdefault("tracking", {})
            raw["tracking"]["backend"] = backend
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            raw.setdefault("tracking", {})
            raw["tracking"]["mlflow_tracking_uri"] = mlflow_uri
        data_dir = os.getenv("QRDP_DATA_DIR")
        if data_dir:
            raw.setdefault("data", {})
            raw["data"].setdefault("raw_dir", str(Path(data_dir) / "raw"))
            raw["data"].setdefault("processed_dir", str(Path(data_dir) / "processed"))
        return raw

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` representation (JSON/YAML friendly)."""
        return self.model_dump(mode="json")

    def to_yaml(self, path: str | os.PathLike) -> None:
        """Persist the (resolved) config to a YAML file."""
        with Path(path).open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)


def load_config(path: str | os.PathLike) -> AppConfig:
    """Convenience wrapper around :meth:`AppConfig.from_yaml`."""
    return AppConfig.from_yaml(path)
