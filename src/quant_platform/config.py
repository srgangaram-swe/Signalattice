"""Typed, validated configuration for the platform.

The entire pipeline is config-driven. A single YAML file is parsed into a tree
of :class:`pydantic.BaseModel` objects so that:

* invalid configs fail fast with clear error messages,
* defaults are explicit and documented,
* downstream code gets autocompletion and type-checking, and
* runs are fully reproducible from a committed config file.

Environment variables prefixed ``SIGNALATTICE_`` can override a small set of
runtime settings; the legacy ``QRDP_`` aliases remain supported for existing
local configurations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
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
    market_autocorrelation: float = Field(default=0.10, gt=-0.95, lt=0.95)
    idiosyncratic_autocorrelation: float = Field(default=-0.05, gt=-0.95, lt=0.95)


class NasdaqDataLinkConfig(_Base):
    """Bounded Nasdaq Data Link table-API configuration.

    The API credential is deliberately absent. Runtime code resolves only the
    ``NASDAQ_DATA_LINK_API_KEY`` environment variable so serialized
    configurations and run manifests cannot contain the secret.
    """

    table: str = "SHARADAR/SEP"
    cache_mode: Literal["prefer_cache", "network", "cache_only"] = "prefer_cache"
    cache_dir: str = "data/vendor/nasdaq-data-link"
    requests_per_minute: int = Field(default=30, ge=1, le=300)
    max_requests: int = Field(default=100, ge=1, le=10_000)
    page_size: int = Field(default=10_000, ge=1, le=10_000)
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    max_retries: int = Field(default=3, ge=0, le=8)
    retry_backoff_seconds: float = Field(default=1.0, gt=0.0, le=60.0)
    availability_lag_hours: int = Field(default=8, ge=1, le=168)

    @field_validator("table")
    @classmethod
    def _validate_table(cls, value: str) -> str:
        normalized = value.strip().upper()
        parts = normalized.split("/")
        if len(parts) != 2 or not all(part.replace("_", "").isalnum() for part in parts):
            raise ValueError("`data.nasdaq_data_link.table` must be DATABASE/TABLE")
        return normalized

    @field_validator("cache_dir")
    @classmethod
    def _safe_cache_dir(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("`data.nasdaq_data_link.cache_dir` must be a safe relative path")
        return value


class DataConfig(_Base):
    """Market-data ingestion configuration."""

    source: Literal["yfinance", "stooq", "nasdaq_data_link", "synthetic", "auto"] = "auto"
    allow_synthetic_fallback: bool = False
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
    nasdaq_data_link: NasdaqDataLinkConfig = Field(default_factory=NasdaqDataLinkConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)

    @model_validator(mode="after")
    def _validate_dates(self) -> DataConfig:
        start = pd.Timestamp(self.start)
        if self.end is not None and pd.Timestamp(self.end) < start:
            raise ValueError("`data.end` must be on or after `data.start`")
        return self

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
    n_splits: int = Field(default=5, ge=1)
    test_size: int = Field(default=252, ge=1)
    min_train_size: int = Field(default=504, ge=20)
    embargo: int = Field(default=5, ge=0)


EnsembleCandidate = Literal["logistic", "random_forest", "gradient_boosting", "xgboost", "lightgbm"]


def _default_ensemble_candidates() -> list[EnsembleCandidate]:
    """Return a freshly allocated, precisely typed default candidate list."""
    return ["logistic", "random_forest", "gradient_boosting"]


class EnsembleConfig(_Base):
    """Leakage-aware probability calibration and ensemble settings."""

    candidates: list[EnsembleCandidate] = Field(default_factory=_default_ensemble_candidates)
    candidate_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    calibration_fraction: float = Field(default=0.20, gt=0.05, lt=0.50)
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    min_calibration_dates: int = Field(default=40, ge=20)

    @field_validator("candidates")
    @classmethod
    def _unique_candidates(cls, values: list[EnsembleCandidate]) -> list[EnsembleCandidate]:
        unique = list(dict.fromkeys(values))
        if len(unique) < 2:
            raise ValueError("`model.ensemble.candidates` must contain at least two models")
        return unique


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
        "tcn",
        "ensemble",
    ] = "random_forest"
    params: dict[str, Any] = Field(default_factory=dict)
    standardize: bool = True
    cv: CVConfig = Field(default_factory=CVConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    feature_blocklist: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_forecast_contract(self) -> ModelConfig:
        expected_target = "direction" if self.task == "classification" else "forward_return"
        if self.target != expected_target:
            raise ValueError(
                f"task='{self.task}' requires target='{expected_target}', got '{self.target}'"
            )
        if self.forward_horizon < 1:
            raise ValueError("`model.forward_horizon` must be >= 1")
        if self.cv.embargo < self.forward_horizon:
            raise ValueError(
                "`model.cv.embargo` must be >= `model.forward_horizon` to separate "
                "overlapping labels from evaluation dates"
            )
        if self.type == "ensemble" and self.task != "classification":
            raise ValueError("the calibrated ensemble currently requires task='classification'")
        return self


class BacktestConfig(_Base):
    """Backtesting configuration."""

    strategy: Literal["long_only", "long_short"] = "long_short"
    signal: Literal["model", "momentum", "ma_crossover"] = "model"
    top_quantile: float = 0.3
    cost_bps: float = Field(default=1.0, ge=0.0)
    slippage_bps: float = Field(default=0.5, ge=0.0)
    initial_capital: float = Field(default=1_000_000.0, gt=0.0)
    max_leverage: float = Field(default=1.0, gt=0.0)
    position_sizing: Literal["equal_weight", "vol_target", "rank"] = "equal_weight"
    signal_selection: Literal["quantile", "probability_threshold"] = "quantile"
    vol_target_annual: float = Field(default=0.10, gt=0.0)
    max_position_weight: float = Field(default=0.25, gt=0.0, le=1.0)
    rebalance_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    long_threshold: float = Field(default=0.55, ge=0.5, le=1.0)
    short_threshold: float = Field(default=0.45, ge=0.0, le=0.5)
    # Close-derived features are observed after close t. A two-row shift means
    # the first held close-to-close interval begins after the next close, which
    # avoids pretending that the strategy filled at an already-observed close.
    execution_lag: int = Field(default=2, ge=2)

    @field_validator("top_quantile")
    @classmethod
    def _check_quantile(cls, v: float) -> float:
        if not 0 < v <= 0.5:
            raise ValueError("`backtest.top_quantile` must be in (0, 0.5]")
        return v

    @model_validator(mode="after")
    def _check_thresholds(self) -> BacktestConfig:
        if self.short_threshold >= self.long_threshold:
            raise ValueError("`short_threshold` must be below `long_threshold`")
        return self


class EvaluationConfig(_Base):
    """Decision-readiness and operational evaluation settings."""

    cost_grid_bps: list[float] = Field(default_factory=lambda: [0.0, 1.5, 3.0, 5.0, 10.0, 25.0])
    delay_bars: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 5])
    aum_grid_usd: list[float] = Field(
        default_factory=lambda: [1_000_000.0, 5_000_000.0, 10_000_000.0, 25_000_000.0]
    )
    readiness_aum_usd: float = Field(default=1_000_000.0, gt=0.0)
    max_participation_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    latency_budget_ms: float = Field(default=1_000.0, gt=0.0)
    min_roc_auc: float = Field(default=0.52, ge=0.5, le=1.0)
    max_ece: float = Field(default=0.05, ge=0.0, le=1.0)
    min_positive_fold_fraction: float = Field(default=0.60, ge=0.0, le=1.0)
    min_net_sharpe: float = 0.50
    min_cost_multiple: float = Field(default=2.0, gt=0.0)

    @field_validator("cost_grid_bps")
    @classmethod
    def _nonnegative_cost_grid(cls, values: list[float]) -> list[float]:
        if not values or any(value < 0 for value in values):
            raise ValueError("`evaluation.cost_grid_bps` must be non-empty and non-negative")
        return sorted(set(values))

    @field_validator("aum_grid_usd")
    @classmethod
    def _positive_aum_grid(cls, values: list[float]) -> list[float]:
        if not values or any(value <= 0 for value in values):
            raise ValueError("`evaluation.aum_grid_usd` must be non-empty and positive")
        return sorted(set(values))

    @field_validator("delay_bars")
    @classmethod
    def _delay_grid(cls, values: list[int]) -> list[int]:
        if not values or any(value < 0 for value in values):
            raise ValueError("`evaluation.delay_bars` values must be >= 0")
        return sorted(set(values))


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
    title: str = "Signalattice — Forecast Decision-Readiness Report"


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
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    # ----- constructors -----------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | os.PathLike, *, apply_env: bool = True) -> AppConfig:
        """Load and validate configuration from a YAML file.

        Selected environment variables override file values when ``apply_env``
        is true:

        * ``SIGNALATTICE_SEED`` -> ``project.seed``
        * ``SIGNALATTICE_TRACKING_BACKEND`` -> ``tracking.backend``
        * ``MLFLOW_TRACKING_URI`` -> ``tracking.mlflow_tracking_uri``
        * ``SIGNALATTICE_DATA_DIR`` -> ``data.raw_dir`` / ``data.processed_dir``
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
        seed = os.getenv("SIGNALATTICE_SEED") or os.getenv("QRDP_SEED")
        if seed:
            raw.setdefault("project", {})
            raw["project"]["seed"] = int(seed)
        backend = os.getenv("SIGNALATTICE_TRACKING_BACKEND") or os.getenv("QRDP_TRACKING_BACKEND")
        if backend:
            raw.setdefault("tracking", {})
            raw["tracking"]["backend"] = backend
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            raw.setdefault("tracking", {})
            raw["tracking"]["mlflow_tracking_uri"] = mlflow_uri
        data_dir = os.getenv("SIGNALATTICE_DATA_DIR") or os.getenv("QRDP_DATA_DIR")
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
