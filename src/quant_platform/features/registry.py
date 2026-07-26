"""Versioned feature definitions and leakage-safe fitted-state contracts.

The registry is intentionally metadata-only: it describes feature semantics
without dynamically importing or executing code from a manifest.  This keeps
materialization identities auditable and prevents a stored definition from
becoming an executable deserialization boundary.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections.abc import Iterable
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_platform.config import FeatureConfig
from quant_platform.features import technical
from quant_platform.features.pipeline import build_features

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
FEATURE_NAME_RE = re.compile(r"^f_[a-z0-9_]+$")

FeatureParameter = str | int | float | bool | None


def canonical_json(value: object) -> bytes:
    """Serialize a JSON-compatible value with deterministic UTF-8 encoding."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def semantic_hash(value: object) -> str:
    """Return the full SHA-256 identity of a canonical JSON-compatible value."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


class _Contract(BaseModel):
    """Strict immutable base for persisted feature contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FittedTransformState(_Contract):
    """Identity and temporal boundary of one learned preprocessing state."""

    method: str = Field(min_length=1, max_length=80)
    state_sha256: str
    fit_start: date
    fit_end: date
    sample_count: int = Field(ge=1)

    @field_validator("state_sha256")
    @classmethod
    def _full_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("state_sha256 must be a lowercase full SHA-256 digest")
        return value

    @model_validator(mode="after")
    def _ordered_interval(self) -> FittedTransformState:
        if self.fit_end < self.fit_start:
            raise ValueError("fit_end must be on or after fit_start")
        return self


class FeatureSpec(_Contract):
    """Complete semantic definition of one materialized feature column."""

    name: str
    version: str = "1.0.0"
    family: str = Field(min_length=1, max_length=80)
    input_columns: tuple[str, ...] = Field(min_length=1)
    parameters: dict[str, FeatureParameter] = Field(default_factory=dict)
    lookback_bars: int = Field(ge=0, le=1_000_000)
    warmup_bars: int = Field(ge=0, le=1_000_000)
    output_dtype: Literal["float64", "float32", "int64", "bool"] = "float64"
    normalization: Literal["none", "rolling", "cross_sectional", "train_fitted"] = "none"
    missing_policy: Literal["drop", "preserve", "fail"] = "drop"
    sampling_frequency: str = Field(min_length=1, max_length=32)
    leakage_risk: Literal["low", "medium", "high"]
    implementation_sha256: str
    fitted_state: FittedTransformState | None = None

    @field_validator("name")
    @classmethod
    def _feature_name(cls, value: str) -> str:
        if not FEATURE_NAME_RE.fullmatch(value):
            raise ValueError("feature names must match ^f_[a-z0-9_]+$")
        return value

    @field_validator("version")
    @classmethod
    def _semantic_version(cls, value: str) -> str:
        if not SEMVER_RE.fullmatch(value):
            raise ValueError("version must use MAJOR.MINOR.PATCH numeric form")
        return value

    @field_validator("implementation_sha256")
    @classmethod
    def _implementation_hash(cls, value: str) -> str:
        if not SHA256_RE.fullmatch(value):
            raise ValueError("implementation_sha256 must be a lowercase full SHA-256 digest")
        return value

    @field_validator("input_columns")
    @classmethod
    def _unique_inputs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or not value.replace("_", "").isalnum() for value in values):
            raise ValueError("input columns must be non-empty alphanumeric identifiers")
        if len(set(values)) != len(values):
            raise ValueError("input columns must be unique")
        return values

    @field_validator("parameters")
    @classmethod
    def _finite_parameters(cls, values: dict[str, FeatureParameter]) -> dict[str, FeatureParameter]:
        for key, value in values.items():
            if not key or not key.replace("_", "").isalnum():
                raise ValueError("parameter names must be alphanumeric identifiers")
            if any(token in key.casefold() for token in ("secret", "token", "password", "api_key")):
                raise ValueError("credential-shaped parameters are forbidden")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"parameter {key!r} must be finite")
        return values

    @model_validator(mode="after")
    def _fitted_boundary(self) -> FeatureSpec:
        if self.normalization == "train_fitted" and self.fitted_state is None:
            raise ValueError("train_fitted features require fitted_state")
        if self.normalization != "train_fitted" and self.fitted_state is not None:
            raise ValueError("fitted_state is only valid for train_fitted normalization")
        if self.warmup_bars < self.lookback_bars:
            raise ValueError("warmup_bars must be greater than or equal to lookback_bars")
        return self


class FeatureRegistry:
    """Deterministically indexed collection of immutable feature definitions."""

    def __init__(self, specs: Iterable[FeatureSpec]) -> None:
        ordered = tuple(sorted(specs, key=lambda spec: (spec.name, spec.version)))
        if not ordered:
            raise ValueError("feature registry must contain at least one feature")
        names = [spec.name for spec in ordered]
        if len(set(names)) != len(names):
            raise ValueError("feature output names must be unique within a registry")
        self._specs = ordered

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        """Return definitions in deterministic output-name order."""
        return self._specs

    @property
    def output_columns(self) -> tuple[str, ...]:
        """Return the exact feature columns governed by this registry."""
        return tuple(spec.name for spec in self._specs)

    @property
    def identity(self) -> str:
        """Return the semantic SHA-256 identity of the registry."""
        return semantic_hash(self.to_payload())

    def to_payload(self) -> list[dict[str, object]]:
        """Return the canonical JSON-compatible registry payload."""
        return [spec.model_dump(mode="json") for spec in self._specs]


def feature_pipeline_implementation_hash() -> str:
    """Hash the public pipeline and technical-transform implementations."""
    source = "\n".join(
        (
            inspect.getsource(build_features),
            inspect.getsource(technical),
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def conventional_feature_registry(
    config: FeatureConfig,
    *,
    price_field: str = "adj_close",
) -> FeatureRegistry:
    """Describe every column emitted by the current conventional pipeline.

    These features are trailing or same-date cross-sectional transforms and do
    not learn state from a future application interval.  Train-fitted
    transformations use :class:`FittedTransformState` through separately
    registered definitions.
    """
    implementation = feature_pipeline_implementation_hash()
    specs: list[FeatureSpec] = []

    def add(
        name: str,
        *,
        family: str,
        inputs: tuple[str, ...],
        lookback: int,
        parameters: dict[str, FeatureParameter] | None = None,
        normalization: Literal["none", "rolling", "cross_sectional"] = "rolling",
        risk: Literal["low", "medium", "high"] = "medium",
    ) -> None:
        specs.append(
            FeatureSpec(
                name=name,
                family=family,
                input_columns=inputs,
                parameters=parameters or {},
                lookback_bars=lookback,
                warmup_bars=lookback,
                normalization=normalization,
                missing_policy="drop" if config.dropna else "preserve",
                sampling_frequency="1d",
                leakage_risk=risk,
                implementation_sha256=implementation,
            )
        )

    add(
        "f_ret_1d",
        family="returns",
        inputs=("return",),
        lookback=1,
        normalization="none",
        risk="low",
    )
    add(
        "f_logret_1d",
        family="returns",
        inputs=("log_return",),
        lookback=1,
        normalization="none",
        risk="low",
    )
    add(
        "f_ret_5d",
        family="returns",
        inputs=(price_field,),
        lookback=5,
        parameters={"window": 5},
    )
    for window in config.vol_windows:
        add(
            f"f_vol_{window}",
            family="volatility",
            inputs=("return",),
            lookback=window,
            parameters={"window": window},
        )
    add(
        f"f_realized_vol_{config.realized_vol_window}",
        family="volatility",
        inputs=("return",),
        lookback=config.realized_vol_window,
        parameters={"window": config.realized_vol_window},
    )
    add(
        f"f_sharpe_{config.sharpe_window}",
        family="risk_adjusted_return",
        inputs=("return",),
        lookback=config.sharpe_window,
        parameters={"window": config.sharpe_window},
    )
    for window in config.ma_windows:
        add(
            f"f_ma_ratio_{window}",
            family="trend",
            inputs=(price_field,),
            lookback=window,
            parameters={"window": window},
        )
    for window in config.ema_windows:
        add(
            f"f_ema_ratio_{window}",
            family="trend",
            inputs=(price_field,),
            lookback=window,
            parameters={"span": window},
        )
    add(
        f"f_rsi_{config.rsi_window}",
        family="oscillator",
        inputs=(price_field,),
        lookback=config.rsi_window,
        parameters={"window": config.rsi_window},
    )
    macd_lookback = max(config.macd.fast, config.macd.slow) + config.macd.signal
    for output in ("f_macd", "f_macd_signal", "f_macd_hist"):
        add(
            output,
            family="oscillator",
            inputs=(price_field,),
            lookback=macd_lookback,
            parameters={
                "fast": config.macd.fast,
                "slow": config.macd.slow,
                "signal": config.macd.signal,
            },
        )
    for output in ("f_bb_pctb", "f_bb_bandwidth"):
        add(
            output,
            family="volatility",
            inputs=(price_field,),
            lookback=config.bollinger.window,
            parameters={
                "window": config.bollinger.window,
                "n_std": config.bollinger.n_std,
            },
        )
    for window in config.momentum_windows:
        add(
            f"f_mom_{window}",
            family="momentum",
            inputs=(price_field,),
            lookback=window,
            parameters={"window": window},
        )
    add(
        "f_mom_12_1",
        family="momentum",
        inputs=(price_field,),
        lookback=252,
        parameters={"window": 252, "skip": 21},
    )
    add(
        "f_reversal_5",
        family="mean_reversion",
        inputs=(price_field,),
        lookback=5,
        parameters={"window": 5},
    )
    add(
        "f_zscore_21",
        family="mean_reversion",
        inputs=(price_field,),
        lookback=21,
        parameters={"window": 21},
    )
    for output in ("f_log_dollar_volume", "f_volume_zscore", "f_relative_volume"):
        add(
            output,
            family="liquidity",
            inputs=("volume", price_field),
            lookback=config.realized_vol_window,
            parameters={"window": config.realized_vol_window},
        )
    add(
        f"f_beta_{config.beta_window}",
        family="dependence",
        inputs=("return",),
        lookback=config.beta_window,
        parameters={"window": config.beta_window},
        risk="high",
    )
    add(
        "f_drawdown",
        family="drawdown",
        inputs=(price_field,),
        # Running peak is an expanding statistic.  The sentinel upper bound
        # forces partition loaders to include all available prior history.
        lookback=1_000_000,
        parameters={"expanding": True},
    )
    add(
        f"f_max_dd_{config.drawdown_window}",
        family="drawdown",
        inputs=(price_field,),
        lookback=config.drawdown_window,
        parameters={"window": config.drawdown_window},
    )
    if config.cross_sectional:
        bases = (
            ("mom_12_1", 252),
            ("reversal_5", 5),
            (f"vol_{config.vol_windows[-1]}", config.vol_windows[-1]),
            (f"rsi_{config.rsi_window}", config.rsi_window),
        )
        for base, lookback in bases:
            for kind in ("rank", "z"):
                add(
                    f"f_cs_{kind}_{base}",
                    family="cross_sectional",
                    inputs=(f"f_{base}",),
                    lookback=lookback,
                    parameters={"method": kind},
                    normalization="cross_sectional",
                    risk="high",
                )
    return FeatureRegistry(specs)
