"""End-to-end pipeline orchestration.

The :class:`Pipeline` ties every module together and caches intermediate
artifacts (Parquet + joblib) so each CLI sub-command can run independently while
``run_full`` executes the whole chain in one process and wraps it in an
experiment-tracking run.

Stage dependency graph::

    ingest -> features -> [train] -> signals -> backtest -> report
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.backtest.engine import BacktestResult, run_backtest
from quant_platform.config import AppConfig
from quant_platform.data.ingest import ingest, load_processed
from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.features.pipeline import build_features, feature_columns
from quant_platform.logging_utils import get_logger
from quant_platform.models.baseline import baseline_signal
from quant_platform.models.train import TrainResult, save_model, walk_forward_train
from quant_platform.reporting.plots import generate_all_figures
from quant_platform.reporting.report import build_report
from quant_platform.risk.analytics import correlation_matrix, stress_test
from quant_platform.tracking import get_tracker
from quant_platform.utils import (
    ensure_dir,
    git_commit_hash,
    hash_dataframe,
    resolve_path,
    set_global_seed,
)

logger = get_logger(__name__)

FEATURES_FILENAME = "features.parquet"


@dataclass
class PipelineArtifacts:
    """In-memory handles to the artifacts produced by a run."""

    panel: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    train_result: TrainResult | None = None
    signals: pd.DataFrame | None = None
    backtest: BacktestResult | None = None
    report_path: Path | None = None
    figures: dict[str, Path] = field(default_factory=dict)
    data_hash: str | None = None


class Pipeline:
    """Config-driven orchestrator for the full research workflow."""

    def __init__(self, config: AppConfig, *, base_dir: str | None = None) -> None:
        self.config = config
        self.base_dir = base_dir
        self.art = PipelineArtifacts()
        set_global_seed(config.project.seed)

    # -- path helpers --------------------------------------------------------
    @property
    def processed_dir(self) -> Path:
        return ensure_dir(resolve_path(self.config.data.processed_dir, self.base_dir))

    @property
    def features_path(self) -> Path:
        return self.processed_dir / FEATURES_FILENAME

    @property
    def model_path(self) -> Path:
        models_dir = ensure_dir(resolve_path("models", self.base_dir))
        return models_dir / f"{self.config.project.name}_model.joblib"

    @property
    def reports_dir(self) -> Path:
        return ensure_dir(resolve_path(self.config.report.output_dir, self.base_dir))

    @property
    def figures_dir(self) -> Path:
        return ensure_dir(resolve_path(self.config.report.figures_dir, self.base_dir))

    # -- stages --------------------------------------------------------------
    def ingest(self, *, force: bool = False) -> pd.DataFrame:
        logger.info("=== Stage: ingest-data ===")
        panel = ingest(self.config.data, base_dir=self.base_dir, force=force)
        self.art.panel = panel
        self.art.data_hash = hash_dataframe(panel)
        return panel

    def _ensure_panel(self) -> pd.DataFrame:
        if self.art.panel is None:
            try:
                self.art.panel = load_processed(self.processed_dir)
            except FileNotFoundError:
                self.art.panel = self.ingest()
            if self.art.data_hash is None:
                self.art.data_hash = hash_dataframe(self.art.panel)
        return self.art.panel

    def build_features(self, *, force: bool = False) -> pd.DataFrame:
        logger.info("=== Stage: build-features ===")
        if self.features_path.exists() and not force:
            logger.info("Loading cached features from %s", self.features_path)
            feats = pd.read_parquet(self.features_path)
            feats[DATE_COL] = pd.to_datetime(feats[DATE_COL])
            self.art.features = feats
            return feats
        panel = self._ensure_panel()
        feats = build_features(
            panel,
            self.config.features,
            benchmark=self.config.data.benchmark,
            price_field=self.config.data.price_field,
            forward_horizon=self.config.model.forward_horizon,
        )
        feats.to_parquet(self.features_path, index=False)
        logger.info("Saved features to %s", self.features_path)
        self.art.features = feats
        return feats

    def _ensure_features(self) -> pd.DataFrame:
        if self.art.features is None:
            self.build_features()
        assert self.art.features is not None
        return self.art.features

    def train(self) -> TrainResult:
        logger.info("=== Stage: train-model ===")
        features = self._ensure_features()
        result = walk_forward_train(features, self.config.model, seed=self.config.project.seed)
        save_model(result, self.model_path)
        self.art.train_result = result
        return result

    def compute_signals(self) -> pd.DataFrame:
        """Produce the signal frame the backtester consumes.

        For ``signal='model'`` these are out-of-sample walk-forward predictions;
        for baselines they are rule-based scores over the full feature window.
        """
        logger.info("=== Stage: signals (%s) ===", self.config.backtest.signal)
        features = self._ensure_features()
        if self.config.backtest.signal == "model":
            if self.art.train_result is None:
                self.train()
            assert self.art.train_result is not None
            signals = self.art.train_result.predictions[[DATE_COL, TICKER_COL, "score"]].copy()
        else:
            score = baseline_signal(features, self.config.backtest.signal)
            signals = features[[DATE_COL, TICKER_COL]].copy()
            signals["score"] = score.to_numpy()
            signals = signals.dropna(subset=["score"])
        self.art.signals = signals
        return signals

    def backtest(self) -> BacktestResult:
        logger.info("=== Stage: run-backtest ===")
        panel = self._ensure_panel()
        signals = self.art.signals if self.art.signals is not None else self.compute_signals()
        result = run_backtest(
            signals,
            panel,
            self.config.backtest,
            benchmark=self.config.data.benchmark,
            risk_config=self.config.risk,
        )
        self.art.backtest = result
        return result

    def _risk_block(self) -> dict[str, Any]:
        bt = self.art.backtest
        assert bt is not None
        portfolio = {
            "ann_volatility": bt.stats["ann_volatility"],
            "sharpe": bt.stats["sharpe"],
            "sortino": bt.stats["sortino"],
            "max_drawdown": bt.stats["max_drawdown"],
            "var_95": bt.stats["var_95"],
            "cvar_95": bt.stats["cvar_95"],
            "beta": bt.stats.get("beta", float("nan")),
            "avg_gross_exposure": bt.trade_summary["avg_gross_exposure"],
            "avg_net_exposure": bt.trade_summary["avg_net_exposure"],
            "annual_turnover": bt.stats["annual_turnover"],
        }
        stress_df = stress_test(
            bt.returns,
            self.config.risk.stress_scenarios,
            beta_to_market=bt.stats.get("beta", 1.0) or 1.0,
            confidence=self.config.risk.var_confidence,
        )
        return {"portfolio": portfolio, "stress": stress_df.to_dict("records")}

    def report(self) -> Path:
        logger.info("=== Stage: generate-report ===")
        panel = self._ensure_panel()
        features = self._ensure_features()
        if self.art.backtest is None:
            self.backtest()
        bt = self.art.backtest
        assert bt is not None

        returns_wide = panel.pivot_table(index=DATE_COL, columns=TICKER_COL, values="return")
        corr = correlation_matrix(returns_wide)

        figures = generate_all_figures(
            panel=panel,
            backtest=bt,
            train_result=self.art.train_result,
            correlation=corr,
            figures_dir=self.figures_dir,
            task=self.config.model.task,
        )
        self.art.figures = figures

        disk_meta: dict[str, Any] = {}
        metadata_path = self.processed_dir / "panel_metadata.json"
        if metadata_path.exists():
            try:
                disk_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Could not parse panel metadata sidecar at %s", metadata_path)

        panel_meta = {
            "source": disk_meta.get("source", "cached"),
            "tickers": sorted(panel[TICKER_COL].unique().tolist()),
            "n_rows": int(len(panel)),
            "date_min": str(panel[DATE_COL].min().date()),
            "date_max": str(panel[DATE_COL].max().date()),
            "data_hash": self.art.data_hash or disk_meta.get("data_hash"),
        }
        report_path = build_report(
            self.config,
            panel_meta=panel_meta,
            feature_names=feature_columns(features),
            train_result=self.art.train_result,
            backtest_result=bt,
            risk_block=self._risk_block(),
            figures=figures,
            output_path=self.reports_dir / f"{self.config.project.name}_report.md",
            data_hash=self.art.data_hash,
            git_commit=git_commit_hash(),
        )
        self.art.report_path = report_path
        return report_path

    # -- full run ------------------------------------------------------------
    def run_full(self, *, force: bool = False) -> PipelineArtifacts:
        logger.info("=== Running FULL pipeline: %s ===", self.config.project.name)
        tracker = get_tracker(self.config.tracking, base_dir=self.base_dir)
        with tracker.run(self.config.project.name) as ctx:
            self.ingest(force=force)
            self.build_features(force=force)
            if self.config.backtest.signal == "model":
                self.train()
            self.compute_signals()
            self.backtest()
            self.report()

            features = self._ensure_features()
            ctx.set_dataset(
                data_hash=self.art.data_hash or "",
                tickers=sorted(self.art.panel[TICKER_COL].unique().tolist()),
                features=feature_columns(features),
            )
            ctx.log_params(self.config.to_dict())
            if self.art.train_result is not None:
                ctx.log_metrics({f"model_{k}": v for k, v in self.art.train_result.metrics.items()})
            bt = self.art.backtest
            assert bt is not None
            ctx.log_metrics({f"bt_{k}": v for k, v in bt.stats.items()})
            ctx.log_metrics({f"bench_{k}": v for k, v in bt.benchmark_stats.items()})
            if self.art.report_path:
                ctx.log_artifact(self.art.report_path)
            for p in self.art.figures.values():
                ctx.log_artifact(p)
        logger.info("=== FULL pipeline complete ===")
        return self.art


def load_pipeline(config_path: str, *, base_dir: str | None = None) -> Pipeline:
    """Build a :class:`Pipeline` from a YAML config path."""
    return Pipeline(AppConfig.from_yaml(config_path), base_dir=base_dir)
