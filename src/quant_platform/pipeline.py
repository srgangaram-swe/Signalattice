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
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.backtest.engine import BacktestResult, run_backtest
from quant_platform.config import AppConfig
from quant_platform.data.ingest import ingest
from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.evaluation import (
    break_even_cost_bps,
    cost_sensitivity,
    execution_delay_sensitivity,
    liquidity_capacity_table,
    readiness_gate,
    warm_inference_benchmark,
)
from quant_platform.features.backfill import BackfillOrchestrator, BackfillPartition, BackfillPlan
from quant_platform.features.integration import build_pipeline_materialization_request
from quant_platform.features.pipeline import build_features, feature_columns
from quant_platform.features.spectral import build_spectral_features
from quant_platform.features.store import FeatureStore
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
    decision_analysis: dict[str, Any] = field(default_factory=dict)
    data_hash: str | None = None
    feature_materialization_id: str | None = None


class Pipeline:
    """Config-driven orchestrator for the full research workflow."""

    def __init__(self, config: AppConfig, *, base_dir: str | None = None) -> None:
        self.config = config
        self.base_dir = base_dir
        self.art = PipelineArtifacts()
        self._feature_store: FeatureStore | None = None
        set_global_seed(config.project.seed)

    # -- path helpers --------------------------------------------------------
    @property
    def processed_dir(self) -> Path:
        return ensure_dir(resolve_path(self.config.data.processed_dir, self.base_dir))

    @property
    def features_path(self) -> Path:
        """Return the immutable object path, or the configured store root."""
        if self.art.feature_materialization_id is None:
            return self.feature_store.root
        return self.feature_store.objects_dir / self.art.feature_materialization_id

    @property
    def features_manifest_path(self) -> Path:
        """Return the current verified materialization manifest path."""
        if self.art.feature_materialization_id is None:
            raise RuntimeError("no feature materialization has been selected")
        return self.features_path / "manifest.json"

    @property
    def feature_store(self) -> FeatureStore:
        """Return the bounded local store configured for this pipeline."""
        if self._feature_store is None:
            store_config = self.config.feature_store
            self._feature_store = FeatureStore(
                resolve_path(store_config.root_dir, self.base_dir),
                max_query_rows=store_config.max_query_rows,
                max_query_columns=store_config.max_query_columns,
            )
        return self._feature_store

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
        panel = ingest(
            self.config.data,
            base_dir=self.base_dir,
            force=force,
            seed=self.config.project.seed,
        )
        self.art.panel = panel
        self.art.data_hash = hash_dataframe(panel)
        return panel

    def _ensure_panel(self) -> pd.DataFrame:
        if self.art.panel is None:
            # Route every disk load through ingestion so the config fingerprint
            # is checked; existence alone is not evidence that a cache is valid.
            self.art.panel = self.ingest()
            if self.art.data_hash is None:
                self.art.data_hash = hash_dataframe(self.art.panel)
        return self.art.panel

    def build_features(self, *, force: bool = False) -> pd.DataFrame:
        logger.info("=== Stage: build-features ===")
        panel = self._ensure_panel()
        metadata: dict[str, Any] = {}
        metadata_path = self.processed_dir / "panel_metadata.json"
        if metadata_path.exists():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata = loaded
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                logger.warning("Ignoring unreadable panel metadata at %s", metadata_path)
        request = build_pipeline_materialization_request(self.config, panel, metadata)
        if self.config.feature_store.enabled and not force:
            existing = self.feature_store.lookup(request)
            if existing is not None:
                logger.info(
                    "Loading verified content-addressed features %s",
                    existing.object_id,
                )
                feats = self.feature_store.read(existing.object_id)
                feats[DATE_COL] = pd.to_datetime(feats[DATE_COL])
                self.art.features = feats
                self.art.feature_materialization_id = existing.object_id
                return feats
        feats = build_features(
            panel,
            self.config.features,
            benchmark=self.config.data.benchmark,
            price_field=self.config.data.price_field,
            forward_horizon=self.config.model.forward_horizon,
        )
        feats = self._attach_spectral_features(panel, feats)
        if self.config.feature_store.enabled:
            registered = tuple(feature.name for feature in request.features)
            produced = tuple(sorted(feature_columns(feats)))
            if produced != registered:
                missing = sorted(set(registered).difference(produced))
                undeclared = sorted(set(produced).difference(registered))
                raise RuntimeError(
                    "feature registry/output mismatch: "
                    f"missing={missing}, undeclared={undeclared}"
                )
            manifest = self.feature_store.materialize(request, feats)
            feats = self.feature_store.read(manifest.object_id)
            feats[DATE_COL] = pd.to_datetime(feats[DATE_COL])
            self.art.feature_materialization_id = manifest.object_id
            logger.info(
                "Published immutable feature materialization %s (%d rows)",
                manifest.object_id,
                manifest.rows,
            )
        else:
            logger.warning("Feature-store persistence is explicitly disabled")
        self.art.features = feats
        return feats

    def _attach_spectral_features(
        self, panel: pd.DataFrame, features: pd.DataFrame
    ) -> pd.DataFrame:
        """Join opt-in causal spectral features onto the conventional matrix.

        Spectral descriptors are computed from the **full** panel rather than
        the already-filtered feature frame, because their causal windows need
        the early history that ``build_features`` drops as conventional warm-up.
        The join is then an explicit ``(date, ticker)`` merge validated as
        one-to-one, so a duplicated observation surfaces as an error instead of
        quietly multiplying rows.
        """
        spectral_config = self.config.features.spectral
        if not spectral_config.enabled:
            return features
        spectral = build_spectral_features(
            panel,
            spectral_config,
            benchmark=self.config.data.benchmark,
        )
        keyed = pd.concat([panel[[DATE_COL, TICKER_COL]], spectral], axis=1)
        merged = features.merge(
            keyed,
            on=[DATE_COL, TICKER_COL],
            how="left",
            validate="one_to_one",
        )
        if self.config.features.dropna:
            merged = merged.dropna(subset=list(spectral.columns)).reset_index(drop=True)
        logger.info(
            "Attached %d causal spectral features (%d -> %d rows after spectral warm-up)",
            len(spectral.columns),
            len(features),
            len(merged),
        )
        return merged

    def _ensure_features(self) -> pd.DataFrame:
        if self.art.features is None:
            self.build_features()
        assert self.art.features is not None
        return self.art.features

    def backfill_features(self) -> pd.DataFrame:
        """Materialize feature partitions through the resumable state machine.

        Each partition receives only its declared application interval plus the
        trailing warm-up rows and forward-label horizon it needs.  Completed
        checkpoints are verified and reused after interruption.
        """
        logger.info("=== Stage: backfill-features ===")
        panel = self._ensure_panel()
        metadata: dict[str, Any] = {}
        metadata_path = self.processed_dir / "panel_metadata.json"
        if metadata_path.exists():
            try:
                loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    metadata = loaded
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                logger.warning("Ignoring unreadable panel metadata at %s", metadata_path)
        request = build_pipeline_materialization_request(self.config, panel, metadata)
        store_config = self.config.feature_store
        plan = BackfillPlan.create(
            request,
            max_workers=store_config.max_workers,
            max_attempts=store_config.max_attempts,
            max_rows_per_partition=store_config.max_rows_per_partition,
        )
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel[DATE_COL]).unique()))
        warmup = max(feature.warmup_bars for feature in request.features)
        horizon = self.config.model.forward_horizon

        def load_partition(partition: BackfillPartition) -> pd.DataFrame:
            start_index = int(dates.searchsorted(pd.Timestamp(partition.start), side="left"))
            end_index = int(dates.searchsorted(pd.Timestamp(partition.end), side="right"))
            warmup_start = dates[max(0, start_index - warmup)]
            future_end = dates[min(len(dates) - 1, end_index - 1 + horizon)]
            subset = panel.loc[
                (pd.to_datetime(panel[DATE_COL]) >= warmup_start)
                & (pd.to_datetime(panel[DATE_COL]) <= future_end)
            ].copy()
            features = build_features(
                subset,
                self.config.features,
                benchmark=self.config.data.benchmark,
                price_field=self.config.data.price_field,
                forward_horizon=horizon,
            )
            feature_dates = pd.to_datetime(features[DATE_COL]).dt.date
            return features.loc[
                (feature_dates >= partition.start) & (feature_dates <= partition.end)
            ].copy()

        result = BackfillOrchestrator(self.feature_store).run(plan, load_partition)
        features = self.feature_store.read(result.object_id)
        features[DATE_COL] = pd.to_datetime(features[DATE_COL])
        self.art.features = features
        self.art.feature_materialization_id = result.object_id
        logger.info(
            "Backfill %s published %s (reused=%d, computed=%d)",
            result.plan_id,
            result.object_id,
            result.reused_partitions,
            result.computed_partitions,
        )
        return features

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

    def _inference_benchmark(self) -> pd.DataFrame:
        train_result = self.art.train_result
        if train_result is None:
            return pd.DataFrame()
        features = self._ensure_features()
        model = train_result.model
        if hasattr(model, "estimator") and hasattr(model, "scaler"):
            from quant_platform.models.torch_lstm import build_panel_sequences

            scaled = model.scaler.transform(features[train_result.feature_names])
            batch = build_panel_sequences(
                scaled,
                features[DATE_COL].to_numpy(),
                features[TICKER_COL].to_numpy(),
                sequence_length=model.sequence_length,
            )
            inputs = batch.X[-min(512, len(batch.X)) :]
            predict = (
                model.estimator.predict_proba
                if self.config.model.task == "classification"
                else model.estimator.predict
            )
        else:
            inputs = features[train_result.feature_names].to_numpy(dtype=float)[-512:]
            predict = (
                model.predict_proba
                if self.config.model.task == "classification" and hasattr(model, "predict_proba")
                else model.predict
            )
        sizes = sorted({1, min(64, len(inputs)), min(512, len(inputs))})
        return warm_inference_benchmark(
            predict,
            inputs,
            sizes,
            warmup_runs=3,
            measured_runs=15,
        )

    def evaluate_decision_readiness(self) -> dict[str, Any]:
        """Quantify implementation headroom and return an explicit go/no-go gate."""
        panel = self._ensure_panel()
        signals = self.art.signals if self.art.signals is not None else self.compute_signals()
        backtest = self.art.backtest if self.art.backtest is not None else self.backtest()
        evaluation = self.config.evaluation
        cost_table = cost_sensitivity(
            signals,
            panel,
            self.config.backtest,
            evaluation.cost_grid_bps,
            benchmark=self.config.data.benchmark,
            risk_config=self.config.risk,
        )
        delay_table = execution_delay_sensitivity(
            signals,
            panel,
            self.config.backtest,
            evaluation.delay_bars,
            benchmark=self.config.data.benchmark,
            risk_config=self.config.risk,
        )
        try:
            break_even_bps = break_even_cost_bps(
                backtest.gross_returns,
                backtest.turnover,
            )
        except ValueError:
            break_even_bps = float("nan")
        capacity_aums = sorted({*evaluation.aum_grid_usd, evaluation.readiness_aum_usd})
        capacity = liquidity_capacity_table(
            panel,
            backtest.weights,
            capacity_aums,
            participation_limit=evaluation.max_participation_rate,
        )
        latency = self._inference_benchmark()

        train_result = self.art.train_result
        if train_result is None:
            ece = float("nan")
            roc_auc = float("nan")
            positive_fold_fraction = float("nan")
        else:
            ece = train_result.metrics.get("expected_calibration_error", float("nan"))
            roc_auc = train_result.metrics.get("roc_auc", float("nan"))
            fold_key = (
                "roc_auc" if train_result.task == "classification" else "directional_accuracy"
            )
            fold_values = [
                float(metrics[fold_key])
                for metrics in train_result.fold_metrics
                if fold_key in metrics and math.isfinite(float(metrics[fold_key]))
            ]
            positive_fold_fraction = (
                float(sum(value > 0.5 for value in fold_values) / len(fold_values))
                if fold_values
                else float("nan")
            )
        latency_p95 = (
            float(latency.iloc[-1]["p95_latency_ms"]) if not latency.empty else float("nan")
        )
        readiness_capacity = capacity.loc[np.isclose(capacity["aum"], evaluation.readiness_aum_usd)]
        capacity_p95 = (
            float(readiness_capacity.iloc[0]["p95_participation_rate"])
            if not readiness_capacity.empty
            else float("nan")
        )
        assumed_cost = self.config.backtest.cost_bps + self.config.backtest.slippage_bps
        gate = readiness_gate(
            expected_calibration_error=ece,
            break_even_one_way_cost_bps=break_even_bps,
            positive_fold_fraction=positive_fold_fraction,
            p95_latency_ms=latency_p95,
            roc_auc=roc_auc,
            net_sharpe=float(backtest.stats.get("sharpe", float("nan"))),
            p95_participation_rate=capacity_p95,
            max_calibration_error=evaluation.max_ece,
            min_roc_auc=evaluation.min_roc_auc,
            min_net_sharpe=evaluation.min_net_sharpe,
            min_break_even_cost_bps=assumed_cost * evaluation.min_cost_multiple,
            min_positive_fold_fraction=evaluation.min_positive_fold_fraction,
            max_p95_latency_ms=evaluation.latency_budget_ms,
            max_p95_participation_rate=evaluation.max_participation_rate,
        )
        result = {
            "cost_sensitivity": cost_table,
            "delay_sensitivity": delay_table,
            "capacity": capacity,
            "inference_latency": latency,
            "break_even_one_way_cost_bps": break_even_bps,
            "readiness_gate": gate,
            "assumed_one_way_cost_bps": assumed_cost,
            "readiness_aum_usd": evaluation.readiness_aum_usd,
        }
        self.art.decision_analysis = result
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
        beta_value = bt.stats.get("beta", float("nan"))
        scenario_beta = beta_value if math.isfinite(beta_value) else 1.0
        stress_df = stress_test(
            bt.returns,
            self.config.risk.stress_scenarios,
            beta_to_market=scenario_beta,
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
        decision = (
            self.art.decision_analysis
            if self.art.decision_analysis
            else self.evaluate_decision_readiness()
        )

        returns_wide = panel.pivot_table(index=DATE_COL, columns=TICKER_COL, values="return")
        corr = correlation_matrix(returns_wide)

        figures = generate_all_figures(
            panel=panel,
            backtest=bt,
            train_result=self.art.train_result,
            correlation=corr,
            figures_dir=self.figures_dir,
            task=self.config.model.task,
            decision_analysis=decision,
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
            "synthetic": bool(disk_meta.get("synthetic", False)),
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
            decision_analysis=decision,
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
            self.evaluate_decision_readiness()
            self.report()

            features = self._ensure_features()
            panel = self.art.panel
            assert panel is not None
            ctx.set_dataset(
                data_hash=self.art.data_hash or "",
                tickers=sorted(panel[TICKER_COL].unique().tolist()),
                features=feature_columns(features),
            )
            if self.art.feature_materialization_id is not None:
                ctx.log_tags({"feature_materialization_id": self.art.feature_materialization_id})
            ctx.log_params(self.config.to_dict())
            if self.art.train_result is not None:
                ctx.log_metrics({f"model_{k}": v for k, v in self.art.train_result.metrics.items()})
            bt = self.art.backtest
            assert bt is not None
            ctx.log_metrics({f"bt_{k}": v for k, v in bt.stats.items()})
            ctx.log_metrics({f"bench_{k}": v for k, v in bt.benchmark_stats.items()})
            gate = self.art.decision_analysis.get("readiness_gate", {})
            ctx.log_metrics(
                {
                    "readiness_passed": float(bool(gate.get("overall_pass", False))),
                    "break_even_cost_bps": self.art.decision_analysis.get(
                        "break_even_one_way_cost_bps", float("nan")
                    ),
                }
            )
            if self.art.report_path:
                ctx.log_artifact(self.art.report_path)
            for p in self.art.figures.values():
                ctx.log_artifact(p)
        logger.info("=== FULL pipeline complete ===")
        return self.art


def load_pipeline(config_path: str, *, base_dir: str | None = None) -> Pipeline:
    """Build a :class:`Pipeline` from a YAML config path."""
    return Pipeline(AppConfig.from_yaml(config_path), base_dir=base_dir)
