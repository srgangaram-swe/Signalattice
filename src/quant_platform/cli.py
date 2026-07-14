"""Command-line interface (Typer).

Sub-commands map to pipeline stages::

    quant-platform ingest-data       --config configs/example.yaml
    quant-platform build-features    --config configs/example.yaml
    quant-platform train-model       --config configs/example.yaml
    quant-platform run-backtest      --config configs/example.yaml
    quant-platform generate-report   --config configs/example.yaml
    quant-platform run-full-pipeline --config configs/example.yaml
    quant-platform list-experiments  --config configs/example.yaml

Also available after `pip install -e .` as the console script ``quant-platform``,
or via ``python -m quant_platform.cli ...``.
"""

from __future__ import annotations

import typer

from quant_platform import __version__
from quant_platform.config import AppConfig
from quant_platform.logging_utils import configure_logging, get_logger
from quant_platform.pipeline import Pipeline

app = typer.Typer(
    name="quant-platform",
    help="Signalattice — probabilistic forecasts, decision evidence, and research reports.",
    add_completion=False,
    no_args_is_help=True,
)
logger = get_logger(__name__)

ConfigOpt = typer.Option(
    "configs/example.yaml",
    "--config",
    "-c",
    help="Path to a YAML config file.",
    exists=False,
)
LogLevelOpt = typer.Option(None, "--log-level", help="Override log level (DEBUG/INFO/...).")
ForceOpt = typer.Option(False, "--force", help="Recompute and overwrite cached artifacts.")


def _load(config: str, log_level: str | None) -> Pipeline:
    configure_logging(log_level, force=log_level is not None)
    cfg = AppConfig.from_yaml(config)
    return Pipeline(cfg)


@app.callback()
def _main() -> None:
    """Signalattice command-line interface."""
    configure_logging()


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"signalattice {__version__}")


@app.command("ingest-data")
def ingest_data(
    config: str = ConfigOpt, log_level: str | None = LogLevelOpt, force: bool = ForceOpt
) -> None:
    """Download/ingest market data, validate it and persist a Parquet panel."""
    pipe = _load(config, log_level)
    panel = pipe.ingest(force=force)
    typer.echo(
        f"Ingested {len(panel):,} rows for {panel['ticker'].nunique()} tickers "
        f"→ {pipe.processed_dir}"
    )


@app.command("build-features")
def build_features_cmd(
    config: str = ConfigOpt, log_level: str | None = LogLevelOpt, force: bool = ForceOpt
) -> None:
    """Engineer time-series & cross-sectional features and persist them."""
    pipe = _load(config, log_level)
    feats = pipe.build_features(force=force)
    from quant_platform.features.pipeline import feature_columns

    typer.echo(
        f"Built {len(feature_columns(feats))} features, {len(feats):,} rows → {pipe.features_path}"
    )


@app.command("train-model")
def train_model(config: str = ConfigOpt, log_level: str | None = LogLevelOpt) -> None:
    """Train the model with walk-forward CV and report out-of-sample metrics."""
    pipe = _load(config, log_level)
    result = pipe.train()
    typer.echo("Out-of-sample metrics:")
    for k, v in result.metrics.items():
        typer.echo(f"  {k:<22} {v:.4f}")
    typer.echo(f"Model saved → {pipe.model_path}")


@app.command("run-backtest")
def run_backtest_cmd(config: str = ConfigOpt, log_level: str | None = LogLevelOpt) -> None:
    """Run the vectorized backtest and print headline performance stats."""
    pipe = _load(config, log_level)
    result = pipe.backtest()
    typer.echo("Backtest stats (strategy vs benchmark):")
    keys = ["cagr", "ann_volatility", "sharpe", "sortino", "max_drawdown", "var_95"]
    for k in keys:
        s = result.stats.get(k, float("nan"))
        b = result.benchmark_stats.get(k, float("nan"))
        typer.echo(f"  {k:<16} strat={s: .4f}  bench={b: .4f}")


@app.command("generate-report")
def generate_report(config: str = ConfigOpt, log_level: str | None = LogLevelOpt) -> None:
    """Generate figures and a Markdown/HTML run report."""
    pipe = _load(config, log_level)
    path = pipe.report()
    typer.echo(f"Report written → {path}")
    typer.echo(f"Figures        → {pipe.figures_dir}")


@app.command("run-full-pipeline")
def run_full_pipeline(
    config: str = ConfigOpt, log_level: str | None = LogLevelOpt, force: bool = ForceOpt
) -> None:
    """Run the complete pipeline end-to-end and track the experiment."""
    pipe = _load(config, log_level)
    art = pipe.run_full(force=force)
    bt = art.backtest
    assert bt is not None
    typer.echo("\n========== SUMMARY ==========")
    typer.echo(
        f"Strategy Sharpe : {bt.stats['sharpe']:.2f}  (benchmark {bt.benchmark_stats['sharpe']:.2f})"
    )
    typer.echo(f"Strategy CAGR   : {bt.stats['cagr'] * 100:.2f}%")
    typer.echo(f"Max drawdown    : {bt.stats['max_drawdown'] * 100:.2f}%")
    typer.echo(f"Report          : {art.report_path}")
    typer.echo(f"Figures         : {pipe.figures_dir} ({len(art.figures)} plots)")


@app.command("list-experiments")
def list_experiments(
    config: str = ConfigOpt, log_level: str | None = LogLevelOpt, limit: int = 10
) -> None:
    """List recorded experiment runs from the tracking backend."""
    from quant_platform.tracking import get_tracker

    cfg = AppConfig.from_yaml(config)
    configure_logging(log_level, force=log_level is not None)
    tracker = get_tracker(cfg.tracking)
    runs = tracker.list_runs()[:limit]
    if not runs:
        typer.echo("No experiment runs recorded yet.")
        raise typer.Exit()
    for r in runs:
        metrics = r.get("metrics", {}) or {}
        sharpe = metrics.get("bt_sharpe", float("nan"))
        typer.echo(
            f"{r.get('started_at', '')[:19]}  {r.get('name', ''):<16} "
            f"status={r.get('status', '')}  bt_sharpe={sharpe}"
        )


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
