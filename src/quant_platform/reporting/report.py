"""Assemble a self-contained Markdown (or HTML) run report."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from quant_platform.config import AppConfig
from quant_platform.logging_utils import get_logger
from quant_platform.utils import ensure_dir

logger = get_logger(__name__)

_DISCLAIMER = (
    "> **Research-use only.** This artifact is not financial advice or an "
    "authorization to trade. Backtests and synthetic experiments are evidence "
    "about software and validation behavior, not proof of live-market alpha."
)

_PCT_KEYS = {
    "cagr",
    "ann_return",
    "ann_volatility",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "hit_rate",
    "alpha_ann",
    "avg_turnover",
    "annual_turnover",
    "best_day",
    "worst_day",
    "pct_positive_days",
    "avg_gross_exposure",
    "avg_net_exposure",
    "expected_calibration_error",
    "liquidity_coverage",
    "median_participation_rate",
    "p95_participation_rate",
    "max_participation_rate",
    "share_trades_above_limit",
}


def _fmt(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float) and (value != value):  # NaN
        return "n/a"
    if key in _PCT_KEYS:
        return f"{value * 100:.2f}%"
    return f"{value:.3f}"


def _metrics_table(
    metrics: dict[str, Any], *, headers: tuple[str, str] = ("Metric", "Value")
) -> str:
    rows = [f"| {k} | {_fmt(k, v)} |" for k, v in metrics.items()]
    head = f"| {headers[0]} | {headers[1]} |\n|---|---|"
    return head + "\n" + "\n".join(rows)


def _compare_table(strat: dict[str, Any], bench: dict[str, Any], keys: list[str]) -> str:
    head = "| Metric | Strategy | Benchmark |\n|---|---|---|"
    rows = [f"| {k} | {_fmt(k, strat.get(k))} | {_fmt(k, bench.get(k))} |" for k in keys]
    return head + "\n" + "\n".join(rows)


def _df_to_md(df: pd.DataFrame, *, floatfmt: str = ".2%", index: bool = True) -> str:
    if df is None or df.empty:
        return "_(no data)_"
    try:
        from tabulate import tabulate

        return str(
            tabulate(
                cast(Any, df),
                headers="keys",
                tablefmt="github",
                floatfmt=floatfmt,
                showindex=index,
            )
        )
    except Exception:  # pragma: no cover - fallback
        return df.to_markdown(index=index)


def build_report(
    config: AppConfig,
    *,
    panel_meta: dict[str, Any],
    feature_names: list[str],
    train_result: Any,
    backtest_result: Any,
    risk_block: dict[str, Any],
    decision_analysis: dict[str, Any],
    figures: dict[str, Path],
    output_path: Path,
    data_hash: str | None = None,
    git_commit: str | None = None,
) -> Path:
    """Render a forecast-quality and decision-readiness research report."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig_dir = config.report.figures_dir

    def fig_link(name: str) -> str:
        p = figures.get(name)
        if p is None:
            return ""
        rel = Path(p)
        try:
            rel = rel.relative_to(output_path.parent)
        except ValueError:
            rel = Path(Path(fig_dir).name) / rel.name
        return f"![{name}]({rel.as_posix()})\n"

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    extra = train_result.extra if train_result is not None else {}
    gate = decision_analysis.get("readiness_gate", {})
    criteria = pd.DataFrame(gate.get("criteria", []))
    verdict = str(gate.get("verdict", "NOT_EVALUATED"))
    passed = int(gate.get("passed_count", 0))
    criterion_count = int(gate.get("criterion_count", 0))
    synthetic = bool(panel_meta.get("synthetic", False))
    lines: list[str] = []
    a = lines.append

    a(f"# {config.report.title}\n")
    a(_DISCLAIMER + "\n")
    a("## 1. Executive decision\n")
    a(
        _metrics_table(
            {
                "Readiness verdict": verdict,
                "Criteria passed": f"{passed}/{criterion_count}",
                "Evidence class": (
                    "declared synthetic engineering experiment"
                    if synthetic
                    else "historical market-data research"
                ),
                "Break-even one-way cost (bps)": decision_analysis.get(
                    "break_even_one_way_cost_bps", float("nan")
                ),
                "Assumed one-way cost (bps)": decision_analysis.get(
                    "assumed_one_way_cost_bps", float("nan")
                ),
            }
        )
        + "\n"
    )
    if verdict != "READY":
        a(
            "**Decision:** do not promote this configuration to capital deployment. "
            "Failed gates identify the experiments required before a shadow-trading "
            "review; they are not averaged away into a composite score.\n"
        )
    else:
        a(
            "**Decision:** the configured research gates passed. This permits only the "
            "next validation stage (for example, shadow evaluation), not live trading.\n"
        )
    a(fig_link("readiness_scorecard"))

    a("## 2. Run provenance\n")
    a(
        _metrics_table(
            {
                "Generated": now,
                "Config / experiment": config.project.name,
                "Random seed": config.project.seed,
                "Data source": panel_meta.get("source", "n/a"),
                "Synthetic data": synthetic,
                "Dataset hash": data_hash or panel_meta.get("data_hash", "n/a"),
                "Git commit": git_commit or "n/a",
                "Requested model": f"{config.model.type} ({config.model.task})",
                "Effective model": extra.get("effective_model", config.model.type),
                "Forecast horizon": f"{config.model.forward_horizon} bar(s)",
                "CV embargo": f"{config.model.cv.embargo} bar(s)",
                "Execution lag": f"{config.backtest.execution_lag} bar(s)",
                "Strategy": f"{config.backtest.strategy} / {config.backtest.signal}",
            }
        )
        + "\n"
    )

    a("## 3. Data and feature contract\n")
    a(
        _metrics_table(
            {
                "Tickers": ", ".join(panel_meta.get("tickers", [])),
                "Benchmark": config.data.benchmark,
                "Date range": f"{panel_meta.get('date_min')} → {panel_meta.get('date_max')}",
                "Observations": panel_meta.get("n_rows", "n/a"),
                "Price field": config.data.price_field,
            }
        )
        + "\n"
    )
    a(fig_link("price_history"))
    a(fig_link("correlation"))
    a(f"- **{len(feature_names)}** features engineered per (date, ticker).\n")
    a(
        "- Families: returns, volatility/realised-vol, moving-average & EMA trend, "
        "RSI/MACD oscillators, Bollinger bands, momentum & 12-1 momentum, mean-reversion "
        "z-scores, volume features, rolling beta, drawdown features, and cross-sectional "
        "ranks/z-scores.\n"
    )
    a("<details><summary>Full feature list</summary>\n\n")
    a("\n".join(f"- `{f}`" for f in feature_names))
    a("\n\n</details>\n")

    a("## 4. Forecast quality (strictly out of sample)\n")
    if train_result is not None:
        a(_metrics_table(train_result.metrics) + "\n")
        if train_result.fold_metrics:
            fold_df = pd.DataFrame(train_result.fold_metrics).set_index("fold")
            a("\n**Per-fold metrics**\n\n")
            a(_df_to_md(fold_df, floatfmt=".3f") + "\n")
        bootstrap = extra.get("accuracy_block_bootstrap_ci")
        if bootstrap:
            a("\n**Date-block bootstrap interval**\n\n")
            a(_metrics_table(bootstrap) + "\n")
        decomposition = extra.get("brier_decomposition")
        if decomposition:
            a("\n**Brier decomposition**\n\n")
            a(_metrics_table(decomposition) + "\n")
        calibration = extra.get("calibration_table")
        if isinstance(calibration, pd.DataFrame):
            a("\n**Reliability table**\n\n")
            a(_df_to_md(calibration, floatfmt=".4f", index=False) + "\n")
        candidate_metrics = extra.get("candidate_metrics")
        if candidate_metrics:
            a("\n**Candidate model comparison**\n\n")
            candidate_df = pd.DataFrame(candidate_metrics).T
            candidate_df.index.name = "candidate"
            a(_df_to_md(candidate_df, floatfmt=".4f") + "\n")
        ensemble_weights = extra.get("ensemble_weights")
        if ensemble_weights:
            a("\n**Final ensemble weights**\n\n")
            a(_metrics_table(ensemble_weights) + "\n")
        a(fig_link("reliability"))
        a(fig_link("score_distribution"))
        a(fig_link("precision_recall"))
        a(fig_link("selective_coverage"))
        a(fig_link("prediction_deciles"))
        a(fig_link("fold_stability"))
        a(fig_link("feature_importance"))
        a(fig_link("feature_stability"))
        a(fig_link("ensemble_weights"))
        if config.model.task == "classification":
            a(fig_link("confusion_matrix"))
            a(fig_link("roc_curve"))
    else:
        a("_Rules-based baseline signal used; no ML model trained._\n")

    a("## 5. Economic translation\n")
    bt = backtest_result
    compare_keys = [
        "cagr",
        "ann_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "hit_rate",
        "beta",
    ]
    a(_compare_table(bt.stats, bt.benchmark_stats, compare_keys) + "\n")
    a("\n**Trading activity**\n\n")
    a(_metrics_table(bt.trade_summary) + "\n")
    a(fig_link("equity_curve"))
    a(fig_link("drawdown"))
    a(fig_link("rolling_sharpe"))
    a(fig_link("returns_distribution"))
    a(fig_link("implementation_drag"))
    a(fig_link("exposure_history"))
    if not bt.monthly_returns.empty:
        a("\n**Monthly returns**\n\n")
        a(_df_to_md(bt.monthly_returns, floatfmt=".2%") + "\n")
        a(fig_link("monthly_returns"))

    a("## 6. Cost, delay, capacity, and latency\n")
    cost_table = decision_analysis.get("cost_sensitivity")
    if isinstance(cost_table, pd.DataFrame):
        a("\n**Transaction-cost frontier**\n\n")
        a(_df_to_md(cost_table, floatfmt=".4f", index=False) + "\n")
    delay_table = decision_analysis.get("delay_sensitivity")
    if isinstance(delay_table, pd.DataFrame):
        a("\n**Incremental execution-delay decay**\n\n")
        a(_df_to_md(delay_table, floatfmt=".4f", index=False) + "\n")
    capacity = decision_analysis.get("capacity")
    if isinstance(capacity, pd.DataFrame):
        a("\n**Dollar-volume capacity proxy**\n\n")
        a(_df_to_md(capacity, floatfmt=".4f", index=False) + "\n")
    latency = decision_analysis.get("inference_latency")
    if isinstance(latency, pd.DataFrame) and not latency.empty:
        a("\n**Warm synchronous inference benchmark**\n\n")
        a(_df_to_md(latency, floatfmt=".4f", index=False) + "\n")
    if not criteria.empty:
        a("\n**Readiness criteria**\n\n")
        a(_df_to_md(criteria, floatfmt=".4f", index=False) + "\n")
    a(fig_link("cost_frontier"))
    a(fig_link("delay_decay"))
    a(fig_link("capacity_participation"))
    a(fig_link("inference_performance"))

    a("## 7. Risk analytics\n")
    a(_metrics_table(risk_block.get("portfolio", {})) + "\n")
    stress = risk_block.get("stress")
    if stress is not None and len(stress):
        a("\n**Stress / scenario analysis**\n\n")
        a(_df_to_md(pd.DataFrame(stress), floatfmt=".2%", index=False) + "\n")

    a("## 8. What this run does not establish\n")
    a(
        "- Daily bars cannot establish intraday fill quality, queue position, spread, "
        "borrow availability, market impact, or exchange-level latency.\n"
        "- The capacity table is a trailing dollar-volume participation proxy, not an "
        "order-book execution simulator.\n"
        "- A current fixed universe can contain selection and survivorship bias; free "
        "vendor data can differ from point-in-time institutional data.\n"
        "- One walk-forward experiment does not control data mining across all ideas a "
        "researcher might have tried. Independent replication and shadow evaluation "
        "remain mandatory.\n"
    )

    a("## 9. Reproduction and promotion contract\n")
    a(
        "The committed configuration, dataset fingerprint, seed, Git commit, fold "
        "definitions, calibrated OOS predictions, and gate thresholds form the audit "
        "contract. Promotion requires every readiness criterion to pass on evidence "
        "appropriate to the next stage; changing a threshold creates a new experiment, "
        "not a reinterpretation of this result.\n"
    )

    content = "\n".join(lines)
    if config.report.format == "html":
        content = _to_html(content, title=config.report.title)
        output_path = output_path.with_suffix(".html")

    output_path.write_text(content, encoding="utf-8")
    logger.info("Wrote report to %s", output_path)
    return output_path


def _to_html(markdown_text: str, *, title: str) -> str:
    """Best-effort Markdown→HTML; falls back to a <pre> block if no converter."""
    body: str
    try:  # pragma: no cover - optional dependency
        import markdown as md

        body = md.markdown(markdown_text, extensions=["tables", "fenced_code"])
    except Exception:
        import html

        body = f"<pre>{html.escape(markdown_text)}</pre>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:system-ui,Arial,sans-serif;max-width:980px;"
        "margin:2rem auto;padding:0 1rem;line-height:1.5} "
        "table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:4px 8px} "
        "img{max-width:100%}</style></head><body>"
        f"{body}</body></html>"
    )
