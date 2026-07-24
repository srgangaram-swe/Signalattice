"""High-signal plots for forecast, execution, and portfolio diagnostics.

The public entry point is intentionally best-effort: each figure is emitted
only when its supporting data is available, and one unavailable diagnostic
does not prevent the rest of a research report from rendering.  All figures
use genuinely out-of-sample inputs supplied by the training and backtest
results; this module performs no model fitting or scenario selection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from quant_platform.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

_BLUE = "#245A7A"
_TEAL = "#2A7F7F"
_ORANGE = "#D97941"
_RED = "#B84A4A"
_GRAY = "#68737D"
_LIGHT_GRAY = "#D9E0E5"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9.5,
        "axes.titleweight": "semibold",
    }
)


def _save(fig: Figure, path: Path) -> Path:
    """Write and close a figure without importing the legacy plot module."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _as_frame(value: Any) -> pd.DataFrame:
    """Return a defensive DataFrame copy, or an empty frame for unusable input."""
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame()
    try:
        return pd.DataFrame(value)
    except (TypeError, ValueError):
        return pd.DataFrame()


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric while retaining their common rows."""
    if frame.empty or not set(columns).issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    result = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(result).replace([np.inf, -np.inf], np.nan).dropna(how="all")


def _percent_axis(axis: Any) -> None:
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))


def _currency(value: float, _position: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if magnitude >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def _is_classification_result(train_result: Any, predictions: pd.DataFrame) -> bool:
    """Honor an explicit task, or infer binary labels for lightweight callers."""
    task = str(getattr(train_result, "task", "")).lower()
    if task:
        return task == "classification"
    if "y_true" not in predictions.columns:
        return False
    observed = pd.to_numeric(predictions["y_true"], errors="coerce").dropna().unique()
    return bool(len(observed) and set(observed).issubset({0.0, 1.0}))


def _calibration_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"y_true", "score"}
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()
    data = _numeric(predictions, ["y_true", "score"]).dropna()
    if data.empty or not data["score"].between(0.0, 1.0).all():
        return pd.DataFrame()
    edges = np.linspace(0.0, 1.0, 11)
    bins = pd.cut(
        data["score"],
        bins=edges.tolist(),
        labels=False,
        include_lowest=True,
    )
    grouped = data.assign(bin=bins).groupby("bin", observed=True)
    return grouped.agg(
        count=("y_true", "size"),
        mean_probability=("score", "mean"),
        observed_rate=("y_true", "mean"),
    ).reset_index()


def _plot_reliability(table: pd.DataFrame, path: Path) -> Path | None:
    data = _numeric(table, ["mean_probability", "observed_rate", "count"]).dropna(
        subset=["mean_probability", "observed_rate"]
    )
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    count_ax = ax.twinx()
    width = max(0.025, min(0.09, 0.75 / max(len(data), 1)))
    count_ax.bar(
        data["mean_probability"],
        data["count"].fillna(0.0),
        width=width,
        color=_LIGHT_GRAY,
        alpha=0.65,
        label="Observations per bin",
        zorder=1,
    )
    count_ax.set_ylabel("Observations per probability bin", color=_GRAY)
    count_ax.grid(False)
    count_ax.tick_params(axis="y", colors=_GRAY)

    # Keep the evidence line legible while the translucent count bars remain
    # visible behind it on the secondary axis.
    ax.set_zorder(count_ax.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.plot([0, 1], [0, 1], linestyle="--", color=_GRAY, linewidth=1.1, label="Ideal calibration")
    ax.plot(
        data["mean_probability"],
        data["observed_rate"],
        marker="o",
        color=_BLUE,
        linewidth=2.0,
        label="Walk-forward observations",
        zorder=3,
    )
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel("Mean predicted probability of an up move")
    ax.set_ylabel("Observed frequency of an up move")
    ax.set_title("Are out-of-sample probabilities calibrated to observed outcomes?")
    handles, labels = ax.get_legend_handles_labels()
    count_handles, count_labels = count_ax.get_legend_handles_labels()
    ax.legend(handles + count_handles, labels + count_labels, loc="upper left", frameon=False)
    return _save(fig, path)


def _plot_score_distribution(predictions: pd.DataFrame, path: Path) -> Path | None:
    data = _numeric(predictions, ["y_true", "score"]).dropna()
    if data.empty or not data["score"].between(0.0, 1.0).all():
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bins = np.linspace(0.0, 1.0, 21).tolist()
    labels = [(0, "Realized down", _ORANGE), (1, "Realized up", _BLUE)]
    for outcome, label, color in labels:
        values = data.loc[data["y_true"].round().astype(int) == outcome, "score"]
        if not values.empty:
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.38,
                color=color,
                edgecolor=color,
                linewidth=1.1,
                label=f"{label} (n={len(values):,})",
            )
    ax.axvline(0.5, color=_GRAY, linestyle="--", linewidth=1.0, label="0.50 decision boundary")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Out-of-sample predicted probability of an up move")
    ax.set_ylabel("Density")
    ax.set_title("Do forecast scores separate realized directional outcomes?")
    ax.legend(frameon=False)
    return _save(fig, path)


def _plot_precision_recall(predictions: pd.DataFrame, path: Path) -> Path | None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    data = _numeric(predictions, ["y_true", "score"]).dropna()
    if data.empty or data["y_true"].nunique() < 2 or not data["score"].between(0.0, 1.0).all():
        return None
    y_true = data["y_true"].round().astype(int).to_numpy()
    scores = data["score"].to_numpy()
    precision, recall, _ = precision_recall_curve(y_true, scores)
    average_precision = average_precision_score(y_true, scores)
    prevalence = float(y_true.mean())

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(
        recall,
        precision,
        color=_BLUE,
        linewidth=2.0,
        label=f"Walk-forward forecast (AP={average_precision:.3f})",
    )
    ax.axhline(
        prevalence,
        color=_GRAY,
        linestyle="--",
        linewidth=1.1,
        label=f"Event-rate reference ({prevalence:.3f})",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel("Recall of realized up moves")
    ax.set_ylabel("Precision among predicted up moves")
    ax.set_title("What precision is retained as directional recall expands?")
    ax.legend(loc="lower left", frameon=False)
    return _save(fig, path)


def _selective_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    data = _numeric(predictions, ["y_true", "score"]).dropna()
    if data.empty or not data["score"].between(0.0, 1.0).all():
        return pd.DataFrame()
    confidence = 2.0 * np.abs(data["score"].to_numpy() - 0.5)
    correct = (data["score"].to_numpy() >= 0.5).astype(int) == data["y_true"].to_numpy(dtype=int)
    order = np.argsort(-confidence, kind="stable")
    rows = []
    for coverage in np.linspace(0.1, 1.0, 10):
        selected_count = max(1, int(np.ceil(coverage * len(data))))
        selected = order[:selected_count]
        rows.append(
            {
                "coverage": selected_count / len(data),
                "accuracy": float(correct[selected].mean()),
                "n_selected": selected_count,
            }
        )
    return pd.DataFrame(rows)


def _plot_selective_coverage(table: pd.DataFrame, path: Path) -> Path | None:
    data = _numeric(table, ["coverage", "accuracy"]).dropna()
    if data.empty:
        return None
    data = data.sort_values("coverage")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(data["coverage"], data["accuracy"], color=_TEAL, marker="o", linewidth=2.0)
    full_coverage = float(
        data.loc[[data["coverage"].idxmax()], "accuracy"].to_numpy(dtype=float)[0]
    )
    ax.axhline(
        full_coverage,
        color=_GRAY,
        linestyle="--",
        linewidth=1.0,
        label=f"Full-coverage accuracy ({full_coverage:.3f})",
    )
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    _percent_axis(ax)
    ax.set_xlabel("Share of most-confident forecasts retained")
    ax.set_ylabel("Directional accuracy on retained forecasts")
    ax.set_title("Does abstaining on low-confidence forecasts improve accuracy?")
    ax.legend(frameon=False)
    return _save(fig, path)


def _deciles_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    data = _numeric(predictions, ["score", "forward_return"]).dropna()
    if len(data) < 2:
        return pd.DataFrame()
    n_bins = min(10, len(data))
    quantile = pd.qcut(data["score"].rank(method="first"), q=n_bins, labels=False) + 1
    grouped = data.assign(quantile=quantile).groupby("quantile", observed=True)
    result = grouped["forward_return"].agg(["count", "mean", "std"]).reset_index()
    result["standard_error"] = result["std"].fillna(0.0) / np.sqrt(result["count"])
    return result.rename(columns={"mean": "mean_return"})


def _plot_prediction_deciles(table: pd.DataFrame, path: Path) -> Path | None:
    required = ["quantile", "mean_return", "standard_error"]
    data = _numeric(table, required).dropna(subset=["quantile", "mean_return"])
    if data.empty:
        return None
    data = data.sort_values("quantile")
    errors = 1.96 * data["standard_error"].fillna(0.0).clip(lower=0.0)
    colors = plt.get_cmap("RdYlBu")(np.linspace(0.15, 0.85, len(data)))

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.bar(data["quantile"], data["mean_return"], color=colors, alpha=0.85, width=0.72)
    ax.errorbar(
        data["quantile"],
        data["mean_return"],
        yerr=errors,
        fmt="none",
        ecolor="#333333",
        elinewidth=1.1,
        capsize=3,
        label="Approx. 95% mean interval",
    )
    ax.axhline(0.0, color=_GRAY, linewidth=1.0)
    ax.set_xticks(data["quantile"].astype(int))
    ax.set_xlabel("Forecast score quantile (lowest to highest)")
    ax.set_ylabel("Mean next-period return")
    _percent_axis(ax)
    ax.set_title("Do forecast-ranked groups separate subsequent returns?")
    ax.legend(frameon=False)
    return _save(fig, path)


def _fold_metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"fold", "y_true", "score"}
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    rows: list[dict[str, float]] = []
    for _, group in predictions.groupby("fold", sort=True):
        data = _numeric(group, ["y_true", "score"]).dropna()
        if data.empty:
            continue
        y_true = data["y_true"].round().astype(int).to_numpy()
        score = data["score"].to_numpy()
        row = {
            "fold": float(group["fold"].to_numpy(dtype=float)[0]),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, (score >= 0.5).astype(int))),
        }
        if np.unique(y_true).size > 1:
            row["roc_auc"] = float(roc_auc_score(y_true, score))
            row["average_precision"] = float(average_precision_score(y_true, score))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_fold_stability(table: pd.DataFrame, path: Path) -> Path | None:
    if table.empty:
        return None
    data = table.copy()
    if "fold" not in data.columns:
        data.insert(0, "fold", data.index)
    metric_preference = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "accuracy",
        "directional_accuracy",
        "r2",
    ]
    metrics = [metric for metric in metric_preference if metric in data.columns][:3]
    if not metrics:
        numeric_candidates = data.drop(columns="fold", errors="ignore").select_dtypes(
            include="number"
        )
        metrics = list(numeric_candidates.columns[:3])
    if not metrics:
        return None
    numeric = _numeric(data, ["fold", *metrics]).dropna(subset=["fold"])
    metrics = [metric for metric in metrics if numeric[metric].notna().any()]
    if not metrics:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    colors = [_BLUE, _TEAL, _ORANGE]
    for metric, color in zip(metrics, colors, strict=False):
        ax.plot(
            numeric["fold"],
            numeric[metric],
            marker="o",
            linewidth=1.8,
            color=color,
            label=metric.replace("_", " ").title(),
        )
    bounded = all(numeric[metric].dropna().between(0.0, 1.0).all() for metric in metrics)
    if bounded:
        ax.set_ylim(0.0, 1.02)
        _percent_axis(ax)
    ax.set_xticks(numeric["fold"].dropna().astype(int).unique())
    ax.set_xlabel("Chronological walk-forward fold")
    ax.set_ylabel("Out-of-sample metric")
    ax.set_title("Does forecast quality persist across walk-forward folds?")
    ax.legend(frameon=False, ncol=min(3, len(metrics)))
    return _save(fig, path)


def _plot_feature_stability(table: pd.DataFrame, path: Path, *, top_n: int = 12) -> Path | None:
    if table.empty:
        return None
    numeric = table.drop(columns="fold", errors="ignore").apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if numeric.empty:
        return None
    magnitude = numeric.abs()
    means = magnitude.mean(axis=0)
    deviations = magnitude.std(axis=0, ddof=1).fillna(0.0)
    selected = means.nlargest(min(top_n, len(means))).sort_values()
    deviations = deviations.reindex(selected.index)

    fig, ax = plt.subplots(figsize=(7.8, max(4.4, 0.36 * len(selected) + 1.8)))
    positions = np.arange(len(selected))
    for fold_number, (_, fold_values) in enumerate(magnitude[selected.index].iterrows()):
        ax.scatter(
            fold_values,
            positions,
            color=_LIGHT_GRAY,
            s=16,
            alpha=0.75,
            label="Individual folds" if fold_number == 0 else None,
            zorder=2,
        )
    ax.errorbar(
        selected,
        positions,
        xerr=deviations,
        fmt="o",
        color=_BLUE,
        ecolor=_BLUE,
        capsize=3,
        linewidth=1.4,
        label="Fold mean ± 1 SD",
        zorder=3,
    )
    ax.set_yticks(positions, labels=selected.index)
    ax.set_xlabel("Absolute model importance")
    ax.set_ylabel("Feature")
    ax.set_title("Which features remain influential across chronological folds?")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, path)


def _coerce_weights(value: Any) -> pd.Series:
    if value is None:
        return pd.Series(dtype=float)
    if isinstance(value, pd.Series):
        series = value.copy()
    elif isinstance(value, Mapping):
        series = pd.Series(dict(value), dtype=float)
    else:
        try:
            series = pd.Series(value, dtype=float)
        except (TypeError, ValueError):
            return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _plot_ensemble_weights(
    fold_table: pd.DataFrame,
    final_weights: pd.Series,
    path: Path,
) -> Path | None:
    fold = fold_table.copy()
    fold_labels: list[str] = []
    if not fold.empty:
        if "fold" in fold.columns:
            fold_labels = [f"Fold {value:g}" for value in pd.to_numeric(fold.pop("fold"))]
        else:
            fold_labels = [f"Fold {value}" for value in fold.index]
        fold = fold.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    candidates = list(fold.columns)
    candidates.extend(name for name in final_weights.index if name not in candidates)
    if not candidates:
        return None

    if fold.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        final_values = final_weights.reindex(candidates).fillna(0.0)
        ax.bar(
            candidates,
            final_values,
            color=plt.get_cmap("Set2")(np.linspace(0.0, 0.8, len(final_values))),
        )
        ax.set_xlabel("Candidate model")
        ax.set_ylabel("Final convex ensemble weight")
        ax.set_title("How is the final calibrated ensemble allocated across models?")
        ax.tick_params(axis="x", rotation=20)
        return _save(fig, path)

    rows = fold.reindex(columns=candidates)
    labels = fold_labels
    if not final_weights.empty:
        rows = pd.concat(
            [rows, final_weights.reindex(candidates).rename("Final fit").to_frame().T],
            axis=0,
        )
        labels = [*labels, "Final fit"]
    rows = rows.fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bottoms = np.zeros(len(rows))
    colors = plt.get_cmap("Set2")(np.linspace(0.0, 0.9, len(candidates)))
    positions = np.arange(len(rows))
    for candidate, color in zip(candidates, colors, strict=True):
        candidate_values = rows[candidate].to_numpy(dtype=float)
        ax.bar(
            positions,
            candidate_values,
            bottom=bottoms,
            color=color,
            label=candidate,
            width=0.72,
        )
        bottoms += candidate_values
    ax.set_xticks(positions, labels=labels)
    ax.set_ylim(0.0, max(1.02, float(np.nanmax(bottoms)) * 1.08))
    _percent_axis(ax)
    ax.set_xlabel("Chronological fit")
    ax.set_ylabel("Convex ensemble weight")
    ax.set_title("Are ensemble allocations stable across walk-forward folds?")
    ax.legend(frameon=False, ncol=min(4, len(candidates)), loc="upper center")
    return _save(fig, path)


def _plot_cost_frontier(table: pd.DataFrame, path: Path) -> Path | None:
    x_column = next(
        (column for column in ["total_one_way_cost_bps", "cost_bps"] if column in table.columns),
        None,
    )
    y_column = next(
        (
            column
            for column in ["net_total_return", "sharpe", "terminal_equity"]
            if column in table.columns
        ),
        None,
    )
    if x_column is None or y_column is None:
        return None
    data = _numeric(table, [x_column, y_column]).dropna().sort_values(x_column)
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(data[x_column], data[y_column], color=_RED, marker="o", linewidth=2.0)
    if y_column == "net_total_return":
        ax.axhline(0.0, color=_GRAY, linewidth=1.0)
        _percent_axis(ax)
        ylabel = "Simulated net total return"
    elif y_column == "sharpe":
        ax.axhline(0.0, color=_GRAY, linewidth=1.0)
        ylabel = "Simulated net Sharpe ratio"
    else:
        ax.axhline(1.0, color=_GRAY, linewidth=1.0)
        ylabel = "Terminal value of one unit of capital"
    ax.set_xlabel("Assumed total one-way implementation cost (basis points)")
    ax.set_ylabel(ylabel)
    ax.set_title("How sensitive is simulated performance to implementation cost?")
    return _save(fig, path)


def _plot_delay_decay(table: pd.DataFrame, path: Path) -> Path | None:
    x_column = next(
        (
            column
            for column in ["additional_delay_bars", "total_lag_bars", "delay_bars"]
            if column in table.columns
        ),
        None,
    )
    metrics = [metric for metric in ["net_total_return", "sharpe"] if metric in table.columns]
    if x_column is None or not metrics:
        return None
    data = _numeric(table, [x_column, *metrics]).dropna(subset=[x_column]).sort_values(x_column)
    metrics = [metric for metric in metrics if data[metric].notna().any()]
    if data.empty or not metrics:
        return None

    fig, axes = plt.subplots(1, len(metrics), figsize=(7.5 if len(metrics) == 1 else 10.2, 4.5))
    axes_array = np.atleast_1d(axes)
    for ax, metric, color in zip(axes_array, metrics, [_ORANGE, _BLUE], strict=False):
        ax.plot(data[x_column], data[metric], marker="o", color=color, linewidth=2.0)
        ax.axhline(0.0, color=_GRAY, linewidth=1.0)
        ax.set_xlabel("Additional execution delay (bars)")
        if metric == "net_total_return":
            ax.set_ylabel("Simulated net total return")
            _percent_axis(ax)
        else:
            ax.set_ylabel("Simulated net Sharpe ratio")
        ax.set_xticks(data[x_column].dropna().astype(int).unique())
    fig.suptitle(
        "How sensitive is simulated performance to delayed execution?", fontweight="semibold"
    )
    return _save(fig, path)


def _plot_capacity(table: pd.DataFrame, path: Path) -> Path | None:
    if "aum" not in table.columns:
        return None
    rate_columns = [
        column
        for column in [
            "median_participation_rate",
            "p95_participation_rate",
            "max_participation_rate",
        ]
        if column in table.columns
    ]
    if not rate_columns:
        return None
    data = _numeric(table, ["aum", *rate_columns]).dropna(subset=["aum"])
    data = data.loc[data["aum"] > 0.0].sort_values("aum")
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    styles = [
        ("median_participation_rate", _TEAL, "Median trade"),
        ("p95_participation_rate", _ORANGE, "95th percentile trade"),
        ("max_participation_rate", _RED, "Maximum observed trade"),
    ]
    for column, color, label in styles:
        if column in rate_columns:
            ax.plot(data["aum"], data[column], marker="o", color=color, linewidth=1.8, label=label)
    if "participation_limit" in table.columns:
        limit = pd.to_numeric(table["participation_limit"], errors="coerce").dropna()
        if not limit.empty:
            ax.axhline(
                float(limit.iloc[0]),
                color=_GRAY,
                linestyle="--",
                linewidth=1.1,
                label=f"Configured limit ({float(limit.iloc[0]):.1%})",
            )
    if data["aum"].max() / data["aum"].min() >= 10.0:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_currency))
    _percent_axis(ax)
    ax.set_xlabel("Assets under management scenario")
    ax.set_ylabel("Trade participation in trailing dollar volume")
    ax.set_title("At what AUM does the dollar-volume participation proxy become restrictive?")
    ax.legend(frameon=False)
    return _save(fig, path)


def _plot_inference(table: pd.DataFrame, path: Path) -> Path | None:
    if "batch_size" not in table.columns:
        return None
    latency_columns = [
        column
        for column in ["p50_latency_ms", "p95_latency_ms", "p99_latency_ms"]
        if column in table.columns
    ]
    throughput_column = "throughput_rows_per_second"
    if not latency_columns and throughput_column not in table.columns:
        return None
    columns = ["batch_size", *latency_columns]
    if throughput_column in table.columns:
        columns.append(throughput_column)
    data = _numeric(table, columns).dropna(subset=["batch_size"]).sort_values("batch_size")
    if data.empty:
        return None

    panels = 1 + int(throughput_column in data.columns and data[throughput_column].notna().any())
    fig, axes = plt.subplots(1, panels, figsize=(7.5 if panels == 1 else 10.2, 4.5))
    axes_array = np.atleast_1d(axes)
    latency_ax = axes_array[0]
    for column, color in zip(latency_columns, [_TEAL, _ORANGE, _RED], strict=False):
        if data[column].notna().any():
            latency_ax.plot(
                data["batch_size"],
                data[column],
                marker="o",
                linewidth=1.8,
                color=color,
                label=column.removesuffix("_latency_ms").upper(),
            )
    latency_ax.set_xlabel("Inference batch size")
    latency_ax.set_ylabel("Warm latency (milliseconds)")
    latency_ax.legend(frameon=False)
    if panels == 2:
        throughput_ax = axes_array[1]
        throughput_ax.plot(
            data["batch_size"],
            data[throughput_column],
            color=_BLUE,
            marker="o",
            linewidth=2.0,
        )
        throughput_ax.set_xlabel("Inference batch size")
        throughput_ax.set_ylabel("Warm throughput (rows per second)")
        throughput_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    fig.suptitle(
        "What latency-throughput trade-off does warm inference provide?", fontweight="semibold"
    )
    return _save(fig, path)


def _series(value: Any, name: str) -> pd.Series:
    try:
        result = pd.Series(value, copy=True, name=name, dtype=float)
    except (TypeError, ValueError):
        return pd.Series(dtype=float, name=name)
    return result.replace([np.inf, -np.inf], np.nan)


def _plot_implementation_drag(backtest: Any, path: Path) -> Path | None:
    gross = _series(getattr(backtest, "gross_returns", None), "gross")
    net = _series(getattr(backtest, "returns", None), "net")
    if gross.empty or net.empty:
        return None
    returns = pd.concat([gross, net], axis=1).dropna()
    if returns.empty or (returns <= -1.0).any().any():
        return None
    costs = _series(getattr(backtest, "costs", None), "cost").reindex(returns.index).fillna(0.0)
    wealth = (1.0 + returns).cumprod()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    axes[0].plot(wealth.index, wealth["gross"], color=_BLUE, linewidth=1.8, label="Before costs")
    axes[0].plot(wealth.index, wealth["net"], color=_RED, linewidth=1.8, label="After costs")
    axes[0].fill_between(
        wealth.index,
        wealth["net"].to_numpy(),
        wealth["gross"].to_numpy(),
        color=_ORANGE,
        alpha=0.18,
        label="Compounded implementation gap",
    )
    axes[0].set_ylabel("Growth of one unit of capital")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(costs.index, costs.cumsum(), color=_ORANGE, linewidth=1.7)
    axes[1].fill_between(costs.index, 0.0, costs.cumsum().to_numpy(), color=_ORANGE, alpha=0.2)
    axes[1].set_xlabel("Realized evaluation date")
    axes[1].set_ylabel("Cumulative charged cost")
    _percent_axis(axes[1])
    fig.suptitle(
        "How much of the simulated path is removed by implementation costs?", fontweight="semibold"
    )
    return _save(fig, path)


def _plot_exposure_history(backtest: Any, path: Path) -> Path | None:
    exposures = _as_frame(getattr(backtest, "exposures", None))
    exposure_columns = [
        column
        for column in ["gross_exposure", "net_exposure", "long_exposure", "short_exposure"]
        if column in exposures.columns
    ]
    if exposures.empty or not exposure_columns:
        return None
    values = exposures[exposure_columns].apply(pd.to_numeric, errors="coerce")
    if values.dropna(how="all").empty:
        return None
    count_columns = [column for column in ["n_long", "n_short"] if column in exposures.columns]
    panels = 2 if count_columns else 1
    fig, axes = plt.subplots(
        panels,
        1,
        figsize=(8.4, 5.0 if panels == 1 else 6.2),
        sharex=panels == 2,
        gridspec_kw={"height_ratios": [2.2, 1.0]} if panels == 2 else None,
    )
    axes_array = np.atleast_1d(axes)
    exposure_ax = axes_array[0]
    styles = [
        ("long_exposure", _TEAL, "Long"),
        ("short_exposure", _ORANGE, "Short"),
        ("gross_exposure", _BLUE, "Gross"),
        ("net_exposure", _GRAY, "Net"),
    ]
    for column, color, label in styles:
        if column in values:
            linestyle = "--" if column in {"gross_exposure", "net_exposure"} else "-"
            exposure_ax.plot(
                values.index,
                values[column],
                color=color,
                linewidth=1.5,
                linestyle=linestyle,
                label=label,
            )
    exposure_ax.axhline(0.0, color="#333333", linewidth=0.8)
    exposure_ax.set_ylabel("Portfolio exposure (weight)")
    exposure_ax.legend(frameon=False, ncol=4)
    if panels == 2:
        breadth = exposures[count_columns].apply(pd.to_numeric, errors="coerce")
        for column, color in zip(count_columns, [_TEAL, _ORANGE], strict=False):
            axes_array[1].plot(
                breadth.index,
                breadth[column],
                color=color,
                linewidth=1.4,
                label=column.replace("n_", "").title(),
            )
        axes_array[1].set_ylabel("Active positions")
        axes_array[1].set_xlabel("Realized evaluation date")
        axes_array[1].legend(frameon=False, ncol=2)
    else:
        exposure_ax.set_xlabel("Realized evaluation date")
    fig.suptitle(
        "How do portfolio exposure and breadth evolve through evaluation?", fontweight="semibold"
    )
    return _save(fig, path)


def generate_diagnostic_figures(
    train_result: Any,
    backtest: Any,
    decision_analysis: Mapping[str, Any] | None,
    figures_dir: str | Path,
) -> dict[str, Path]:
    """Generate available model-to-execution diagnostics as PNG files.

    Missing or partially unavailable inputs are expected for baseline and
    regression runs.  Each diagnostic is isolated so those runs still retain
    every figure supported by their artifacts.
    """
    destination = Path(figures_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}
    predictions = _as_frame(getattr(train_result, "predictions", None))
    extra = getattr(train_result, "extra", {}) or {}
    if not isinstance(extra, Mapping):
        extra = {}
    analysis = decision_analysis if isinstance(decision_analysis, Mapping) else {}

    def emit(key: str, filename: str, build: Callable[[Path], Path | None]) -> None:
        figures_before = set(plt.get_fignums())
        try:
            result = build(destination / filename)
        except Exception:  # pragma: no cover - protects best-effort report generation
            for figure_number in set(plt.get_fignums()).difference(figures_before):
                plt.close(figure_number)
            logger.exception("Unable to render diagnostic figure %s", key)
            return
        if result is not None:
            figures[key] = result

    calibration = _as_frame(extra.get("calibration_table"))
    if calibration.empty:
        calibration = _calibration_from_predictions(predictions)
    emit(
        "reliability", "reliability_diagram.png", lambda path: _plot_reliability(calibration, path)
    )
    emit(
        "score_distribution",
        "score_by_outcome.png",
        lambda path: _plot_score_distribution(predictions, path),
    )
    emit(
        "precision_recall",
        "precision_recall.png",
        lambda path: _plot_precision_recall(predictions, path),
    )

    selective = _as_frame(extra.get("selective_prediction_frontier"))
    if selective.empty:
        selective = _selective_from_predictions(predictions)
    emit(
        "selective_coverage",
        "selective_coverage_accuracy.png",
        lambda path: _plot_selective_coverage(selective, path),
    )

    deciles = _as_frame(extra.get("prediction_deciles"))
    if deciles.empty:
        deciles = _deciles_from_predictions(predictions)
    emit(
        "prediction_deciles",
        "prediction_decile_returns.png",
        lambda path: _plot_prediction_deciles(deciles, path),
    )

    fold_metrics = _as_frame(getattr(train_result, "fold_metrics", None))
    if fold_metrics.empty:
        fold_metrics = _fold_metrics_from_predictions(predictions)
    emit(
        "fold_stability",
        "walk_forward_fold_stability.png",
        lambda path: _plot_fold_stability(fold_metrics, path),
    )
    emit(
        "feature_stability",
        "fold_feature_importance.png",
        lambda path: _plot_feature_stability(
            _as_frame(extra.get("fold_feature_importances")), path
        ),
    )
    emit(
        "ensemble_weights",
        "ensemble_weights.png",
        lambda path: _plot_ensemble_weights(
            _as_frame(extra.get("fold_ensemble_weights")),
            _coerce_weights(extra.get("ensemble_weights")),
            path,
        ),
    )

    emit(
        "cost_frontier",
        "cost_frontier.png",
        lambda path: _plot_cost_frontier(_as_frame(analysis.get("cost_sensitivity")), path),
    )
    emit(
        "delay_decay",
        "execution_delay_decay.png",
        lambda path: _plot_delay_decay(_as_frame(analysis.get("delay_sensitivity")), path),
    )
    emit(
        "capacity_participation",
        "capacity_participation.png",
        lambda path: _plot_capacity(_as_frame(analysis.get("capacity")), path),
    )
    emit(
        "inference_performance",
        "inference_latency_throughput.png",
        lambda path: _plot_inference(_as_frame(analysis.get("inference_latency")), path),
    )
    emit(
        "implementation_drag",
        "gross_net_cost_drag.png",
        lambda path: _plot_implementation_drag(backtest, path),
    )
    emit(
        "exposure_history",
        "exposure_history.png",
        lambda path: _plot_exposure_history(backtest, path),
    )
    logger.info("Generated %d diagnostic figures in %s", len(figures), destination)
    return figures


__all__ = ["generate_diagnostic_figures"]
