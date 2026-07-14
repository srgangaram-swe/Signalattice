"""Matplotlib plotting helpers.

All plots use the non-interactive ``Agg`` backend so figures render in headless
environments (CI, Docker) and are written straight to PNG. Functions are small,
return the output path, and never call ``plt.show()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from quant_platform.data.schema import DATE_COL, TICKER_COL  # noqa: E402
from quant_platform.logging_utils import get_logger  # noqa: E402
from quant_platform.reporting.diagnostic_plots import (  # noqa: E402
    generate_diagnostic_figures,
)
from quant_platform.utils import ensure_dir  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from quant_platform.backtest.engine import BacktestResult
    from quant_platform.models.train import TrainResult

logger = get_logger(__name__)

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def _save(fig: Figure, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved figure %s", path)
    return path


def plot_price_history(panel: pd.DataFrame, path: Path, *, price_col: str = "adj_close") -> Path:
    """Normalised (rebased to 100) price history for all tickers."""
    fig, ax = plt.subplots()
    for ticker, g in panel.groupby(TICKER_COL):
        g = g.sort_values(DATE_COL)
        series = g.set_index(DATE_COL)[price_col]
        rebased = 100.0 * series / series.iloc[0]
        ax.plot(rebased.index, rebased.values, label=ticker, linewidth=1.2)
    ax.set_title("Price history (rebased to 100)")
    ax.set_ylabel("Index level")
    ax.legend(ncol=4, fontsize=8)
    return _save(fig, path)


def plot_returns_distribution(returns: pd.Series, path: Path, *, label: str = "Strategy") -> Path:
    """Histogram of returns with a fitted normal overlay."""
    r = pd.Series(returns).dropna()
    fig, ax = plt.subplots()
    ax.hist(r, bins=60, density=True, alpha=0.6, color="#3b6ea5")
    if len(r) > 2:
        from scipy.stats import norm

        x = np.linspace(r.min(), r.max(), 200)
        ax.plot(x, norm.pdf(x, r.mean(), r.std()), "r--", linewidth=1.2, label="Normal fit")
        ax.legend()
    ax.set_title(f"{label} daily-return distribution")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Density")
    return _save(fig, path)


def plot_rolling_volatility(returns: pd.Series, path: Path, *, window: int = 63) -> Path:
    """Rolling annualised volatility."""
    r = pd.Series(returns).dropna()
    vol = r.rolling(window, min_periods=max(2, window // 2)).std() * np.sqrt(252)
    fig, ax = plt.subplots()
    ax.plot(vol.index, vol.values, color="#a5523b", linewidth=1.2)
    ax.set_title(f"Rolling annualised volatility ({window}d)")
    ax.set_ylabel("Annualised volatility")
    return _save(fig, path)


def plot_equity_curve(equity: pd.Series, benchmark_equity: pd.Series, path: Path) -> Path:
    """Strategy vs benchmark equity curves (log-scale y)."""
    fig, ax = plt.subplots()
    ax.plot(
        equity.index,
        equity.to_numpy(dtype=float),
        label="Strategy",
        linewidth=1.5,
        color="#2e7d32",
    )
    if benchmark_equity is not None and len(benchmark_equity):
        ax.plot(
            benchmark_equity.index,
            benchmark_equity.to_numpy(dtype=float),
            label="Benchmark (buy & hold)",
            linewidth=1.2,
            color="#888888",
        )
    ax.set_yscale("log")
    ax.set_title("Equity curve")
    ax.set_ylabel("Portfolio value (log scale)")
    ax.legend()
    return _save(fig, path)


def plot_drawdown(drawdown: pd.Series, path: Path) -> Path:
    """Underwater (drawdown) plot."""
    dd = pd.Series(drawdown).dropna()
    fig, ax = plt.subplots()
    ax.fill_between(
        dd.index,
        dd.to_numpy(dtype=float) * 100.0,
        0.0,
        color="#c0392b",
        alpha=0.4,
    )
    ax.set_title("Drawdown (underwater plot)")
    ax.set_ylabel("Drawdown (%)")
    return _save(fig, path)


def plot_rolling_sharpe(returns: pd.Series, path: Path, *, window: int = 126) -> Path:
    """Rolling annualised Sharpe ratio."""
    from quant_platform.risk.analytics import rolling_sharpe

    rs = rolling_sharpe(pd.Series(returns).dropna(), window=window)
    fig, ax = plt.subplots()
    ax.plot(rs.index, rs.to_numpy(dtype=float), color="#6a3d9a", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Rolling Sharpe ratio ({window}d)")
    ax.set_ylabel("Annualised Sharpe")
    return _save(fig, path)


def plot_feature_importance(importances: pd.Series, path: Path, *, top_n: int = 20) -> Path:
    """Horizontal bar chart of the top-N feature importances."""
    imp = pd.Series(importances).dropna().head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(imp))))
    ax.barh(imp.index, imp.to_numpy(dtype=float), color="#3b6ea5")
    ax.set_title(f"Top {len(imp)} feature importances")
    ax.set_xlabel("Importance")
    return _save(fig, path)


def plot_confusion_matrix(y_true: Any, y_pred: Any, path: Path) -> Path:
    """Confusion matrix heatmap for the binary direction classifier."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(
        np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1]
    )
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Down", "Up"])
    ax.set_yticks([0, 1], labels=["Down", "Up"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def plot_roc_curve(y_true: Any, y_proba: Any, path: Path) -> Path:
    """ROC curve with AUC annotation."""
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = np.asarray(y_true).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, color="#2e7d32", linewidth=1.5, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (out-of-sample)")
    ax.legend()
    return _save(fig, path)


def plot_correlation_heatmap(corr: pd.DataFrame, path: Path) -> Path:
    """Correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.index)), labels=corr.index, fontsize=8)
    ax.set_title("Return correlation matrix")
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, path)


def plot_monthly_returns_heatmap(monthly: pd.DataFrame, path: Path) -> Path:
    """Heatmap of monthly returns (Year x Month)."""
    if monthly.empty:
        return path
    data = monthly.drop(columns=[c for c in ["YEAR"] if c in monthly.columns])
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(data))))
    im = ax.imshow(data.values * 100.0, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
    ax.set_xticks(range(len(data.columns)), labels=data.columns, fontsize=8)
    ax.set_yticks(range(len(data.index)), labels=data.index, fontsize=8)
    ax.set_title("Monthly returns (%)")
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val * 100:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return _save(fig, path)


def generate_all_figures(
    *,
    panel: pd.DataFrame,
    backtest: BacktestResult,
    train_result: TrainResult | None,
    correlation: pd.DataFrame,
    figures_dir: Path,
    task: str = "classification",
    decision_analysis: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Generate the full set of report figures and return a name -> path map."""
    figures_dir = ensure_dir(figures_dir)
    figs: dict[str, Path] = {}

    figs["price_history"] = plot_price_history(panel, figures_dir / "price_history.png")
    figs["returns_distribution"] = plot_returns_distribution(
        backtest.returns, figures_dir / "returns_distribution.png"
    )
    figs["rolling_volatility"] = plot_rolling_volatility(
        backtest.returns, figures_dir / "rolling_volatility.png"
    )
    figs["equity_curve"] = plot_equity_curve(
        backtest.equity_curve, backtest.benchmark_equity, figures_dir / "equity_curve.png"
    )
    figs["drawdown"] = plot_drawdown(backtest.drawdown, figures_dir / "drawdown.png")
    figs["rolling_sharpe"] = plot_rolling_sharpe(
        backtest.returns, figures_dir / "rolling_sharpe.png"
    )
    figs["correlation"] = plot_correlation_heatmap(
        correlation, figures_dir / "correlation_heatmap.png"
    )
    if not backtest.monthly_returns.empty:
        figs["monthly_returns"] = plot_monthly_returns_heatmap(
            backtest.monthly_returns, figures_dir / "monthly_returns.png"
        )

    if train_result is not None:
        figs["feature_importance"] = plot_feature_importance(
            train_result.feature_importances, figures_dir / "feature_importance.png"
        )
        preds = train_result.predictions
        if task == "classification" and "y_true" in preds.columns:
            y_pred = (preds["score"] > 0.5).astype(int)
            figs["confusion_matrix"] = plot_confusion_matrix(
                preds["y_true"], y_pred, figures_dir / "confusion_matrix.png"
            )
            figs["roc_curve"] = plot_roc_curve(
                preds["y_true"], preds["score"], figures_dir / "roc_curve.png"
            )
    figs.update(
        generate_diagnostic_figures(
            train_result,
            backtest,
            decision_analysis or {},
            figures_dir,
        )
    )
    logger.info("Generated %d figures in %s", len(figs), figures_dir)
    return figs
