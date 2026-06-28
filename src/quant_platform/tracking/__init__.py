"""Lightweight, pluggable experiment tracking.

Backends:

- ``sqlite`` (default) — a single self-contained ``experiments.sqlite`` DB.
- ``json``   — one JSON file per run under ``experiments/runs/``.
- ``mlflow`` — optional, if MLflow is installed.
- ``none``   — no-op (useful for tests).

Every run records the dataset hash, tickers, features, model & backtest params,
metrics, artifact paths, a timestamp and the git commit — enough to reproduce
and compare research runs.
"""

from __future__ import annotations

from quant_platform.tracking.experiment import (
    ExperimentTracker,
    RunContext,
    get_tracker,
)

__all__ = ["ExperimentTracker", "RunContext", "get_tracker"]
