"""Signalattice probabilistic forecasting platform.

An end-to-end quantitative research platform covering data lineage, causal
panel features, calibrated and temporal forecasting, conservative backtesting,
decision-readiness evaluation, experiment tracking, and evidence reporting.

This package is organised into focused sub-packages:

- :mod:`quant_platform.data`       — market-data ingestion & validation
- :mod:`quant_platform.features`   — technical / cross-sectional feature pipeline
- :mod:`quant_platform.models`     — baseline & ML models, time-series CV
- :mod:`quant_platform.backtest`   — vectorized backtesting engine
- :mod:`quant_platform.evaluation` — cost, delay, capacity, latency & gates
- :mod:`quant_platform.risk`       — risk & performance analytics
- :mod:`quant_platform.tracking`   — lightweight experiment tracking
- :mod:`quant_platform.reporting`  — plots and Markdown report generation

DISCLAIMER: Research-use software only. Not financial advice and not an
authorization for live trading.
"""

from __future__ import annotations

__version__ = "0.2.1"

__all__ = ["__version__"]
