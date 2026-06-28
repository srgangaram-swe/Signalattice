"""Quant Research Data Platform.

An end-to-end research platform for quantitative finance covering data
ingestion, time-series feature engineering, factor research, machine-learning
modelling, vectorized backtesting, risk analytics, experiment tracking and
reporting.

This package is organised into focused sub-packages:

- :mod:`quant_platform.data`       — market-data ingestion & validation
- :mod:`quant_platform.features`   — technical / cross-sectional feature pipeline
- :mod:`quant_platform.models`     — baseline & ML models, time-series CV
- :mod:`quant_platform.backtest`   — vectorized backtesting engine
- :mod:`quant_platform.risk`       — risk & performance analytics
- :mod:`quant_platform.tracking`   — lightweight experiment tracking
- :mod:`quant_platform.reporting`  — plots and Markdown report generation

DISCLAIMER: For educational / portfolio purposes only. Not financial advice
and not intended for live trading.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
