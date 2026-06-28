# Resume Positioning

This project is intended to demonstrate quant research engineering range: data pipelines,
time-series features, ML infrastructure, backtesting, risk, tests, and production-style
packaging.

## Short project description

Built a production-style quant finance research data platform that ingests and validates
market data, engineers factor features, trains walk-forward ML models, runs
transaction-cost-aware vectorized backtests, computes risk analytics, tracks experiments,
and generates reproducible reports.

## Resume bullets

- Built an end-to-end quant research data platform in Python with typed YAML configs,
  Parquet datasets, factor engineering, walk-forward ML, vectorized backtesting, risk
  analytics, experiment tracking, reports, tests, Docker, and CI.
- Implemented leakage-aware time-series training with walk-forward splits, embargo support,
  out-of-sample signal generation, feature importance, and benchmark-aware financial
  metrics.
- Engineered reusable quant feature pipelines covering volatility, momentum, mean
  reversion, beta, drawdown, volume, technical indicators, and cross-sectional ranks.
- Developed a transaction-cost-aware portfolio backtester supporting long-only and
  long/short strategies, position caps, turnover, slippage, drawdowns, monthly returns,
  and benchmark comparison.
- Added lightweight experiment tracking with dataset hashes, model/backtest parameters,
  metrics, artifacts, timestamps, and git commit metadata for reproducible research runs.

## LinkedIn summary

I built an end-to-end quant finance research platform that turns market data into validated
Parquet datasets, factor features, walk-forward ML signals, cost-aware backtests, risk
reports, experiment records, and CI-tested reproducible artifacts.

## Interview talking points

- How the backtester avoids lookahead bias by shifting weights forward one period.
- Why time-series CV differs from random train/test splits.
- How dataset hashes and config snapshots improve experiment reproducibility.
- Tradeoffs of free market data and synthetic fallback for portfolio projects.
- How transaction costs, slippage, turnover, and benchmark comparison change strategy
  evaluation.
