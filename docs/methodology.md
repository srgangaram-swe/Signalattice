# Methodology

The research workflow is designed to be realistic while remaining reproducible on a
personal machine.

## Data ingestion

The ingestion layer supports Yahoo Finance via `yfinance`, Stooq via `pandas-datareader`,
and a deterministic synthetic generator. `source: auto` attempts public data first and
falls back to synthetic data when optional dependencies or network access are unavailable.

Every source is normalized into a canonical long OHLCV panel and validated before feature
engineering. Validation checks include required columns, duplicate `(date, ticker)` rows,
positive prices, OHLC sanity bounds, non-negative volume, minimum observations, and NaN
accounting.

## Feature engineering

Features are trailing and ticker-local unless explicitly cross-sectional. This prevents
one asset's future values or another date's values from contaminating a row.

Implemented feature groups:

- return and log-return horizons
- rolling volatility and realized volatility
- rolling Sharpe
- moving-average and EMA ratios
- RSI, MACD, Bollinger percentage/bandwidth
- momentum, 12-1 momentum, reversal, z-score
- volume trend and dollar-volume features
- rolling beta to a benchmark
- drawdown and rolling max drawdown
- cross-sectional ranks and z-scores by date

Targets are forward-looking by construction:

- `target_forward_return`: future return from `t` to `t + horizon`
- `target_direction`: binary label for whether that forward return is positive

## Modeling

The training harness uses walk-forward or expanding-window splits with an optional embargo.
Each fold fits on past rows and predicts a future holdout block. The resulting out-of-sample
signal frame is used by the backtester, while a final model fitted on all rows is saved for
feature-importance inspection.

Supported model families include logistic regression, ridge, random forest, gradient
boosting, optional XGBoost/LightGBM, and optional PyTorch LSTM.

## Evaluation

Classification metrics include accuracy, precision, recall, F1, ROC-AUC, and Brier score
where applicable. Financial evaluation happens in the backtest layer and includes Sharpe,
Sortino, max drawdown, CAGR, turnover, hit rate, VaR, CVaR, beta, and benchmark comparison.

## Reproducibility

Runs are controlled by YAML configs. The tracker records config parameters, dataset hash,
tickers, feature list, model/backtest metrics, artifacts, timestamps, and the git commit
when the code is inside a git repository.
