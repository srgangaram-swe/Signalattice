# Methodology

Signalattice treats forecasting as a sequence of falsifiable contracts. Predictive
discrimination, probability quality, economic sensitivity, and operational feasibility
are measured separately so a favorable result in one dimension cannot hide a failure in
another.

## 1. Declare the experiment

A typed YAML config fixes the data source, universe, date window, feature parameters,
forecast target, outer cross-validation, inner calibrator-fit and weighting partitions,
portfolio policy, friction grids, readiness thresholds, artifact paths, and random seed.
Unknown fields and invalid combinations fail validation.

Important cross-field contracts include:

- classification uses the direction target and regression uses forward return;
- the embargo is at least the forward horizon;
- the calibrated ensemble is currently classification-only; and
- close-derived backtests use an execution lag of at least two rows.

## 2. Establish data identity

Yahoo Finance and Stooq are optional public sources. Synthetic data is an explicit source
for offline and CI experiments; automatic source selection does not silently synthesize a
missing universe unless `allow_synthetic_fallback` is deliberately enabled.

Each source is normalized into a canonical long OHLCV panel. Validation checks required
columns, unique `(date, ticker)` keys, positive prices, OHLC bounds, non-negative volume,
minimum history, and missingness. All requested tickers must be present.

The processed cache is keyed by the resolved data config and seed. The metadata sidecar
records source identity, universe, row count, date range, data hash, config hash, seed, and
whether the data is synthetic. Writes use temporary files followed by atomic replacement.

The deterministic synthetic process combines a common market factor, heterogeneous
betas, idiosyncratic noise, volatility regimes, and configurable market/idiosyncratic
AR(1) coefficients. The default nonzero coefficients are a declared test signal, not a
market discovery. Both can be set to zero for a directional null experiment.

## 3. Build causal features and targets

Features are trailing and ticker-local unless explicitly cross-sectional. Same-date
cross-sectional ranks and z-scores do not access later dates. Families include:

- simple and log returns;
- rolling and realized volatility and trailing Sharpe;
- moving-average and exponential-moving-average ratios;
- RSI, MACD, and Bollinger statistics;
- momentum, 12-1 momentum, reversal, and rolling z-scores;
- volume and dollar-volume transforms;
- rolling benchmark beta;
- drawdown and rolling maximum drawdown; and
- same-date cross-sectional ranks and z-scores.

Targets are forward-looking by definition and are never inputs:

- `target_forward_return`: return over the configured future horizon; and
- `target_direction`: indicator that the forward return is positive.

Rows without a known future target are excluded from fitting. A feature-manifest
fingerprint prevents an existing matrix from being reused under incompatible feature,
model-horizon, panel, or seed settings.

## 4. Produce honest out-of-sample forecasts

Outer walk-forward or expanding splits operate on unique sorted dates, not shuffled rows.
All tickers for a date move into the same partition. Every emitted test block is later than
its training block and separated by the configured embargo.

Tabular runs support linear, forest, sklearn gradient-boosting, and optional XGBoost or
LightGBM estimators. The ensemble fits heterogeneous candidates on the earlier portion of
each outer training fold. Its trailing, whole-date inner holdout is divided into an earlier
calibrator-fit sub-window and a later independent weighting sub-window. Sigmoid or isotonic
mappings are frozen before calibrated candidate log loss is measured for ensemble weights.
That candidate-fit/calibrator-fit/weighting sequence is repeated independently for every
outer fold.

The optional temporal path builds fixed-length histories within each ticker and trains a
causal dilated temporal convolutional network. A chronological validation tail controls
early stopping; AdamW, gradient clipping, deterministic CPU execution, and persistable
preprocessing/model state make the experiment repeatable. Gradient-times-input values are
reported as sensitivity diagnostics, not causal explanations.

Only concatenated outer-test predictions feed model diagnostics and the backtest. A final
all-history model is persisted for later inference, but its fitted values are not inserted
into reported out-of-sample performance.

## 5. Evaluate forecast quality

Classification evidence includes:

- ROC-AUC and average precision for discrimination;
- balanced accuracy and Matthews correlation at the declared threshold;
- log loss, Brier score, Brier skill, and expected calibration error;
- reliability tables and Brier reliability/resolution/uncertainty decomposition;
- score distributions conditioned on realized outcome;
- selective coverage versus accuracy when abstaining near `0.5`;
- prediction-decile forward-return separation;
- per-fold candidate and ensemble comparisons;
- date-block bootstrap intervals; and
- fold feature-importance stability where the estimator exposes a meaningful measure.

Regression evidence includes RMSE, MAE, R-squared, and directional agreement. Economic
metrics are never described as predictive calibration metrics.

## 6. Test decision value and operational fit

Out-of-sample scores are converted to constrained target weights, shifted according to the
conservative timing contract, and evaluated net of turnover-based cost and slippage.
Signalattice then measures cost sensitivity, incremental delay decay, break-even cost,
gross-to-net drag, fold profitability stability, and a trailing-dollar-volume participation
proxy across AUM scenarios.

The persisted model is also measured under warm, deterministic inference batches. Reported
p50/p95/p99 latency and throughput exclude upstream data access, feature materialization,
network transit, orchestration, risk services, and order handling.

## 7. Apply an auditable readiness gate

The gate compares observed values with configured thresholds for calibration, predictive
discrimination, net economic quality, break-even cost headroom, positive-fold fraction,
and warm inference latency. Each criterion is `PASS` or `FAIL`; missing and non-finite
metrics fail. Overall status is `READY` only when every included criterion passes.

`READY` means the run cleared its declared research screen. It does not mean approved for
capital, externally replicated, or safe for deployment. The full escalation requirements
are in the [validation protocol](validation_protocol.md).

## 8. Preserve lineage

The tracker records resolved config values, dataset hash, tickers, feature list, model and
backtest metrics, artifacts, timestamps, and Git commit when available. Model identity
includes the effective calibrated ensemble and weights where applicable. Optional backends
fail closed instead of silently substituting a different estimator.

## Interpretation rules

- Synthetic outcomes are engineering evidence only.
- A single historical universe is exploratory evidence only.
- Calibration does not imply discrimination; discrimination does not imply net value.
- Backtest performance after a parameter search is not untouched out-of-sample evidence.
- Bootstrap intervals measure sampling variation under their block assumptions, not model,
  data-vendor, regime, or execution uncertainty.
- Every result inherits the limitations in the [data card](data_card.md) and
  [model card](model_card.md).
