# Modeling

The modeling layer emphasizes leakage control and explainability.

## Target definitions

For every `(date, ticker)` row:

- `target_forward_return` is the return from `t` to `t + horizon`
- `target_direction` is 1 when that forward return is positive, otherwise 0

The final `horizon` rows for each ticker have unknown future returns and are dropped before
training.

## Time-series splits

`TimeSeriesSplitter` supports:

- `walk_forward`: fixed-size train and test windows that move forward
- `expanding`: growing train window with fixed test blocks

An embargo gap can be inserted between train and test blocks to reduce label overlap.

## Estimators

The estimator factory supports:

- logistic regression
- ridge / ridge classifier
- random forest
- sklearn gradient boosting
- optional XGBoost
- optional LightGBM
- optional PyTorch LSTM wrapper

Optional backends degrade gracefully when unavailable where reasonable.

## Out-of-sample signals

Each fold fits a fresh model on historical rows and predicts a future test block. Fold
predictions are concatenated into the signal frame consumed by the backtester. This means
the backtest is driven by out-of-sample model predictions rather than fitted in-sample
scores.

## Metrics and artifacts

Classification runs report accuracy, precision, recall, F1, ROC-AUC, and Brier score.
Regression runs report MAE, RMSE, R-squared, and rank correlation when available. The final
model is persisted with the exact feature list, and feature importance is plotted in the
report.
