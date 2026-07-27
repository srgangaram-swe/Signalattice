# Modeling

The modeling layer produces one auditable object: a panel of genuinely out-of-sample
forecasts. Calibration, diagnostics, and decision simulation are downstream consumers of
that object.

## Targets and inputs

For each `(date, ticker)` row:

- `target_forward_return` is the configured future simple return; and
- `target_direction` is one when that return is positive and zero otherwise.

The final horizon rows of each ticker have no observable target and are excluded.
Identifier, raw target, and future-return fields never enter the feature set. A config
validation error is raised if target and task disagree.

## Nested chronology

For ensemble classification, each fold has five ordered regions:

```text
candidate fit dates -> calibrator-fit dates -> weighting dates -> embargo -> outer test dates
```

The outer splitter groups every ticker on a date together and supports fixed-length
walk-forward or expanding histories. The embargo is constrained to be at least the
forecast horizon. The trailing inner holdout keeps each date intact and is split once more:
an earlier sub-window fits the probability calibrators, while a later independent
sub-window measures calibrated candidate log loss and determines ensemble weights.

The whole nested sequence is repeated from scratch for every outer fold. No calibrator,
candidate weight, standardizer, or early-stopping decision is carried forward from an
outer test block.

## Chronological calibrated ensemble

The default candidate set can combine logistic regression, random forest, and sklearn
gradient boosting; XGBoost and LightGBM may be added when their optional dependencies are
installed.

Within an outer training fold:

1. each candidate is cloned and fitted on the early dates;
2. raw candidate scores on the earlier portion of the trailing holdout fit sigmoid (Platt)
   or isotonic calibrators;
3. those frozen calibrators transform scores on a later, disjoint weighting sub-window,
   where each candidate's log loss is measured; and
4. calibrated probabilities are exponentially weighted by relative log loss.

Returned probabilities are clipped away from exact zero and one. Single-class fit or
calibration regions use documented constant/identity behavior rather than producing
undefined estimates. The persisted model identity includes calibration method and fitted
candidate weights.

Isotonic calibration has more flexibility and should be reserved for sufficiently large
calibrator-fit windows. Sigmoid calibration is the conservative default for smaller
samples.
Neither method can repair a model with no discrimination or a shifted deployment
distribution.

## Causal panel TCN

`model.type: tcn` activates an optional PyTorch temporal convolutional model. Sequences are
built as `[sample, time, feature]` tensors from a single ticker's history; panel row order
cannot create cross-asset pseudo-sequences.

The network uses:

- an input projection followed by residual causal convolution blocks;
- exponentially increasing dilations;
- left-only padding, so an output cannot access a later timestep;
- layer normalization, GELU activation, and dropout;
- AdamW optimization and gradient clipping;
- a trailing, whole-date validation window with patience-based early stopping; and
- deterministic CPU execution for reproducible research evidence.

Scaler, feature order, sequence length, estimator state, epoch history, and best epoch are
persisted. Local gradient-times-input attributions can rank time/feature sensitivity, but
must not be interpreted as causal effects.

The legacy `lstm` config key resolves to the same safe temporal estimator for compatibility;
there is no row-windowing LSTM that mixes adjacent assets.

## Other estimators

Single-model experiments support logistic/ridge, random forest, sklearn gradient boosting,
and optional XGBoost/LightGBM for classification or regression where applicable. Standard
scaling is fitted inside the fold for linear models.

Optional model requests fail closed when their dependency is unavailable. Silent fallback
would make the stored experiment identity false and is therefore prohibited.

## Out-of-sample prediction frame

Every fold emits test rows with:

```text
date | ticker | y_true | forward_return | score | fold
```

Ensemble runs also retain calibrated `candidate_*` probabilities. Fold outputs are
concatenated in time and are the only model predictions accepted by reporting and
backtesting. The separately fitted final model supports persistence and future inference;
it is not used to rewrite outer-test scores.

## Metrics and diagnostics

Classification runs report threshold, ranking, and probability metrics rather than relying
on accuracy alone:

- balanced accuracy, precision, recall, F1, and Matthews correlation;
- ROC-AUC and average precision;
- Brier score, log loss, Brier skill, and expected calibration error;
- Brier reliability, resolution, and uncertainty;
- reliability diagram and score distributions;
- selective coverage/accuracy and prediction-decile return separation;
- date-block bootstrap intervals and per-fold stability; and
- candidate metrics, ensemble weights, and fold feature-importance stability.

Regression runs report MAE, RMSE, R-squared, and sign agreement. The
[model card](model_card.md) describes intended use and limitations; the
[validation protocol](validation_protocol.md) specifies how these diagnostics enter a
decision.

## Interpretable dynamic-state and risk references

The modeling package also exposes causal local-level and dynamic-linear Kalman filters,
fixed-parameter EWMA/GARCH(1,1) variance forecasts, Gaussian interval diagnostics, and an
OAS shrinkage-covariance estimator. These are standalone reference contracts rather than
new `model.type` dispatch options: callers must fit or select their parameters inside each
training fold and pass only frozen parameters into an evaluation interval.

See [State-space, volatility, and covariance baselines](state_space_volatility.md) for
equations, shapes, chronology, numerical safeguards, deterministic simulation evidence,
complexity, and limitations.
