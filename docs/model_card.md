# Model Card: Signalattice Forecast Models

## Summary

Signalattice estimates either the probability of a positive forward return or the forward
return itself for a small cross-asset daily panel. Its primary v0.2 model is a
chronologically calibrated heterogeneous classifier ensemble. An optional causal temporal
convolutional network (TCN) provides a deep-learning sequence path.

The models are research components. They do not place orders, forecast executable prices,
or represent an approved trading strategy.

## Intended use

- compare probability estimators under date-grouped walk-forward evaluation;
- inspect calibration, discrimination, uncertainty, stability, and return separation;
- test how an out-of-sample score stream degrades under conservative timing and costs;
- exercise reproducible ForecastOps and model-risk controls; and
- generate hypotheses for external validation.

Out-of-scope uses include autonomous trading, client suitability, market manipulation,
credit decisions, or any high-stakes decision based only on this model output.

## Inputs and outputs

Inputs are configured trailing technical, risk, volume, benchmark, and same-date
cross-sectional features keyed by `(date, ticker)`. Temporal inputs contain only the
history of the sample ticker.

Outputs are:

- `P(target_forward_return > 0)` for classification; or
- a point estimate of `target_forward_return` for regression.

Classification probabilities are decision scores, not expected returns. The calibrated
ensemble currently supports binary classification only.

## Model families

### Calibrated ensemble

Each outer training fold is divided chronologically into candidate-fit dates and a trailing
inner holdout. That holdout is split into earlier calibrator-fit dates and later independent
weighting dates. Candidate probabilities are calibrated with sigmoid or isotonic mappings;
the frozen mappings are then scored on the weighting dates, where log loss determines the
ensemble weights. Candidate fit, calibrator fit, weighting, embargo, and outer test dates
do not overlap.

Supported candidates include logistic regression, random forest, sklearn gradient
boosting, and optional XGBoost/LightGBM. Missing optional dependencies fail the declared
experiment rather than trigger model substitution.

### Causal TCN

The optional PyTorch model consumes `[sample, time, feature]` sequences. Residual dilated
convolutions use left-only padding. A trailing date-grouped validation window controls
early stopping. Training uses AdamW, gradient clipping, and deterministic CPU execution.
Gradient-times-input values describe local sensitivity only.

## Evaluation

All reported forecast metrics use concatenated outer walk-forward test predictions.
Evaluation covers:

- ROC-AUC, average precision, balanced accuracy, Matthews correlation, precision/recall,
  and F1;
- Brier score/skill, log loss, expected calibration error, reliability, resolution, and
  uncertainty;
- score-by-outcome distributions, selective coverage/accuracy, and prediction-decile
  forward returns;
- per-fold metrics, candidate comparisons, feature-importance stability, and date-block
  bootstrap intervals; and
- net decision performance under configured cost, delay, and portfolio constraints.

The separately persisted all-history model is not used to generate outer-test metrics.

## Readiness policy

The report applies independent configured thresholds for calibration, ROC-AUC, net Sharpe,
break-even cost, positive-fold fraction, and warm p95 inference latency. Non-finite or
missing observations fail. `READY` means only that one run cleared its declared research
screen; it is not production approval.

## Known limitations and risks

- Financial labels are noisy, non-stationary, and sensitive to source revisions.
- A current fixed universe creates selection and survivorship bias.
- Daily technical inputs omit news, fundamentals, microstructure, spread, and borrow.
- Calibration is conditional on the evaluation distribution and can fail under drift.
- Log-loss weighting on a short trailing weighting sub-window can be unstable in small
  samples, even though it is independent of calibrator fitting.
- Flexible modeling and repeated experiments create selection/multiple-testing risk.
- TCN attribution is not a causal or economic explanation.
- Flat-bps backtests and dollar-volume proxies do not establish executable capacity.
- Synthetic recovery demonstrates code behavior, not live-market alpha.

## Responsible interpretation

Review the [data card](data_card.md), source label, per-fold evidence, readiness failures,
and [validation protocol](validation_protocol.md) together. A result should be rejected or
sent back for investigation when it depends on one fold, one candidate, an implausible
cost assumption, weak calibration, a narrow universe, or an undocumented data change.

## Versioning

Model behavior is versioned through the Git commit, resolved YAML config, feature list,
dataset/config fingerprints, random seed, and persisted effective model identity. Any
change to target, source, feature manifest, calibrator-fit/weighting split, estimator,
friction model, or readiness threshold creates a distinct experiment.
