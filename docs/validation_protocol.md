# Validation Protocol

This protocol defines the minimum evidence required to move a Signalattice experiment
from an implementation check to a research candidate. It separates what the repository
can verify today from controls that require institutional data and execution systems.

## Evidence levels

| Level | Meaning | Capital implication |
|---|---|---|
| E0 — unit evidence | Local invariants and deterministic calculations pass. | None. |
| E1 — synthetic evidence | Known-signal and null DGP runs behave as expected. | None. |
| E2 — historical research | Untouched walk-forward public/licensed data clears declared gates. | Hypothesis only. |
| E3 — external replication | Independent data, researcher, and implementation reproduce the result. | Candidate for shadow evaluation. |
| E4 — shadow operations | Point-in-time feeds, production features, paper decisions, and cost attribution meet an SLA. | Candidate for governed capital review. |
| E5 — controlled deployment | Risk, compliance, operations, and staged-capital controls approve monitoring and rollback. | Limited mandate only. |

The repository's committed synthetic report is E1. A local public-data run is at most E2.
The software does not claim E3–E5.

## 1. Freeze the experiment contract

Before observing test results, record:

- forecast target and horizon;
- universe, benchmark, source, price convention, and date range;
- feature list and transformation parameters;
- outer split, embargo, inner calibrator-fit/weighting partition, and estimator
  configuration;
- decision timing, selection, sizing, exposure, and rebalance rules;
- cost, delay, AUM, participation, and latency grids;
- readiness thresholds; and
- seed, code revision, and artifact paths.

A post-result change creates a new experiment. It cannot be described as confirmation on
the original holdout.

## 2. Validate data and leakage boundaries

Required automated checks:

- schema, type, uniqueness, OHLC, volume, missingness, and minimum-history validation;
- actual source and full requested-universe coverage;
- matching panel/config/seed and feature-manifest fingerprints;
- trailing ticker-local rolling features and same-date-only cross-sectional transforms;
- no target/future fields in model inputs;
- whole-date outer and inner partitions;
- candidate-training dates strictly before calibrator-fit dates;
- calibrator-fit dates strictly before independent weighting dates;
- weighting dates strictly before the embargo and outer test block;
- embargo greater than or equal to the forecast horizon; and
- temporal histories confined to the same ticker and causal row order.

For real-data E2 and beyond, also require point-in-time universe membership, corporate-
action reconciliation, as-of availability timestamps, delisting treatment, vendor
reconciliation, and a documented missing-data policy. The current public adapters do not
satisfy those institutional controls.

## 3. Verify implementation mechanics

The unit and integration suite must cover at least:

- deterministic config and cache behavior;
- candidate-fit/calibrator-fit/weighting separation and ensemble persistence;
- calibration/proper-score calculations on hand-checkable examples;
- panel sequence construction, causal TCN shape contracts, early stopping, and round-trip
  persistence when PyTorch is installed;
- conservative close-signal execution lag;
- missing signal dates preserving the decision calendar;
- missing held-asset or benchmark returns failing closed;
- position/gross constraints before and after volatility targeting;
- drawdown including an initial-capital high-water mark;
- exact cost, delay, break-even-cost, participation, and latency calculations; and
- deterministic end-to-end artifact generation.

Static formatting, linting, type checking, package build/import, and supported-Python CI
must pass on the exact revision used for evidence.

## 4. Evaluate predictive evidence

Use only concatenated outer-test forecasts. Report, at minimum:

- ROC-AUC and average precision with class prevalence;
- balanced accuracy and Matthews correlation at the declared threshold;
- Brier score/skill, log loss, expected calibration error, and reliability diagram;
- Brier reliability, resolution, and uncertainty;
- per-fold metrics and positive-fold fraction;
- date-block bootstrap intervals;
- candidate versus ensemble metrics and weights; and
- prediction-decile forward returns and selective coverage/accuracy.

Investigate—not average away—sign changes, isolated successful folds, collapsed ensemble
weights, empty confidence regions, or calibration bins with little support. Calibration
does not rescue weak ranking; high ROC-AUC does not guarantee useful probabilities.

## 5. Evaluate decision evidence

Use the exact outer-test score stream. Confirm:

- close-derived scores incur the declared execution lag;
- all positions obey per-name and gross limits;
- transaction cost and slippage are charged on absolute weight change;
- gross and net P&L, cost drag, and turnover reconcile arithmetically;
- net results remain interpretable across the full cost grid;
- added execution delay uses a common evaluation window;
- break-even cost materially exceeds the configured base assumption;
- performance is not concentrated in one fold, ticker, month, or tail event; and
- benchmark, exposure, drawdown, VaR/CVaR, and stress outputs are reviewed together.

A favorable net Sharpe is insufficient if return separation is absent, turnover is
implausible, or results disappear under a small added delay.

## 6. Evaluate operational evidence

The current warm inference benchmark records batch size, warmup/measured runs, timer,
p50/p95/p99, and throughput. An E3+ evidence package must also capture hardware, thread
settings, and dependency versions. The benchmark excludes feature materialization and
network/order latency; that exclusion must remain visible.

Dollar-volume scenarios must report liquidity coverage and label capacity as a proxy. E3+
requires calibrated spread/impact estimates, borrow and financing constraints, intraday
volume profiles, venue/auction assumptions, and an event-driven replay.

## 7. Apply the readiness gate

Each configured criterion is independent:

- predictive calibration;
- predictive discrimination;
- net economic quality;
- break-even cost headroom;
- cross-validation stability; and
- warm operational latency.

Missing or non-finite observations fail. Overall `READY` requires all included criteria to
pass. Thresholds and observed values must be printed in the report; no weighted composite
may hide a failure.

The gate is necessary but not sufficient. At E2 it means **ready for deeper research**,
not ready for live trading.

## 8. Replication and shadow requirements

Before E3/E4, require:

- an untouched later time period and materially different regimes;
- independent reproduction from config and source snapshot;
- sensitivity to universe construction and alternative vendors;
- explicit experiment registry and multiple-testing accounting;
- frozen model/data cards and signed artifact lineage;
- point-in-time feature parity between research and shadow inference;
- paper decisions with timestamped end-to-end latency and executable-price attribution;
- drift, calibration, data-quality, and cost monitors with alert thresholds; and
- rollback, kill-switch, incident, and model-change procedures.

## Stop conditions

Return an experiment to research when any of the following occurs:

- data lineage or as-of availability cannot be reconstructed;
- a leakage or timing invariant fails;
- calibration, discrimination, or net value falls below its frozen threshold;
- the result depends on one test block or undocumented selection;
- cost/delay/capacity conclusions rely on unsupported market assumptions;
- live/shadow feature parity cannot be established; or
- monitoring, ownership, or rollback is undefined.

The correct output of this protocol is often `NOT_READY`. Making that conclusion explicit
is a feature of Signalattice, not a failed demonstration.
