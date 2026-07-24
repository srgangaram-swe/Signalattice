# ADR 0001: Center Signalattice on Probabilistic ForecastOps

- **Status:** Accepted
- **Date:** 2026-07-14

## Context

A conventional quant portfolio repository can appear impressive while leaving its most
important questions unanswered: Were scores produced out of sample? Are probabilities
calibrated? Does performance survive delay and costs? Are failure criteria explicit? Is the
reported model the model that actually ran?

This repository also needs a technical identity distinct from AlphaForge's emphasis on
systematic alpha, regimes, portfolio construction, and execution-oriented research.
Duplicating that surface area would add code without adding a different research claim.

## Decision

Signalattice will be a calibrated probabilistic forecasting and model-risk system whose
primary deliverable is a decision-readiness evidence report.

The architecture will therefore prioritize:

1. fingerprinted data and feature contracts;
2. whole-date outer walk-forward evaluation;
3. chronology-separated candidate fitting, probability-calibrator fitting, and independent
   model weighting;
4. a safe causal panel temporal model rather than row-windowed sequence modeling;
5. proper scoring, reliability, selective prediction, decile, bootstrap, and fold-stability
   diagnostics;
6. conservative score-to-position timing and explicit portfolio invariants;
7. cost, delay, break-even-cost, dollar-volume, and warm-latency analysis; and
8. an auditable all-criteria readiness gate that fails on missing evidence.

Synthetic data will be treated as declared known-signal/null engineering evidence. Public
historical data will be treated as exploratory. Neither will be described as proof of live
profitability.

## Consequences

Positive consequences:

- probability quality becomes a first-class output instead of an incidental Brier score;
- model identity and failure modes are visible;
- deep learning demonstrates causal panel construction and reproducible training;
- economic and operational assumptions can be falsified independently; and
- the project complements rather than clones an execution-focused quant stack.

Costs and tradeoffs:

- independent inner calibrator-fit and weighting windows reduce the sample available for
  candidate fitting and for each selection stage;
- conservative execution lag can make headline performance less attractive;
- all-criteria gating will often return `NOT_READY`;
- daily bars and dollar-volume proxies cannot answer microstructure questions; and
- additional diagnostics increase runtime and report complexity.

These costs are accepted because optimistic evidence would undermine the project's purpose.

## Alternatives considered

### Maximize backtest breadth

Adding more strategy families, optimizers, and execution knobs would overlap AlphaForge
and encourage comparison by headline Sharpe rather than forecast validity. Rejected.

### Lead with a large deep network

A transformer-sized model on a small daily panel would add parameters faster than
identifiable information. The compact causal TCN provides a meaningful temporal modeling
path while keeping chronology and reproducibility inspectable. Rejected as the default.

### Use random cross-validation and post-hoc calibration

Random folds destroy temporal order; calibration on outer-test predictions leaks the
evaluation set. Rejected.

### Collapse evidence into one score

A weighted composite could allow strong latency or Sharpe to compensate for failed
calibration or stability. Independent criteria are easier to audit and harder to game.
Rejected.

## Follow-up

Future milestones may add point-in-time data contracts, Bayesian cross-asset structure,
drift monitoring, shadow inference, and reproducible release artifacts. Event-driven
execution and broker integration remain outside this ADR unless the product boundary is
explicitly reconsidered.
