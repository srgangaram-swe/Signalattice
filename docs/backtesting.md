# Backtesting Contract

The backtester converts an already out-of-sample score panel into a daily research P&L.
It is designed to expose decision and implementation assumptions, not to approximate an
exchange matching engine.

## Timing convention

Features derived from a daily close are available only after that close. Signalattice
therefore refuses a same-close fill and enforces an execution lag of at least two panel
rows:

```text
close t observed
    -> score and desired weight formed after close t
    -> wait through close t+1
    -> first held close-to-close return ends at t+2
```

For configured lag `L`, the engine computes:

```text
held_weight[t + L] = desired_weight[t]
gross_return[t + L] = sum_i held_weight[t + L, i] * asset_return[t + L, i]
```

This is conservative for daily-bar research because it never assumes a fill at a close
already used to construct the signal. When the label horizon is one day and execution lag
is two rows, the portfolio test measures whether forecast information persists beyond the
exact label interval. Predictive metrics and tradable-decision metrics should therefore be
interpreted separately.

Missing score dates retain the previous desired portfolio so a sparse signal frame cannot
silently delete P&L dates. A missing return for an active position, a missing benchmark
return, duplicate keys, or no common universe raises an error rather than being converted
to zero.

## Selection and sizing

Supported strategy policies are:

- `long_only`: allocate only to bullish selections; and
- `long_short`: allocate equal gross budget to bullish and bearish selections when using
  threshold selection.

Scores can be selected by cross-sectional quantile or by calibrated probability
thresholds. A long/short policy remains flat if the available universe cannot support both
sides; it does not silently turn a nominally market-neutral policy into a directional
position.

Sizing modes are:

- `equal_weight`: equal allocation within selected sides;
- `rank`: allocation from cross-sectional score ranks; and
- `vol_target`: equal-weight base portfolio followed by a trailing volatility overlay.

The volatility estimate is shifted so only prior realized strategy returns determine the
current multiplier. Individual weights are clipped by `max_position_weight`, gross
exposure is capped by `max_leverage`, and both limits are re-applied after the overlay.
When clipping is asymmetric, the larger side is scaled down to restore neutrality; capital
is never scaled up or redistributed in a way that could violate another position cap.

An optional `rebalance_threshold` creates a causal per-position no-trade band: a target is
updated only when its absolute change reaches the threshold.

## Costs and turnover

Turnover is absolute one-way weight change, including the initial transition from cash:

```text
turnover[t] = sum_i |weight[t, i] - weight[t - 1, i]|
cost[t] = turnover[t] * (cost_bps + slippage_bps) / 10,000
net_return[t] = gross_return[t] - cost[t]
```

This flat-bps model is transparent and useful for sensitivity analysis. It does not model
spread variation, nonlinear impact, borrow, financing, auction mechanics, partial fills,
or venue fees.

## Implementation diagnostics

The evaluation layer reruns or transforms the same out-of-sample decision stream to
produce:

- net performance across a grid of total one-way costs;
- incremental execution delays beyond the baseline lag, evaluated on a common date
  window;
- arithmetic break-even one-way cost from aggregate gross P&L and traded notional;
- gross-to-net cost drag;
- turnover and exposure histories; and
- AUM participation scenarios using trailing median dollar volume.

The capacity output is explicitly labeled a **dollar-volume proxy**. It estimates required
trade notional divided by observable trailing liquidity and reports coverage, median/p95/
maximum participation, the fraction above a limit, and the implied AUM at that limit. It
does not estimate market impact or executable capacity.

## Outputs

`BacktestResult` retains:

- gross and net returns;
- explicit costs and turnover;
- strategy and benchmark equity curves;
- held weights and exposure history;
- drawdown path, monthly returns, and trade summary; and
- strategy and benchmark performance statistics.

The cost, delay, capacity, stability, calibration, and latency results are evaluated
independently by the readiness gate. A strong Sharpe ratio cannot compensate for failed
calibration, unstable folds, inadequate cost headroom, or missing operational evidence.

## What this does not establish

A vectorized daily-bar simulation cannot answer whether an order would have filled, how
its impact scales, whether a short was borrowable, or whether signal generation completed
before a venue deadline. It is a screening layer. Event-driven replay, point-in-time
reference data, broker constraints, paper trading, and post-trade reconciliation would be
required before a live-trading claim.
