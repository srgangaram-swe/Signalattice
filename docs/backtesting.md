# Backtesting

The backtester is a vectorized cross-sectional engine for daily bars. It is designed for
research credibility rather than live execution.

## Timing convention

A score observed at the close of date `t` creates target weights for date `t`. Those
weights are shifted and earn returns on `t + 1`.

```text
signal[t] -> weights[t] -> realized return[t + 1]
```

This shift is central: it prevents same-day close prices from being used both to generate a
signal and earn the return.

## Strategies

- `long_only`: buy the highest-ranked names.
- `long_short`: buy the highest-ranked names and short the lowest-ranked names.

## Sizing

- `equal_weight`: equally weight selected names.
- `rank`: allocate by cross-sectional rank tilt.
- `vol_target`: apply a trailing volatility overlay using only past returns.

Position weights are capped by `max_position_weight`, and gross exposure is bounded by
`max_leverage`.

## Costs

Transaction costs and slippage are modeled as basis points of turnover:

```text
cost = turnover * (cost_bps + slippage_bps) / 10000
```

Turnover is measured as the absolute change in target weights. Costs are charged when the
new position becomes active.

## Outputs

The engine returns:

- net and gross strategy returns
- strategy and benchmark equity curves
- weights and turnover
- drawdown path
- exposure summary
- monthly return table
- trade summary
- performance and benchmark statistics

Plots generated for the report include equity curve, drawdown, rolling Sharpe, return
distribution, and monthly returns.
