# Risk Metrics

Risk analytics operate on periodic simple returns and annualize with a configurable period
count, defaulting to 252 trading days. These are descriptive statistics for a simulated
return stream, not a complete market or enterprise risk model.

## Performance

- geometric annualized return / CAGR;
- annualized volatility;
- Sharpe and Sortino ratios;
- Calmar ratio;
- positive-period hit rate; and
- skew and excess kurtosis.

The configured annual risk-free rate is converted to a per-period value before Sharpe and
Sortino calculations. Ratios with zero or numerically negligible denominators return
`NaN`; they are not reported as infinite skill.

## Drawdown

Drawdown uses an explicit initial capital point of `1.0`:

```text
equity[t] = product_(s <= t) (1 + return[s])
peak[t] = max(1.0, equity[0], ..., equity[t])
drawdown[t] = equity[t] / peak[t] - 1
```

Including initial capital matters when the first observed return is negative: the first
loss is a drawdown, not a new high-water mark. Maximum drawdown is the most negative point
on this path.

## Tail loss

Value-at-Risk is a positive one-period loss magnitude at the configured confidence level.
The report uses the historical empirical quantile. Conditional VaR / expected shortfall is
the mean loss in observations at or beyond that threshold.

Historical VaR and CVaR inherit the observed sample's regime coverage and serial
dependence. They do not extrapolate unseen crises or establish a capital reserve.

## Benchmark and exposure

Beta is covariance with aligned benchmark returns divided by benchmark variance. The
performance block also reports a simple annualized alpha residual. Per-date exposure
diagnostics include gross, net, long, and short exposure plus active position counts.
Pairwise asset-return correlations are descriptive full-window estimates.

## Stress calculations

Configured scenario names select a transparent calculation:

- equity/market shocks apply `portfolio_beta * market_shock`;
- volatility shocks scale historical VaR; and
- custom shocks pass through the declared value.

These are first-order sensitivity illustrations, not a repricing engine. They omit
nonlinear instrument payoffs, changing correlation and beta, liquidity feedback, path
dependence, margin, financing, and forced deleveraging.

## Interpretation

Risk statistics are shown alongside turnover, cost, delay, capacity-proxy, calibration,
and fold-stability evidence. No risk-adjusted return ratio can by itself clear the
decision-readiness gate or support a live-trading claim.
