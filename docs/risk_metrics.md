# Risk Metrics

Risk analytics operate on periodic simple returns and annualize with a configurable trading
day count, defaulting to 252.

## Performance metrics

- CAGR / annualized return
- annualized volatility
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- hit rate
- skew and kurtosis

## Drawdown

Drawdown is computed from the cumulative return curve:

```text
drawdown[t] = equity[t] / running_peak[t] - 1
```

Maximum drawdown is the most negative drawdown over the evaluation window.

## Tail risk

Value-at-Risk is reported as a positive loss magnitude at the configured confidence level.
The default method is historical quantile. Conditional VaR, also called expected shortfall,
is the average loss beyond the VaR threshold.

## Market risk

Beta is computed as covariance with benchmark returns divided by benchmark variance.
Correlation matrices are generated from the return panel and plotted as a heatmap.

## Exposure and stress

The backtester records gross exposure, net exposure, long exposure, short exposure, and
position counts through time. Stress tests apply configured scenario shocks to estimate
portfolio sensitivity under equity selloff and volatility-shock assumptions.
