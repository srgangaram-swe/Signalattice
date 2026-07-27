# State-space, volatility, and covariance baselines

Sprint 3 MR5 adds deliberately interpretable numerical references before more expressive
sequence or latent-state models are evaluated. These baselines answer a narrow question:
does a more complex candidate add stable out-of-sample value beyond a causal,
uncertainty-aware statistical reference?

They do not select a strategy, estimate deployable alpha, authorize a trade, or turn
synthetic recovery into market evidence.

## Chronology contract

Every state or variance forecast at index \(t\) is fixed before observation \(y_t\) or
residual \(\epsilon_t\) is incorporated:

```text
posterior through t-1 -> one-step forecast for t -> observe t -> posterior through t
```

Changing any suffix beginning at index \(k\) therefore cannot change forecasts through
\(k\), nor filtered states strictly before \(k\). Tests enforce this prefix invariant.
Model parameters and initial conditions must be selected from a training interval that
ends before the evaluated observations. None of the functions estimates a parameter from
its evaluation input.

## Local-level Kalman filter

The scalar local-level model is

\[
\ell_t = \ell_{t-1} + \eta_t,\qquad
y_t = \ell_t + \varepsilon_t,
\]

with \(\eta_t \sim N(0,q)\) and \(\varepsilon_t \sim N(0,r)\). The implementation returns
the one-step mean and variance, innovation, filtered level and variance, Gaussian interval,
and total predictive log likelihood. The covariance update uses Joseph form so rounding
error does not silently create a negative posterior variance.

Runtime and retained-output memory are both \(O(n)\).

## Dynamic linear regression

For declared information vector \(x_t\), coefficients follow a random walk:

\[
\beta_t = \beta_{t-1} + \eta_t,\qquad
y_t = x_t^\top\beta_t + \varepsilon_t.
\]

The process covariance is currently isotropic and fixed. The dense Kalman recursion
retains every coefficient posterior and covariance for auditability. Joseph-form updates,
symmetrization, and a validated numerical variance floor preserve covariance structure;
material loss of positive-semidefinite structure fails closed.

For \(p\) coefficients, runtime is \(O(np^3)\) under dense covariance algebra and retained
output memory is \(O(np^2)\). The contract is intended for small interpretable state
vectors such as a time-varying intercept and benchmark beta, not an unbounded feature
matrix.

## EWMA and GARCH(1,1)

EWMA uses

\[
h_{t+1} = \lambda h_t + (1-\lambda)\epsilon_t^2,
\]

while fixed-parameter GARCH(1,1) uses

\[
h_{t+1} = \omega + \alpha\epsilon_t^2 + \beta h_t.
\]

The value returned at \(t\) is \(h_t\), before \(\epsilon_t\) is consumed. An explicit
initial variance is mandatory so evaluation data cannot silently initialize the model.
GARCH requires positive \(\omega\), non-negative \(\alpha,\beta\), and
\(\alpha+\beta<1\), making its unconditional variance finite. Both algorithms are
\(O(n)\) time and memory.

These are conditional-variance baselines, not a claim to implement a general latent
stochastic-volatility posterior. Non-Gaussian state evolution, leverage effects, and
particle filtering remain outside this bounded slice and require their own justified
model-selection and compute evidence.

## Shrinkage covariance

The covariance baseline centers \(n \times p\) observations, computes the maximum-
likelihood empirical covariance \(S\), and shrinks toward a scaled identity:

\[
\widehat{\Sigma} = (1-\rho)S + \rho\mu I,\qquad
\mu = \operatorname{tr}(S)/p.
\]

If \(\rho\) is omitted, the deterministic Oracle Approximating Shrinkage closed form
selects it. A caller may instead freeze a value chosen on an earlier training interval.
The result records the sample matrix, target, location, shrinkage intensity, minimum
eigenvalue, and before/after condition numbers. A small declared variance floor protects
the degenerate all-constant case.

Runtime is \(O(np^2+p^3)\), including eigenspectrum and conditioning diagnostics; memory is
\(O(np+p^2)\). This reference is not a point-in-time portfolio covariance service.

## Interval diagnostics

`gaussian_interval_diagnostics` consumes already-fixed out-of-sample means and variances
and reports:

- empirical interval coverage and mean width;
- standardized-innovation mean and sample standard deviation; and
- mean Gaussian negative log score, a proper scoring rule.

It performs no recalibration. Coverage evaluated on the same observations used to tune
parameters is optimistically biased and must not be used for model selection.

## Deterministic evidence

Regenerate the machine-readable benchmark and Seaborn plot:

```bash
.venv/bin/python scripts/benchmark_state_space_baselines.py \
  --output-json docs/benchmarks/state_space_baselines_2026-07-26.json \
  --output-plot docs/assets/state_space_baselines_2026-07-26.png
```

The pinned workload uses seed `20260726`, 5,000 observations, seven timing repetitions,
known linear-Gaussian states, a fixed-parameter GARCH process, and an intentionally
ill-conditioned synthetic factor matrix. The plot shows all timing samples and
error/conditioning ratios against declared naive references. Lower ratios demonstrate
implementation recovery or numerical conditioning on that controlled process only.

The evidence is single-process laptop-scale and synthetic. It excludes parameter
estimation, provider data, corporate actions, point-in-time universe membership,
transaction costs, execution, regime selection, capacity, and any live operational
boundary. It cannot establish alpha, profitability, paper-trading readiness, or
live-trading readiness.

## Failure and security boundaries

All numerical inputs are treated as untrusted. Empty, non-finite, incompatible-shape, and
unsafe-parameter inputs raise actionable exceptions before semantic use. Returned arrays
are defensive read-only copies. The modules perform no network access, credential access,
filesystem writes, logging, or global-state mutation. Benchmark output contains only
synthetic aggregates, dependency versions, and platform identity; licensed observations
and local credentials never enter it.
