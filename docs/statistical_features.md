# Volatility, distribution, and dependence contracts (SF-S2-MR2)

This document describes the SF-S2-MR2 statistical feature contracts. They extend
the versioned, causal contract framework from
[SF-S2-MR1](feature_contracts.md) with volatility estimators, robust
distributional statistics, and dependence/memory estimators. The new estimators
live in
[`quant_platform.features.statistics`](../src/quant_platform/features/statistics.py);
the contract wrappers live in the same registry as the conventional features
([`contracts.py`](../src/quant_platform/features/contracts.py)), so
`build_contract_features`, `get_contract`, `list_contracts`, and
`contract_metadata_frame` cover both slices, and `statistical_contracts()`
returns just this set.

Every estimator obeys the same invariants as MR1: **causal** (trailing windows
only), **full-window warm-up** (`NaN` before position `warmup - 1`), **fail
closed** on malformed input, **propagate-NaN** through gaps, and **deterministic**
regardless of input row order.

## The default contract set

| name | family | unit | lookback | warmup | inputs |
|------|--------|------|---------:|-------:|--------|
| `parkinson_vol_21` | volatility | annualized_vol | 21 | 21 | high, low |
| `garman_klass_vol_21` | volatility | annualized_vol | 21 | 21 | open, high, low, close |
| `rogers_satchell_vol_21` | volatility | annualized_vol | 21 | 21 | open, high, low, close |
| `ewma_vol_21` | volatility | annualized_vol | 21 | 21 | adj_close |
| `skew_63` | distribution | dimensionless | 63 | 63 | adj_close |
| `kurt_63` | distribution | dimensionless | 63 | 63 | adj_close |
| `downside_dev_63` | distribution | annualized_vol | 63 | 63 | adj_close |
| `mad_63` | distribution | simple_return | 63 | 63 | adj_close |
| `autocorr_1_63` | dependence | correlation | 64 | 64 | adj_close |
| `pacf_2_63` | dependence | correlation | 65 | 65 | adj_close |
| `hurst_128` | dependence | dimensionless | 128 | 128 | adj_close |
| `var_ratio_5_63` | dependence | dimensionless | 63 | 63 | adj_close |
| `beta_mkt_63` | dependence | dimensionless | 64 | 64 | adj_close |
| `corr_mkt_63` | dependence | correlation | 64 | 64 | adj_close |
| `mutual_info_mkt_63` | dependence | information_nats | 64 | 64 | adj_close |

## Mathematics, assumptions, and stability

Windows are trailing and bar-based. `H`, `L`, `O`, `C` are the OHLC bars, `r_t`
the one-bar adjusted-close return, and annualisation uses `sqrt(252)`.

### Volatility

* **Parkinson** — `sigma^2 = 1/(4 ln2) * mean((ln(H/L))^2)`. Assumes positive
  prices and continuous trading; ignores overnight gaps, so it underestimates
  when jumps dominate. More efficient than close-to-close under those
  assumptions.
* **Garman-Klass** — `sigma^2 = mean(0.5(ln(H/L))^2 - (2 ln2 - 1)(ln(C/O))^2)`.
  The windowed variance is clipped at zero before the root to absorb
  small-sample negativity. Assumes no overnight drift/jump between close and
  next open.
* **Rogers-Satchell** — `sigma^2 = mean(ln(H/C)ln(H/O) + ln(L/C)ln(L/O))`. Each
  per-bar term is non-negative and the estimator is **drift-independent**, so it
  stays valid under a trend.
* **EWMA** — RiskMetrics-style `sqrt(EWMA(r^2))` with `alpha = 2/(span+1)`,
  `adjust=False`.

Minimum samples equal the window (or span). All are `>= 0`.

### Distribution

* **Skewness / excess kurtosis** — bias-adjusted sample moments (Fisher-Pearson;
  excess kurtosis is 0 for a normal). Need `>= 3` and `>= 4` observations.
  Validated against `scipy.stats`.
* **Downside deviation** — annualised RMS of negative returns (`>= 0`).
* **Median absolute deviation** — `median(|x - median(x)|)`, a robust dispersion
  in return units that is insensitive to outliers (validated against
  `scipy.stats.median_abs_deviation`, `scale=1`).

### Dependence and memory

* **Autocorrelation (lag k)** — trailing Pearson correlation of `r_t` and
  `r_{t-k}`, clipped to `[-1, 1]`.
* **Partial autocorrelation (lag 2)** — closed form `(r2 - r1^2)/(1 - r1^2)`
  from the first two autocorrelations, denominator guarded, clipped to
  `[-1, 1]`.
* **Hurst exponent** — slope of `log(std of L-step log-price differences)` on
  `log(L)` over lags `L in {2,4,8,16,32}` inside the window (structure-function
  estimator). `~0.5` random walk, `>0.5` persistent, `<0.5` mean-reverting.
  Complexity `O(window * n_lags)` per bar.
* **Variance ratio (Lo-MacKinlay)** — `VR(q) = Var_q / Var_1` with the overlapping,
  bias-corrected per-period `q`-return variance. `~1` random walk, `>1`
  trending, `<1` mean-reverting. Undefined when the one-period variance is zero.
* **Beta / correlation to market** — rolling beta and correlation of returns to
  an **equal-weight cross-sectional mean-return market proxy** (self-contained;
  needs no designated benchmark ticker). On a single-name date the proxy is that
  name's own return.
* **Mutual information to market** — plug-in binned MI (nats, `>= 0`) between
  returns and the market proxy over the window; captures nonlinear dependence
  the correlation misses. Plug-in MI is positively biased for small samples and
  many bins, so `bins` is kept small (8) relative to the window.

## Configuration and usage

```python
from quant_platform.features import build_contract_features, statistical_contracts, get_contract

# Just the SF-S2-MR2 statistical set.
matrix = build_contract_features(panel, statistical_contracts())

# A named subset.
matrix = build_contract_features(panel, [get_contract("garman_klass_vol_21"), get_contract("hurst_128")])

# Introspect an estimator's declared metadata.
print(get_contract("var_ratio_5_63").metadata())
```

`build_contract_features(panel)` with no contract argument evaluates the full
registry (conventional MR1 + statistical MR2).

## Reproducibility and benchmark evidence

Deterministic (no randomness in the kernels; order-independent). Reproduce with
`pytest tests/test_statistical_features.py -q` and
`python scripts/benchmark_statistical_contracts.py`.

Laptop-scale, single process, synthetic deterministic data (seed 7),
Python 3.13 / numpy 2.5 / pandas 3.0 on Apple Silicon. The per-window kernels
(Hurst, variance ratio, mutual information) dominate; absolute numbers depend on
hardware and warm state.

| tickers × dates | rows | seconds | rows/s | peak memory |
|-----------------|-----:|--------:|-------:|------------:|
| 16 × 504 | 8,064 | 3.6 | ~2,240 | ~2 MB |
| 32 × 1,008 | 32,256 | 14.9 | ~2,170 | ~8 MB |
| 64 × 1,260 | 80,640 | 38.1 | ~2,120 | ~19 MB |

## Security and data

No credential, licensed observation, proprietary dataset, generated run, or
final-holdout artifact is committed. Tests and benchmark use in-process
synthetic data.

## Risks, rollback, and residual limitations

* **Rollback.** Purely additive: a new `statistics.py` module and new registry
  entries; MR1 behaviour, `build_features`, and existing contracts are
  unchanged. Reverting the commit restores the previous state with no migration.
* **Per-window kernels are the cost centre.** Hurst, variance ratio, and mutual
  information use `rolling().apply` / per-window loops (`~2,000 rows/s`).
  Vectorising them is tracked follow-up work; the vectorised MR1 features remain
  ~100× faster.
* **Market proxy is equal-weight.** Beta/correlation/MI are measured against the
  equal-weight cross-sectional mean return, not a designated index; this is a
  deliberate self-contained choice, documented per contract.
* **Range-based volatility assumptions.** Parkinson/Garman-Klass ignore overnight
  gaps and assume continuous trading; Rogers-Satchell is drift-independent but
  still range-based. They complement, not replace, close-to-close volatility.
* **Plug-in MI bias.** Small-sample, histogram-based MI is positively biased;
  treat magnitudes comparatively, not absolutely.
* **No predictive claim.** These are conventional statistical baselines; edge is
  established only by the SF-S2 modelling and governance slices against an
  untouched holdout.
* **Sprint 1 integration dependency.** Built on the current `dev` price-panel
  schema and the MR1 contract interface; final integration with the Sprint 1
  point-in-time/lineage/calendar contracts follows once those merge.
