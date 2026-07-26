# Volume, liquidity, spread, and impact contracts (SF-S2-MR3)

This document describes the SF-S2-MR3 liquidity feature contracts. They extend
the causal contract framework from [SF-S2-MR1](feature_contracts.md) and
[SF-S2-MR2](statistical_features.md) with **daily-bar proxies** for volume,
liquidity, bid-ask spread, and price impact. The estimators live in
[`quant_platform.features.liquidity`](../src/quant_platform/features/liquidity.py);
the contract wrappers live in the shared registry
([`contracts.py`](../src/quant_platform/features/contracts.py)).

These are proxies computed from OHLCV. Per the issue non-goal, they are **not**
observed order-book spread, depth, or executable capacity; each contract's
description states its source fields, units, zero-volume behaviour, and
interpretation limit.

## The contract set

`liquidity_contracts()` returns the OHLCV-computable contracts (included in
`default_contracts()`). **Turnover** additionally needs a `shares_outstanding`
column, so it is *registered* (reachable via `get_contract("turnover_21")`) but
excluded from the default OHLCV build; requesting it without the column fails
closed.

| name | unit | lookback | warmup | inputs |
|------|------|---------:|-------:|--------|
| `volume_change_1` | dimensionless | 2 | 2 | volume |
| `dollar_volume_21` | log_dollar_volume | 21 | 21 | close, volume |
| `rel_volume_21` | dimensionless | 21 | 21 | volume |
| `amihud_21` | illiquidity | 22 | 22 | close, volume |
| `volume_imbalance_21` | dimensionless | 22 | 22 | close, volume |
| `corwin_schultz_21` | spread_fraction | 22 | 22 | high, low |
| `roll_spread_21` | spread_fraction | 23 | 23 | close |
| `turnover_21` (registered, not default) | dimensionless | 21 | 21 | volume, shares_outstanding |

VWAP deviation, also part of the roadmap's volume/liquidity family, already ships
as `vwap_dev_21` in SF-S2-MR1 and is not duplicated here.

## Mathematics, assumptions, and interpretation limits

Windows are trailing and bar-based; `r_t` is the one-bar close-to-close return.

* **Volume change** — `ln(V_t) - ln(V_{t-p})`. Volume-momentum proxy. Zero-volume
  bars yield `NaN`.
* **Dollar volume** — trailing mean of `ln(close · volume)`. A liquidity *level*
  in log dollars; **not** scale-free (it carries the price scale). Non-positive
  traded value yields `NaN`.
* **Relative volume** — `volume / trailing_mean(volume)` (unitless, `>= 0`).
* **Amihud illiquidity** — trailing mean of `|r_t| / dollar_volume_t`, scaled to
  return per \\$1M (a **price-impact** proxy, `>= 0`). In dollar units, so it is
  not price-scale invariant. A zero-volume bar yields `NaN` for that bar.
  Interpretation limit: a coarse average impact, not an order-book impact curve.
* **Volume imbalance** — `(up-day volume - down-day volume) / total volume` over
  the window, in `[-1, 1]`, using the return sign as a crude order-flow-direction
  proxy. A zero-volume window yields `NaN`.
* **Corwin-Schultz spread** — the 2012 high-low estimator over each adjacent day
  pair `(t-1, t)` (causal), negative per-pair estimates floored at zero, then
  averaged over the window. Fractional, `>= 0`. Assumes the high is a buy and the
  low a sell; not the quoted spread.
* **Roll spread** — Roll (1984) `2·sqrt(-Cov(dp_t, dp_{t-1}))` over the window
  (`dp` = log return); when the serial covariance is non-negative the estimator
  is undefined and returned as `0`. Fractional, `>= 0`. Assumes bid-ask bounce is
  the only source of negative serial correlation.
* **Turnover** — trailing mean of `volume / shares_outstanding` (`>= 0`). The one
  contract with an explicit extra data requirement.

## Configuration, usage, and lineage

```python
from quant_platform.features import build_contract_features, liquidity_contracts, get_contract

# OHLCV liquidity set (in the default build).
matrix = build_contract_features(panel, liquidity_contracts())

# Turnover: supply the extra column, then request it explicitly.
panel["shares_outstanding"] = shares
matrix = build_contract_features(panel, [get_contract("turnover_21")])
```

Each contract's cache/lineage identity is its `name`, `version`, and `params`.
The parameters that change the numbers — window and (for Amihud) the dollar
`scale` — are part of `params`, so a cache key derived from `metadata()` changes
whenever a liquidity parameter changes. Adjustment policy lives upstream in the
Sprint 1 dataset contract; changing it changes the input panel's identity and
therefore any derived feature identity.

## Reproducibility and benchmark evidence

Deterministic and order-independent. Reproduce with
`pytest tests/test_liquidity_features.py -q` and
`python scripts/benchmark_liquidity_contracts.py`.

Laptop-scale, single process, synthetic deterministic data (seed 7),
Python 3.13 / numpy 2.5 / pandas 3.0 on Apple Silicon. The estimators are
vectorised rolling reductions, so they scale like the conventional MR1 set.

| tickers × dates | rows | seconds | rows/s | peak memory |
|-----------------|-----:|--------:|-------:|------------:|
| 16 × 504 | 8,064 | 0.10 | ~77k | ~2 MB |
| 64 × 1,260 | 80,640 | 0.35 | ~227k | ~12 MB |
| 256 × 2,520 | 645,120 | 1.48 | ~436k | ~94 MB |

## Security and data

No credential, licensed observation, proprietary dataset, generated run, or
final-holdout artifact is committed. Tests and benchmark use in-process synthetic
data.

## Risks, rollback, and residual limitations

* **Rollback.** Purely additive: a new `liquidity.py` module and new registry
  entries; MR1/MR2 behaviour, `build_features`, and existing contracts are
  unchanged. Reverting the commit restores the previous state with no migration.
* **Daily-bar proxies only.** These approximate spread/impact/liquidity from
  OHLCV. They are not observed quotes, depth, or executable capacity — do not
  represent them as such.
* **Amihud and dollar volume are dollar-denominated**, hence not price-scale
  invariant (unlike the fractional spread and volume-ratio features); this is by
  construction and the MR1 scale-invariance test excludes them.
* **Direction proxy.** Volume imbalance uses the return sign as a stand-in for
  signed order flow — crude on a daily bar.
* **Roll undefined region.** When serial covariance is non-negative Roll's spread
  is set to `0`; treat frequent zeros as "no bid-ask-bounce signal", not zero
  spread.
* **Turnover needs shares outstanding**, which is not in the OHLCV panel; it is
  registered but excluded from the default build and fails closed without the
  column.
* **No predictive claim.** These are baselines; edge is established only by the
  SF-S2 modelling and governance slices against an untouched holdout.
* **Sprint 1 integration dependency.** Built on the current `dev` price-panel
  schema and verified volume/adjustment semantics; final integration with the
  Sprint 1 point-in-time/lineage contracts follows once merged.
