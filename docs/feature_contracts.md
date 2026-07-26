# Conventional feature contracts (SF-S2-MR1)

This document describes the versioned, self-describing feature contracts that
turn a validated point-in-time price panel into a leakage-safe matrix of
conventional price, return, trend, and mean-reversion signals.

It is the SF-S2-MR1 vertical slice of the Signal Foundry roadmap. It adds a
metadata and registry layer over the audited primitives in
[`quant_platform.features.technical`](../src/quant_platform/features/technical.py)
and
[`quant_platform.features.cross_sectional`](../src/quant_platform/features/cross_sectional.py);
it introduces no new predictive claim and no new hidden mathematics.

## Why contracts

The existing feature pipeline computes signals procedurally. That is fine for a
single run, but a research platform needs to *reason about* a feature without
reading its code: what unit is it in, how much history does it consume, when is
its value knowable, what does it do when data is missing, and what values can it
legally take. A `FeatureContract` binds a causal computation kernel to exactly
that metadata, so features become inspectable, comparable, and safe to cache and
version.

## The contract

Each entry in the registry is a
[`FeatureContract`](../src/quant_platform/features/contracts.py) declaring:

| field | meaning |
|-------|---------|
| `name`, `version` | stable identity for caching and lineage (`1.0.0` at introduction) |
| `family` | `return`, `momentum`, `trend`, or `mean_reversion` |
| `scope` | `per_asset` (within one asset's history) or `cross_sectional` (across the universe on a date) |
| `unit` | measurement unit of the output (see below) |
| `inputs` | the panel columns the feature consumes |
| `params` | the frozen parameters (windows, spans, skips) |
| `lookback` | bars of history read |
| `warmup` | bars until the first defined value; output is `NaN` before position `warmup - 1` |
| `frequency` | sampling frequency assumed (`daily`) |
| `temporal_availability` | `causal` — uses only bars up to and including *t* |
| `missing_data_policy` | `propagate_nan` — missing input yields missing output; never imputed |
| `numerical_range` | closed interval the defined values are guaranteed to lie in, when bounded |

`FeatureContract.metadata()` returns this as a JSON-friendly dict, and
`contract_metadata_frame()` returns the whole registry as a table.

### Units

* `simple_return` — fractional change, `0.01 == 1%`.
* `log_return` — continuously compounded return.
* `ratio_deviation` — `value / reference − 1` (dimensionless).
* `zscore` — standardised, dimensionless.
* `rank_pct` — cross-sectional percentile in `[0, 1]`.
* `log_residual` — residual in log-price units (≈ fractional deviation).

## Invariants

1. **Causality.** Every contract reads only information available up to and
   including bar *t*. A feature at *t* never references *t+1*. This is verified
   by a metamorphic test: corrupting a future bar changes only that bar's
   feature and never a past one.
2. **Full-window warm-up.** A contract emits `NaN` until it has observed a
   complete lookback window for that asset. Partial-window values are
   suppressed, so a feature means the same thing at every timestamp and across
   assets. The first defined value for an asset appears at position
   `warmup − 1`.
3. **Fail closed.** Missing required columns, non-finite `inf` inputs, duplicate
   `(date, ticker)` keys, and empty panels raise before any feature is computed.
   Genuine gaps (`NaN`) propagate rather than being silently imputed.
4. **Scale invariance.** Every feature is invariant to a positive multiplicative
   rescaling of an asset's prices, so a split/dividend adjustment applied as a
   scale factor leaves the features unchanged. This is verified by test.
5. **Determinism.** Identical inputs produce byte-identical outputs regardless of
   input row order (the panel is validated and sorted internally).

## The default contract set

`default_contracts()` returns the frozen set below. `build_contract_features`
evaluates it (or any subset) and returns a `date, ticker` frame plus one
`fc_<name>` column per contract, row-aligned to the validated panel.

| name | family | scope | unit | lookback | warmup | inputs |
|------|--------|-------|------|---------:|-------:|--------|
| `ret_1d` | return | per_asset | simple_return | 1 | 1 | adj_close |
| `ret_5d` | return | per_asset | simple_return | 5 | 5 | adj_close |
| `ret_21d` | return | per_asset | simple_return | 21 | 21 | adj_close |
| `logret_1d` | return | per_asset | log_return | 1 | 1 | adj_close |
| `roc_10` | momentum | per_asset | simple_return | 10 | 10 | adj_close |
| `mom_63` | momentum | per_asset | simple_return | 63 | 63 | adj_close |
| `mom_126` | momentum | per_asset | simple_return | 126 | 126 | adj_close |
| `mom_252` | momentum | per_asset | simple_return | 252 | 252 | adj_close |
| `mom_252_21` | momentum | per_asset | simple_return | 252 | 252 | adj_close |
| `ma_dist_20` | trend | per_asset | ratio_deviation | 20 | 20 | adj_close |
| `ma_dist_50` | trend | per_asset | ratio_deviation | 50 | 50 | adj_close |
| `ma_dist_200` | trend | per_asset | ratio_deviation | 200 | 200 | adj_close |
| `ema_dist_12` | trend | per_asset | ratio_deviation | 12 | 12 | adj_close |
| `ema_dist_26` | trend | per_asset | ratio_deviation | 26 | 26 | adj_close |
| `zscore_21` | mean_reversion | per_asset | zscore | 21 | 21 | adj_close |
| `reversal_5` | mean_reversion | per_asset | simple_return | 5 | 5 | adj_close |
| `vwap_dev_21` | mean_reversion | per_asset | ratio_deviation | 21 | 21 | high, low, close, volume |
| `resid_63` | mean_reversion | per_asset | log_residual | 63 | 63 | adj_close |
| `cs_rank_mom_63` | momentum | cross_sectional | rank_pct | 63 | 63 | adj_close |
| `cs_z_mom_63` | momentum | cross_sectional | zscore | 63 | 63 | adj_close |

## Mathematics

For an asset with adjusted close `P_t` (or OHLCV bar `t`), window `w`, span `s`:

* **Simple return / ROC / reversal.** `ret_w(t) = P_t / P_{t−w} − 1`;
  `reversal_w(t) = −ret_w(t)`.
* **Log return.** `logret_1(t) = ln(P_t / P_{t−1})`.
* **Momentum.** `mom_w(t) = P_t / P_{t−w} − 1`; with skip `k`,
  `mom_{w,k}(t) = P_{t−k} / P_{t−w} − 1` (the 12−1 convention skips the most
  recent month to avoid short-term reversal).
* **MA distance.** `ma_dist_w(t) = P_t / SMA_w(t) − 1`.
* **EMA distance.** `ema_dist_s(t) = P_t / EMA_s(t) − 1`, EMA with
  `α = 2/(s+1)`, `adjust=False`. The EMA has infinite memory; warm-up is the
  nominal span.
* **Rolling z-score.** `zscore_w(t) = (P_t − mean_w(t)) / std_w(t)`; undefined
  (`NaN`) when the window has zero dispersion — never `±inf`.
* **VWAP deviation.** typical price `TP = (H + L + C)/3`;
  `VWAP_w(t) = Σ TP·V / Σ V` over the trailing window; deviation
  `= C_t / VWAP_w(t) − 1`. A zero-volume window yields `NaN`.
* **Log-linear residual.** fit ordinary least squares of `ln P` on an integer
  time index over the trailing window; the feature is the residual of the last
  observation (actual − fitted log level). Because the residual of the last
  window point is a fixed linear functional of the window, the full series is
  computed with a single strided matrix product rather than a per-window Python
  loop. It is validated differentially against `numpy.linalg.lstsq`.
* **Cross-sectional rank / z-score.** rank (`pct`, tie-averaged) or standardise
  the per-asset momentum base across all names present on each date. Names
  entering or leaving the universe are handled per date, so delistings and late
  listings need no special casing.

## Configuration and usage

Feature selection and parameters are code-level configuration, not run YAML: the
default set is `default_contracts()`, and callers pick a subset by name.

```python
from quant_platform.features import (
    build_contract_features, default_contracts, get_contract, list_contracts,
)

# Whole default set.
matrix = build_contract_features(panel)

# A named subset.
subset = [get_contract(n) for n in ("ret_1d", "mom_63", "vwap_dev_21")]
matrix = build_contract_features(panel, subset)

# Inspect the registry.
print(list_contracts())
print(get_contract("resid_63").metadata())
```

The input `panel` is the canonical long OHLCV panel
([`quant_platform.data.schema`](../src/quant_platform/data/schema.py)):
one row per `(date, ticker)`, timezone-naive `date`, lower-snake OHLCV columns.

## Reproducibility

* Deterministic: no randomness in the kernels; output is independent of input
  row order.
* Run the tests: `make test` (or
  `pytest tests/test_feature_contracts.py -q`).
* Run the benchmark: `python scripts/benchmark_feature_contracts.py`.

### Benchmark evidence

Laptop-scale, single process, synthetic deterministic data (seed 7),
Python 3.13 / numpy 2.5 / pandas 3.0 on Apple Silicon. Reproduce with the script
above; absolute numbers depend on hardware and warm cache state.

| tickers × dates | rows | seconds | rows/s | peak memory |
|-----------------|-----:|--------:|-------:|------------:|
| 16 × 504 | 8,064 | 0.17 | ~48k | ~2.6 MB |
| 64 × 1,260 | 80,640 | 0.60 | ~134k | ~23 MB |
| 256 × 2,520 | 645,120 | 2.63 | ~245k | ~184 MB |

## Security and data

No credential, licensed observation, proprietary dataset, generated run, or
final-holdout artifact is committed. Tests and the benchmark use synthetic data
generated in process. The feature layer reads only the validated panel handed to
it.

## Risks, rollback, and residual limitations

* **Rollback.** The layer is purely additive: it introduces new modules and a new
  public entry point without changing `build_features` or any existing behaviour.
  Reverting the commit restores the previous state with no migration.
* **Full-window warm-up is intentionally strict.** Contracts suppress
  partial-window values that the raw primitives in `technical.py` would emit
  under their `min_periods` heuristics. This trades a few early observations for
  a feature that means the same thing everywhere.
* **`daily` frequency assumption.** Windows are bar-based, not calendar-based;
  contracts assume a daily bar. Intraday or mixed-frequency panels are out of
  scope for this slice.
* **Cross-sectional features need a populated universe.** Rank and z-score are
  only meaningful with more than one name present on a date; a single-name date
  yields a degenerate rank of `1.0`.
* **No predictive claim.** These are conventional baseline signals. Nothing here
  asserts out-of-sample edge; that is the job of the SF-S2 modelling and
  statistical-governance slices, evaluated against an untouched holdout.
* **Sprint 1 integration dependency.** This slice is built on the current `dev`
  price-panel schema. It is designed to consume the Sprint 1 point-in-time
  dataset, feature-lineage, and calendar contracts once those are merged and
  verified; until then it operates on the existing panel contract.
