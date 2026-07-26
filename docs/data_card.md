# Data Card

## Dataset purpose

Signalattice uses daily OHLCV panels to test forecast calibration, temporal modeling,
decision sensitivity, and research controls. It supports optional public market adapters
and a deterministic synthetic data-generating process (DGP).

No dataset shipped or fetched by this project should be assumed institutionally licensed,
survivorship-bias-free, or historically point-in-time.

## Sources

| Source | Intended role | Important caveat |
|---|---|---|
| Yahoo Finance | Optional exploratory public-data run | Adjustments, revisions, availability, and redistribution terms are vendor-dependent. |
| Stooq | Optional alternative public-data run | Coverage and adjustment conventions can differ from other vendors. |
| Synthetic DGP | Offline CI and known-signal/null engineering experiments | It is generated evidence, not observed market evidence. |

`source: auto` tries configured public adapters. Synthetic substitution is disabled by
default and must be requested with `source: synthetic` or deliberately allowed. Every run
records the actual source and a synthetic flag.

## Canonical schema

The long price panel has one row per `(date, ticker)`:

```text
date, ticker, open, high, low, close, adj_close, volume, return, log_return
```

Simple and log returns are calculated within ticker after source normalization. The
feature matrix retains the key, adds `f_*` columns, and adds forward-return/direction
targets. Raw identifiers and targets are not model features.

## Validation and lineage

Ingestion checks:

- required fields and type coercion;
- unique `(date, ticker)` keys;
- positive prices and internally consistent OHLC bounds;
- non-negative volume;
- minimum per-ticker history and missingness; and
- complete coverage of the requested ticker list.

The processed metadata records actual source, universe, row count, date range, price field,
dataframe hash, data-config/seed hash, random seed, and synthetic status. A cached panel is
reused only when its fingerprint matches. Raw snapshots, processed panel, and metadata are
written through temporary files and atomic replacement.

The feature cache has a separate manifest tied to the panel, feature config, target
horizon, and seed. Generated source data is ignored by Git.

## Synthetic DGP

The synthetic panel contains:

- a common market factor with heterogeneous per-ticker beta;
- idiosyncratic shocks and heterogeneous volatility;
- a smooth volatility regime;
- configurable market and idiosyncratic AR(1) coefficients;
- plausible but simulated OHLC ranges and volumes; and
- deterministic generation under a fixed seed.

The committed experiment's nonzero autocorrelation coefficients plant a moderate, declared
causal dependency; library defaults remain deliberately smaller.
They exist so tests can verify that a valid pipeline recovers known structure. Setting both
coefficients to zero creates a directional null DGP. Neither experiment supports a claim
about real-market profitability or realistic market impact.

## Temporal semantics

Features use trailing information within ticker. Same-date cross-sectional transforms may
compare tickers on that date. Outer train/test, inner calibrator-fit, and later weighting
boundaries move whole dates together. Temporal sequences never cross ticker boundaries.

This chronology is not equivalent to a complete point-in-time data system. Public adjusted
prices may be revised, current ticker lists omit delisted constituents, and no as-of
corporate-action, index-membership, fundamental-publication, or symbol-history database is
implemented.

## Known gaps

- small, manually configured universe;
- no delisted securities or historical constituent membership;
- no exchange calendar/half-day audit beyond observed rows;
- no quote, spread, auction, order-book, borrow, financing, or intraday volume data;
- no vendor reconciliation or corporate-action ledger;
- no feature-level as-of join framework for fundamentals or alternative data; and
- no source SLA, entitlements, retention policy, or production access controls.

These gaps constrain both model validity and the capacity claims described in the
[backtesting contract](backtesting.md).

The optional Nasdaq Data Link adapter improves source provenance, immutable caching, and
temporal contract enforcement, but the availability timestamp is a conservative policy
assumption unless the selected table supplies an authoritative publication timestamp.
The adapter does not by itself remove survivorship, historical-revision, constituent, or
corporate-action limitations. Its redacted source manifest records these gaps.

The local feature store carries those source limitations forward into every immutable
materialization. Its quality SLAs can reject missing tickers, duplicate keys, non-finite
features, excessive business-date gaps, staleness, and configured distribution drift.
Passing these checks proves the declared mechanics and thresholds only; it cannot supply
missing point-in-time source evidence or demonstrate predictive value.

## Appropriate use

Use synthetic data for deterministic engineering validation and public data for explicitly
labeled exploratory research. Before external or capital-allocation use, replace public
adapters with licensed point-in-time data, define data-quality SLAs, reconcile corporate
actions, construct survivorship-aware universes, and rerun the full
[validation protocol](validation_protocol.md).
