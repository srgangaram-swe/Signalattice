# Architecture

Signalattice is a config-driven research DAG with explicit contracts between data,
forecasting, decision simulation, and evidence generation. Stages can run independently
from the CLI; `Pipeline.run_full()` executes them in one tracked experiment.

## Component flow

```mermaid
flowchart TD
    A[AppConfig YAML] --> B[CLI / Pipeline]
    B --> C[Yahoo, Stooq, or explicit synthetic source]
    C --> D[Canonical OHLCV panel]
    D --> E[Schema and market-data validation]
    E --> F[Atomic Parquet + metadata fingerprint]
    F --> G[Trailing feature and target pipeline]
    G --> H[Typed registry + quality and drift gates]
    H --> U[Immutable partitioned Parquet + DuckDB catalog]
    U --> V[Verified predicate read]
    V --> I[Whole-date walk-forward splits]
    I --> J[Tabular candidate models]
    I --> K[Causal per-ticker TCN]
    J --> L[Earlier calibrator-fit dates]
    L --> W[Later independent weighting dates]
    W --> M[Out-of-sample probabilities]
    K --> M
    M --> N[Forecast diagnostics]
    M --> O[Conservative lagged backtest]
    O --> P[Risk and implementation diagnostics]
    M --> Q[Warm inference benchmark]
    N --> R[Decision-readiness gate]
    P --> R
    Q --> R
    R --> S[Markdown / HTML report + figures]
    S --> T[Experiment tracker]
```

## Core packages

- `quant_platform.config`: Pydantic contracts for YAML configuration and selected
  environment overrides.
- `quant_platform.data`: source adapters, canonical schema, quality validation, explicit
  synthetic data generation, atomic persistence, and cache fingerprints.
- `quant_platform.features`: ticker-local trailing features, same-date cross-sectional
  transforms, forward targets, typed feature/fitted-state registry, quality and drift
  evidence, an immutable DuckDB/Parquet store, and resumable backfills.
- `quant_platform.models`: panel-aware splits, estimator construction, chronological
  calibration, heterogeneous ensembling, causal temporal convolution, walk-forward
  training, probability diagnostics, and model persistence.
- `quant_platform.backtest`: vectorized long-only or long/short decision simulation with
  conservative timing, turnover costs, no-trade bands, and portfolio limits.
- `quant_platform.evaluation`: cost/delay frontiers, break-even costs, dollar-volume
  participation, warm inference benchmarks, and independent readiness criteria.
- `quant_platform.risk`: performance, drawdown, VaR/CVaR, beta, exposures, correlations,
  and scenario calculations.
- `quant_platform.tracking`: SQLite, JSON, MLflow, and no-op experiment backends.
- `quant_platform.reporting`: diagnostic figures and self-contained run reports.
- `quant_platform.cli`: Typer commands for individual stages and full runs.

## Data contracts

The canonical price panel is long format with a unique `(date, ticker)` key:

```text
date | ticker | open | high | low | close | adj_close | volume | return | log_return
```

The feature matrix preserves the key and adds:

```text
f_* | target_forward_return | target_direction
```

Feature selection is prefix-based and explicit. Identifier, target, and future-return
columns are not inferred as model inputs. Temporal inputs are constructed as
`[sample, time, feature]` tensors from the history of the same ticker; sequence metadata
retains the prediction row and contributing history rows.

The out-of-sample prediction frame is the sole model signal accepted by the model-backed
backtest:

```text
date | ticker | y_true | forward_return | score | fold | candidate_*
```

`score` is `P(up)` for classification or a point forecast for regression. Candidate
columns are present for the calibrated ensemble when available.

## Invariants

| Boundary | Enforced invariant |
|---|---|
| Data source | All requested tickers must be returned; synthetic substitution is explicit and labeled. |
| Cache | Reuse requires a matching data/config/seed fingerprint. |
| Panel | Keys are unique and prices, OHLC bounds, volume, and minimum history are validated. |
| Features | Rolling inputs are trailing; cross-sectional transforms use only the same date; fitted state must end before application. |
| Feature store | Identity binds source, registry, fit interval, code/runtime, policy, logical content, and verified immutable partitions. |
| Outer evaluation | Entire dates move together and every test date is after training plus embargo. |
| Ensemble selection | Candidate fit dates precede calibrator-fit dates; independent weighting dates follow before embargo and outer test. |
| Temporal model | Every history element belongs to the sample ticker and occurs no later than the prediction row. |
| Backtest | Close-derived decisions incur at least the configured two-row execution lag. |
| Portfolio | Per-name and gross limits are re-applied after volatility scaling; missing held returns raise. |
| Readiness | Missing or non-finite evidence fails its criterion; there is no compensating weighted score. |

These invariants are executable contracts covered by unit or integration tests, not just
diagram annotations.

## Artifact lineage

Typical outputs are:

- raw per-ticker snapshots: `data/raw/*.parquet`;
- immutable licensed-provider snapshots: `data/vendor/nasdaq-data-link/`;
- versioned AlphaForge exchange bundles: `data/signal-foundry-bundles/<bundle-id>/`;
- panel and source metadata: `data/processed/price_panel.parquet` and
  `panel_metadata.json`;
- immutable feature objects, DuckDB catalog, and resumable checkpoints under the ignored
  `data/feature-store/` root;
- persisted preprocessing/model bundle: `models/{project_name}_model.joblib`;
- report and figures: the configured `reports/` path; and
- experiment metadata: SQLite, JSON, or MLflow according to configuration.

Data and model artifacts are ignored because public-vendor redistribution and stale binary
outputs are poor reproducibility mechanisms. The repository commits only a documented
example report that can be regenerated from its config and seed.

## Deployment boundary

Signalattice stops at research decision-readiness. It now implements a bounded historical
Nasdaq Data Link adapter, a versioned as-of dataset exchange contract, and a local offline
feature store. These are not a real-time feed handler, online/distributed feature service,
or proof of complete point-in-time history. It does not implement a portfolio optimizer,
broker adapter, order management system, pre-trade risk service, or post-trade ledger.
Its inference and feature-store timings are laptop-scale local evidence. Its capacity
result covers trailing dollar-volume participation only. Those exclusions remain visible
in the report, [feature-store contract](feature_store.md), and [data card](data_card.md).
