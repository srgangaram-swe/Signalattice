# Data Directory

This directory is intentionally versioned only by structure. Generated market data is not
committed.

```text
data/
├── raw/         # per-ticker source snapshots as Parquet
├── processed/   # validated price panels and feature matrices
└── external/    # optional external datasets
```

The ingestion stage writes:

- `data/raw/{TICKER}.parquet`
- `data/processed/price_panel.parquet`
- `data/processed/panel_metadata.json`

The feature stage writes:

- `data/processed/features.parquet`

Metadata identifies the actual source, universe, date range, data hash, resolved
data-config/seed fingerprint, and whether the panel is synthetic. A processed cache is
reused only when that fingerprint matches. Feature reuse is governed by a separate
manifest tied to the panel and feature/target contract.

Synthetic data is an explicit engineering source. It must not be interpreted as observed
market evidence, and public-data failures do not silently become synthetic runs by
default. See the [data card](../docs/data_card.md) for lineage and point-in-time limitations.

All generated data files are ignored by git so the repository remains lightweight and free
of vendor data redistribution issues.
