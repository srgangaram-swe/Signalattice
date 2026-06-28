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

All generated data files are ignored by git so the repository remains lightweight and free
of vendor data redistribution issues.
