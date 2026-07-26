# Scripts

Small reproducibility helpers live here. Core logic stays in the package and CLI.

```bash
bash scripts/run_example_pipeline.sh
python scripts/summarize_experiments.py
python scripts/benchmark_feature_store.py \
  --output-json docs/benchmarks/feature_store_2026-07-25.json \
  --output-plot docs/assets/feature_store_latency_2026-07-25.png \
  --output-example-manifest docs/examples/feature_store_manifest.json
```

The feature-store benchmark uses deterministic synthetic data, records every timing
sample and environment limitation, and generates its visual evidence through Seaborn.
