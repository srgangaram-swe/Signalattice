# Signal Foundry market-data contract

Signalattice publishes data; AlphaForge consumes it. The boundary is an immutable file
bundle so the repositories can evolve independently and neither imports the other's
internal code.

## Bundle layout

```text
<bundle-id>/
├── manifest.json
└── prices/
    ├── year=2024/part-00000.parquet
    └── year=2025/part-00000.parquet
```

`bundle-id` is the SHA-256 digest of the semantic manifest: contract/schema version,
source snapshot and manifest hashes, producer Git SHA, exact file hashes/counts,
columns, coverage, universe, temporal rules, point-in-time limitations, licensing policy,
and redacted source provenance. Re-exporting identical canonical inputs with the same
producer/source identity returns the same bundle. A data or safety-metadata change
produces a new directory and never overwrites the old bundle.

## Schema 1.0.0

| Field | Meaning |
|---|---|
| `date`, `ticker` | Canonical daily observation key |
| `open`, `high`, `low`, `close`, `adj_close`, `volume` | Validated market values |
| `effective_at` | Economic event time |
| `available_at` | Earliest time research may use the observation |
| `observed_at` | Provider retrieval time |
| `provider_updated_at` | Provider revision metadata, nullable when unavailable |
| `instrument_id` | Provider/canonical instrument identity |
| `currency` | Quote currency |
| `exchange_calendar` | Calendar identifier |
| `adjustment_state` | Explicit price-adjustment semantics |
| `source`, `source_table` | Provider lineage |

All timestamps except the canonical daily `date` are UTC. The consumer's as-of rule is
strictly `available_at <= decision_timestamp`.

## Publication transaction

Publication validates required columns, finite values, positive prices, non-negative
volume, OHLC bounds, unique keys, temporal ordering, identity fields, and provider secret
attestation. Partitions are written under `.publishing`, hashed, described by a canonical
manifest, then atomically renamed to the bundle identity. A failed publication removes
staging and leaves no valid-looking partial bundle.

Licensed observations are never committed. CI uses provider-shaped synthetic fixtures.

## Consumer validation

Before reading any row, the standalone validator checks:

1. contract and supported schema version;
2. directory name equals bundle identity;
3. every path is relative and remains beneath the bundle root;
4. every partition exists and matches SHA-256 and row count;
5. exact column order and temporal/market invariants;
6. aggregate counts, coverage, universe, and semantic identity.

Corruption, truncation, path traversal, duplicate keys, unsupported evolution, future
availability, or ambiguous provenance fails closed with
`SignalFoundryContractError`.

`load_signal_foundry_bundle(..., as_of=...)` validates the complete bundle and then
filters by the availability rule. Mutation tests prove that a future observation change
cannot alter an earlier as-of view.

## Reference fixture and benchmark

`tests/fixtures/signal_foundry_v1/current.json` points to a committed, deterministic,
redistribution-safe two-symbol bundle spanning a year partition boundary. Consumer
repositories use it for compatibility tests without importing Signalattice or accessing
the network. Rebuild it with:

```bash
python scripts/build_signal_foundry_fixture.py
```

The local benchmark exercises export, full validation, and an as-of read over 48,000
synthetic rows:

```bash
python scripts/benchmark_signal_foundry_contract.py
```

Raw observed evidence and machine/runtime context are recorded in
[`docs/benchmarks/signal_foundry_contract_2026-07-23.json`](benchmarks/signal_foundry_contract_2026-07-23.json).
The single Apple M4 run observed approximately 58,479 exported rows/second, 0.412 seconds
for complete validation, and 0.767 seconds for verified as-of loading. This is a
laptop-scale reference, not a regression threshold or distributed-storage claim.

## Compatibility policy

- `1.x`: additive manifest metadata may be introduced when older consumers can ignore
  it safely. Data columns remain ordered and semantically stable.
- Breaking column, type, timestamp, identity, adjustment, or as-of changes require a
  new major version and an explicit AlphaForge consumer update.
- Consumers reject unknown major/minor schema versions until compatibility is reviewed
  and tested.
- No consumer may silently infer missing temporal or adjustment metadata.

## Known limitations

The contract preserves the provenance supplied by the source; it cannot create historical
revisions, delisted constituents, corporate actions, or provider publication timestamps
that the subscription does not contain. Those limitations are part of the manifest and
must remain visible in AlphaForge reports and readiness decisions.
