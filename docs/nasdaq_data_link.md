# Nasdaq Data Link ingestion boundary

Signalattice can ingest a configured Nasdaq Data Link Tables or Time-Series product
through a bounded, cache-first adapter. The implementation uses the provider's HTTPS
JSON contract directly, so the runtime does not need the optional `nasdaq-data-link`
Python package. This is a data-integration boundary, not permission to redistribute
provider observations or claim that a particular subscription contains institutional
point-in-time history.

## Credential setup on macOS

Never paste the API key into chat, YAML, a CLI argument, an issue, or a shell command.
Store it through the interactive Keychain prompt:

```bash
security add-generic-password \
  -U \
  -a "$USER" \
  -s com.signal-foundry.nasdaq-data-link \
  -w
```

`security` prompts for the secret because `-w` is the final option. The value therefore
does not enter shell history. Expose it only to the bounded command and remove it from
the current shell afterward:

```bash
export NASDAQ_DATA_LINK_API_KEY="$(
  security find-generic-password \
    -a "$USER" \
    -s com.signal-foundry.nasdaq-data-link \
    -w
)"
signalattice ingest-data --config configs/nasdaq_smoke.yaml --force
unset NASDAQ_DATA_LINK_API_KEY
```

The SEP smoke profile requests one liquid common-stock ticker over five market dates,
permits at most two HTTP requests, retries at most once, and stores all results under
ignored local directories. The configured `SHARADAR/SEP` table is an equity-price
table; ETF and fund symbols such as SPY belong to a separate provider table and are
not valid SEP entitlement probes.
Run it only after the offline suite passes. To remove the Keychain item:

```bash
security delete-generic-password \
  -a "$USER" \
  -s com.signal-foundry.nasdaq-data-link
```

## Request and cache contract

`NasdaqDataLinkConfig` forbids unknown fields and validates:

- an explicit `tables` or `time_series` endpoint kind and matching provider code;
- adjusted or unadjusted time-series semantics;
- currency, exchange calendar, and conservative UTC market-close policy;
- `prefer_cache`, `network`, or `cache_only` operation;
- a safe relative cache root;
- requests per minute, hard total-request budget, page size, timeout, retry count,
  retry backoff, and availability lag.

The credential is not a configuration field. The HTTPS URL contains it only because the
provider API requires a query parameter. The URL is never logged or included in an
exception. HTTP errors are converted into bounded messages that exclude the request URL.
Authentication and entitlement failures are terminal. Only HTTP 429 and selected 5xx
responses are retried; `Retry-After` is honored and exponential backoff includes jitter.

Each redacted request has a stable SHA-256 identity. Pages first land in a staging
directory. A completed response becomes an immutable snapshot identified by the ordered
page hashes, with an atomic `latest.json` pointer. Cache replay verifies pointer,
manifest, page hashes, schemas, and row counts before parsing. Partial or corrupt
snapshots never masquerade as valid data.

The default research profile permits 30 requests per minute and 100 total requests.
Those are local safety caps, not statements of the provider's current entitlement.
Configure a lower value when the account limit is lower. Cache replay and
`cache_only` mode should serve repeated research runs without new API traffic.

## Canonical mapping and temporal limits

The adapter targets daily OHLCV table semantics compatible with `SHARADAR/SEP` and
the standard daily OHLCV Time-Series schema used by products such as `XDUS`. It
requires ticker identity plus date, open, high, low, close, and volume, and uses
`closeadj` as `adj_close` when present. For an adjusted time series, the provider's
adjusted close is used as `adj_close`. It rejects identity mismatches,
missing/non-finite values, non-positive prices, negative volume, invalid OHLC bounds,
duplicate keys, malformed pages, and schema changes between pages or series.

One ingestion configuration addresses one provider table or time-series database. A
research universe must therefore contain instruments represented by that product.
Combining equity, ETF/fund, fundamental, or corporate-action products requires separate
immutable acquisitions and a versioned point-in-time join; the current profile
deliberately does not guess or silently union those semantics.

Each observation records:

- `effective_at`: the market date at the configured conservative UTC close;
- `available_at`: `effective_at` plus the configured lag;
- `observed_at`: retrieval time;
- `provider_updated_at`: provider revision metadata when present;
- source table, instrument, currency, calendar, and adjustment state.

The SEP default is 21:00 UTC; the XDUS sample uses a conservative 17:00 UTC close and
12-hour availability lag. `available_at` is a documented policy assumption, not a
provider publication timestamp.
The manifest explicitly records that historical revisions, point-in-time universe
membership, and corporate actions are not proven complete. A backtest must retain these
limitations and must not relabel current-vintage adjusted history as survivorship-free.

## Free XDUS engineering sample

`configs/nasdaq_xdus_sample.yaml` requests five series that the provider product page
advertises as free samples from the Düsseldorf Stock Exchange product over 2016–2018.
The entire acquisition is bounded to five requests, with no retry traffic, and
subsequent runs are cache-first.
The profile records EUR, the provider product code, adjustment state, response
metadata, immutable page hashes, and an explicit conservative close/availability
policy.

A successful sample acquisition can exercise the network, schema, cache, lineage,
validation, and persistence mechanics without purchasing the intended production
dataset. It does **not** prove a U.S. equity universe, point-in-time membership,
historical fundamentals, current prices, live tradability, or strategy profitability.
Do not compare its backtest results with a U.S. investable benchmark or use it as a
live-trading qualification artifact.

Provider-side account enablement remains authoritative. A 401/403 is terminal: do not
retry, bypass access controls, or claim that the sample was acquired. Resolve access with
Nasdaq Data Link before treating the network acceptance criterion as complete.

## Threat model and failure policy

| Threat or failure | Control |
|---|---|
| Credential disclosure | Keychain-backed environment lookup; no config field; URL/error redaction; secret tests |
| API quota exhaustion | Single-threaded client, minimum interval, hard request cap, cache-first and offline modes |
| Oversized or partial response/write | 64 MiB response bound; staging plus atomic content-addressed promotion |
| Schema or semantic drift | Required-column and page-schema validation; fail closed |
| Cache tampering/corruption | SHA-256 verification of pointer, manifest, and every page |
| Path traversal | Strict relative cache configuration and resolved-root checks |
| Silent data revision | Immutable snapshots and retrieval/provider revision provenance |
| License violation | Vendor responses and bundles stay in ignored local roots |
| Temporal leakage | Explicit effective/available/observed timestamps and downstream as-of rule |

## Commands

```bash
# Pull the bounded free XDUS engineering sample; cache-first after success.
signalattice ingest-data --config configs/nasdaq_xdus_sample.yaml --force

# Full configured pull; cache-first after the initial successful request.
signalattice ingest-data --config configs/nasdaq_data_link.yaml

# Publish the verified processed panel for AlphaForge.
signalattice export-signal-foundry-bundle \
  --config configs/nasdaq_data_link.yaml \
  --output data/signal-foundry-bundles

# Validate before any consumer reads observations.
signalattice validate-signal-foundry-bundle \
  data/signal-foundry-bundles/<bundle-id>
```

Do not commit generated files. A public evidence report may contain safe aggregates and
hashes only when the data license permits and observations cannot be reconstructed.
