# Versioned time-frequency tensors (SF-S3-MR2)

Deterministic, content-addressed spectrogram and scalogram tensors with immutable
metadata, explicit missingness masks, train-only normalization state, and
integrity-checked storage.

Opt-in via `features.time_frequency.enabled` (default `false`). Enabling it adds
**no columns to any existing artifact** — a tensor is a separate object with its
own identity — so disabling it again leaves prior evidence untouched.

> **Not evidence of predictability.** Per the issue's non-goals, this MR trains
> no image model and publishes no picture as a finding. Every rendered figure
> carries a caption saying so.

---

## 1. Where this sits

SF-S3-MR1 reduced a causal window to scalars (`f_spec_*` columns). This MR keeps
the **whole time-frequency surface** for consumers that want the structure
rather than a summary. The two share one channel definition and one window
contract, so a descriptor and a tensor built from the same configuration
describe the same support — two definitions of "the volatility channel" would
eventually disagree.

| | SF-S3-MR1 descriptors | SF-S3-MR2 tensors |
|---|---|---|
| shape per observation | ~14 scalars | `(channels, frequency, time)` |
| joins the feature matrix | yes | **never** |
| storage | feature store, Parquet | content-addressed object store, `.npy` |
| normalization | none needed (scale-free) | train-fit log z-score |

---

## 2. The tensor contract

Shape is always `(n_samples, n_channels, n_frequency, n_time)`.

* **samples** — one per `(ticker, date)`, in the order of the returned alignment
  index. The index is returned *alongside* the tensor rather than embedded in
  metadata, so metadata stays bounded regardless of panel size.
* **channels** — return / volatility / volume / residual, all sharing one
  frequency grid. That shared grid is what makes a multi-channel tensor
  genuinely stackable instead of a ragged collection glued together.
* **frequency** — one-sided FFT bins in cycles per bar (spectrogram) or Morlet
  periods in bars (scalogram). Always strictly ascending; the metadata names the
  axis (`cycles_per_bar` vs `period_bars`) so the unit is never ambiguous, and a
  contract validator rejects a representation/axis mismatch.
* **time** — sub-segments *inside* the causal window, oldest first, so index
  `-1` is always the most recent slice.

`TimeFrequencyMetadata` carries everything needed to interpret, reproduce, and
verify the tensor: representation, channels, frequency grid, the full window
contract from MR1, coverage dates, tickers, normalization and its fitted state,
observed fraction, and SHA-256 digests of both arrays. The arrays themselves are
not self-describing, which is exactly why the metadata must be complete and
hash-bound to them.

**Identity** is the semantic SHA-256 of that metadata. Because the array digests
are *part of* the metadata, two tensors with the same identity necessarily have
the same bytes — that is what makes the cache safe.

---

## 3. Causality and masking

Every slice comes from MR1's trailing windows. There is no centred window and no
forward fill anywhere.

* **Warm-up is masked, not padded.** The first `length - 1` bars per ticker have
  `mask = False` and `values = NaN`. Padding them with zeros or with the first
  available window would manufacture a plausible surface out of nothing.
* **A sample/channel is observed only when its entire surface is finite.** A
  partially finite window is an incomplete estimate, not a usable one.
* **No imputation.** A consumer that ignores the mask sees NaN, never a
  convenient zero.

The scalogram's time axis is built by stepping the causal wavelet *backwards*
from the most recent bar, so every column is itself a causal estimate over a
strictly earlier sub-window and the last column is the only one touching the
evaluation bar.

The governing test rewrites every bar after a cutoff and asserts that all
earlier samples and mask entries are bit-identical.

---

## 4. Train-only normalization

Power is non-negative and heavily right-skewed across orders of magnitude, so a
raw z-score would be dominated by a handful of high-power windows. The default
`train_log_zscore` therefore takes a logarithm first — turning multiplicative
spread into additive spread — and then z-scores per `(channel, frequency, time)`
bin.

Statistics come **only** from observed samples inside an explicitly supplied
`[fit_start, fit_end]` interval. The interval, the sample count, and a digest of
the fitted statistics are recorded as a `FittedTransformState` in the metadata.
Fitting over the whole tensor is the leakage bug this API shape exists to make
impossible: there is no code path that normalizes without being told an interval.

Guards, all of which fail closed:

* an inverted interval, or one selecting no samples / no *observed* samples;
* a bin with fewer than two observed training samples (a standard deviation
  estimated from one point is not an estimate);
* normalizing an already-normalized tensor;
* a misaligned index;
* a non-positive log floor.

The floor (`1e-300`) is stated, recorded in metadata, and applied identically at
fit and apply time. Power can legitimately underflow to zero in a quiet band, and
`log(0)` would poison every downstream statistic.

A constant bin gets `std = 1.0` rather than a division by ~0, which would amplify
float noise into a large standardized value.

---

## 5. Storage

`TimeFrequencyStore` is a local immutable object store. A cache of scientific
artifacts is only useful if a hit is provably what a rebuild would produce:

* **Content addressing** — the object id *is* the metadata identity, so a stale
  artifact cannot be served under a changed configuration.
* **Verification on read** — every load re-hashes both arrays and re-derives the
  manifest identity. Bit-rot, a truncated write, or a hand-edited manifest is an
  error at load, not a slightly different result three stages later.
* **Atomic publication** — staged in a temporary directory, moved with a single
  rename, fsynced. A crash mid-write leaves no half-object that a later run
  would treat as a hit; a failed write removes its staging directory.
* **Path containment** — the root may not be a symlink (a symlinked root would
  let a contained relative path resolve outside the tree the caller believes it
  is writing to), and object ids are validated as full lowercase SHA-256 digests
  *before* touching the filesystem, so `../escape` never reaches a path join.
* **No pickle** — arrays are written and read with `allow_pickle=False`, and the
  dtype is checked on load. A stored tensor must never be an executable
  deserialization boundary.
* **Bounded** — a write above `max_object_bytes` is refused before it is
  attempted.

---

## 6. Rendering

`quant_platform.reporting.time_frequency_plots` renders through Seaborn's API and
theme system.

* Sequential, perceptually uniform, colourblind-safe colour maps (`mako`). A
  diverging map on a non-normalized power surface would invent a meaningful
  midpoint that does not exist.
* Masked cells render in a distinct grey — "no data" and "low power" must not
  look alike.
* Axes are always labelled with units; an unlabelled frequency axis makes an
  image unreadable and therefore unfalsifiable.
* Every caption carries channel, window geometry, taper, detrend, normalization
  *and its fit interval*, plus an explicit "not evidence of predictability" note.
* Rendering a masked sample raises rather than emitting a blank image that would
  read as real.
* `plot_channel_coverage` exists because coverage is the first thing to check:
  a 40%-masked channel produces individually fine-looking images and a biased
  aggregate.

---

## 7. Compute and bounds

Laptop-scale, labelled as such. Python 3.13.11, macOS 15.5 arm64, numpy 2.5.1,
pandas 2.3.3, single process, 3 repetitions, median. Default 4 channels, window
length 64, segment 32, hop 16.

| panel | representation | shape | build | write | verify+read | memory | on disk |
|---|---|---|---|---|---|---|---|
| 10 × 1260 bars | spectrogram | (12600, 4, 33, 3) | 0.10 s | 0.04 s | 0.03 s | 40.0 MB | 40.0 MB |
| 10 × 1260 bars | scalogram | (12600, 4, 8, 3) | 0.04 s | 0.01 s | 0.01 s | 9.7 MB | 9.7 MB |
| 25 × 1260 bars | spectrogram | (31500, 4, 33, 3) | 0.24 s | 0.10 s | 0.06 s | 99.9 MB | 99.9 MB |

**Tensors are large.** A spectrogram is roughly `33 × 3 × 8` bytes per
channel-observation — about 3.2 KB — so a 25-name decade is ~100 MB. That is the
reason for the ceilings, and the reason tensors never join the feature matrix.

Refusal thresholds (configuration mistakes, not tuning knobs):

* `max_tensor_cells` (default 2e8) on `samples × channels × frequency × time`,
  checked **before** allocation;
* `max_object_bytes` (default 2e9) per stored object;
* `scalogram_periods` must be sorted, unique, and above the Nyquist period of 2
  bars.

---

## 8. Evidence

`tests/test_time_frequency.py` — 60 tests:

* **Identity/replay** — identical inputs give identical metadata, identity, and
  bytes across both representations; a configuration change changes identity;
  the digest distinguishes shape and dtype from bytes alone.
* **Causality/masking** — future-mutation bit-identity; warm-up masked and NaN;
  NaN returns mask their covering windows; channels share one grid; the index
  aligns with samples.
* **Normalization** — fit interval only; tampering outside the interval does not
  move the statistics; finite where observed and NaN where masked; every
  rejection path.
* **Store** — round-trip; idempotent re-write; corrupt values, corrupt mask,
  edited manifest, unreadable manifest, missing array, wrong dtype; object-id
  validation including `../escape`; symlinked root; byte ceiling with no partial
  object left behind; no staging residue after a failed write.
* **Bounds/config** — cell ceiling, opt-in, malformed panels, missing benchmark,
  unsorted/duplicate/aliased periods, unsafe cache paths.
* **Metadata contract** — ascending finite grid, unique identifiers, coverage
  ordering, digest format, normalization/state consistency, axis/representation
  agreement, shape mismatch, in-memory tamper detection.
* **Rendering** — caption contents; figures written; masked/unknown/out-of-range
  selections refused.

Coverage: `time_frequency.py` 95%, `time_frequency_store.py` 93%,
`time_frequency_plots.py` 98% (branch).

---

## 9. Planned ablations — not yet run

Designed, deliberately not executed: running it before the Sprint 2 baselines are
frozen would be selection on an unfrozen benchmark. Under identical folds and
cost assumptions:

1. frozen Sprint 2 conventional baseline;
2. baseline + MR1 spectral descriptors;
3. baseline + tensor-derived summaries (pooled over the time axis);
4. spectrogram vs scalogram at matched compute;
5. single-channel vs multi-channel tensors.

SF-S3-MR11 publishes the decision. Any image model over these tensors is out of
scope for this sprint and would need its own capacity and overfitting controls.

---

## 10. Residual limitations

* **No incremental-value evidence exists**, and none is implied.
* **Tensors are big.** Storage grows linearly in samples × channels ×
  frequency × time; budget before enabling on a wide universe.
* **Scalogram columns inherit MR1's cone-of-influence caveat** — each is a causal
  wavelet estimate at the edge of its own sub-window.
* **Bar time, not calendar time.** A 5-bar period is not 5 calendar days.
* **The mask is per `(sample, channel)`, not per cell.** A window with one bad
  bar masks the whole surface. That is deliberately conservative; a per-cell mask
  would let a partially corrupted estimate through.
* **Normalization is per-bin across samples**, so it assumes the training
  interval is representative of the applied interval. A regime shift between them
  degrades the standardization silently — the fit interval is recorded precisely
  so this can be audited.
* **Synthetic evidence only.** Every number here comes from synthetic panels.
* **Depends on SF-S3-MR1** (PR #53), which is not yet merged.
