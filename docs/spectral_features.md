# Causal spectral transform and descriptor engine (SF-S3-MR1)

Frequency-domain features for the Signal Foundry research platform: rolling FFT,
Welch PSD, STFT, continuous and discrete wavelets, reduced to versioned,
scale-free descriptors over four analysis channels.

The engine is **opt-in and disabled by default** (`features.spectral.enabled`).
It emits only `f_spec_*` columns, redefines nothing, and can be switched off
again without invalidating previously materialized evidence.

> **No predictive claim is made here.** This MR delivers the representation and
> its correctness evidence. Whether spectral features add value over the frozen
> Sprint 2 baselines is a separate, controlled experiment (SF-S3-MR11). Nothing
> in this document should be read as a finding about returns.

---

## 1. Architecture

Three modules, split by responsibility:

| module | responsibility | depends on |
|---|---|---|
| [`spectral_transforms.py`](../src/quant_platform/features/spectral_transforms.py) | causal windowing and the transforms themselves | numpy only |
| [`spectral_descriptors.py`](../src/quant_platform/features/spectral_descriptors.py) | spectrum → named scalar descriptors | numpy, config contracts |
| [`spectral.py`](../src/quant_platform/features/spectral.py) | channels, engine, registry specs, fitted normalizer | the above + feature registry |

Window and descriptor **contracts** live in [`config.py`](../src/quant_platform/config.py)
alongside every other configuration model, which keeps the dependency direction
pointing inward: the transforms depend on nothing, and the engine depends on the
stable contracts rather than the reverse.

The transforms are implemented rather than delegated so their windowing,
normalization, and edge behaviour are auditable. `scipy.signal` appears in the
**tests** as an independent reference, never in the implementation.

---

## 2. Causality

This is the property everything else is subordinate to. The value at bar *t* is a
function of `x[t-L+1 .. t]` and nothing later.

* **Trailing windows only.** `causal_windows` builds row *t* from the *L* bars
  ending at *t*. There is no centred window anywhere in the engine — a centred
  window is how a time-frequency feature usually acquires lookahead, because the
  leakage hides inside a convolution kernel instead of an obvious `shift(-1)`.
* **Warm-up is missing, not zero.** The first `L-1` bars per ticker are NaN.
* **Per-ticker grouping.** Windows are built inside each ticker, so one asset's
  history cannot enter another's spectrum.
* **No fitted state.** Every descriptor is scale-free (§4), so nothing is fit
  across the sample and there is no train/test boundary to violate.

The governing test rewrites every bar after a cutoff with arbitrary values and
asserts that every feature before the cutoff is **bit-identical**
(`test_future_bars_cannot_change_past_features`). Any centred window, any
forward-reaching transform, or any full-sample normalization fails it.

---

## 3. The window contract

`SpectralWindow` records the whole reproducibility surface, and every field is
copied onto every emitted feature's registry spec:

| field | meaning |
|---|---|
| `length` | causal lookback in bars |
| `segment_length` | sub-segment length for Welch/STFT averaging |
| `hop` | advance between sub-segments |
| `overlap` | derived: `segment_length - hop` |
| `n_segments` | derived: sub-segments averaged per window |
| `n_fft` | FFT length; above `segment_length` this zero-pads |
| `padding` | derived: `none` or `zero` — cannot disagree with the transform |
| `sampling_frequency` | samples per unit time (`1.0` for daily bars) |
| `frequency_unit` | `cycles_per_bar` |
| `warmup_bars` | derived: `length - 1` |
| `detrend` | `none` / `mean` / `linear`, applied per segment before tapering |
| `taper` | `boxcar` / `hann` / `hamming` / `blackman` |

Defaults: `length=64`, `segment_length=32`, `hop=16`, `n_fft=64`, Hann taper,
mean detrend — four half-overlapping segments per window.

### Bar time is not calendar time

Frequencies are **cycles per bar**. A "period of 5" means five trading bars, not
five calendar days. Weekends and holidays do not make the sampling irregular in
this coordinate; they make bar time a non-linear reparameterization of calendar
time. That is a modelling choice, and it is why the unit is named explicitly on
every spec rather than left implicit.

`test_calendar_gaps_do_not_change_bar_time_spectra` stretches the calendar
unevenly while preserving observation order and asserts the output is unchanged —
the assumption is tested, not just asserted in prose.

---

## 4. Mathematics

### Welch power spectral density

A single periodogram of an *L*-sample window is an **inconsistent** estimator: its
variance does not shrink as *L* grows. Welch's method averages *K* tapered
sub-periodograms, cutting variance by roughly *K* at the cost of a main lobe
`L/segment_length` times wider. For series with the signal-to-noise ratio of
daily returns, that trade is strongly worth taking.

Per segment: detrend → taper → `rfft` to `n_fft` → `|X_k|²`, scaled by
`1 / (fs · Σw²)`, with interior bins doubled for the one-sided fold, then
averaged over segments. The result integrates to the signal variance and matches
`scipy.signal.welch(..., scaling="density")` to **7e-15** absolute
(`test_welch_matches_scipy_reference`, all three detrend modes).

### Causal Morlet wavelet

The standard Morlet `π^(-1/4)·e^(iω₀t)·e^(-t²/2)` is symmetric about its centre,
so rolling it over a series reads the future. Truncating it to the past makes it
usable — and introduces two problems that are corrected rather than ignored:

1. **Admissibility.** A truncated wavelet no longer integrates to zero and would
   respond to a constant offset. The discrete mean is subtracted, restoring the
   zero-mean property on the truncated support.
2. **Normalization.** Energy is renormalized to unity *after* truncation, so
   responses are comparable across scales instead of decaying with the fraction
   of the wavelet that was cut away.

Scale ↔ frequency is published exactly: `f = ω₀ / (2πs)` cycles per bar. A scale
index that cannot be converted to a frequency is not interpretable evidence.

**Known limitation:** the coefficient is evaluated at the most recent bar, which
sits at the edge of the analysed support — inside the cone of influence — so it
is edge-affected by construction. This is inherent to *any* causal wavelet
estimate. A centred estimate would be cleaner and would read the future.

### Discrete wavelet transform

A Mallat cascade with orthogonal Daubechies filters (`haar`, `db2`), coefficients
written out literally so they can be checked against published values by eye.
Each level convolves with the scaling and wavelet filters and decimates by two.

**Periodic extension** is chosen deliberately: it is the only extension that keeps
the transform exactly orthogonal, which makes Parseval's identity an *exact*
invariant the tests assert rather than a tolerance — verified to 1e-15 across
both wavelets and three levels. The price is wrap-around contamination between a
window's oldest and newest samples. Every sample involved is still strictly in
the past, so this is an edge artefact, not a causality violation.

Outputs are per-level energy **fractions**, hence scale-free.

### Descriptors

Every descriptor is invariant to the amplitude of its input. That is the central
design rule, not a stylistic preference: a raw band power scales with the
variance of the series, so it would encode "this asset was volatile recently" — a
fact the volatility features already carry, and one whose distribution shifts
violently across regimes. Normalizing the spectrum to a probability mass
`p_k = P_k / ΣP` first means each descriptor answers the question the
representation exists for: *how is variation distributed across frequencies*,
independent of how much of it there was.

| descriptor | definition | range |
|---|---|---|
| `centroid` | `Σ f·p` — spectral centre of gravity | `[0, 0.5]` |
| `bandwidth` | `√(Σ (f-centroid)²·p)` | ≥ 0 |
| `entropy` | `-Σ p·log p / log K` — 1 is white, 0 a pure tone | `[0, 1]` |
| `flatness` | geometric ÷ arithmetic mean of the PSD (Wiener entropy) | `(0, 1]` |
| `rolloff` | lowest `f` capturing `q` (default 0.85) of the energy | `[0, 0.5]` |
| `peak_frequency` | `argmax_f p` — the dominant frequency | `[0, 0.5]` |
| `sparsity` | Hoyer: `(√K − ‖P‖₁/‖P‖₂)/(√K − 1)` | `[0, 1]` |
| `concentration` | energy share of the strongest `n` bins (default 3) | `[0, 1]` |
| `band_{low,mid,high}` | relative energy per band | `[0, 1]`, sums to 1 |
| `ratio_low_mid`, `ratio_mid_high` | adjacent band-power ratios | ≥ 0 |
| `flux` | mean L2 change between consecutive normalized STFT slices | ≥ 0 |

`entropy` and `sparsity` are both concentration measures but are not redundant:
entropy responds to the *shape* of the distribution over bins, Hoyer sparsity to
the *number* of active bins. `flux` is the only descriptor that needs the STFT
rather than an averaged spectrum — it separates a stably oscillating regime from
one whose frequency content is reorganizing, and each slice is normalized before
differencing so it reports shape change rather than amplitude change.

Default bands split at periods of 8 and 3 bars. These are a **stated convention,
not a tuned result**: no band edge was chosen by looking at an outcome. The band
ending at Nyquist is closed at the top so the bands are a true partition —
otherwise every band silently drops the Nyquist bin and the shares fail to sum
to one.

Wavelet channels additionally emit `cwt_peak_period`, `cwt_concentration`, and
`dwt_d1..dN` / `dwt_approx`.

---

## 5. Channels

| channel | series | extra lookback |
|---|---|---|
| `return` | raw returns | — |
| `volatility` | trailing std of returns (`volatility_window`, default 21) | +20 |
| `volume` | `log1p(volume)`, compressing the heavy right tail | — |
| `residual` | `r − β·r_market` using the trailing rolling beta (`beta_window`, default 63) | +62 |

The residual channel exists because without removing the common factor, every
asset's spectrum is close to a rescaled copy of the index's. It is the only
channel with a cross-asset dependency and is registered at `leakage_risk:
medium` accordingly; the rest are `low`.

`channel_lookback` adds the channel's own transform lookback to the window
length, and that total is what lands in `lookback_bars`/`warmup_bars` on the
registry spec. Understating it is how a feature ends up materialized from a
partition that never held enough history.

**Order imbalance is not implemented.** The sprint plan lists it "when
available"; the canonical panel is OHLCV with no quote or trade-side data.
Emitting a proxy and labelling it imbalance would be inventing evidence.

---

## 6. Failure modes — all fail closed

| condition | behaviour |
|---|---|
| engine disabled | `ValueError` on build and on registry construction |
| empty panel / missing column | `ValueError` naming the columns |
| benchmark absent while `residual` requested | `ValueError` — never a silently all-NaN channel |
| NaN anywhere in a window | NaN output for every window covering it; **never imputed** |
| constant window (zero AC power) | NaN descriptors, not a convenient-looking zero |
| numerically constant window | detrended residual collapsed to exact zero, so no spectrum is manufactured from rounding error (see below) |
| empty frequency band | NaN, not a structural zero |
| fewer than two STFT slices | `flux` is NaN |
| request over the compute ceiling | `ValueError` before allocation |
| invalid window/config geometry | `ValueError` at construction |

### The rounding-residue guard

Detrending a constant series does not give exactly zero: `4.2 − mean(4.2)` leaves
~1e-16 of floating-point residue. Left alone, that residue normalizes into a
perfectly plausible-looking spectrum computed **entirely from rounding error** —
a descriptor with a real value, a real peak frequency, and no information in it
whatsoever. Residuals below `CONSTANT_RELATIVE_TOLERANCE` (1e-12) relative to the
segment's own magnitude are collapsed to exact zero so the window reports as
degenerate instead. This was found by a test asserting the constant-series case,
not by inspection.

---

## 7. Compute and bounds

Measured on this MR's implementation; laptop-scale evidence, labelled as such.

**Environment:** Python 3.13.11, macOS 15.5 arm64 (Apple silicon), numpy 2.5.1,
pandas 2.3.3, single process. **Harness:** `build_spectral_features` on a
synthetic panel, 3 repetitions, median reported.

| panel | config | columns | median wall time | output size |
|---|---|---|---|---|
| 10 × 2520 bars (25.2k rows) | 4 channels | 56 | 0.31 s | 11.3 MB |
| 10 × 2520 bars (25.2k rows) | 4 channels + wavelets | 62 | 0.34 s | 12.5 MB |
| 50 × 2520 bars (126k rows) | 4 channels | 56 | 1.57 s | 56.4 MB |
| 50 × 2520 bars (126k rows) | 4 channels + wavelets | 62 | 1.77 s | 62.5 MB |

Cost is linear in rows and roughly linear in channels; wavelets add ~10%. These
are laptop numbers on synthetic data and are **not** a throughput guarantee for a
production panel.

Hard bounds, all refusal thresholds rather than tuning knobs — a request past
them is a configuration mistake, and accepting it silently would turn a typo into
an out-of-memory kill part-way through a backfill:

* `MAX_WINDOW_CELLS = 200_000_000` on `rows × length × channels`
* window `length ≤ 1024`, `n_fft ≤ 4096`, `dwt_levels ≤ 10`
* `cwt_periods` must exceed the Nyquist period of 2 bars — below that a "period"
  describes the sampling grid, not the market

---

## 8. Evidence

`tests/test_spectral_features.py` — 85 tests, grouped by the property each
defends:

* **Reference** — Welch vs `scipy.signal.welch` (7e-15, three detrend modes);
  STFT slices vs a direct segment transform; all four tapers vs
  `scipy.signal.get_window`; Haar level-1 details vs the closed form
  `(x[2k]−x[2k+1])/√2`; Parseval to 1e-15 across wavelets and levels; dominant
  frequency and CWT scale recovered from known tones.
* **Causality** — the future-mutation test; warm-up is NaN; windows are copies,
  not shared views; tickers are isolated.
* **Property/metamorphic** — scale invariance across 1e-4 … 1e4; bar-time
  semantics under calendar stretching; descriptor ranges and band partition;
  tones are sparser and lower-entropy than noise.
* **Numerical stability** — constant series, extreme amplitudes, NaN
  propagation, degenerate bands and windows.
* **Deterministic replay** — repeated builds bit-identical; row order
  irrelevant; implementation hash stable.
* **Integration** — registry output columns equal the emitted frame exactly;
  the full window contract is on every spec; enabling the family changes the
  materialization request identity.
* **Malformed input and bounds** — every guard above.

Coverage: `spectral.py` 100%, `spectral_descriptors.py` 96%,
`spectral_transforms.py` 92% (branch).

---

## 9. Planned ablations — not yet run

The AC asks for the ablation design to be documented. It is **designed, not
executed**; running it before the Sprint 2 baselines and study are frozen would
be selection on an unfrozen benchmark.

Under identical folds, embargo, and cost assumptions, comparing:

1. conventional time-domain features only (the frozen Sprint 2 baseline);
2. baseline + Welch descriptors;
3. baseline + wavelet descriptors;
4. baseline + all spectral channels;
5. spectral only.

with per-channel and per-descriptor-family drops to attribute any difference,
and the rejection rule fixed in advance. SF-S3-MR3 adds the adaptive-decomposition
arms; SF-S3-MR11 publishes the decision.

---

## 10. Residual limitations

* **No incremental-value evidence exists yet.** Correctness is demonstrated;
  usefulness is not, and must not be implied.
* **CWT coefficients are cone-of-influence affected** at the evaluation bar, by
  construction. Treat `cwt_peak_period` as indicative, not precise.
* **DWT periodic extension** wraps a window's oldest and newest samples together.
  Orthogonality was chosen over edge purity; the artefact is bounded to the
  window edge.
* **Bar time ≠ calendar time.** A 5-bar period is not a 5-day period, and a
  holiday-heavy stretch compresses calendar time relative to bar time. Any
  cross-market comparison on differing calendars must account for this.
* **Descriptors are per-asset and per-window.** No cross-sectional
  standardization is applied; a model that needs comparability across the
  cross-section should add it explicitly.
* **Synthetic evidence only.** Every number in this document comes from
  synthetic panels or closed-form references. Nothing here is market evidence.
* **The default band edges and window geometry are conventions**, not optimized
  values. They were fixed before any outcome was inspected and should stay
  frozen until an ablation with a pre-registered rejection rule says otherwise.
* **Sprint 2 dependency.** The issue specifies implementation from current `dev`.
  The Sprint 2 conventional feature-contract MRs (Signalattice #42–#44) are still
  open, so the "frozen Sprint 2 baseline" this representation is eventually to be
  measured against does not exist yet. That is a sequencing fact, not a blocker
  for the engine, and is exactly why no comparison is claimed here.
