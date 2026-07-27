# Adaptive decomposition and the representation contract (SF-S3-MR3)

Bounded, seeded, convergence-reporting implementations of EMD, EEMD, CEEMDAN,
and VMD, their mode descriptors, and the contract that makes raw, spectral,
wavelet, EMD, and VMD arms comparable on identical folds.

> **No incremental-value claim.** This MR delivers the decompositions and their
> correctness evidence. Whether any of them beats the frozen Sprint 2 baselines
> is SF-S3-MR11's controlled experiment. The issue's non-goal — treating unstable
> modes as economic signals — is respected throughout.

---

## 1. Why adaptive decomposition needs more guard rails than a transform

Fourier and wavelet transforms project onto a basis chosen **in advance**. EMD
and its relatives derive their basis **from the data**. That is what makes them
attractive for non-stationary series, and it is exactly what makes them
dangerous: the basis is a function of the sample, so an unstable decomposition
produces modes that look like structure and are an artefact of one noise
realization.

Three refusals follow, and they shape the whole module:

* **Nothing runs unbounded.** Sifting iterations, mode counts, ensemble sizes,
  ADMM iterations, and series length all have explicit ceilings. An adaptive
  method with no iteration cap can spin on a pathological window forever.
* **Non-convergence is never silent.** Every call returns a
  `DecompositionReport` with the stopping criterion that fired, iterations used
  against budget, and an explicit `converged` flag. **Spending a budget is not
  convergence** — an exhausted mode ceiling reports `converged=False`, because
  the residual still has extrema and the decomposition is truncated.
* **Randomness is seeded and recorded.** Each realization draws from a named
  child stream of `(seed, method, index)`, so realization *i* is identical
  whether the ensemble runs in order, in parallel, or resumed — a property a
  single advancing generator cannot offer.

---

## 2. The algorithms

### EMD — Huang et al. (1998)

Sift the highest-frequency intrinsic mode out of the residual, subtract, repeat
until the residual is monotonic or the mode budget is spent.

* **Envelopes** are cubic splines through local maxima and minima.
* **Boundary treatment** mirrors the two outermost extrema about each endpoint.
  A spline through interior extrema only is wildly unconstrained at the edges,
  and that envelope error propagates inward through every later sift — the
  classic EMD end effect. Mirroring pins the spline without inventing amplitude.
* **Stopping** uses Huang's Cauchy-type criterion: iterate until the relative
  change between successive candidates falls below `sd_tolerance`.

### EEMD — Wu & Huang (2009)

Plain EMD suffers **mode mixing**: sifting follows whichever extrema happen to
exist, so an intermittent component makes one IMF carry wildly different scales.
Adding white noise populates scale space uniformly, so each realization is
anchored by a dense unbiased extrema set; averaging cancels the noise at
`1/sqrt(n_ensembles)` while true components survive.

Residual noise does not fully vanish, so EEMD's modes do not reconstruct exactly
the way EMD's do. This implementation returns the residual as
`input - sum(modes)`, which makes reconstruction exact by construction and puts
the ensemble's leftover noise into `residual_energy_fraction` where it is
visible instead of hidden.

### CEEMDAN — Torres et al. (2011)

EEMD averages *independent* decompositions, so mode counts differ between
realizations. CEEMDAN extracts one mode at a time from a **shared** residual, so
every realization contributes to the same mode index and reconstruction is exact
by construction. `beta_k` scales injected noise to the current residual, so late
stages are not swamped by noise sized for the original series.

### VMD — Dragomiretskiy & Zosso (2014)

Not greedy sifting but a single variational problem: find `n_modes` band-limited
components, each compact around its own centre frequency, summing to the signal.
Robust to noise and free of mode mixing by construction — at the cost of needing
the mode count in advance, which EMD infers.

ADMM in the frequency domain: each mode is a Wiener-filtered residual
`(f - sum_{i≠k} u_i + λ/2) / (1 + α(ω - ω_k)²)`, so `alpha` directly sets the
bandwidth penalty; each centre frequency moves to its mode's power centroid; `λ`
enforces exact reconstruction when `tau > 0` (default `0`, the noise-tolerant
mode, with the leftover visible in the residual).

**Initialization is log-spaced, deliberately.** Linear spacing puts almost every
starting centre frequency in the top decade. On a signal whose components span
decades — the normal case in markets, where a weekly and an annual cycle differ
by ~50× — several modes start above every true component, converge onto the same
one, and the decomposition silently reports a duplicated mode. This was observed
directly: with linear initialization, a 6/24/96-bar test signal produced two
modes at period 6 and lost the 24 entirely. Log spacing recovers all three.

The series is mirror-extended before transforming so the periodic FFT does not
wrap the window's end onto its start.

---

## 3. Reconstruction is the governing invariant

For all four methods, modes plus residual sum back to the input. A decomposition
that does not reconstruct is not a decomposition. Measured at **< 1e-9** for
every method and published in the report as `reconstruction_error` rather than
assumed.

---

## 4. Mode order is not a timescale

EMD sifts highest-frequency first, and VMD is explicitly sorted to match, so for
those two the mode index *is* a frequency rank — tested.

**The noise-assisted variants offer no such guarantee.** Injecting noise at each
stage can leave a later mode with a higher dominant frequency than an earlier
one. This is asserted as a documented limitation in the test suite rather than
papered over.

The consequence for consumers is concrete: every mode carries a measured
`dominant_period`, and modes must be ordered and interpreted by that, never by
trusting the index to mean a timescale.

---

## 5. Mode descriptors

A decomposition yields a variable number of modes; a feature vector needs a fixed
width. Modes beyond `n_reported_modes` fold into the residual accounting, and
**missing modes are NaN, never zero** — "zero energy" and "this mode did not
exist" are different facts.

Per mode, all scale-free:

| descriptor | meaning |
|---|---|
| `energy_fraction` | share of total signal energy |
| `dominant_period` | bars per cycle at the mode's spectral peak — what makes an adaptive mode interpretable at all |
| `spectral_entropy` | how tonal the mode is; a genuine IMF is narrowband, so high entropy is the signature of mode mixing |
| `stability` | `tanh(log(late energy / early energy))` in `[-1, 1]`; an intermittent mode that exists in only part of the window is not a component of the whole window |

Plus `residual_energy_fraction`, `max_cross_correlation` (orthogonal modes should
barely correlate, so a high value means one component was split across modes),
and `converged`.

Reporting entropy, stability, and cross-correlation is what lets an unstable mode
be **rejected** rather than used — the issue's non-goal made operational.

---

## 6. The comparison contract

The sprint's question is whether an advanced representation beats a conventional
one. That is only meaningful if the arms differ in *exactly one* respect. Two
teams comparing "wavelets vs raw" while one detrends and the other does not have
measured their preprocessing, not their representations.

`representations.py` removes that failure mode **structurally**: every family
consumes the same `causal_windows` output, built once from one `SpectralWindow`.
There is no per-family preprocessing argument to get wrong, because there is no
per-family preprocessing at all. `shared_window_identity()` publishes the shared
contract so a comparison can *prove* it — if two runs disagree there, their
descriptors are not comparable whatever the metrics say.

Families: `raw` (scale-free time-domain shape statistics), `spectral` (MR1 Welch
descriptors), `wavelet` (MR1 DWT band energies), `emd`, `vmd`.

Every family is warm-up aligned to the same bar — tested for all five.

---

## 7. Compute and bounds

Adaptive decomposition is orders of magnitude costlier than an FFT. Measured:
Python 3.13.11, macOS 15.5 arm64, numpy 2.5.1, single process, median of 5.

**Per 64-bar window:**

| method | median |
|---|---|
| VMD | 0.69 ms |
| EMD | 1.32 ms |
| EEMD (20 ensembles) | 28.2 ms |
| CEEMDAN (20 ensembles) | 60.1 ms |

**Per 1260-bar series, all bars, by stride:**

| stride | raw | spectral | wavelet | emd | vmd |
|---|---|---|---|---|---|
| 1 | 0.03 s | 0.00 s | 0.00 s | 1.86 s | 0.98 s |
| 5 | 0.01 s | 0.00 s | 0.00 s | 0.37 s | 0.20 s |
| 21 | 0.00 s | 0.00 s | 0.00 s | 0.09 s | 0.04 s |

EMD is ~500× the cost of the spectral arm and VMD ~250×. That is why `stride`
exists. **Skipped bars stay NaN** — an honest gap, never a forward fill, which
would smear a later window's information backwards into earlier bars.

EEMD and CEEMDAN are deliberately *not* wired into the rolling contract: at 28–60
ms per window they are 20–45× EMD, which is not viable per-bar over a panel. They
are available directly for analysis and for the MR11 study on sampled windows.

Ceilings (refusal thresholds, not tuning knobs): `MAX_MODES` 20,
`MAX_SIFT_ITERATIONS` 2000, `MAX_ENSEMBLES` 500, `MAX_VMD_ITERATIONS` 5000,
`MAX_SERIES_LENGTH` 100 000, `MIN_SERIES_LENGTH` 16.

---

## 8. Evidence

`tests/test_decomposition.py` — 78 tests:

* **Reference** — VMD recovers 6/24/96-bar components (tolerance derived from
  the `period²/N` resolution limit, not chosen for convenience); EMD separates
  well-spaced tones; larger `alpha` demonstrably narrows modes.
* **Reconstruction** — all four methods to < 1e-9.
* **Ordering** — frequency ordering asserted for EMD and VMD; the noise-assisted
  caveat asserted as a *limitation*, with a guard that fails loudly if the
  fixture ever stops demonstrating it.
* **Scale** — EMD and VMD equivariant across 1e-3 … 1e3; mode descriptors
  invariant.
* **Noise** — same seed bit-identical, different seed different, seed/ensemble/
  noise recorded; `noise_std=0` reduces EEMD exactly to EMD.
* **Mode mixing** — the classic demonstration, measured: on an intermittent
  signal EMD's IMF1 carries **75%** of its energy at the carrier frequency;
  CEEMDAN's carries **0.1%**. Also asserts the fixture genuinely exhibits the
  pathology, so the test cannot pass vacuously.
* **Degenerate** — short, non-finite, multidimensional, constant, and monotonic
  series; constant and ramp terminate with zero modes and still reconstruct.
* **Bounds/convergence** — 14 out-of-range parameter refusals; exhausted mode and
  ADMM budgets reported as `converged=False`; report and shape invariants.
* **Comparison contract** — declared columns per family; identical warm-up across
  all five; stride gaps are NaN not forward-filled; determinism; unknown family
  refused; NaN windows yield missing descriptors.

Coverage: `decomposition.py` 94%, `representations.py` 90% (branch).

---

## 9. Planned ablations — not yet run

Designed, deliberately not executed before the Sprint 2 baselines are frozen.
Under the shared window contract, identical folds, and identical costs:

1. `raw` (conventional arm);
2. `spectral` (fixed basis, Fourier);
3. `wavelet` (fixed basis, multiresolution);
4. `emd` (adaptive, greedy);
5. `vmd` (adaptive, variational);

with mode-count and `alpha` sensitivity as inner ablations, and the rejection
rule fixed in advance. SF-S3-MR11 publishes the decision.

---

## 10. Residual limitations

* **No incremental-value evidence exists**, and none is implied.
* **The basis is data-dependent.** Modes from adjacent windows are not the same
  basis, so a mode-indexed feature is not a stable coordinate over time. This is
  intrinsic to adaptive decomposition and is why `dominant_period`, `stability`,
  and `max_cross_correlation` are emitted alongside every energy.
* **Mode index is not a frequency rank for EEMD/CEEMDAN** (§4).
* **VMD needs its mode count chosen in advance**, and the choice matters. It is
  a declared configuration value, not something inferred from the data.
* **EEMD/CEEMDAN are not available per-bar** at viable cost (§7).
* **Boundary effects remain.** Mirror extension reduces but does not remove EMD's
  end effect, and the evaluation bar sits at the window edge where it is worst.
* **`stride > 1` leaves genuine gaps.** Downstream consumers must handle NaN
  rather than assume a dense column.
* **Synthetic and closed-form evidence only.** Nothing here is market evidence.
* **Depends on unmerged PRs #53 and #54.**
