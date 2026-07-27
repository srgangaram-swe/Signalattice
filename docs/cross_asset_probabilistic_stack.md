# Cross-asset probabilistic stack (#14, #15, #16)

Three coupled capabilities that close Signal Foundry Sprint 3: a hierarchical
Bayesian return-distribution forecaster, causal dynamic asset graphs with a
temporal message-passing baseline, and a pre-registered scenario and decision
laboratory.

They ship as one merge request because they are one vertical slice: the graph
and the laboratory both consume the posterior, and the laboratory's whole
purpose is to test whether the uncertainty the forecaster produces changes a
decision. Reviewing or reverting them separately would leave a half-connected
chain.

> **No edge is claimed.** Every number below comes from synthetic panels with
> planted structure. This MR delivers the machinery and demonstrates it recovers
> what was planted. It is not evidence that any of it works on markets.

---

## 1. Hierarchical Bayesian forecaster (#14)

### Why partial pooling

Two bad extremes bracket cross-asset modelling. Fit each asset independently and
every estimate is starved — a year of daily returns is ~252 points, so an
asset-level mean is mostly noise. Pool everything and asset identity vanishes,
which is worse, because differing is the whole point.

Partial pooling is the principled middle:

```
r_it        ~ StudentT(nu, mu_i, sigma_i)
mu_i        ~ Normal(mu_pop, tau^2)
log sigma_i ~ Normal(psi_pop, omega^2)
```

The shrinkage weight is **not a tuning knob** — it falls out of the ratio of
within-asset to between-asset variance. `HierarchicalPosterior.shrinkage()`
reports it per asset, which is what makes pooling auditable: an asset shrunk to
~1 is being described by its peers rather than its own data, and any claim about
it should say so.

Student-t rather than Normal, because a Gaussian forced onto daily returns
inflates sigma to cover the tails and then understates ordinary-day risk. `nu` is
**estimated on a bounded grid**, not assumed; a fit landing near 30 would itself
be evidence the tails are mild.

### Inference

Gibbs sampling written from scratch, so priors, diagnostics, and compute are
inspectable. It exploits the Student-t's scale-mixture representation
(`t = Normal / sqrt(Gamma)`): the latent per-observation weights make `mu` and
`sigma` conditionally conjugate, and the weights are themselves interpretable —
a small one is the sampler declaring an observation an outlier, so robustness
falls out of the model rather than being bolted on.

Convergence is **reported, never assumed**. Split R-hat (split so within-chain
drift cannot hide behind chains that drift together) and effective sample size
via Geyer's initial positive sequence. `converged` is a conjunction over *every*
monitored parameter — an aggregate would let one badly mixed parameter hide.

### Measured on a planted panel

5 assets, 400 days, true `nu = 5`, 300-day train / 100-day test:

| quantity | result |
|---|---|
| sigma recovery | within 25% relative on every asset |
| nu recovery | posterior mean ~5.1 against truth 5 |
| held-out log predictive density vs no-pooling Gaussian | **positive on aggregate**, majority of assets improved |
| posterior predictive checks (std, kurtosis, q05, q95) | p-values away from 0/1 |

**The aggregate claim is the honest one.** Partial pooling promises better
population-level held-out density; it does *not* promise per-asset dominance, and
an asset with abundant data and a genuinely idiosyncratic mean can legitimately
score slightly worse after being shrunk toward peers it does not resemble. The
test asserts the aggregate plus a majority, not a sweep — an earlier draft
asserted a sweep and was wrong.

---

## 2. Causal dynamic asset graphs (#15)

A static correlation matrix assumes dependence is the same in 2015 and in a
crash. These graphs are re-estimated on a rolling trailing window.

Two leakage hazards are specific to graphs and both are closed **by
construction**:

* **Time leakage** — every edge at date `t` is estimated from a window ending
  strictly before `t`. Tested by rewriting all future returns and asserting the
  earlier adjacency is bit-identical.
* **Graph leakage** — the subtle one. Message passing lets information flow
  between assets, so an edge built from a target's own future could carry that
  future back through a neighbour. Edges are functions of lagged returns only,
  so no path length can leak. Tested at two rounds of propagation.

Three edge estimators, and the difference matters:

| method | directed | finds a planted lead-lag? |
|---|---|---|
| `correlation` | no | no — picks the co-driven asset |
| `partial_correlation` | no | no — removes the common factor but stays symmetric |
| `lagged_causality` | **yes** | **yes** — recovers A0→A1 at weight 0.69 |

Partial correlation is ridge-regularized before inversion: a short-window
empirical correlation matrix is frequently near-singular, and inverting it
unregularized turns estimation noise into enormous spurious edges.

The adjacency is **row-normalized with a zero diagonal**, so message passing
averages neighbours rather than summing them — otherwise a high-degree node's
update grows with degree rather than with information.

The model is deliberately compact: one or two rounds of neighbour averaging with
no fitted readout layer. The question is whether cross-asset structure carries
incremental information *at all*, and a regression head would absorb the very
effect being measured. `self_weight = 1.0` reduces **exactly** to the
lagged-return baseline, which is what makes the ablation meaningful; that
identity is asserted by test.

---

## 3. Scenario and decision laboratory (#16)

A point forecast plus a covariance matrix yields an allocation optimal for a
world that will not occur. This asks a different question: does an allocation
that *knows* about estimation uncertainty beat one that ignores it, under costs,
constraints, and tail risk?

### Coherent scenarios

Drawing each asset's return from its own marginal would destroy the dependence
that makes diversification meaningful and would flatter every portfolio.
Scenarios use a Gaussian copula on the training-window correlation, with
marginals from the posterior predictive — so fat tails *and* estimation
uncertainty both survive.

`scenario_coherence_report()` measures realized scenario correlation against the
correlation the copula was given (max error < 0.03 on the reference run).
Without it, "coherent" would be an assertion rather than a measurement.

**Stated limitation:** a Gaussian copula reproduces linear dependence but *not
tail dependence*, so joint extreme moves are under-represented. Tail figures from
this set are a **floor, not a ceiling**.

### Pre-registration

`ExperimentDesign` freezes policies, cost model, constraints, horizon, and
metrics, and hashes them. Changing the cost model after seeing results produces a
different identity, so a retuned experiment cannot be presented as the original.
`evaluate_policies` sorts rows **by policy name, not by performance** — ranking
by a metric invites reading the top row as a winner.

### A bug this surfaced

The first implementation clipped weights to the cap and then renormalized by the
L1 norm. That is *not* a projection: renormalizing pushes a clipped weight back
above the cap, and the reference run returned `max_weight = 0.333` under a
`0.25` constraint. It now alternates clip and renormalize to a fixed point,
asserts the post-condition, and **refuses an infeasible cap** (one that cannot
fund a fully invested book) rather than silently rescaling.

### Reference comparison

5 assets, 3000 scenarios, 5-day horizon, 10bps cost, 25% cap:

| policy | net expected | volatility | CVaR(5%) | effective assets |
|---|---|---|---|---|
| deterministic mean-variance | highest gross | highest | worst | 3.0 → 5.0 after fix |
| equal weight | lowest | lowest | best | 6.0 |
| risk parity | ≈ equal weight | ≈ equal weight | ≈ equal weight | 6.0 |
| robust CVaR | between | between | between | 5.9 |

On the realized out-of-sample vector, plug-in mean-variance did **worst** —
the textbook estimation-error result, reproduced rather than hidden. Risk parity
is included precisely because means are the hardest moment to estimate and the
one mean-variance is most sensitive to; a policy that declines to use them is a
serious competitor and omitting it would be flattering.

---

## 4. Evidence

`tests/test_cross_asset_stack.py` — 46 tests covering parameter recovery,
aggregate predictive-density improvement, shrinkage reporting, predictive
uncertainty composition, posterior predictive checks, reproducibility and
seed-sensitivity, non-convergence reporting, split-R-hat sensitivity to
disagreeing chains, time and graph leakage, the `self_weight = 1` reduction,
next-period alignment, dependence preservation, constraint satisfaction, cost
accounting, infeasible-cap refusal, design-identity stability, and every
documented bound.

Coverage: `hierarchical.py` 93%, `asset_graph.py` 91%, `scenario_lab.py` 93%
(branch). Repository total 86.25% against an unchanged 80% floor.

---

## 5. Residual limitations

* **Synthetic evidence only.** No market claim is made or implied.
* **Means remain hard.** The posterior recovers scale and tail index well;
  asset-level *means* stay noisy, which is a property of 300 daily observations,
  not a defect of the sampler. Any strategy leaning on estimated means inherits
  that.
* **Gaussian copula understates joint tails** (§3).
* **The message-passing model has no fitted readout**, so it measures whether
  graph structure carries information, not the best achievable use of it.
* **Graph staleness.** The adjacency is re-estimated every `refit_every` bars and
  is stale in between; the trade is explicit, not hidden.
* **Univariate emissions.** The hierarchical model pools means and scales but
  does not model a joint likelihood; cross-asset dependence enters only through
  the copula at scenario time.
* **`nu` is shared across assets**, so one heavy-tailed name pulls the common
  tail index. A per-asset `nu` is a natural extension and is not implemented.
