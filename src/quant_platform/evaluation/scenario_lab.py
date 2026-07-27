"""Posterior scenario generation and a robust decision laboratory (SA #16).

A point forecast plus a covariance matrix produces an allocation that is optimal
for a world that will not occur. This module does the other thing: it draws
*coherent cross-asset scenarios* from the hierarchical posterior, and then asks
whether an allocation that knows about that uncertainty actually beats one that
ignores it — under costs, constraints, and tail risk.

"Coherent" is doing real work. Drawing each asset's return from its own marginal
would destroy the cross-asset dependence that makes diversification meaningful,
and would make every portfolio look better than it is. Scenarios here are drawn
with a Gaussian copula whose correlation is estimated from the training window,
so marginals come from the posterior predictive (fat-tailed, with estimation
uncertainty) while the dependence structure is preserved.

The comparison is **pre-registered**: policies, cost model, constraints, horizon,
and metrics are declared in a frozen :class:`ExperimentDesign` before any
scenario is drawn. Nothing here selects a winner — it produces the comparison,
and the rejection rule belongs to the study that froze the design.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quant_platform.models.hierarchical import HierarchicalPosterior

FloatArray = NDArray[np.float64]

#: Refusal thresholds, not tuning knobs.
MAX_SCENARIOS = 200_000
MAX_ASSETS = 200


class ScenarioLabError(ValueError):
    """Raised when a scenario or decision request is unusable."""


@dataclass(frozen=True)
class ExperimentDesign:
    """A frozen pre-registration for one decision experiment.

    Declared and hashed *before* scenarios are drawn. The identity is what makes
    the comparison auditable: changing the cost model or the constraint set after
    seeing results produces a different identity, so a silently retuned
    experiment cannot be presented as the original one.
    """

    name: str
    n_scenarios: int
    horizon_days: int
    seed: int
    #: Round-trip transaction cost in basis points applied to turnover.
    cost_bps: float = 10.0
    #: Maximum absolute weight per asset.
    max_weight: float = 0.25
    #: Whether shorting is permitted.
    allow_short: bool = False
    #: Tail quantile for CVaR.
    tail_quantile: float = 0.05
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ScenarioLabError("experiment design requires a name")
        if not 1 <= self.n_scenarios <= MAX_SCENARIOS:
            raise ScenarioLabError(f"n_scenarios must be in [1, {MAX_SCENARIOS}]")
        if self.horizon_days < 1:
            raise ScenarioLabError("horizon_days must be at least one")
        if self.seed < 0:
            raise ScenarioLabError("seed must be non-negative")
        if self.cost_bps < 0.0 or not np.isfinite(self.cost_bps):
            raise ScenarioLabError("cost_bps must be finite and non-negative")
        if not 0.0 < self.max_weight <= 1.0:
            raise ScenarioLabError("max_weight must lie in (0, 1]")
        if not 0.0 < self.tail_quantile < 0.5:
            raise ScenarioLabError("tail_quantile must lie in (0, 0.5)")

    @property
    def identity(self) -> str:
        """Return the deterministic SHA-256 identity of this pre-registration."""
        payload = {
            "name": self.name,
            "n_scenarios": self.n_scenarios,
            "horizon_days": self.horizon_days,
            "seed": self.seed,
            "cost_bps": self.cost_bps,
            "max_weight": self.max_weight,
            "allow_short": self.allow_short,
            "tail_quantile": self.tail_quantile,
            "notes": list(self.notes),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScenarioSet:
    """Coherent cross-asset scenarios drawn from a posterior.

    ``returns`` is ``(n_scenarios, n_assets)`` cumulative returns over the
    design's horizon.
    """

    assets: tuple[str, ...]
    returns: FloatArray
    design_identity: str
    correlation: FloatArray

    def __post_init__(self) -> None:
        if self.returns.ndim != 2 or self.returns.shape[1] != len(self.assets):
            raise ScenarioLabError("scenario matrix must be (n_scenarios, n_assets)")
        if not np.isfinite(self.returns).all():
            raise ScenarioLabError("scenarios must be finite")


def estimate_correlation(panel: pd.DataFrame, assets: tuple[str, ...]) -> FloatArray:
    """Return the training-window return correlation for ``assets``.

    Estimated on the training panel only. A correlation fitted on the evaluation
    window would let the scenario set know how assets co-moved in the period the
    allocation is being judged on.
    """
    wide = panel.pivot_table(index="date", columns="ticker", values="return", aggfunc="last")
    missing = [asset for asset in assets if asset not in wide.columns]
    if missing:
        raise ScenarioLabError(f"correlation panel is missing assets: {missing}")
    matrix = np.corrcoef(wide.loc[:, list(assets)].dropna().to_numpy(float), rowvar=False)
    matrix = np.nan_to_num(np.atleast_2d(matrix), nan=0.0)
    np.fill_diagonal(matrix, 1.0)
    return np.asarray(matrix, dtype=np.float64)


def generate_scenarios(
    posterior: HierarchicalPosterior,
    correlation: FloatArray,
    design: ExperimentDesign,
) -> ScenarioSet:
    """Draw horizon-cumulative scenarios coupling posterior marginals.

    A Gaussian copula: draw correlated uniforms from the training correlation,
    then push them through each asset's posterior predictive quantile. The
    dependence comes from the copula and the marginals from the posterior, so
    fat tails and estimation uncertainty both survive into the scenario set.

    The copula is Gaussian, which is a real limitation and is documented: it
    does not reproduce tail *dependence*, so joint crashes are under-represented
    relative to reality. It is a floor on tail risk, not a ceiling.
    """
    n_assets = len(posterior.assets)
    if correlation.shape != (n_assets, n_assets):
        raise ScenarioLabError("correlation must be square over the posterior assets")
    if n_assets > MAX_ASSETS:
        raise ScenarioLabError(f"scenario set exceeds the {MAX_ASSETS}-asset ceiling")

    generator = np.random.default_rng(np.random.SeedSequence([design.seed, design.horizon_days]))
    # Nearest positive-definite repair: an empirical correlation can have tiny
    # negative eigenvalues that make the Cholesky factorization fail outright.
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    repaired = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-8, None)) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)
    factor = np.linalg.cholesky(repaired)

    horizon_returns = np.zeros((design.n_scenarios, n_assets), dtype=np.float64)
    for _ in range(design.horizon_days):
        normals = generator.standard_normal((design.n_scenarios, n_assets)) @ factor.T
        # Posterior predictive per step, coupled through the normal ranks.
        draw_indices = generator.integers(0, posterior.n_draws, size=design.n_scenarios)
        mu = posterior.mu[draw_indices]
        sigma = posterior.sigma[draw_indices]
        nu = posterior.nu[draw_indices][:, None]
        # Transform correlated normals to t-marginals with the drawn nu, keeping
        # the rank dependence the copula supplies.
        chi = generator.chisquare(np.broadcast_to(nu, normals.shape))
        student = normals * np.sqrt(nu / np.maximum(chi, 1e-12))
        horizon_returns += mu + sigma * student

    return ScenarioSet(
        assets=posterior.assets,
        returns=horizon_returns,
        design_identity=design.identity,
        correlation=repaired,
    )


# ---------------------------------------------------------------------------
# Allocation policies
# ---------------------------------------------------------------------------

Policy = Callable[[ScenarioSet, ExperimentDesign], FloatArray]


def _project(weights: FloatArray, design: ExperimentDesign) -> FloatArray:
    """Project onto the constraint set: bounded weights summing to one.

    Clip-then-renormalize is not a projection — dividing by the L1 norm can push
    a clipped weight straight back above the cap, silently returning a book that
    violates the constraint it was asked to respect. Alternating the two to a
    fixed point restores the invariant, and the post-condition is asserted so a
    violation can never be returned instead of raised.
    """
    lower = -design.max_weight if design.allow_short else 0.0
    if design.max_weight * weights.size < 1.0:
        raise ScenarioLabError(
            f"max_weight {design.max_weight} cannot fund a book across "
            f"{weights.size} assets; raise the cap or widen the universe"
        )
    bounded = np.asarray(weights, dtype=np.float64)
    for _ in range(100):
        bounded = np.clip(bounded, lower, design.max_weight)
        total = float(np.sum(np.abs(bounded)))
        if total <= 0.0:
            bounded = np.full(weights.size, 1.0 / weights.size)
            break
        bounded = bounded / total
        if float(np.max(np.abs(bounded))) <= design.max_weight + 1e-12:
            break
    else:  # pragma: no cover - the alternating projection converges in practice
        bounded = np.full(weights.size, 1.0 / weights.size)
    if float(np.max(np.abs(bounded))) > design.max_weight + 1e-9:
        raise ScenarioLabError("projection failed to satisfy the weight cap")
    return np.asarray(bounded, dtype=np.float64)


def equal_weight_policy(scenarios: ScenarioSet, design: ExperimentDesign) -> FloatArray:
    """The naive diversification benchmark every other policy must beat."""
    return np.full(len(scenarios.assets), 1.0 / len(scenarios.assets))


def deterministic_mean_variance_policy(
    scenarios: ScenarioSet, design: ExperimentDesign
) -> FloatArray:
    """Plug-in mean-variance on the scenario mean and covariance.

    The "ignores uncertainty" arm: it treats the estimated moments as if they
    were the truth. This is the policy the uncertainty-aware ones must beat, and
    its weakness is well known — it concentrates precisely where estimation error
    is largest.
    """
    mean = scenarios.returns.mean(axis=0)
    covariance = np.cov(scenarios.returns, rowvar=False)
    covariance = np.atleast_2d(covariance) + 1e-8 * np.eye(len(scenarios.assets))
    weights = np.linalg.solve(covariance, mean)
    return _project(weights, design)


def robust_cvar_policy(scenarios: ScenarioSet, design: ExperimentDesign) -> FloatArray:
    """Uncertainty-aware allocation penalising conditional tail loss.

    Maximises mean return minus a multiple of CVaR, solved by projected gradient
    ascent on the scenario set. Optimising the tail *directly on scenarios* is
    the point: a variance penalty is symmetric and cannot distinguish a fat left
    tail from a fat right one, whereas CVaR only ever charges for the left.
    """
    n_assets = len(scenarios.assets)
    weights = np.full(n_assets, 1.0 / n_assets)
    cut = max(int(design.tail_quantile * scenarios.returns.shape[0]), 1)
    step = 0.05
    for _ in range(200):
        portfolio = scenarios.returns @ weights
        order = np.argsort(portfolio)[:cut]
        # Gradient of (mean - tail penalty) with respect to the weights.
        gradient = scenarios.returns.mean(axis=0) - scenarios.returns[order].mean(axis=0)
        weights = _project(weights + step * gradient, design)
    return weights


def risk_parity_policy(scenarios: ScenarioSet, design: ExperimentDesign) -> FloatArray:
    """Inverse-volatility weights — uncertainty-aware without estimating means.

    Included because means are the hardest moment to estimate and the one
    mean-variance is most sensitive to. A policy that simply declines to use them
    is a serious competitor, and saying so is more honest than omitting it.
    """
    volatility = np.maximum(scenarios.returns.std(axis=0), 1e-12)
    return _project(1.0 / volatility, design)


DEFAULT_POLICIES: dict[str, Policy] = {
    "equal_weight": equal_weight_policy,
    "deterministic_mean_variance": deterministic_mean_variance_policy,
    "risk_parity": risk_parity_policy,
    "robust_cvar": robust_cvar_policy,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_policies(
    scenarios: ScenarioSet,
    design: ExperimentDesign,
    *,
    policies: dict[str, Policy] | None = None,
    realized: FloatArray | None = None,
    previous_weights: FloatArray | None = None,
) -> pd.DataFrame:
    """Compare policies on the scenario set, net of costs.

    Returns one row per policy with expected return, volatility, CVaR, the
    turnover cost actually charged, and — when a realized out-of-sample return
    vector is supplied — what the policy would have earned on it.

    Rows are sorted by policy name, not by performance. Sorting by a metric
    invites reading the top row as a winner, which is the selection the frozen
    design exists to prevent.
    """
    policies = policies or DEFAULT_POLICIES
    if realized is not None and realized.shape != (len(scenarios.assets),):
        raise ScenarioLabError("realized returns must have one value per asset")
    baseline = (
        previous_weights
        if previous_weights is not None
        else np.full(len(scenarios.assets), 1.0 / len(scenarios.assets))
    )
    cut = max(int(design.tail_quantile * scenarios.returns.shape[0]), 1)

    rows: list[dict[str, Any]] = []
    for name in sorted(policies):
        weights = policies[name](scenarios, design)
        if weights.shape != (len(scenarios.assets),):
            raise ScenarioLabError(f"policy {name!r} returned the wrong weight shape")
        portfolio = scenarios.returns @ weights
        turnover = float(np.sum(np.abs(weights - baseline)))
        cost = turnover * design.cost_bps / 10_000.0
        tail = float(np.mean(np.sort(portfolio)[:cut]))
        row: dict[str, Any] = {
            "policy": name,
            "expected_return_gross": float(portfolio.mean()),
            "expected_return_net": float(portfolio.mean() - cost),
            "volatility": float(portfolio.std()),
            "cvar": tail,
            "turnover": turnover,
            "cost": cost,
            "max_weight": float(np.max(np.abs(weights))),
            "effective_assets": float(1.0 / np.sum(weights**2)) if np.any(weights) else 0.0,
        }
        if realized is not None:
            row["realized_return_net"] = float(realized @ weights - cost)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs["design_identity"] = design.identity
    return frame


def scenario_coherence_report(scenarios: ScenarioSet, design: ExperimentDesign) -> dict[str, Any]:
    """Return evidence that the scenario set preserved its dependence structure.

    Without this, "coherent" is an assertion. The realized scenario correlation
    is compared against the correlation the copula was given; a large gap means
    the marginals or the horizon accumulation broke the dependence, and any
    diversification conclusion drawn from the set would be unfounded.
    """
    realized = np.corrcoef(scenarios.returns, rowvar=False)
    realized = np.nan_to_num(np.atleast_2d(realized), nan=0.0)
    offdiag = ~np.eye(len(scenarios.assets), dtype=bool)
    error = np.abs(realized[offdiag] - scenarios.correlation[offdiag])
    return {
        "design_identity": design.identity,
        "n_scenarios": int(scenarios.returns.shape[0]),
        "horizon_days": design.horizon_days,
        "max_correlation_error": float(error.max()) if error.size else 0.0,
        "mean_correlation_error": float(error.mean()) if error.size else 0.0,
        "copula": "gaussian",
        "limitation": (
            "A Gaussian copula reproduces linear dependence but not tail "
            "dependence, so joint extreme moves are under-represented; tail "
            "figures from this set are a floor, not a ceiling."
        ),
    }
