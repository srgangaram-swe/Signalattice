"""Hierarchical Bayesian cross-asset return-distribution forecaster (SA #14).

Two bad extremes bracket cross-asset modelling. Fit every asset independently
and each estimate is starved: a year of daily returns is ~252 points, so an
asset-level mean is mostly noise. Pool everything into one model and asset
identity vanishes, which is worse — the whole point is that assets differ.

Partial pooling is the principled middle. Each asset's parameters are drawn from
a population distribution whose parameters are themselves estimated, so an asset
with little data is shrunk hard toward the population and an asset with plenty
is left near its own estimate. The shrinkage weight is not a tuning knob; it
falls out of the ratio of within-asset to between-asset variance.

The model, for asset ``i`` on day ``t``:

    r_it  ~ StudentT(nu, mu_i, sigma_i)
    mu_i  ~ Normal(mu_pop, tau^2)
    log sigma_i ~ Normal(psi_pop, omega^2)

Student-t rather than Normal for the observation model, because daily equity
returns have tails a Gaussian cannot represent; forcing a Gaussian inflates
sigma to cover the tails and then understates ordinary-day risk. ``nu`` is
estimated, not assumed, and a fitted ``nu`` near 30 would itself be evidence
that the tails are mild.

Inference is a Gibbs sampler with conjugate updates where they exist and a
bounded griddy step for ``nu``, written from scratch so priors, diagnostics, and
the compute budget are all inspectable. Everything is seeded and bounded: chain
count, iterations, and warm-up are declared, and the sampler reports R-hat,
effective sample size, and divergence-free status rather than assuming
convergence.

**Causality.** The fit consumes a training window only. Posterior predictive
draws for a later window use parameters estimated strictly before it, and the
public entry point takes an explicit train/evaluate split rather than inferring
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Refusal thresholds, not tuning knobs.
MAX_ASSETS = 200
MAX_CHAINS = 8
MAX_DRAWS = 50_000
MIN_OBSERVATIONS_PER_ASSET = 20

#: Degrees-of-freedom grid for the Student-t observation model. Bounded below at
#: 2.5 so the variance exists, and above at 50 where the t is Gaussian for any
#: practical purpose — beyond that the likelihood is flat and sampling it is
#: wasted compute.
_NU_GRID = np.array([2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50], dtype=np.float64)


class HierarchicalModelError(ValueError):
    """Raised when a hierarchical fit request is unusable or out of bounds."""


@dataclass(frozen=True)
class HierarchicalPriors:
    """Weakly informative priors on the population parameters.

    Deliberately weak, not flat. A flat prior on a variance component lets the
    sampler wander into regions where the population variance is enormous and
    partial pooling silently degenerates into no pooling at all. These scales
    are stated in daily-return units so they can be argued with.
    """

    #: Population mean return: centred at zero, scale 1% daily.
    mean_location: float = 0.0
    mean_scale: float = 0.01
    #: Between-asset spread of means (half-Normal scale), 1% daily.
    tau_scale: float = 0.01
    #: Population mean of log volatility: exp(-4) ~ 1.8% daily.
    log_sigma_location: float = -4.0
    log_sigma_scale: float = 1.0
    #: Between-asset spread of log volatility (half-Normal scale).
    omega_scale: float = 0.5

    def __post_init__(self) -> None:
        for name in ("mean_scale", "tau_scale", "log_sigma_scale", "omega_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise HierarchicalModelError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class PosteriorDiagnostics:
    """Convergence evidence for one fit.

    ``converged`` is a *conjunction*: every parameter must satisfy both the
    split R-hat and effective-sample-size thresholds. Reporting a single
    aggregate would let one badly mixed parameter hide behind well-behaved ones.
    """

    n_chains: int
    n_draws: int
    n_warmup: int
    max_r_hat: float
    min_ess: float
    r_hat_threshold: float
    ess_threshold: float
    seed: int

    @property
    def converged(self) -> bool:
        """Whether every monitored parameter met both thresholds."""
        return bool(self.max_r_hat <= self.r_hat_threshold and self.min_ess >= self.ess_threshold)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record for the run manifest."""
        return {
            "n_chains": self.n_chains,
            "n_draws": self.n_draws,
            "n_warmup": self.n_warmup,
            "max_r_hat": self.max_r_hat,
            "min_ess": self.min_ess,
            "r_hat_threshold": self.r_hat_threshold,
            "ess_threshold": self.ess_threshold,
            "converged": self.converged,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class HierarchicalPosterior:
    """Posterior draws over asset-level and population parameters.

    ``mu`` and ``sigma`` are ``(n_draws, n_assets)``; the population parameters
    are ``(n_draws,)``. Draws are the deliverable, not point estimates: the
    scenario laboratory consumes the whole posterior, and collapsing to a mean
    would discard exactly the estimation uncertainty it exists to propagate.
    """

    assets: tuple[str, ...]
    mu: FloatArray
    sigma: FloatArray
    nu: FloatArray
    mu_population: FloatArray
    tau: FloatArray
    diagnostics: PosteriorDiagnostics

    def __post_init__(self) -> None:
        draws, assets = self.mu.shape
        if self.sigma.shape != (draws, assets):
            raise HierarchicalModelError("mu and sigma must share their shape")
        if len(self.assets) != assets:
            raise HierarchicalModelError("asset labels must match the posterior width")
        for name in ("nu", "mu_population", "tau"):
            if getattr(self, name).shape != (draws,):
                raise HierarchicalModelError(f"{name} must have one value per draw")

    @property
    def n_draws(self) -> int:
        """Number of retained posterior draws."""
        return int(self.mu.shape[0])

    def shrinkage(self, observed_means: FloatArray) -> FloatArray:
        """Return each asset's shrinkage toward the population mean, in ``[0, 1]``.

        ``0`` means the posterior sits on the asset's own sample mean and ``1``
        that it was pulled entirely to the population. This is the number that
        makes partial pooling auditable: an asset whose shrinkage is ~1 is being
        described by its peers, not by its own data, and any claim about it
        should say so.
        """
        posterior_mean = self.mu.mean(axis=0)
        population = float(self.mu_population.mean())
        spread = observed_means - population
        with np.errstate(divide="ignore", invalid="ignore"):
            retained = np.where(np.abs(spread) > 0.0, (posterior_mean - population) / spread, 0.0)
        return np.clip(1.0 - retained, 0.0, 1.0)

    def predictive_draws(self, generator: np.random.Generator, *, n_paths: int) -> FloatArray:
        """Draw ``(n_paths, n_assets)`` one-step posterior predictive returns.

        Each path first samples a posterior draw and then samples the Student-t
        observation given it, so the result carries **both** estimation and
        observation uncertainty. Sampling the observation at a fixed posterior
        mean instead would understate predictive spread — the classic way an
        interval ends up too narrow.
        """
        if n_paths < 1:
            raise HierarchicalModelError("n_paths must be at least one")
        indices = generator.integers(0, self.n_draws, size=n_paths)
        mu = self.mu[indices]
        sigma = self.sigma[indices]
        nu = self.nu[indices][:, None]
        standard = generator.standard_t(np.broadcast_to(nu, mu.shape))
        return np.asarray(mu + sigma * standard, dtype=np.float64)


def _panel_matrix(
    panel: pd.DataFrame, value_column: str
) -> tuple[tuple[str, ...], list[FloatArray]]:
    """Return asset labels and their finite observation vectors."""
    required = {"date", "ticker", value_column}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise HierarchicalModelError(f"panel is missing required columns: {missing}")
    assets: list[str] = []
    observations: list[FloatArray] = []
    for ticker, group in panel.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        values = group[value_column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size < MIN_OBSERVATIONS_PER_ASSET:
            continue
        assets.append(str(ticker))
        observations.append(finite)
    if not assets:
        raise HierarchicalModelError(
            f"no asset has the required {MIN_OBSERVATIONS_PER_ASSET} finite observations"
        )
    if len(assets) > MAX_ASSETS:
        raise HierarchicalModelError(f"panel exceeds the {MAX_ASSETS}-asset ceiling")
    return tuple(assets), observations


def _student_t_log_likelihood(values: FloatArray, mu: float, sigma: float, nu: float) -> float:
    """Return the Student-t log likelihood of one asset's observations."""
    from scipy.special import gammaln

    standardized = (values - mu) / sigma
    return float(
        values.size
        * (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * nu) - np.log(sigma))
        - (nu + 1.0) / 2.0 * np.sum(np.log1p(standardized**2 / nu))
    )


def fit_hierarchical_returns(
    panel: pd.DataFrame,
    *,
    value_column: str = "return",
    priors: HierarchicalPriors | None = None,
    n_chains: int = 4,
    n_draws: int = 1_000,
    n_warmup: int = 500,
    seed: int = 42,
    r_hat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
) -> HierarchicalPosterior:
    """Fit the partial-pooling model by Gibbs sampling on a training panel.

    The sampler exploits the Student-t's scale-mixture representation: writing
    ``t = Normal / sqrt(Gamma)`` introduces per-observation latent weights that
    make ``mu`` and ``sigma`` conditionally conjugate. Those weights are also
    interpretable — a small weight is the sampler declaring an observation an
    outlier and down-weighting it, which is robustness falling out of the model
    rather than being bolted on.

    Args:
        panel: Long panel with ``date``, ``ticker``, and ``value_column``.
        priors: Population priors; weakly informative defaults.
        n_chains: Independent chains, needed for split R-hat.
        n_draws: Retained draws per chain.
        n_warmup: Discarded warm-up draws per chain.
        seed: Root seed; each chain draws from a named child stream.

    Raises:
        HierarchicalModelError: On an unusable panel or out-of-range budget.
    """
    priors = priors or HierarchicalPriors()
    if not 2 <= n_chains <= MAX_CHAINS:
        raise HierarchicalModelError(f"n_chains must be in [2, {MAX_CHAINS}] for split R-hat")
    if not 1 <= n_draws <= MAX_DRAWS or not 0 <= n_warmup <= MAX_DRAWS:
        raise HierarchicalModelError(f"draw counts must be within [0, {MAX_DRAWS}]")
    if seed < 0:
        raise HierarchicalModelError("seed must be non-negative")

    assets, observations = _panel_matrix(panel, value_column)
    n_assets = len(assets)
    chains: list[dict[str, FloatArray]] = []

    for chain in range(n_chains):
        generator = np.random.default_rng(np.random.SeedSequence([seed, chain]))
        state = _run_chain(
            observations,
            priors,
            generator,
            n_draws=n_draws,
            n_warmup=n_warmup,
        )
        chains.append(state)

    monitored = {
        "mu_population": np.stack([chain["mu_population"] for chain in chains]),
        "tau": np.stack([chain["tau"] for chain in chains]),
        "nu": np.stack([chain["nu"] for chain in chains]),
    }
    for index in range(n_assets):
        monitored[f"mu[{index}]"] = np.stack([chain["mu"][:, index] for chain in chains])
        monitored[f"sigma[{index}]"] = np.stack([chain["sigma"][:, index] for chain in chains])

    r_hats = [split_r_hat(values) for values in monitored.values()]
    ess_values = [effective_sample_size(values) for values in monitored.values()]
    diagnostics = PosteriorDiagnostics(
        n_chains=n_chains,
        n_draws=n_draws,
        n_warmup=n_warmup,
        max_r_hat=float(np.max(r_hats)),
        min_ess=float(np.min(ess_values)),
        r_hat_threshold=r_hat_threshold,
        ess_threshold=ess_threshold,
        seed=seed,
    )
    return HierarchicalPosterior(
        assets=assets,
        mu=np.concatenate([chain["mu"] for chain in chains], axis=0),
        sigma=np.concatenate([chain["sigma"] for chain in chains], axis=0),
        nu=np.concatenate([chain["nu"] for chain in chains]),
        mu_population=np.concatenate([chain["mu_population"] for chain in chains]),
        tau=np.concatenate([chain["tau"] for chain in chains]),
        diagnostics=diagnostics,
    )


def _run_chain(
    observations: list[FloatArray],
    priors: HierarchicalPriors,
    generator: np.random.Generator,
    *,
    n_draws: int,
    n_warmup: int,
) -> dict[str, FloatArray]:
    """Run one Gibbs chain and return its retained draws."""
    n_assets = len(observations)
    counts = np.array([values.size for values in observations], dtype=np.float64)
    sample_means = np.array([float(values.mean()) for values in observations])
    # Guard a zero-variance asset: a constant series would drive sigma to zero
    # and the likelihood to infinity, the standard degenerate solution.
    sample_scales = np.array([max(float(values.std()), 1e-8) for values in observations])

    mu = sample_means.copy()
    sigma = sample_scales.copy()
    nu = 6.0
    mu_population = float(sample_means.mean())
    tau = max(float(sample_means.std()), 1e-6)

    kept_mu = np.empty((n_draws, n_assets))
    kept_sigma = np.empty((n_draws, n_assets))
    kept_nu = np.empty(n_draws)
    kept_mu_pop = np.empty(n_draws)
    kept_tau = np.empty(n_draws)

    for iteration in range(n_warmup + n_draws):
        # --- latent scale mixture weights: w ~ Gamma((nu+1)/2, (nu + z^2)/2) ---
        weights = []
        for index, values in enumerate(observations):
            standardized = (values - mu[index]) / sigma[index]
            shape = (nu + 1.0) / 2.0
            rate = (nu + standardized**2) / 2.0
            weights.append(generator.gamma(shape, 1.0 / rate))

        # --- asset means: conjugate Normal given weights and population ---
        for index, values in enumerate(observations):
            weight = weights[index]
            precision_data = float(weight.sum()) / sigma[index] ** 2
            precision_prior = 1.0 / tau**2
            precision = precision_data + precision_prior
            centre = (
                float(np.sum(weight * values)) / sigma[index] ** 2 + mu_population * precision_prior
            ) / precision
            mu[index] = generator.normal(centre, np.sqrt(1.0 / precision))

        # --- asset scales: conjugate inverse-gamma on sigma^2 -----------------
        for index, values in enumerate(observations):
            weight = weights[index]
            residual = values - mu[index]
            shape = counts[index] / 2.0 + 2.0
            rate = float(np.sum(weight * residual**2)) / 2.0 + np.exp(
                2.0 * priors.log_sigma_location
            )
            sigma[index] = float(np.sqrt(1.0 / generator.gamma(shape, 1.0 / rate)))
            sigma[index] = max(sigma[index], 1e-10)

        # --- population mean: conjugate Normal --------------------------------
        precision_prior = 1.0 / priors.mean_scale**2
        precision_data = n_assets / tau**2
        precision = precision_prior + precision_data
        centre = (priors.mean_location * precision_prior + float(mu.sum()) / tau**2) / precision
        mu_population = float(generator.normal(centre, np.sqrt(1.0 / precision)))

        # --- between-asset spread: half-Normal prior, inverse-gamma draw ------
        shape = n_assets / 2.0 + 1.0
        rate = float(np.sum((mu - mu_population) ** 2)) / 2.0 + priors.tau_scale**2
        tau = float(np.sqrt(1.0 / generator.gamma(shape, 1.0 / rate)))
        tau = max(tau, 1e-10)

        # --- degrees of freedom: bounded griddy Gibbs -------------------------
        log_posterior = np.array(
            [
                sum(
                    _student_t_log_likelihood(values, mu[index], sigma[index], candidate)
                    for index, values in enumerate(observations)
                )
                for candidate in _NU_GRID
            ]
        )
        log_posterior -= log_posterior.max()
        probabilities = np.exp(log_posterior)
        probabilities /= probabilities.sum()
        nu = float(generator.choice(_NU_GRID, p=probabilities))

        if iteration >= n_warmup:
            position = iteration - n_warmup
            kept_mu[position] = mu
            kept_sigma[position] = sigma
            kept_nu[position] = nu
            kept_mu_pop[position] = mu_population
            kept_tau[position] = tau

    return {
        "mu": kept_mu,
        "sigma": kept_sigma,
        "nu": kept_nu,
        "mu_population": kept_mu_pop,
        "tau": kept_tau,
    }


def split_r_hat(draws: FloatArray) -> float:
    """Return the split-chain potential scale reduction factor.

    Splitting each chain in half before comparing is what makes the statistic
    sensitive to *within-chain* drift: an unsplit R-hat is happy with several
    chains that are each slowly trending, as long as they trend together.
    """
    chains, length = draws.shape
    if length < 4:
        return float("inf")
    half = length // 2
    split = np.concatenate([draws[:, :half], draws[:, half : 2 * half]], axis=0)
    n_split, n_samples = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    between = n_samples * float(np.var(chain_means, ddof=1))
    within = float(np.mean(chain_vars))
    if within <= 0.0:
        return 1.0
    estimate = ((n_samples - 1) / n_samples) * within + between / n_samples
    return float(np.sqrt(estimate / within))


def effective_sample_size(draws: FloatArray) -> float:
    """Return a bounded effective sample size from the autocorrelation sum.

    Truncated at the first lag whose summed consecutive autocorrelation pair is
    negative (Geyer's initial positive sequence), because summing further adds
    noise rather than signal.
    """
    chains, length = draws.shape
    if length < 4:
        return 0.0
    flat = draws - draws.mean(axis=1, keepdims=True)
    variance = float(np.mean(np.var(draws, axis=1, ddof=1)))
    if variance <= 0.0:
        return float(chains * length)
    total = 0.0
    for lag in range(1, min(length // 2, 200)):
        correlation = float(np.mean([np.mean(row[:-lag] * row[lag:]) for row in flat]) / variance)
        if correlation < 0.0:
            break
        total += correlation
    return float(chains * length / max(1.0 + 2.0 * total, 1.0))


def posterior_predictive_checks(
    posterior: HierarchicalPosterior,
    panel: pd.DataFrame,
    *,
    value_column: str = "return",
    seed: int = 7,
    n_replications: int = 200,
) -> pd.DataFrame:
    """Compare replicated statistics against the observed ones.

    A model can fit its own training mean perfectly and still be wrong about
    everything that matters. These checks target the statistics a return model
    is actually used for — dispersion and tails — and report a Bayesian p-value
    per asset per statistic. Values near 0 or 1 are the model failing to
    reproduce the data it was fit on; near 0.5 is unremarkable.

    Reported and never thresholded here: what counts as a failing check belongs
    to the pre-registered study, not to the function computing it.
    """
    generator = np.random.default_rng(seed)
    assets, observations = _panel_matrix(panel, value_column)
    if assets != posterior.assets:
        raise HierarchicalModelError("check panel assets must match the fitted posterior")

    statistics = {
        "std": lambda values: float(np.std(values)),
        "kurtosis": lambda values: float(
            np.mean(((values - values.mean()) / max(values.std(), 1e-12)) ** 4)
        ),
        "q05": lambda values: float(np.quantile(values, 0.05)),
        "q95": lambda values: float(np.quantile(values, 0.95)),
    }
    rows: list[dict[str, Any]] = []
    for index, asset in enumerate(assets):
        observed = observations[index]
        draw_indices = generator.integers(0, posterior.n_draws, size=n_replications)
        replicated = {name: np.empty(n_replications) for name in statistics}
        for replication, draw in enumerate(draw_indices):
            sample = posterior.mu[draw, index] + posterior.sigma[
                draw, index
            ] * generator.standard_t(posterior.nu[draw], size=observed.size)
            for name, function in statistics.items():
                replicated[name][replication] = function(sample)
        for name, function in statistics.items():
            rows.append(
                {
                    "ticker": asset,
                    "statistic": name,
                    "observed": function(observed),
                    "replicated_mean": float(replicated[name].mean()),
                    "bayesian_p_value": float(np.mean(replicated[name] >= function(observed))),
                }
            )
    return pd.DataFrame(rows)


def independent_baseline(panel: pd.DataFrame, *, value_column: str = "return") -> pd.DataFrame:
    """Return per-asset maximum-likelihood Gaussian estimates.

    The no-pooling comparison the hierarchical model must justify itself
    against. If partial pooling does not beat this on held-out predictive
    density, the extra machinery bought nothing and should be said so.
    """
    assets, observations = _panel_matrix(panel, value_column)
    return pd.DataFrame(
        {
            "ticker": list(assets),
            # Deliberately not named `mean`/`std`: those shadow DataFrame method
            # names, so `baseline.mean` silently resolves to the method rather
            # than the column and the mistake type-checks.
            "mean_return": [float(values.mean()) for values in observations],
            "volatility": [max(float(values.std()), 1e-12) for values in observations],
            "n_observations": [int(values.size) for values in observations],
        }
    )


def held_out_log_predictive_density(
    posterior: HierarchicalPosterior,
    evaluation_panel: pd.DataFrame,
    *,
    value_column: str = "return",
) -> pd.DataFrame:
    """Return mean held-out log predictive density per asset.

    The honest comparison metric: it rewards a model for putting probability
    mass where future observations actually landed, and penalises both
    over-confidence and vagueness. Computed on a window the posterior never saw.
    """
    from scipy.special import gammaln

    frame = evaluation_panel.sort_values(["ticker", "date"])
    rows: list[dict[str, Any]] = []
    for position, asset in enumerate(posterior.assets):
        raw = np.asarray(
            frame.loc[frame["ticker"] == asset, value_column].to_numpy(), dtype=np.float64
        )
        values = raw[np.isfinite(raw)]
        if values.size == 0:
            continue
        mu = posterior.mu[:, position][:, None]
        sigma = posterior.sigma[:, position][:, None]
        nu = posterior.nu[:, None]
        standardized = (values[None, :] - mu) / sigma
        log_density = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(np.pi * nu)
            - np.log(sigma)
            - (nu + 1.0) / 2.0 * np.log1p(standardized**2 / nu)
        )
        # Mixture over draws in log space, then average across observations.
        pointwise = _log_mean_exp(log_density, axis=0)
        rows.append(
            {
                "ticker": asset,
                "mean_log_predictive_density": float(np.mean(pointwise)),
                "n_observations": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def _log_mean_exp(values: FloatArray, axis: int) -> FloatArray:
    """Numerically stable log of a mean of exponentials."""
    peak = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - peak)
    return np.asarray(
        np.squeeze(np.log(np.mean(shifted, axis=axis, keepdims=True)) + peak, axis=axis),
        dtype=np.float64,
    )


def gaussian_baseline_log_predictive_density(
    baseline: pd.DataFrame, evaluation_panel: pd.DataFrame, *, value_column: str = "return"
) -> pd.DataFrame:
    """Return the independent Gaussian baseline's held-out log density."""
    frame = evaluation_panel.sort_values(["ticker", "date"])
    rows: list[dict[str, Any]] = []
    for position in range(len(baseline)):
        ticker = str(baseline["ticker"].iloc[position])
        location = float(baseline["mean_return"].iloc[position])
        scale = float(baseline["volatility"].iloc[position])
        raw = np.asarray(
            frame.loc[frame["ticker"] == ticker, value_column].to_numpy(), dtype=np.float64
        )
        values = raw[np.isfinite(raw)]
        if values.size == 0:
            continue
        density = -0.5 * np.log(2.0 * np.pi * scale**2) - 0.5 * ((values - location) / scale) ** 2
        rows.append(
            {
                "ticker": ticker,
                "mean_log_predictive_density": float(np.mean(density)),
                "n_observations": int(values.size),
            }
        )
    return pd.DataFrame(rows)
