"""Tests for the cross-asset probabilistic stack (SA #14, #15, #16).

Three coupled capabilities, tested by the property each has to earn:

* the hierarchical model must **recover planted parameters**, beat a no-pooling
  baseline on held-out predictive density, and report its own non-convergence;
* the graph must be **causal by construction** — future returns cannot reach an
  earlier edge through any path — and its lead-lag estimator must find a planted
  lead-lag that symmetric estimators miss;
* the scenario laboratory must preserve **cross-asset dependence**, respect its
  constraint set exactly, and refuse to be re-tuned after the design is frozen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.evaluation.scenario_lab import (
    DEFAULT_POLICIES,
    ExperimentDesign,
    ScenarioLabError,
    ScenarioSet,
    estimate_correlation,
    evaluate_policies,
    generate_scenarios,
    scenario_coherence_report,
)
from quant_platform.models.asset_graph import (
    AssetGraphError,
    build_dynamic_graph,
    evaluate_cross_asset_generalization,
    temporal_message_passing,
)
from quant_platform.models.hierarchical import (
    HierarchicalModelError,
    HierarchicalPosterior,
    HierarchicalPriors,
    fit_hierarchical_returns,
    gaussian_baseline_log_predictive_density,
    held_out_log_predictive_density,
    independent_baseline,
    posterior_predictive_checks,
    split_r_hat,
)

N_ASSETS = 5
N_DAYS = 400
SPLIT = 300


@pytest.fixture(scope="module")
def planted() -> dict[str, object]:
    """A panel with known population parameters and a planted lead-lag."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    true_mu = rng.normal(0.0004, 0.0003, N_ASSETS)
    true_sigma = np.exp(rng.normal(-4.0, 0.25, N_ASSETS))
    values = np.column_stack(
        [true_mu[i] + true_sigma[i] * rng.standard_t(5, N_DAYS) for i in range(N_ASSETS)]
    )
    # Asset 0 leads assets 1 and 2 by exactly one bar.
    values[1:, 1] += 0.6 * values[:-1, 0]
    values[1:, 2] += 0.5 * values[:-1, 0]
    wide = pd.DataFrame(values, index=dates, columns=[f"A{i}" for i in range(N_ASSETS)])
    panel = (
        wide.reset_index()
        .melt(id_vars="index", var_name="ticker", value_name="return")
        .rename(columns={"index": "date"})
    )
    return {
        "wide": wide,
        "panel": panel,
        "dates": dates,
        "true_sigma": true_sigma,
        "train": panel[panel["date"] < dates[SPLIT]],
        "test": panel[panel["date"] >= dates[SPLIT]],
    }


@pytest.fixture(scope="module")
def posterior(planted: dict[str, object]) -> HierarchicalPosterior:
    return fit_hierarchical_returns(planted["train"], n_chains=2, n_draws=300, n_warmup=150, seed=3)


# ---------------------------------------------------------------------------
# #14 Hierarchical Bayesian forecaster
# ---------------------------------------------------------------------------


def test_posterior_recovers_planted_scale_and_tails(
    posterior: HierarchicalPosterior, planted: dict[str, object]
) -> None:
    """Scale and tail index are identifiable from 300 daily observations."""
    recovered = posterior.sigma.mean(axis=0)
    truth = np.asarray(planted["true_sigma"])
    np.testing.assert_allclose(recovered, truth, rtol=0.25)
    # The observation model is Student-t with nu=5; a fit that landed at the
    # Gaussian end of the grid would mean the tails were not detected.
    assert 3.0 <= float(posterior.nu.mean()) <= 12.0


def test_partial_pooling_beats_no_pooling_out_of_sample(
    posterior: HierarchicalPosterior, planted: dict[str, object]
) -> None:
    """The claim the extra machinery has to earn.

    The promise of partial pooling is an *aggregate* one: borrowing strength
    improves held-out predictive density across the population. It deliberately
    does **not** claim per-asset dominance, and asserting that would be wrong —
    an asset with abundant data and a genuinely idiosyncratic mean is shrunk
    toward peers it does not resemble and can legitimately score slightly worse.
    The honest test is therefore the aggregate plus a majority, not a sweep.
    """
    baseline = independent_baseline(planted["train"])
    hierarchical = held_out_log_predictive_density(posterior, planted["test"])
    gaussian = gaussian_baseline_log_predictive_density(baseline, planted["test"])
    merged = hierarchical.merge(gaussian, on="ticker", suffixes=("_h", "_g"))
    delta = merged["mean_log_predictive_density_h"] - merged["mean_log_predictive_density_g"]
    assert float(delta.mean()) > 0.0
    assert int((delta > 0).sum()) > len(merged) / 2


def test_shrinkage_is_reported_per_asset(
    posterior: HierarchicalPosterior, planted: dict[str, object]
) -> None:
    baseline = independent_baseline(planted["train"])
    shrinkage = posterior.shrinkage(baseline["mean_return"].to_numpy())
    assert shrinkage.shape == (N_ASSETS,)
    assert np.all((shrinkage >= 0.0) & (shrinkage <= 1.0))


def test_predictive_draws_carry_estimation_and_observation_uncertainty(
    posterior: HierarchicalPosterior,
) -> None:
    """Predictive spread must exceed the average observation scale alone."""
    generator = np.random.default_rng(0)
    draws = posterior.predictive_draws(generator, n_paths=4000)
    assert draws.shape == (4000, N_ASSETS)
    assert np.isfinite(draws).all()
    assert float(draws.std(axis=0).mean()) > float(posterior.sigma.mean()) * 0.5


def test_posterior_predictive_checks_are_not_degenerate(
    posterior: HierarchicalPosterior, planted: dict[str, object]
) -> None:
    checks = posterior_predictive_checks(posterior, planted["train"], n_replications=40)
    assert set(checks["statistic"]) == {"std", "kurtosis", "q05", "q95"}
    # A p-value pinned at 0 or 1 everywhere means the model cannot reproduce the
    # data it was fit on.
    assert checks["bayesian_p_value"].between(0.01, 0.99).mean() > 0.7


def test_fit_is_reproducible_and_seed_sensitive(planted: dict[str, object]) -> None:
    first = fit_hierarchical_returns(planted["train"], n_chains=2, n_draws=60, n_warmup=30, seed=5)
    second = fit_hierarchical_returns(planted["train"], n_chains=2, n_draws=60, n_warmup=30, seed=5)
    np.testing.assert_array_equal(first.mu, second.mu)
    other = fit_hierarchical_returns(planted["train"], n_chains=2, n_draws=60, n_warmup=30, seed=6)
    assert not np.array_equal(first.mu, other.mu)


def test_short_chains_report_non_convergence(planted: dict[str, object]) -> None:
    """Non-convergence must be visible, not assumed away."""
    starved = fit_hierarchical_returns(
        planted["train"], n_chains=2, n_draws=20, n_warmup=5, seed=1, ess_threshold=400.0
    )
    assert starved.diagnostics.converged is False
    assert starved.diagnostics.min_ess < 400.0
    assert "converged" in starved.diagnostics.to_dict()


def test_split_r_hat_detects_disagreeing_chains() -> None:
    agreeing = np.random.default_rng(0).normal(size=(4, 500))
    assert split_r_hat(agreeing) < 1.05
    # Chains centred in different places must not be called converged.
    disagreeing = agreeing + np.array([[0.0], [5.0], [10.0], [15.0]])
    assert split_r_hat(disagreeing) > 1.2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_chains": 1}, "n_chains"),
        ({"n_chains": 99}, "n_chains"),
        ({"n_draws": 0}, "draw counts"),
        ({"seed": -1}, "seed"),
    ],
)
def test_hierarchical_bounds_are_enforced(
    planted: dict[str, object], kwargs: dict, message: str
) -> None:
    with pytest.raises(HierarchicalModelError, match=message):
        fit_hierarchical_returns(planted["train"], **kwargs)


def test_priors_must_be_positive() -> None:
    with pytest.raises(HierarchicalModelError, match="tau_scale"):
        HierarchicalPriors(tau_scale=0.0)


def test_unusable_panels_are_refused(planted: dict[str, object]) -> None:
    with pytest.raises(HierarchicalModelError, match="missing required columns"):
        fit_hierarchical_returns(planted["train"].drop(columns=["return"]))
    thin = planted["train"].head(10)
    with pytest.raises(HierarchicalModelError, match="finite observations"):
        fit_hierarchical_returns(thin)


# ---------------------------------------------------------------------------
# #15 Dynamic asset graph
# ---------------------------------------------------------------------------


def test_lagged_estimator_recovers_a_planted_lead_lag(planted: dict[str, object]) -> None:
    """The directed estimator must find what symmetric ones structurally cannot."""
    graph = build_dynamic_graph(
        planted["wide"],
        as_of=planted["dates"][250],
        window=120,
        method="lagged_causality",
        threshold=0.2,
    )
    leaders = dict(graph.neighbours("A1", top=3))
    assert "A0" in leaders, leaders
    assert leaders["A0"] == max(leaders.values())


@pytest.mark.parametrize("method", ["correlation", "partial_correlation", "lagged_causality"])
def test_graph_is_row_normalized_with_zero_diagonal(
    planted: dict[str, object], method: str
) -> None:
    graph = build_dynamic_graph(
        planted["wide"], as_of=planted["dates"][250], window=120, method=method, threshold=0.2
    )
    assert np.allclose(np.diag(graph.adjacency), 0.0)
    sums = graph.adjacency.sum(axis=1)
    for total in sums:
        assert total == pytest.approx(0.0) or total == pytest.approx(1.0)
    assert 0.0 <= graph.density <= 1.0


def test_graph_never_reads_the_forecast_date_or_later(planted: dict[str, object]) -> None:
    """Time leakage: rewriting the future must not change an earlier graph."""
    wide = planted["wide"]
    as_of = planted["dates"][250]
    baseline = build_dynamic_graph(wide, as_of=as_of, window=120, method="lagged_causality")
    mutated = wide.copy()
    rng = np.random.default_rng(3)
    mutated.loc[mutated.index >= as_of] = rng.normal(
        0.0, 0.5, mutated.loc[mutated.index >= as_of].shape
    )
    perturbed = build_dynamic_graph(mutated, as_of=as_of, window=120, method="lagged_causality")
    np.testing.assert_array_equal(baseline.adjacency, perturbed.adjacency)


def test_message_passing_predictions_never_use_the_current_bar(
    planted: dict[str, object],
) -> None:
    """Graph leakage: no path length may carry a future return backwards."""
    wide = planted["wide"]
    cutoff = 320
    baseline = temporal_message_passing(wide, window=120, n_rounds=2, threshold=0.2)
    mutated = wide.copy()
    rng = np.random.default_rng(7)
    mutated.iloc[cutoff:] = rng.normal(0.0, 0.5, mutated.iloc[cutoff:].shape)
    perturbed = temporal_message_passing(mutated, window=120, n_rounds=2, threshold=0.2)
    left = baseline.predictions.iloc[:cutoff].to_numpy()
    right = perturbed.predictions.iloc[:cutoff].to_numpy()
    np.testing.assert_allclose(np.nan_to_num(left), np.nan_to_num(right), rtol=1e-12, atol=1e-12)


def test_full_self_weight_reduces_to_the_lagged_baseline(planted: dict[str, object]) -> None:
    """The ablation that makes the graph's contribution measurable."""
    wide = planted["wide"]
    solo = temporal_message_passing(wide, window=120, self_weight=1.0, threshold=0.2)
    lagged = wide.shift(1).reindex_like(solo.predictions)
    observed = solo.predictions.notna().all(axis=1)
    np.testing.assert_allclose(
        solo.predictions[observed].to_numpy(), lagged[observed].to_numpy(), rtol=1e-12
    )


def test_generalization_scores_against_the_next_period(planted: dict[str, object]) -> None:
    result = temporal_message_passing(
        planted["wide"], window=120, method="lagged_causality", threshold=0.2
    )
    scores = evaluate_cross_asset_generalization(result.predictions, planted["wide"])
    assert set(scores.columns) == {
        "ticker",
        "n_observations",
        "information_coefficient",
        "hit_rate",
    }
    assert scores["hit_rate"].between(0.0, 1.0).all()
    assert result.to_dict()["edge_method"] == "lagged_causality"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window": 5}, "window"),
        ({"threshold": 1.5}, "threshold"),
        ({"max_degree": 0}, "max_degree"),
    ],
)
def test_graph_bounds_are_enforced(planted: dict[str, object], kwargs: dict, message: str) -> None:
    with pytest.raises(AssetGraphError, match=message):
        build_dynamic_graph(planted["wide"], as_of=planted["dates"][250], **kwargs)


def test_insufficient_history_is_refused(planted: dict[str, object]) -> None:
    with pytest.raises(AssetGraphError, match="strictly before"):
        build_dynamic_graph(planted["wide"], as_of=planted["dates"][30], window=120)


def test_unknown_asset_lookup_is_refused(planted: dict[str, object]) -> None:
    graph = build_dynamic_graph(planted["wide"], as_of=planted["dates"][250], window=120)
    with pytest.raises(AssetGraphError, match="unknown asset"):
        graph.neighbours("NOPE")


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"n_rounds": 9}, "n_rounds"), ({"self_weight": 2.0}, "self_weight")]
)
def test_message_passing_bounds_are_enforced(
    planted: dict[str, object], kwargs: dict, message: str
) -> None:
    with pytest.raises(AssetGraphError, match=message):
        temporal_message_passing(planted["wide"], window=120, **kwargs)


# ---------------------------------------------------------------------------
# #16 Scenario and decision laboratory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def design() -> ExperimentDesign:
    return ExperimentDesign(name="sprint-3-decision-lab", n_scenarios=3000, horizon_days=5, seed=11)


@pytest.fixture(scope="module")
def scenarios(
    posterior: HierarchicalPosterior, planted: dict[str, object], design: ExperimentDesign
) -> ScenarioSet:
    correlation = estimate_correlation(planted["train"], posterior.assets)
    return generate_scenarios(posterior, correlation, design)


def test_design_identity_changes_when_the_design_changes(design: ExperimentDesign) -> None:
    """A silently retuned experiment cannot masquerade as the frozen one."""
    assert len(design.identity) == 64
    assert (
        design.identity
        == ExperimentDesign(
            name="sprint-3-decision-lab", n_scenarios=3000, horizon_days=5, seed=11
        ).identity
    )
    retuned = ExperimentDesign(
        name="sprint-3-decision-lab", n_scenarios=3000, horizon_days=5, seed=11, cost_bps=0.0
    )
    assert retuned.identity != design.identity


def test_scenarios_preserve_cross_asset_dependence(
    scenarios: ScenarioSet, design: ExperimentDesign
) -> None:
    """Without this, 'coherent' would be an assertion rather than a measurement."""
    report = scenario_coherence_report(scenarios, design)
    assert report["max_correlation_error"] < 0.15
    assert report["copula"] == "gaussian"
    assert "tail dependence" in report["limitation"]


def test_scenarios_are_reproducible(
    posterior: HierarchicalPosterior, planted: dict[str, object], design: ExperimentDesign
) -> None:
    correlation = estimate_correlation(planted["train"], posterior.assets)
    first = generate_scenarios(posterior, correlation, design)
    second = generate_scenarios(posterior, correlation, design)
    np.testing.assert_array_equal(first.returns, second.returns)


def test_every_policy_respects_the_constraint_set(
    scenarios: ScenarioSet, design: ExperimentDesign
) -> None:
    """The bug this caught: clip-then-renormalize is not a projection."""
    frame = evaluate_policies(scenarios, design)
    assert set(frame["policy"]) == set(DEFAULT_POLICIES)
    assert (frame["max_weight"] <= design.max_weight + 1e-9).all()
    assert list(frame["policy"]) == sorted(frame["policy"]), "rows must not be ranked by metric"
    assert frame.attrs["design_identity"] == design.identity


def test_long_only_policies_never_short(scenarios: ScenarioSet, design: ExperimentDesign) -> None:
    for policy in DEFAULT_POLICIES.values():
        weights = policy(scenarios, design)
        assert (weights >= -1e-12).all()
        assert float(np.sum(np.abs(weights))) == pytest.approx(1.0, abs=1e-6)


def test_costs_reduce_net_return_by_exactly_the_charged_amount(
    scenarios: ScenarioSet, design: ExperimentDesign
) -> None:
    frame = evaluate_policies(scenarios, design)
    difference = frame["expected_return_gross"] - frame["expected_return_net"]
    np.testing.assert_allclose(difference.to_numpy(), frame["cost"].to_numpy(), atol=1e-15)


def test_an_infeasible_weight_cap_is_refused(scenarios: ScenarioSet) -> None:
    """A cap that cannot fund the book must raise, not silently rescale."""
    infeasible = ExperimentDesign(
        name="infeasible", n_scenarios=100, horizon_days=1, seed=0, max_weight=0.05
    )
    with pytest.raises(ScenarioLabError, match="cannot fund a book"):
        evaluate_policies(scenarios, infeasible)


def test_realized_evaluation_requires_matching_assets(
    scenarios: ScenarioSet, design: ExperimentDesign
) -> None:
    with pytest.raises(ScenarioLabError, match="one value per asset"):
        evaluate_policies(scenarios, design, realized=np.zeros(N_ASSETS + 3))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": " "}, "requires a name"),
        ({"n_scenarios": 0}, "n_scenarios"),
        ({"horizon_days": 0}, "horizon_days"),
        ({"seed": -1}, "seed"),
        ({"cost_bps": -1.0}, "cost_bps"),
        ({"max_weight": 0.0}, "max_weight"),
        ({"tail_quantile": 0.9}, "tail_quantile"),
    ],
)
def test_design_validation(kwargs: dict, message: str) -> None:
    fields = {"name": "x", "n_scenarios": 10, "horizon_days": 1, "seed": 0}
    fields.update(kwargs)
    with pytest.raises(ScenarioLabError, match=message):
        ExperimentDesign(**fields)  # type: ignore[arg-type]


def test_correlation_requires_the_posterior_assets(
    planted: dict[str, object], posterior: HierarchicalPosterior
) -> None:
    with pytest.raises(ScenarioLabError, match="missing assets"):
        estimate_correlation(planted["train"], (*posterior.assets, "ABSENT"))


def test_scenario_shape_is_validated(design: ExperimentDesign) -> None:
    with pytest.raises(ScenarioLabError, match="n_scenarios, n_assets"):
        ScenarioSet(
            assets=("A", "B"),
            returns=np.zeros(4),
            design_identity=design.identity,
            correlation=np.eye(2),
        )
    with pytest.raises(ScenarioLabError, match="finite"):
        ScenarioSet(
            assets=("A",),
            returns=np.full((2, 1), np.nan),
            design_identity=design.identity,
            correlation=np.eye(1),
        )
