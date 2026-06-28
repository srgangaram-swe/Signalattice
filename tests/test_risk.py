"""Tests for risk and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.risk import metrics as rm
from quant_platform.risk.analytics import correlation_matrix, exposure_summary, stress_test


def test_annualized_vol_scaling():
    daily = pd.Series(np.full(252, 0.0)).copy()
    rng = np.random.default_rng(0)
    daily = pd.Series(rng.normal(0, 0.01, 5000))
    ann = rm.annualized_volatility(daily)
    # ~ 0.01 * sqrt(252)
    assert np.isclose(ann, 0.01 * np.sqrt(252), rtol=0.1)


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.005, 2000))
    assert rm.sharpe_ratio(r) > 0


def test_sharpe_zero_vol_is_nan():
    r = pd.Series(np.full(100, 0.001))
    assert np.isnan(rm.sharpe_ratio(r))


def test_max_drawdown_known_path():
    # +100% then -50% -> drawdown of -50%
    returns = pd.Series([1.0, -0.5])
    assert np.isclose(rm.max_drawdown(returns), -0.5)


def test_drawdown_series_non_positive(daily_returns):
    dd = rm.drawdown_series(daily_returns)
    assert (dd <= 1e-12).all()


def test_var_cvar_positive_and_ordered(daily_returns):
    var = rm.value_at_risk(daily_returns, 0.95)
    cvar = rm.conditional_value_at_risk(daily_returns, 0.95)
    assert var >= 0
    assert cvar >= var  # expected shortfall is at least as large as VaR


def test_var_gaussian_vs_historical(daily_returns):
    g = rm.value_at_risk(daily_returns, 0.95, method="gaussian")
    h = rm.value_at_risk(daily_returns, 0.95, method="historical")
    assert g > 0 and h > 0


def test_beta_of_self_is_one(daily_returns):
    assert np.isclose(rm.beta(daily_returns, daily_returns), 1.0, atol=1e-9)


def test_sortino_geq_relationship():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0008, 0.01, 3000))
    assert np.isfinite(rm.sortino_ratio(r))


def test_performance_summary_keys(daily_returns):
    s = rm.performance_summary(daily_returns, benchmark_returns=daily_returns)
    for k in (
        "cagr",
        "ann_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "hit_rate",
        "beta",
    ):
        assert k in s


def test_correlation_matrix_symmetric():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 0.01, (300, 4)), columns=list("ABCD"))
    corr = correlation_matrix(df)
    assert np.allclose(corr.values, corr.values.T)
    assert np.allclose(np.diag(corr.values), 1.0)


def test_exposure_summary():
    weights = pd.DataFrame(
        {"A": [0.5, 0.3], "B": [-0.5, -0.3], "C": [0.0, 0.4]},
        index=pd.bdate_range("2020-01-01", periods=2),
    )
    exp = exposure_summary(weights)
    assert np.isclose(exp["gross_exposure"].iloc[0], 1.0)
    assert np.isclose(exp["net_exposure"].iloc[0], 0.0)
    assert exp["n_long"].iloc[1] == 2


def test_stress_test_scenarios(daily_returns):
    scenarios = {"equity_-10pct": -0.10, "vol_spike_2x": 2.0}
    df = stress_test(daily_returns, scenarios, beta_to_market=1.2)
    assert len(df) == 2
    eq = df[df["scenario"] == "equity_-10pct"]["estimated_pnl"].iloc[0]
    assert np.isclose(eq, 1.2 * -0.10)
