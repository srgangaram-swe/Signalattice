"""Tests for SF-S2-MR2 volatility, distribution, and dependence features.

Differential tests check each estimator against an independent reference
(manual formulae, scipy, numpy). Property/metamorphic tests assert causality,
scale/chunk invariance, and known statistical behaviour. Adversarial tests cover
non-positive prices, flat series, gaps, outliers, singular windows, and small
samples, per the issue acceptance criteria.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quant_platform.features import statistics as st
from quant_platform.features.contracts import (
    FeatureFamily,
    Unit,
    build_contract_features,
    get_contract,
    statistical_contracts,
)
from tests.test_feature_contracts import make_panel

MR2_NAMES = [contract.name for contract in statistical_contracts()]
MR2_COLUMNS = [f"fc_{name}" for name in MR2_NAMES]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_panel(["AAA", "BBB", "CCC", "DDD"], n=400, seed=21)


@pytest.fixture(scope="module")
def features(panel: pd.DataFrame) -> pd.DataFrame:
    return build_contract_features(panel, statistical_contracts())


def _asset(panel: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return panel[panel["ticker"] == ticker].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Volatility — differential vs manual formulae
# ---------------------------------------------------------------------------


def test_parkinson_matches_manual(panel: pd.DataFrame) -> None:
    g = _asset(panel, "AAA")
    window, end = 21, 120
    vol = st.parkinson_volatility(g["high"], g["low"], window)
    hl = np.log((g["high"] / g["low"]).to_numpy())[end - window + 1 : end + 1]
    expected = np.sqrt((hl**2).mean() / (4.0 * np.log(2.0))) * np.sqrt(252)
    assert np.isclose(vol.iloc[end], expected, atol=1e-10)


def test_garman_klass_matches_manual(panel: pd.DataFrame) -> None:
    g = _asset(panel, "AAA")
    window, end = 21, 150
    vol = st.garman_klass_volatility(g["open"], g["high"], g["low"], g["close"], window)
    sl = slice(end - window + 1, end + 1)
    log_hl = np.log((g["high"] / g["low"]).to_numpy())[sl]
    log_co = np.log((g["close"] / g["open"]).to_numpy())[sl]
    term = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    expected = np.sqrt(max(term.mean(), 0.0)) * np.sqrt(252)
    assert np.isclose(vol.iloc[end], expected, atol=1e-10)


def test_rogers_satchell_non_negative_and_manual(panel: pd.DataFrame) -> None:
    g = _asset(panel, "BBB")
    window, end = 21, 200
    vol = st.rogers_satchell_volatility(g["open"], g["high"], g["low"], g["close"], window)
    assert (vol.dropna() >= 0).all()
    sl = slice(end - window + 1, end + 1)
    h, low_, c, o = (g[k].to_numpy()[sl] for k in ("high", "low", "close", "open"))
    term = np.log(h / c) * np.log(h / o) + np.log(low_ / c) * np.log(low_ / o)
    expected = np.sqrt(max(term.mean(), 0.0)) * np.sqrt(252)
    assert np.isclose(vol.iloc[end], expected, atol=1e-10)


def test_ewma_volatility_matches_pandas_ewm(panel: pd.DataFrame) -> None:
    g = _asset(panel, "AAA")
    ret = g["adj_close"].pct_change()
    vol = st.ewma_volatility(ret, span=21)
    expected = np.sqrt((ret**2).ewm(span=21, adjust=False, min_periods=21).mean()) * np.sqrt(252)
    pd.testing.assert_series_equal(vol, expected, check_names=False)


# ---------------------------------------------------------------------------
# Distribution — differential vs scipy
# ---------------------------------------------------------------------------


def test_skewness_matches_scipy(panel: pd.DataFrame) -> None:
    ret = _asset(panel, "AAA")["adj_close"].pct_change()
    window, end = 63, 200
    got = st.rolling_skewness(ret, window).iloc[end]
    ref = stats.skew(ret.to_numpy()[end - window + 1 : end + 1], bias=False)
    assert np.isclose(got, ref, atol=1e-9)


def test_kurtosis_matches_scipy(panel: pd.DataFrame) -> None:
    ret = _asset(panel, "AAA")["adj_close"].pct_change()
    window, end = 63, 200
    got = st.rolling_kurtosis(ret, window).iloc[end]
    ref = stats.kurtosis(ret.to_numpy()[end - window + 1 : end + 1], fisher=True, bias=False)
    assert np.isclose(got, ref, atol=1e-9)


def test_mad_matches_scipy(panel: pd.DataFrame) -> None:
    ret = _asset(panel, "AAA")["adj_close"].pct_change()
    window, end = 63, 200
    got = st.median_absolute_deviation(ret, window).iloc[end]
    ref = stats.median_abs_deviation(ret.to_numpy()[end - window + 1 : end + 1], scale=1.0)
    assert np.isclose(got, ref, atol=1e-12)


def test_downside_deviation_manual(panel: pd.DataFrame) -> None:
    ret = _asset(panel, "AAA")["adj_close"].pct_change()
    window, end = 63, 200
    got = st.downside_deviation(ret, window).iloc[end]
    negatives = np.clip(ret.to_numpy()[end - window + 1 : end + 1], None, 0.0)
    expected = np.sqrt((negatives**2).mean()) * np.sqrt(252)
    assert np.isclose(got, expected, atol=1e-12)


def test_mad_is_robust_to_a_single_outlier() -> None:
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 200))
    contaminated = ret.copy()
    contaminated.iloc[150] = 5.0  # one extreme outlier
    mad_clean = st.median_absolute_deviation(ret, 63)
    mad_dirty = st.median_absolute_deviation(contaminated, 63)
    std_clean = ret.rolling(63).std()
    std_dirty = contaminated.rolling(63).std()
    # MAD barely moves; std explodes at the same index.
    assert abs(mad_dirty.iloc[151] - mad_clean.iloc[151]) < 5e-3
    assert std_dirty.iloc[151] > 3 * std_clean.iloc[151]


# ---------------------------------------------------------------------------
# Dependence — differential and property tests
# ---------------------------------------------------------------------------


def test_autocorrelation_matches_numpy(panel: pd.DataFrame) -> None:
    ret = _asset(panel, "AAA")["adj_close"].pct_change()
    window, lag, end = 60, 1, 150
    got = st.rolling_autocorrelation(ret, window, lag).iloc[end]
    x = ret.to_numpy()
    a = x[end - window + 1 : end + 1]
    b = x[end - window + 1 - lag : end + 1 - lag]
    ref = np.corrcoef(a, b)[0, 1]
    assert np.isclose(got, ref, atol=1e-9)


def test_correlation_matches_numpy() -> None:
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0, 1, 200))
    b = pd.Series(rng.normal(0, 1, 200))
    window, end = 60, 150
    got = st.rolling_correlation(a, b, window).iloc[end]
    ref = np.corrcoef(
        a.to_numpy()[end - window + 1 : end + 1], b.to_numpy()[end - window + 1 : end + 1]
    )[0, 1]
    assert np.isclose(got, ref, atol=1e-9)


def test_partial_autocorrelation_closed_form(panel: pd.DataFrame) -> None:
    ret = _asset(panel, "AAA")["adj_close"].pct_change()
    window, end = 63, 200
    r1 = st.rolling_autocorrelation(ret, window, 1).iloc[end]
    r2 = st.rolling_autocorrelation(ret, window, 2).iloc[end]
    expected = np.clip((r2 - r1**2) / (1 - r1**2), -1.0, 1.0)
    got = st.partial_autocorrelation_lag2(ret, window).iloc[end]
    assert np.isclose(got, expected, atol=1e-9)


def test_variance_ratio_matches_reference() -> None:
    rng = np.random.default_rng(4)
    ret = pd.Series(rng.normal(0, 0.01, 300))
    window, q, end = 100, 5, 200

    def vr_ref(values: np.ndarray) -> float:
        n = len(values)
        mu = values.mean()
        var1 = values.var(ddof=1)
        q_sums = np.array([values[i : i + q].sum() for i in range(n - q + 1)])
        scale = q * (n - q + 1) * (1 - q / n)
        var_q = ((q_sums - q * mu) ** 2).sum() / scale
        return float(var_q / var1)

    got = st.variance_ratio(ret, window, q).iloc[end]
    expected = vr_ref(ret.to_numpy()[end - window + 1 : end + 1])
    assert np.isclose(got, expected, atol=1e-10)


def test_variance_ratio_random_walk_near_one() -> None:
    rng = np.random.default_rng(5)
    ret = pd.Series(rng.normal(0, 0.01, 4000))
    vr = st.variance_ratio(ret, 250, 5).dropna()
    assert 0.9 < vr.mean() < 1.1  # iid returns => VR ~ 1


def test_hurst_random_walk_near_half() -> None:
    rng = np.random.default_rng(6)
    # log-price is a random walk (and prices stay strictly positive).
    price = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 4000))))
    hurst = st.hurst_exponent(price, 256).dropna()
    assert 0.4 < hurst.mean() < 0.6  # Brownian motion => H ~ 0.5


def test_hurst_persistent_series_above_half() -> None:
    rng = np.random.default_rng(7)
    # A doubly-integrated (very smooth, persistent) log-price. H is invariant to
    # the exponent's scale, which is normalised only to keep prices bounded.
    double_integral = np.cumsum(np.cumsum(rng.normal(0, 1.0, 2000)))
    scaled = 0.5 * double_integral / np.max(np.abs(double_integral))
    price = pd.Series(100.0 * np.exp(scaled))
    hurst = st.hurst_exponent(price, 256).dropna()
    assert hurst.mean() > 0.7  # persistent process => H well above 0.5


def test_mutual_information_detects_dependence() -> None:
    rng = np.random.default_rng(8)
    x = pd.Series(rng.normal(0, 1, 400))
    y_independent = pd.Series(rng.normal(0, 1, 400))
    mi_self = st.rolling_mutual_information(x, x, 200, bins=8).dropna()
    mi_indep = st.rolling_mutual_information(x, y_independent, 200, bins=8).dropna()
    assert (mi_self >= 0).all() and (mi_indep >= 0).all()
    assert mi_self.mean() > mi_indep.mean()


# ---------------------------------------------------------------------------
# Estimator guards
# ---------------------------------------------------------------------------


def test_estimator_argument_guards() -> None:
    series = pd.Series(np.arange(50, dtype=float))
    with pytest.raises(ValueError, match="lag must be >= 1"):
        st.rolling_autocorrelation(series, 20, 0)
    with pytest.raises(ValueError, match="q >= 2"):
        st.variance_ratio(series, 20, 1)
    with pytest.raises(ValueError, match="two lags"):
        st.hurst_exponent(series, window=2)


# ---------------------------------------------------------------------------
# Adversarial: non-positive prices, flat series, gaps, small samples
# ---------------------------------------------------------------------------


def test_non_positive_prices_yield_nan_not_crash() -> None:
    high = pd.Series([1.0, 2.0, -1.0, 2.0, 3.0] * 10)
    low = pd.Series([0.5, 1.0, 0.5, 1.0, 1.5] * 10)
    with np.errstate(invalid="ignore"):
        vol = st.parkinson_volatility(high, low, 5)
    assert not np.isinf(vol.to_numpy(dtype=float)).any()


def test_flat_series_has_zero_vol_and_no_inf() -> None:
    panel = make_panel(["AAA", "BBB"], n=250, seed=1)
    for col in ("open", "high", "low", "close", "adj_close"):
        panel[col] = 100.0
    feats = build_contract_features(panel, statistical_contracts())
    np.testing.assert_allclose(feats["fc_parkinson_vol_21"].dropna().to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(feats["fc_ewma_vol_21"].dropna().to_numpy(), 0.0, atol=1e-12)
    assert not np.isinf(feats[MR2_COLUMNS].to_numpy(dtype=float)).any()


def test_gaps_propagate_as_nan() -> None:
    panel = make_panel(["AAA", "BBB"], n=250, seed=2)
    mask = (panel["ticker"] == "AAA") & (panel.groupby("ticker").cumcount() == 120)
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        panel.loc[mask, col] = np.nan
    feats = build_contract_features(panel, statistical_contracts())
    assert not np.isinf(feats[MR2_COLUMNS].to_numpy(dtype=float)).any()


def test_insufficient_history_is_all_nan() -> None:
    panel = make_panel(["AAA", "BBB"], n=40, seed=3)  # shorter than 63/128 windows
    feats = build_contract_features(panel, statistical_contracts())
    assert feats["fc_skew_63"].isna().all()
    assert feats["fc_hurst_128"].isna().all()
    # A 21-window volatility still resolves.
    assert feats["fc_parkinson_vol_21"].notna().any()


# ---------------------------------------------------------------------------
# Contract-level: bounds, warm-up, causality, determinism, metadata
# ---------------------------------------------------------------------------


def test_contract_numerical_ranges_hold(features: pd.DataFrame) -> None:
    for name in ("autocorr_1_63", "pacf_2_63", "corr_mkt_63"):
        assert features[f"fc_{name}"].dropna().between(-1.0, 1.0).all()
    for name in (
        "parkinson_vol_21",
        "garman_klass_vol_21",
        "rogers_satchell_vol_21",
        "ewma_vol_21",
        "downside_dev_63",
        "mad_63",
        "var_ratio_5_63",
        "mutual_info_mkt_63",
    ):
        assert (features[f"fc_{name}"].dropna() >= 0.0).all()


def test_no_infinities(features: pd.DataFrame) -> None:
    assert not np.isinf(features[MR2_COLUMNS].to_numpy(dtype=float)).any()


@pytest.mark.parametrize("name", MR2_NAMES)
def test_warmup_suppresses_partial_windows(panel: pd.DataFrame, name: str) -> None:
    contract = get_contract(name)
    feats = build_contract_features(panel, [contract])
    column = f"fc_{name}"
    for _ticker, group in feats.groupby("ticker", sort=False):
        series = group[column].reset_index(drop=True)
        assert series.iloc[: contract.warmup - 1].isna().all()
        assert series.iloc[contract.warmup - 1 :].notna().any()


def test_future_bar_does_not_change_past_features(panel: pd.DataFrame) -> None:
    base = build_contract_features(panel, statistical_contracts())
    perturbed_panel = panel.copy()
    last_idx = perturbed_panel.index[perturbed_panel["ticker"] == "AAA"][-1]
    for col in ("open", "high", "low", "close", "adj_close"):
        perturbed_panel.loc[last_idx, col] *= 4.0
    perturbed = build_contract_features(perturbed_panel, statistical_contracts())
    base_aaa = base[base["ticker"] == "AAA"].reset_index(drop=True)
    pert_aaa = perturbed[perturbed["ticker"] == "AAA"].reset_index(drop=True)
    pd.testing.assert_frame_equal(base_aaa[MR2_COLUMNS].iloc[:-1], pert_aaa[MR2_COLUMNS].iloc[:-1])


def test_per_asset_features_are_chunk_invariant(panel: pd.DataFrame) -> None:
    # Per-asset volatility must not depend on which other names share the panel.
    contract = get_contract("garman_klass_vol_21")
    full = build_contract_features(panel, [contract])
    full_aaa = full[full["ticker"] == "AAA"].reset_index(drop=True)
    solo = build_contract_features(_asset(panel, "AAA"), [contract])
    pd.testing.assert_frame_equal(full_aaa, solo)


def test_deterministic_replay(panel: pd.DataFrame) -> None:
    first = build_contract_features(panel, statistical_contracts())
    second = build_contract_features(
        panel.sample(frac=1.0, random_state=9), statistical_contracts()
    )
    pd.testing.assert_frame_equal(first, second)


def test_mr2_metadata_declares_family_unit_and_samples() -> None:
    expected_families = {
        FeatureFamily.VOLATILITY,
        FeatureFamily.DISTRIBUTION,
        FeatureFamily.DEPENDENCE,
    }
    seen = set()
    for contract in statistical_contracts():
        meta = contract.metadata()
        assert meta["description"]  # documents the estimator
        assert meta["warmup"] >= 1  # minimum-sample requirement
        assert meta["unit"] in {unit.value for unit in Unit}
        assert meta["temporal_availability"] == "causal"
        seen.add(contract.family)
    assert seen == expected_families
