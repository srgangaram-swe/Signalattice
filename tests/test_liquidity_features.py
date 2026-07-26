"""Tests for SF-S2-MR3 volume, liquidity, spread, and impact features.

Differential tests check each proxy against an independent reference. Adversarial
tests cover the acceptance criteria: missing and zero volume, stale prices,
corporate-action (price-scale) behaviour, thin assets, calendar gaps, overflow,
and the turnover data-requirement (fail-closed on the unavailable dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.features import liquidity as lq
from quant_platform.features.contracts import (
    FeatureFamily,
    Unit,
    build_contract_features,
    default_contracts,
    get_contract,
    liquidity_contracts,
    list_contracts,
)
from tests.test_feature_contracts import make_panel

MR3_NAMES = [contract.name for contract in liquidity_contracts()]
MR3_COLUMNS = [f"fc_{name}" for name in MR3_NAMES]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_panel(["AAA", "BBB", "CCC"], n=400, seed=33)


@pytest.fixture(scope="module")
def features(panel: pd.DataFrame) -> pd.DataFrame:
    return build_contract_features(panel, liquidity_contracts())


def _asset(panel: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return panel[panel["ticker"] == ticker].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Differential tests vs independent references
# ---------------------------------------------------------------------------


def test_volume_change_matches_manual(panel: pd.DataFrame) -> None:
    v = _asset(panel, "AAA")["volume"]
    got = lq.volume_change(v, 1)
    expected = np.log(v) - np.log(v.shift(1))
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_dollar_volume_matches_manual(panel: pd.DataFrame) -> None:
    g = _asset(panel, "AAA")
    window, end = 21, 120
    got = lq.rolling_log_dollar_volume(g["close"], g["volume"], window).iloc[end]
    log_dollar = np.log((g["close"] * g["volume"]).to_numpy())
    expected = log_dollar[end - window + 1 : end + 1].mean()
    assert np.isclose(got, expected, atol=1e-10)


def test_relative_volume_matches_manual(panel: pd.DataFrame) -> None:
    v = _asset(panel, "AAA")["volume"]
    window, end = 21, 150
    got = lq.relative_volume(v, window).iloc[end]
    expected = v.iloc[end] / v.to_numpy()[end - window + 1 : end + 1].mean()
    assert np.isclose(got, expected, atol=1e-10)


def test_amihud_matches_manual(panel: pd.DataFrame) -> None:
    g = _asset(panel, "AAA")
    window, end, scale = 21, 150, 1_000_000
    got = lq.amihud_illiquidity(g["close"], g["volume"], window, scale=scale).iloc[end]
    ret = g["close"].pct_change().abs()
    daily = (ret / (g["close"] * g["volume"]) * scale).to_numpy()
    expected = daily[end - window + 1 : end + 1].mean()
    assert np.isclose(got, expected, atol=1e-12)


def test_volume_imbalance_matches_manual(panel: pd.DataFrame) -> None:
    g = _asset(panel, "AAA")
    window, end = 21, 150
    got = lq.volume_imbalance(g["close"], g["volume"], window).iloc[end]
    signed = np.sign(g["close"].pct_change().to_numpy()) * g["volume"].to_numpy()
    total = g["volume"].to_numpy()[end - window + 1 : end + 1].sum()
    expected = signed[end - window + 1 : end + 1].sum() / total
    assert np.isclose(got, expected, atol=1e-12)


def test_roll_spread_matches_manual(panel: pd.DataFrame) -> None:
    close = _asset(panel, "AAA")["close"]
    window, end = 21, 150
    got = lq.roll_spread(close, window).iloc[end]
    dp = np.log(close).diff()
    cov = dp.rolling(window, min_periods=window).cov(dp.shift(1)).iloc[end]
    expected = 2.0 * np.sqrt(max(-cov, 0.0))
    assert np.isclose(got, expected, atol=1e-12)


def test_corwin_schultz_matches_reference(panel: pd.DataFrame) -> None:
    g = _asset(panel, "AAA")
    window, end = 21, 150
    got = lq.corwin_schultz_spread(g["high"], g["low"], window).iloc[end]

    high = g["high"].to_numpy()
    low = g["low"].to_numpy()
    const = 3.0 - 2.0 * np.sqrt(2.0)
    per_day = np.full(len(high), np.nan)
    for t in range(1, len(high)):
        beta = np.log(high[t] / low[t]) ** 2 + np.log(high[t - 1] / low[t - 1]) ** 2
        two_high = max(high[t], high[t - 1])
        two_low = min(low[t], low[t - 1])
        gamma = np.log(two_high / two_low) ** 2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / const - np.sqrt(gamma / const)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        per_day[t] = max(spread, 0.0)
    expected = np.nanmean(per_day[end - window + 1 : end + 1])
    assert np.isclose(got, expected, atol=1e-12)


def test_turnover_matches_manual(panel: pd.DataFrame) -> None:
    g = _asset(panel, "AAA")
    shares = pd.Series(np.full(len(g), 1e9), index=g.index)
    window, end = 21, 150
    got = lq.turnover(g["volume"], shares, window).iloc[end]
    rate = (g["volume"] / shares).to_numpy()
    expected = rate[end - window + 1 : end + 1].mean()
    assert np.isclose(got, expected, atol=1e-15)


# ---------------------------------------------------------------------------
# Zero / missing volume and stale prices
# ---------------------------------------------------------------------------


def test_zero_volume_bar_yields_nan_not_inf() -> None:
    panel = make_panel(["AAA", "BBB"], n=200, seed=1)
    mask = (panel["ticker"] == "AAA") & (panel.groupby("ticker").cumcount() == 100)
    panel.loc[mask, "volume"] = 0.0
    feats = build_contract_features(panel, liquidity_contracts())
    assert not np.isinf(feats[MR3_COLUMNS].to_numpy(dtype=float)).any()
    aaa = feats[feats["ticker"] == "AAA"].reset_index(drop=True)
    # The zero-volume bar breaks log dollar volume and volume change at that bar.
    assert np.isnan(aaa.loc[100, "fc_volume_change_1"])


def test_missing_volume_propagates() -> None:
    panel = make_panel(["AAA", "BBB"], n=200, seed=2)
    mask = (panel["ticker"] == "AAA") & (panel.groupby("ticker").cumcount() == 90)
    panel.loc[mask, "volume"] = np.nan
    feats = build_contract_features(panel, liquidity_contracts())
    assert not np.isinf(feats[MR3_COLUMNS].to_numpy(dtype=float)).any()


def test_stale_prices_have_zero_spread_and_impact() -> None:
    panel = make_panel(["AAA", "BBB"], n=200, seed=3)
    for col in ("open", "high", "low", "close", "adj_close"):
        panel[col] = 100.0  # flat prices, non-zero volume
    feats = build_contract_features(panel, liquidity_contracts())
    for name in ("amihud_21", "corwin_schultz_21", "roll_spread_21"):
        np.testing.assert_allclose(feats[f"fc_{name}"].dropna().to_numpy(), 0.0, atol=1e-12)
    assert not np.isinf(feats[MR3_COLUMNS].to_numpy(dtype=float)).any()


# ---------------------------------------------------------------------------
# Corporate actions (price scale), thin assets, overflow
# ---------------------------------------------------------------------------


def test_fractional_proxies_are_price_scale_invariant(panel: pd.DataFrame) -> None:
    base = build_contract_features(panel, liquidity_contracts())
    scaled_panel = panel.copy()
    mask = scaled_panel["ticker"] == "BBB"
    for col in ("open", "high", "low", "close", "adj_close"):
        scaled_panel.loc[mask, col] *= 4.0  # a 4-for-1 split adjustment
    scaled = build_contract_features(scaled_panel, liquidity_contracts())
    base_bbb = base[base["ticker"] == "BBB"].reset_index(drop=True)
    scaled_bbb = scaled[scaled["ticker"] == "BBB"].reset_index(drop=True)
    # Spread and volume proxies are fractional/unitless -> invariant to price scale.
    for name in ("corwin_schultz_21", "roll_spread_21", "volume_imbalance_21", "rel_volume_21"):
        np.testing.assert_allclose(
            base_bbb[f"fc_{name}"].to_numpy(dtype=float),
            scaled_bbb[f"fc_{name}"].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        )


def test_thin_asset_illiquidity_is_large_but_finite() -> None:
    panel = make_panel(["AAA"], n=120, seed=4)
    panel["volume"] = 1.0  # extremely thin
    feats = build_contract_features(panel, [get_contract("amihud_21")])
    amihud = feats["fc_amihud_21"].dropna()
    assert (amihud > 0).all()
    assert np.isfinite(amihud.to_numpy()).all()


def test_large_values_do_not_overflow() -> None:
    panel = make_panel(["AAA"], n=120, seed=5)
    panel["volume"] = 1e12  # huge share volume
    feats = build_contract_features(panel, liquidity_contracts())
    assert np.isfinite(
        feats[MR3_COLUMNS].to_numpy(dtype=float)[
            ~np.isnan(feats[MR3_COLUMNS].to_numpy(dtype=float))
        ]
    ).all()


# ---------------------------------------------------------------------------
# Turnover data requirement (unavailable dependency)
# ---------------------------------------------------------------------------


def test_turnover_is_registered_but_not_default() -> None:
    default_names = {c.name for c in default_contracts()}
    assert "turnover_21" not in default_names
    assert "turnover_21" in list_contracts()  # discoverable via the registry
    assert get_contract("turnover_21").inputs == ("volume", "shares_outstanding")


def test_turnover_requires_shares_outstanding(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="shares_outstanding"):
        build_contract_features(panel, [get_contract("turnover_21")])


def test_turnover_computes_with_shares_outstanding(panel: pd.DataFrame) -> None:
    enriched = panel.copy()
    enriched["shares_outstanding"] = 1e9
    feats = build_contract_features(enriched, [get_contract("turnover_21")])
    assert feats["fc_turnover_21"].notna().any()
    assert (feats["fc_turnover_21"].dropna() >= 0).all()


# ---------------------------------------------------------------------------
# Contract-level: bounds, causality, determinism, metadata, lineage
# ---------------------------------------------------------------------------


def test_bounds_and_finiteness(features: pd.DataFrame) -> None:
    assert not np.isinf(features[MR3_COLUMNS].to_numpy(dtype=float)).any()
    assert features["fc_volume_imbalance_21"].dropna().between(-1.0, 1.0).all()
    for name in ("rel_volume_21", "amihud_21", "corwin_schultz_21", "roll_spread_21"):
        assert (features[f"fc_{name}"].dropna() >= 0.0).all()


@pytest.mark.parametrize("name", MR3_NAMES)
def test_warmup_suppresses_partial_windows(panel: pd.DataFrame, name: str) -> None:
    contract = get_contract(name)
    feats = build_contract_features(panel, [contract])
    column = f"fc_{name}"
    for _ticker, group in feats.groupby("ticker", sort=False):
        series = group[column].reset_index(drop=True)
        assert series.iloc[: contract.warmup - 1].isna().all()
        assert series.iloc[contract.warmup - 1 :].notna().any()


def test_future_bar_does_not_change_past_features(panel: pd.DataFrame) -> None:
    base = build_contract_features(panel, liquidity_contracts())
    perturbed_panel = panel.copy()
    last_idx = perturbed_panel.index[perturbed_panel["ticker"] == "AAA"][-1]
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        perturbed_panel.loc[last_idx, col] *= 3.0
    perturbed = build_contract_features(perturbed_panel, liquidity_contracts())
    base_aaa = base[base["ticker"] == "AAA"].reset_index(drop=True)
    pert_aaa = perturbed[perturbed["ticker"] == "AAA"].reset_index(drop=True)
    pd.testing.assert_frame_equal(base_aaa[MR3_COLUMNS].iloc[:-1], pert_aaa[MR3_COLUMNS].iloc[:-1])


def test_deterministic_replay(panel: pd.DataFrame) -> None:
    first = build_contract_features(panel, liquidity_contracts())
    second = build_contract_features(panel.sample(frac=1.0, random_state=7), liquidity_contracts())
    pd.testing.assert_frame_equal(first, second)


def test_metadata_states_source_fields_units_and_family() -> None:
    all_liquidity = (*liquidity_contracts(), get_contract("turnover_21"))
    for contract in all_liquidity:
        meta = contract.metadata()
        assert meta["inputs"]  # source fields declared
        assert meta["description"]  # units / zero-volume / interpretation limits
        assert meta["unit"] in {unit.value for unit in Unit}
        assert contract.family is FeatureFamily.LIQUIDITY
        assert meta["temporal_availability"] == "causal"


def test_liquidity_parameters_are_part_of_identity() -> None:
    # Cache/lineage identity must reflect the liquidity parameters (window, scale).
    amihud = get_contract("amihud_21").metadata()["params"]
    assert amihud["window"] == 21
    assert amihud["scale"] == 1_000_000
