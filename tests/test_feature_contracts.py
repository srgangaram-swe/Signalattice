"""Tests for the versioned conventional feature contracts (SF-S2-MR1).

Coverage spans metadata completeness, registry integrity, strict causality,
warm-up suppression, differential checks against independent references, and the
edge cases the acceptance criteria call out: constant series, gaps, splits
(scale invariance), irregular calendars, cross-sectional ties, insufficient
history, universe-membership changes, and fail-closed input validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.features import technical as ta
from quant_platform.features.contracts import (
    CONTRACT_REGISTRY,
    NumericalRange,
    TemporalAvailability,
    Unit,
    build_contract_features,
    contract_metadata_frame,
    default_contracts,
    get_contract,
    list_contracts,
    validate_contract_panel,
)

FEATURE_COLUMNS = [f"fc_{contract.name}" for contract in default_contracts()]


def make_panel(
    tickers: list[str],
    n: int = 320,
    seed: int = 0,
    *,
    freq: str = "B",
) -> pd.DataFrame:
    """Build a deterministic multi-ticker OHLCV panel (adj_close == close)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq=freq)
    frames = []
    for ticker in tickers:
        steps = rng.normal(0.0, 0.01, n)
        close = 100.0 * np.exp(np.cumsum(steps))
        high = close * (1.0 + np.abs(rng.normal(0.0, 0.003, n)))
        low = close * (1.0 - np.abs(rng.normal(0.0, 0.003, n)))
        open_ = close * (1.0 + rng.normal(0.0, 0.002, n))
        volume = rng.uniform(1e6, 5e6, n)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_panel(["AAA", "BBB", "CCC", "DDD"], n=320, seed=11)


@pytest.fixture(scope="module")
def features(panel: pd.DataFrame) -> pd.DataFrame:
    return build_contract_features(panel)


# ---------------------------------------------------------------------------
# Metadata & registry
# ---------------------------------------------------------------------------


def test_registry_names_unique_and_ordered() -> None:
    names = list_contracts()
    assert len(names) == len(set(names))
    assert names == tuple(CONTRACT_REGISTRY.keys())
    assert get_contract(names[0]) is CONTRACT_REGISTRY[names[0]]


def test_get_contract_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_contract("does_not_exist")


def test_every_contract_declares_full_metadata() -> None:
    for contract in default_contracts():
        meta = contract.metadata()
        # The acceptance criteria: units, lookback, warm-up, frequency,
        # missing-data policy, temporal availability, numerical behaviour.
        assert meta["unit"] in {unit.value for unit in Unit}
        assert meta["lookback"] >= 1
        assert meta["warmup"] >= 1
        assert meta["frequency"] == "daily"
        assert meta["missing_data_policy"] == "propagate_nan"
        assert meta["temporal_availability"] == TemporalAvailability.CAUSAL.value
        assert meta["inputs"], "a contract must declare its input columns"
        assert isinstance(meta["params"], dict)


def test_metadata_frame_shape() -> None:
    frame = contract_metadata_frame()
    assert len(frame) == len(default_contracts())
    assert {"name", "unit", "lookback", "warmup", "temporal_availability"} <= set(frame.columns)


def test_numerical_range_contains() -> None:
    rng = NumericalRange(lower=0.0, upper=1.0)
    assert rng.contains(pd.Series([0.0, 0.5, 1.0, np.nan]))
    assert not rng.contains(pd.Series([0.0, 1.5]))
    assert not rng.contains(pd.Series([-0.1, 0.4]))


# ---------------------------------------------------------------------------
# Shape, finiteness, bounds
# ---------------------------------------------------------------------------


def test_build_produces_all_contract_columns(features: pd.DataFrame) -> None:
    assert list(features.columns[:2]) == ["date", "ticker"]
    assert set(FEATURE_COLUMNS) <= set(features.columns)
    assert len(features.columns) == len(FEATURE_COLUMNS) + 2


def test_defined_values_are_finite(features: pd.DataFrame) -> None:
    values = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    assert not np.isinf(values).any()
    defined = values[~np.isnan(values)]
    assert np.isfinite(defined).all()


def test_rank_is_bounded_unit_interval(features: pd.DataFrame) -> None:
    rank = features["fc_cs_rank_mom_63"].dropna()
    assert rank.between(0.0, 1.0).all()
    contract = get_contract("cs_rank_mom_63")
    assert contract.numerical_range == NumericalRange(0.0, 1.0)
    assert contract.numerical_range.contains(rank)


# ---------------------------------------------------------------------------
# Warm-up (full-window suppression)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [c.name for c in default_contracts()])
def test_warmup_suppresses_partial_windows(panel: pd.DataFrame, name: str) -> None:
    contract = get_contract(name)
    features = build_contract_features(panel, [contract])
    column = f"fc_{name}"
    for _ticker, group in features.groupby("ticker", sort=False):
        series = group[column].reset_index(drop=True)
        # Everything strictly before the warm-up boundary must be NaN.
        assert series.iloc[: contract.warmup - 1].isna().all()
        # A full-length panel yields at least one defined value after warm-up.
        assert series.iloc[contract.warmup - 1 :].notna().any()


# ---------------------------------------------------------------------------
# Causality (metamorphic): a future bar cannot change a past feature
# ---------------------------------------------------------------------------


def test_future_bar_does_not_change_past_features(panel: pd.DataFrame) -> None:
    base = build_contract_features(panel)
    perturbed_panel = panel.copy()
    # Corrupt the most recent bar of one ticker by a large multiplicative shock.
    mask = perturbed_panel["ticker"] == "AAA"
    last_idx = perturbed_panel.index[mask][-1]
    for col in ("open", "high", "low", "close", "adj_close"):
        perturbed_panel.loc[last_idx, col] *= 5.0
    perturbed = build_contract_features(perturbed_panel)

    base_aaa = base[base["ticker"] == "AAA"].reset_index(drop=True)
    pert_aaa = perturbed[perturbed["ticker"] == "AAA"].reset_index(drop=True)
    # Every date except the corrupted final bar must be identical.
    pd.testing.assert_frame_equal(
        base_aaa[FEATURE_COLUMNS].iloc[:-1],
        pert_aaa[FEATURE_COLUMNS].iloc[:-1],
    )


# ---------------------------------------------------------------------------
# Scale invariance (corporate-action / split robustness)
# ---------------------------------------------------------------------------


def test_features_are_invariant_to_price_scaling(panel: pd.DataFrame) -> None:
    base = build_contract_features(panel)
    scaled_panel = panel.copy()
    mask = scaled_panel["ticker"] == "BBB"
    for col in ("open", "high", "low", "close", "adj_close"):
        scaled_panel.loc[mask, col] *= 7.5  # a clean 7.5-for-1 split adjustment
    scaled = build_contract_features(scaled_panel)

    base_bbb = base[base["ticker"] == "BBB"].reset_index(drop=True)
    scaled_bbb = scaled[scaled["ticker"] == "BBB"].reset_index(drop=True)
    np.testing.assert_allclose(
        base_bbb[FEATURE_COLUMNS].to_numpy(dtype=float),
        scaled_bbb[FEATURE_COLUMNS].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-12,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# Differential tests against independent references
# ---------------------------------------------------------------------------


def test_regression_residual_matches_lstsq_reference() -> None:
    rng = np.random.default_rng(1)
    price = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    window = 40
    residual = ta.rolling_regression_residual(price, window)
    log_price = np.log(price.to_numpy())
    design = np.column_stack((np.ones(window), np.arange(window, dtype=float)))
    for end in (window, 100, 199):
        y = log_price[end - window + 1 : end + 1]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        expected = y[-1] - (design @ coef)[-1]
        assert np.isclose(residual.iloc[end], expected, atol=1e-9)


def test_regression_residual_requires_min_window() -> None:
    with pytest.raises(ValueError, match="window >= 3"):
        ta.rolling_regression_residual(pd.Series([1.0, 2.0, 3.0]), window=2)


def test_vwap_deviation_matches_manual_reference() -> None:
    rng = np.random.default_rng(2)
    n, window = 60, 10
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)))
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series(rng.uniform(1e5, 2e5, n))
    dev = ta.vwap_deviation(high, low, close, volume, window)
    typical = (high + low + close) / 3.0
    end = 30
    num = (typical.iloc[end - window + 1 : end + 1] * volume.iloc[end - window + 1 : end + 1]).sum()
    den = volume.iloc[end - window + 1 : end + 1].sum()
    expected = close.iloc[end] / (num / den) - 1.0
    assert np.isclose(dev.iloc[end], expected, atol=1e-12)


def test_vwap_zero_volume_window_is_nan() -> None:
    close = pd.Series(np.full(30, 100.0))
    volume = pd.Series(np.zeros(30))
    dev = ta.vwap_deviation(close, close, close, volume, 10)
    assert dev.dropna().empty


# ---------------------------------------------------------------------------
# Constant series and gaps
# ---------------------------------------------------------------------------


def test_constant_series_behaviour() -> None:
    panel = make_panel(["AAA", "BBB"], n=300, seed=5)
    for col in ("open", "high", "low", "close", "adj_close"):
        panel[col] = 100.0  # flat price
    features = build_contract_features(panel)
    # Returns / momentum / roc of a flat series are exactly zero (post warm-up).
    for name in ("ret_1d", "ret_5d", "roc_10", "mom_63", "ma_dist_20", "vwap_dev_21"):
        defined = features[f"fc_{name}"].dropna()
        np.testing.assert_allclose(defined.to_numpy(), 0.0, atol=1e-12)
    # z-score and residual have zero dispersion -> undefined (NaN), never inf.
    assert features["fc_zscore_21"].dropna().eq(0.0).all() or features["fc_zscore_21"].isna().all()
    assert not np.isinf(features["fc_resid_63"].to_numpy(dtype=float)).any()


def test_missing_values_propagate_as_nan() -> None:
    panel = make_panel(["AAA", "BBB"], n=300, seed=6)
    mask = (panel["ticker"] == "AAA") & (panel.groupby("ticker").cumcount() == 150)
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        panel.loc[mask, col] = np.nan
    features = build_contract_features(panel)
    aaa = features[features["ticker"] == "AAA"].reset_index(drop=True)
    # The one-bar return straddling the gap must be NaN (not imputed).
    assert np.isnan(aaa.loc[151, "fc_ret_1d"])
    # No spurious infinities anywhere.
    assert not np.isinf(features[FEATURE_COLUMNS].to_numpy(dtype=float)).any()


# ---------------------------------------------------------------------------
# Irregular calendars and cross-sectional ties / membership
# ---------------------------------------------------------------------------


def test_irregular_calendar_is_positional_and_stable() -> None:
    # Daily (includes weekends) vs business-day calendars: features are bar-based,
    # so an identical return path yields identical features regardless of spacing.
    rng = np.random.default_rng(7)
    steps = rng.normal(0, 0.01, 260)
    close = 100.0 * np.exp(np.cumsum(steps))

    def panel_with_freq(freq: str) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=260, freq=freq)
        return pd.DataFrame(
            {
                "date": dates,
                "ticker": "AAA",
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "adj_close": close,
                "volume": 1e6,
            }
        )

    business = build_contract_features(panel_with_freq("B"))
    daily = build_contract_features(panel_with_freq("D"))
    np.testing.assert_allclose(
        business[FEATURE_COLUMNS].to_numpy(dtype=float),
        daily[FEATURE_COLUMNS].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_cross_sectional_rank_handles_ties() -> None:
    # Three tickers on an identical path -> tied momentum -> average rank 0.5.
    rng = np.random.default_rng(8)
    steps = rng.normal(0, 0.01, 120)
    close = 100.0 * np.exp(np.cumsum(steps))
    frames = []
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    for ticker in ("AAA", "BBB", "CCC"):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "adj_close": close,
                    "volume": 1e6,
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    features = build_contract_features(panel)
    ranks = features["fc_cs_rank_mom_63"].dropna()
    # Identical inputs must receive an identical (tie-averaged) rank. pandas
    # percentile rank of n tied values is the mean rank / n = (n + 1) / (2n);
    # for n == 3 that is 2/3. The invariant that matters is determinism + ties
    # collapsing to one shared value.
    assert ranks.nunique() == 1
    np.testing.assert_allclose(ranks.to_numpy(), 2.0 / 3.0, atol=1e-12)


def test_universe_membership_changes_do_not_crash() -> None:
    # BBB is delisted halfway; CCC lists late. Cross-sectional features must be
    # computed only over names present on each date.
    panel = make_panel(["AAA", "BBB", "CCC"], n=200, seed=9).reset_index(drop=True)
    order = panel.groupby("ticker").cumcount()
    delisted = (panel["ticker"] == "BBB") & (order >= 100)
    late_listed = (panel["ticker"] == "CCC") & (order < 100)
    panel = panel[~(delisted | late_listed)].reset_index(drop=True)
    features = build_contract_features(panel)
    ranks = features["fc_cs_rank_mom_63"].dropna()
    assert ranks.between(0.0, 1.0).all()
    # Every produced row maps back to a surviving observation.
    assert len(features) == len(panel)


# ---------------------------------------------------------------------------
# Insufficient history
# ---------------------------------------------------------------------------


def test_insufficient_history_yields_nan_not_error() -> None:
    panel = make_panel(["AAA", "BBB"], n=30, seed=10)  # far shorter than 252
    features = build_contract_features(panel)
    # Long-horizon contracts have no full window -> entirely NaN, no exception.
    assert features["fc_mom_252"].isna().all()
    assert features["fc_resid_63"].isna().all()
    # Short-horizon contracts still produce values.
    assert features["fc_ret_1d"].notna().any()


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------


def test_validate_rejects_missing_columns() -> None:
    panel = make_panel(["AAA"], n=50).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        build_contract_features(panel)


def test_validate_rejects_inf_inputs() -> None:
    panel = make_panel(["AAA"], n=50)
    panel.loc[10, "adj_close"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        build_contract_features(panel)


def test_validate_rejects_duplicate_keys() -> None:
    panel = make_panel(["AAA"], n=50)
    panel = pd.concat([panel, panel.iloc[[5]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_contract_features(panel, [get_contract("ret_1d")])


def test_validate_rejects_empty_panel() -> None:
    empty = make_panel(["AAA"], n=50).iloc[0:0]
    with pytest.raises(ValueError, match="empty panel"):
        build_contract_features(empty)


def test_build_requires_at_least_one_contract() -> None:
    panel = make_panel(["AAA"], n=50)
    with pytest.raises(ValueError, match="at least one"):
        build_contract_features(panel, [])


def test_contract_compute_rejects_missing_input_column() -> None:
    panel = validate_contract_panel(make_panel(["AAA"], n=40), ["adj_close"])
    contract = get_contract("vwap_dev_21")
    with pytest.raises(ValueError, match="requires columns"):
        contract.compute(panel.drop(columns=["volume"]))


def test_validate_contract_panel_sorts_and_resets_index() -> None:
    panel = make_panel(["BBB", "AAA"], n=20, seed=3)
    shuffled = panel.sample(frac=1.0, random_state=0)
    validated = validate_contract_panel(shuffled, ["adj_close"])
    assert list(validated.index) == list(range(len(validated)))
    assert validated["ticker"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_replay(panel: pd.DataFrame) -> None:
    first = build_contract_features(panel)
    second = build_contract_features(panel.sample(frac=1.0, random_state=1))
    pd.testing.assert_frame_equal(first, second)
