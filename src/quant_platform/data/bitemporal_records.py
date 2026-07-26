"""Bitemporal universe and corporate-action records for Signal Foundry bundles.

The records in this module are provider-neutral boundary types. They preserve
economic time independently from information-availability and ingestion time,
so downstream research can apply the same fail-closed as-of rule to prices,
universe membership, and corporate actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

UNIVERSE_COLUMNS: Final[tuple[str, ...]] = (
    "membership_id",
    "universe_id",
    "instrument_id",
    "ticker",
    "effective_at",
    "available_at",
    "observed_at",
    "provider_updated_at",
    "is_member",
    "reason",
    "source",
    "source_table",
)
CORPORATE_ACTION_COLUMNS: Final[tuple[str, ...]] = (
    "action_id",
    "instrument_id",
    "ticker",
    "action_type",
    "effective_at",
    "available_at",
    "observed_at",
    "provider_updated_at",
    "cash_amount",
    "split_ratio",
    "currency",
    "old_ticker",
    "new_ticker",
    "adjustment_state",
    "source",
    "source_table",
)
ACTION_TYPES: Final[frozenset[str]] = frozenset(
    {"cash_dividend", "delisting", "merger", "spinoff", "split", "symbol_change"}
)


class BitemporalRecordError(ValueError):
    """Raised when an auxiliary point-in-time record set is unsafe to consume."""


@dataclass(frozen=True)
class SignalFoundryBundleView:
    """Verified record families visible at one optional decision timestamp."""

    prices: pd.DataFrame
    universe: pd.DataFrame
    corporate_actions: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], family: str) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    unknown = sorted(set(frame.columns).difference(columns))
    if missing or unknown:
        raise BitemporalRecordError(
            f"{family} columns do not match the contract; missing={missing}, unknown={unknown}"
        )
    return frame.loc[:, list(columns)].copy()


def _strict_utc(frame: pd.DataFrame, columns: tuple[str, ...], family: str) -> None:
    """Normalize timezone-aware values while rejecting ambiguous naive timestamps."""
    for column in columns:
        parsed = pd.to_datetime(frame[column], errors="raise", utc=False)
        if column == "provider_updated_at" and parsed.isna().all():
            frame[column] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
            continue
        if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise BitemporalRecordError(
                f"{family}.{column} must contain explicit timezone-aware timestamps"
            )
        frame[column] = parsed.dt.tz_convert("UTC")


def _validate_temporal_order(frame: pd.DataFrame, family: str) -> None:
    if frame[["effective_at", "available_at", "observed_at"]].isna().any().any():
        raise BitemporalRecordError(
            f"{family} effective_at, available_at, and observed_at are required"
        )
    if (frame["observed_at"] < frame["available_at"]).any():
        raise BitemporalRecordError(f"{family} observed_at precedes available_at")
    provider_time = frame["provider_updated_at"]
    if (provider_time.notna() & provider_time.gt(frame["observed_at"])).any():
        raise BitemporalRecordError(f"{family} provider_updated_at follows observed_at")


def _validate_text(frame: pd.DataFrame, columns: tuple[str, ...], family: str) -> None:
    for column in columns:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise BitemporalRecordError(f"{family}.{column} contains an empty value")
        frame[column] = frame[column].astype(str)


def empty_universe_records() -> pd.DataFrame:
    """Return an empty frame with the exact universe-membership schema."""
    return pd.DataFrame(columns=list(UNIVERSE_COLUMNS))


def empty_corporate_action_records() -> pd.DataFrame:
    """Return an empty frame with the exact corporate-action schema."""
    return pd.DataFrame(columns=list(CORPORATE_ACTION_COLUMNS))


def coerce_universe_records(records: pd.DataFrame | None) -> pd.DataFrame:
    """Validate and canonicalize versioned point-in-time membership events."""
    if records is None or records.empty:
        return empty_universe_records()
    frame = _require_columns(records, UNIVERSE_COLUMNS, "universe")
    _strict_utc(
        frame,
        ("effective_at", "available_at", "observed_at", "provider_updated_at"),
        "universe",
    )
    _validate_temporal_order(frame, "universe")
    _validate_text(
        frame,
        (
            "membership_id",
            "universe_id",
            "instrument_id",
            "ticker",
            "reason",
            "source",
            "source_table",
        ),
        "universe",
    )
    if not frame["is_member"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise BitemporalRecordError("universe.is_member must be boolean")
    frame["is_member"] = frame["is_member"].astype(bool)
    if frame.duplicated(["membership_id", "available_at"]).any():
        raise BitemporalRecordError(
            "universe contains a duplicate (membership_id, available_at) revision"
        )
    return frame.sort_values(
        ["universe_id", "instrument_id", "effective_at", "available_at"], kind="stable"
    ).reset_index(drop=True)


def coerce_corporate_action_records(records: pd.DataFrame | None) -> pd.DataFrame:
    """Validate and canonicalize versioned corporate-action events."""
    if records is None or records.empty:
        return empty_corporate_action_records()
    frame = _require_columns(records, CORPORATE_ACTION_COLUMNS, "corporate_actions")
    _strict_utc(
        frame,
        ("effective_at", "available_at", "observed_at", "provider_updated_at"),
        "corporate_actions",
    )
    _validate_temporal_order(frame, "corporate_actions")
    _validate_text(
        frame,
        (
            "action_id",
            "instrument_id",
            "ticker",
            "action_type",
            "currency",
            "adjustment_state",
            "source",
            "source_table",
        ),
        "corporate_actions",
    )
    optional_text = ("old_ticker", "new_ticker")
    for column in optional_text:
        frame[column] = frame[column].fillna("").astype(str)
    if not frame["action_type"].isin(ACTION_TYPES).all():
        invalid = sorted(set(frame["action_type"]).difference(ACTION_TYPES))
        raise BitemporalRecordError(f"corporate_actions has unsupported action types: {invalid}")
    for column in ("cash_amount", "split_ratio"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].dropna() <= 0).any():
            raise BitemporalRecordError(f"corporate_actions.{column} must be positive when present")
    splits = frame["action_type"].eq("split")
    if frame.loc[splits, "split_ratio"].isna().any():
        raise BitemporalRecordError("split actions require split_ratio")
    dividends = frame["action_type"].eq("cash_dividend")
    if frame.loc[dividends, "cash_amount"].isna().any():
        raise BitemporalRecordError("cash-dividend actions require cash_amount")
    symbol_changes = frame["action_type"].eq("symbol_change")
    if frame.loc[symbol_changes, "new_ticker"].str.strip().eq("").any():
        raise BitemporalRecordError("symbol-change actions require new_ticker")
    if frame.duplicated(["action_id", "available_at"]).any():
        raise BitemporalRecordError(
            "corporate_actions contains a duplicate (action_id, available_at) revision"
        )
    return frame.sort_values(
        ["instrument_id", "effective_at", "available_at"], kind="stable"
    ).reset_index(drop=True)


def visible_revisions(
    frame: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    identity_column: str,
) -> pd.DataFrame:
    """Return the latest visible revision for each stable record identity."""
    if frame.empty:
        return frame.copy()
    visible = frame.loc[
        frame["effective_at"].le(as_of) & frame["available_at"].le(as_of)
    ].sort_values([identity_column, "available_at", "observed_at"], kind="stable")
    return visible.drop_duplicates(identity_column, keep="last").reset_index(drop=True)
