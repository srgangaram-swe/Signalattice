"""Adapter from pipeline configuration and panel metadata to store contracts."""

from __future__ import annotations

from datetime import UTC
from typing import Any

import pandas as pd

from quant_platform.config import AppConfig
from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.features.quality import FeatureQualityPolicy
from quant_platform.features.registry import conventional_feature_registry, semantic_hash
from quant_platform.features.store import (
    DatasetLineage,
    FeatureMaterializationRequest,
    FeatureOutputContract,
)
from quant_platform.utils import hash_dataframe


def build_pipeline_materialization_request(
    config: AppConfig,
    panel: pd.DataFrame,
    metadata: dict[str, Any],
) -> FeatureMaterializationRequest:
    """Translate verified ingestion evidence into a feature-store request.

    The adapter intentionally allowlists source metadata.  Provider manifests
    may contain additional transport details, but feature lineage receives only
    non-secret identifiers, completeness flags, and license classification.
    """
    if panel.empty:
        raise ValueError("cannot create feature lineage from an empty panel")
    dates = pd.to_datetime(panel[DATE_COL], errors="raise")
    source_manifest = metadata.get("source_manifest")
    source_manifest = source_manifest if isinstance(source_manifest, dict) else {}
    point_limits = source_manifest.get("point_in_time_limits")
    point_limits = point_limits if isinstance(point_limits, dict) else {}
    data_sha256 = hash_dataframe(panel, length=64)
    configured_request = {
        "source": config.data.source,
        "tickers": config.data.tickers,
        "benchmark": config.data.benchmark,
        "start": config.data.start,
        "end": config.data.end,
        "price_field": config.data.price_field,
        "nasdaq_data_link": config.data.nasdaq_data_link.model_dump(mode="json"),
        "seed": config.project.seed,
    }
    request_sha256 = str(source_manifest.get("request_hash", ""))
    if len(request_sha256) != 64:
        request_sha256 = semantic_hash(configured_request)
    source_revision = str(
        source_manifest.get("snapshot_hash") or metadata.get("data_hash") or data_sha256
    )
    retrieved_raw = source_manifest.get("retrieved_at")
    if retrieved_raw is None:
        # Synthetic/public adapters without provider retrieval metadata use the
        # final observation boundary as deterministic evidence time.
        retrieved_at = pd.Timestamp(dates.max()).tz_localize(UTC)
    else:
        retrieved_at = pd.Timestamp(str(retrieved_raw))
        if retrieved_at.tzinfo is None:
            raise ValueError("source retrieved_at must be timezone-aware")
        retrieved_at = retrieved_at.tz_convert(UTC)
    requested = tuple(config.data.tickers)
    returned = tuple(sorted(panel[TICKER_COL].astype(str).str.upper().unique()))
    lineage = DatasetLineage(
        dataset_sha256=data_sha256,
        source=str(metadata.get("source") or config.data.source),
        source_revision=source_revision,
        request_sha256=request_sha256,
        schema_version=str(source_manifest.get("schema_version") or "canonical-panel-1.0.0"),
        retrieved_at=retrieved_at.to_pydatetime(),
        coverage_start=dates.min().date(),
        coverage_end=dates.max().date(),
        requested_tickers=requested,
        returned_tickers=returned,
        observations_redistributable=bool(
            source_manifest.get(
                "observations_redistributable",
                config.data.source == "synthetic",
            )
        ),
        historical_revisions_complete=bool(
            point_limits.get("historical_revisions_complete", False)
        ),
        universe_membership_point_in_time=bool(
            point_limits.get("universe_membership_point_in_time", False)
        ),
        corporate_actions_complete=bool(point_limits.get("corporate_actions_complete", False)),
    )
    store = config.feature_store
    quality = FeatureQualityPolicy(
        max_missing_fraction=store.max_missing_fraction,
        max_business_day_gap=store.max_business_day_gap,
        max_staleness_days=store.max_staleness_days,
        min_drift_samples=store.min_drift_samples,
        max_ks_statistic=store.max_ks_statistic,
        max_psi=store.max_psi,
        psi_bins=store.psi_bins,
    )
    return FeatureMaterializationRequest.create(
        lineage=lineage,
        registry=conventional_feature_registry(
            config.features,
            price_field=config.data.price_field,
        ),
        output_contract=FeatureOutputContract(
            benchmark=config.data.benchmark,
            price_field=config.data.price_field,
            forward_horizon=config.model.forward_horizon,
            target_columns=("target_forward_return", "target_direction"),
        ),
        application_start=dates.min().date(),
        application_end=dates.max().date(),
        expected_end=dates.max().date(),
        partition_by=store.partition_by,
        quality_policy=quality,
        evidence_time=retrieved_at.to_pydatetime(),
    )
