"""Leakage-aware walk-forward training for tabular and temporal forecasts.

Every reported prediction is produced by a model whose fit, probability
calibration, and early-stopping windows end before the outer test block. Panel
splits move entire dates together. The returned prediction frame is the only
model signal consumed by the backtester.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_platform.config import ModelConfig
from quant_platform.data.schema import DATE_COL, TICKER_COL
from quant_platform.features.pipeline import (
    TARGET_DIRECTION,
    TARGET_RETURN,
    feature_columns,
)
from quant_platform.logging_utils import get_logger
from quant_platform.models.diagnostics import (
    brier_decomposition,
    calibration_table,
    date_block_bootstrap_ci,
    prediction_decile_return_table,
    selective_prediction_frontier,
)
from quant_platform.models.ensemble import ChronologicalCalibratedEnsemble
from quant_platform.models.factory import build_estimator, supports_proba
from quant_platform.models.metrics import classification_metrics, regression_metrics
from quant_platform.models.splits import TimeSeriesSplitter

logger = get_logger(__name__)

SIGNAL_COL = "score"
FORWARD_RETURN_COL = "forward_return"


@dataclass
class TemporalModelBundle:
    """Persistable preprocessing and causal temporal estimator bundle."""

    scaler: Any
    estimator: Any
    sequence_length: int
    feature_names: list[str]
    effective_identity: str = "causal-panel-tcn"


@dataclass
class TrainResult:
    """Container for walk-forward predictions, diagnostics, and final model."""

    predictions: pd.DataFrame
    metrics: dict[str, float]
    fold_metrics: list[dict[str, Any]]
    feature_importances: pd.Series
    feature_names: list[str]
    model: Any
    task: str
    n_splits: int
    extra: dict[str, Any] = field(default_factory=dict)


def _select_features(features: pd.DataFrame, config: ModelConfig) -> list[str]:
    cols = feature_columns(features)
    blocked = set(config.feature_blocklist)
    selected = [column for column in cols if column not in blocked]
    if not selected:
        raise ValueError("No feature columns left after applying feature_blocklist")
    return selected


def _make_pipeline(config: ModelConfig, seed: int) -> tuple[Any, Any]:
    estimator = build_estimator(config, seed=seed)
    if config.standardize and config.type in {"logistic", "ridge"}:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([("scaler", StandardScaler()), ("model", estimator)]), estimator
    return estimator, estimator


def _build_ensemble(config: ModelConfig, *, seed: int, n_dates: int) -> Any:
    estimators: dict[str, Any] = {}
    for offset, model_type in enumerate(config.ensemble.candidates):
        candidate = config.model_copy(
            update={
                "type": model_type,
                "params": config.ensemble.candidate_params.get(model_type, {}),
            }
        )
        estimators[model_type], _ = _make_pipeline(candidate, seed + offset)

    minimum_fraction = config.ensemble.min_calibration_dates / max(n_dates, 1)
    calibration_fraction = min(
        0.45,
        max(config.ensemble.calibration_fraction, minimum_fraction),
    )
    return ChronologicalCalibratedEnsemble(
        estimators,
        calibration_fraction=calibration_fraction,
        calibration_method=config.ensemble.calibration_method,
    )


def _predict_scores(model: Any, X: np.ndarray, task: str) -> np.ndarray:
    """Return P(up) for classification or a point forecast for regression."""
    if task == "classification":
        if supports_proba(model):
            probabilities = model.predict_proba(X)
            classes = list(getattr(model, "classes_", [0.0, 1.0]))
            positive_index = classes.index(1.0) if 1.0 in classes else -1
            return np.asarray(probabilities[:, positive_index], dtype=float)
        if hasattr(model, "decision_function"):
            decision = np.asarray(model.decision_function(X), dtype=float)
            return 1.0 / (1.0 + np.exp(-decision))
        return np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _base_importances(estimator: Any, feature_names: list[str]) -> pd.Series:
    model = estimator
    named_steps = getattr(estimator, "named_steps", {})
    if "model" in named_steps:
        model = named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = np.abs(coefficients).ravel()
        if len(values) != len(feature_names):
            values = np.abs(coefficients).mean(axis=0)
    else:
        values = np.full(len(feature_names), np.nan)
    return pd.Series(values, index=feature_names, name="importance")


def _importances(estimator: Any, feature_names: list[str]) -> pd.Series:
    """Extract weighted candidate importance for ensembles and simple models."""
    if hasattr(estimator, "estimators_") and hasattr(estimator, "weights_"):
        combined = pd.Series(0.0, index=feature_names, name="importance")
        observed_weight = 0.0
        for name, candidate in estimator.estimators_.items():
            importance = _base_importances(candidate, feature_names)
            if importance.notna().any():
                # Coefficients and tree importances live on different scales.
                # Normalize each candidate before applying its ensemble weight
                # so the combined ranking is interpretable as relative mass.
                importance = importance.fillna(0.0)
                magnitude = float(importance.abs().sum())
                if magnitude <= 0.0:
                    continue
                importance /= magnitude
                weight = float(estimator.weights_[name])
                combined = combined.add(importance * weight, fill_value=0.0)
                observed_weight += weight
        if observed_weight == 0.0:
            combined[:] = np.nan
        else:
            combined /= observed_weight
        return combined.sort_values(ascending=False)
    return _base_importances(estimator, feature_names).sort_values(ascending=False)


def _prediction_block(
    features: pd.DataFrame,
    indices: np.ndarray,
    scores: np.ndarray,
    fold: int,
) -> pd.DataFrame:
    block = features.iloc[indices][[DATE_COL, TICKER_COL, TARGET_DIRECTION, TARGET_RETURN]].copy()
    block = block.rename(columns={TARGET_DIRECTION: "y_true", TARGET_RETURN: FORWARD_RETURN_COL})
    block[SIGNAL_COL] = scores
    block["fold"] = fold
    return block


def _classification_extra(
    predictions: pd.DataFrame,
    fold_importances: list[pd.Series],
    fold_weights: list[dict[str, Any]],
    final_model: Any,
) -> dict[str, Any]:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    probabilities = predictions[SIGNAL_COL].to_numpy(dtype=float)
    dates = pd.to_datetime(predictions[DATE_COL])
    correctness = ((probabilities >= 0.5).astype(int) == y_true).astype(float)
    extra: dict[str, Any] = {
        "calibration_table": calibration_table(y_true, probabilities, n_bins=10),
        "brier_decomposition": brier_decomposition(y_true, probabilities, n_bins=10),
        "selective_prediction_frontier": selective_prediction_frontier(y_true, probabilities),
        "prediction_deciles": prediction_decile_return_table(
            probabilities,
            predictions[FORWARD_RETURN_COL].to_numpy(dtype=float),
        ),
        "accuracy_block_bootstrap_ci": date_block_bootstrap_ci(
            correctness,
            dates,
            n_bootstrap=300,
            block_size=min(5, dates.nunique()),
            random_state=17,
        ),
        "effective_model": getattr(final_model, "effective_identity", type(final_model).__name__),
    }
    if fold_importances:
        importance_frame = pd.concat(fold_importances, axis=1).T
        importance_frame.index.name = "fold"
        extra["fold_feature_importances"] = importance_frame
    if fold_weights:
        extra["fold_ensemble_weights"] = pd.DataFrame(fold_weights)
    if hasattr(final_model, "weights"):
        extra["ensemble_weights"] = final_model.weights
        extra["candidate_calibration_log_loss"] = final_model.calibration_log_losses_

    candidate_columns = [
        str(column) for column in predictions.columns if str(column).startswith("candidate_")
    ]
    if candidate_columns:
        extra["candidate_metrics"] = {
            column.removeprefix("candidate_"): classification_metrics(
                y_true,
                (predictions[column].to_numpy() >= 0.5).astype(int),
                predictions[column].to_numpy(dtype=float),
            )
            for column in candidate_columns
        }
    return extra


def _walk_forward_tabular(
    features: pd.DataFrame,
    config: ModelConfig,
    feature_names: list[str],
    *,
    seed: int,
) -> TrainResult:
    target_column = TARGET_DIRECTION if config.task == "classification" else TARGET_RETURN
    X_all = features[feature_names].to_numpy(dtype=float)
    y_all = features[target_column].to_numpy(dtype=float)
    dates = pd.to_datetime(features[DATE_COL])
    splitter = TimeSeriesSplitter(
        scheme=config.cv.scheme,
        n_splits=config.cv.n_splits,
        test_size=config.cv.test_size,
        min_train_size=config.cv.min_train_size,
        embargo=config.cv.embargo,
    )

    oos_rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    fold_importances: list[pd.Series] = []
    fold_weights: list[dict[str, Any]] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(dates)):
        if config.type == "ensemble":
            model = _build_ensemble(
                config,
                seed=seed + fold,
                n_dates=dates.iloc[train_indices].nunique(),
            )
        else:
            model, _ = _make_pipeline(config, seed + fold)
        y_train = y_all[train_indices]
        if config.task == "classification" and len(np.unique(y_train)) < 2:
            logger.warning("Fold %d skipped: training target has a single class", fold)
            continue

        if config.type == "ensemble":
            model.fit(X_all[train_indices], y_train, dates=dates.iloc[train_indices].to_numpy())
        else:
            model.fit(X_all[train_indices], y_train)
        scores = _predict_scores(model, X_all[test_indices], config.task)
        block = _prediction_block(features, test_indices, scores, fold)
        if config.type == "ensemble":
            for name, probabilities in model.candidate_probabilities(X_all[test_indices]).items():
                block[f"candidate_{name}"] = probabilities
            fold_weights.append({"fold": fold, **model.weights})
        oos_rows.append(block)

        metrics = _fold_metrics(y_all[test_indices], scores, config.task, fold)
        fold_metrics.append(metrics)
        importance = _importances(model, feature_names)
        importance.name = fold
        fold_importances.append(importance)
        logger.info(
            "Fold %d: train=%d test=%d %s",
            fold,
            len(train_indices),
            len(test_indices),
            _fmt_metrics(metrics),
        )

    if not oos_rows:
        raise RuntimeError("No folds produced predictions; check CV configuration")
    predictions = (
        pd.concat(oos_rows, ignore_index=True)
        .sort_values([DATE_COL, TICKER_COL])
        .reset_index(drop=True)
    )
    aggregate_metrics = _aggregate_metrics(predictions, config.task)

    if config.type == "ensemble":
        final_model = _build_ensemble(
            config,
            seed=seed,
            n_dates=dates.nunique(),
        )
        final_model.fit(X_all, y_all, dates=dates.to_numpy())
    else:
        final_model, _ = _make_pipeline(config, seed)
        final_model.fit(X_all, y_all)
    final_importance = _importances(final_model, feature_names)
    extra: dict[str, Any] = {}
    if config.task == "classification":
        extra = _classification_extra(
            predictions,
            fold_importances,
            fold_weights,
            final_model,
        )

    return TrainResult(
        predictions=predictions,
        metrics=aggregate_metrics,
        fold_metrics=fold_metrics,
        feature_importances=final_importance,
        feature_names=feature_names,
        model=final_model,
        task=config.task,
        n_splits=len(fold_metrics),
        extra=extra,
    )


def _temporal_sequences(
    frame: pd.DataFrame,
    scaled_features: np.ndarray,
    target_column: str,
    sequence_length: int,
) -> Any:
    from quant_platform.models.torch_lstm import build_panel_sequences

    return build_panel_sequences(
        scaled_features,
        frame[DATE_COL].to_numpy(),
        frame[TICKER_COL].to_numpy(),
        targets=frame[target_column].to_numpy(dtype=float),
        sequence_length=sequence_length,
    )


def _fit_temporal_scaler(
    frame: pd.DataFrame,
    feature_names: list[str],
    validation_fraction: float,
) -> Any:
    """Fit preprocessing strictly before the temporal validation tail."""
    from sklearn.preprocessing import StandardScaler

    unique_dates = np.sort(pd.to_datetime(frame[DATE_COL]).unique())
    fit_frame = frame
    if validation_fraction > 0.0 and len(unique_dates) >= 2:
        n_validation_dates = min(
            len(unique_dates) - 1,
            max(1, int(np.ceil(len(unique_dates) * validation_fraction))),
        )
        validation_start = unique_dates[-n_validation_dates]
        fit_frame = frame.loc[pd.to_datetime(frame[DATE_COL]) < validation_start]
    if fit_frame.empty:
        raise ValueError("temporal preprocessing window is empty")
    return StandardScaler().fit(fit_frame[feature_names])


def _walk_forward_temporal(
    features: pd.DataFrame,
    config: ModelConfig,
    feature_names: list[str],
    *,
    seed: int,
) -> TrainResult:
    from quant_platform.models.torch_lstm import build_temporal_estimator

    target_column = TARGET_DIRECTION if config.task == "classification" else TARGET_RETURN
    dates = pd.to_datetime(features[DATE_COL])
    sequence_length = int(config.params.get("sequence_length", config.params.get("seq_len", 20)))
    validation_fraction = float(config.params.get("validation_fraction", 0.2))
    splitter = TimeSeriesSplitter(
        scheme=config.cv.scheme,
        n_splits=config.cv.n_splits,
        test_size=config.cv.test_size,
        min_train_size=config.cv.min_train_size,
        embargo=config.cv.embargo,
    )
    oos_rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for fold, (train_indices, test_indices) in enumerate(splitter.split(dates)):
        train_frame = features.iloc[train_indices].copy()
        test_frame = features.iloc[test_indices].copy()
        combined = pd.concat([train_frame, test_frame], ignore_index=True)
        scaler = _fit_temporal_scaler(
            train_frame,
            feature_names,
            validation_fraction,
        )
        train_batch = _temporal_sequences(
            train_frame,
            scaler.transform(train_frame[feature_names]),
            target_column,
            sequence_length,
        )
        combined_batch = _temporal_sequences(
            combined,
            scaler.transform(combined[feature_names]),
            target_column,
            sequence_length,
        )
        model = build_temporal_estimator(config, seed=seed + fold)
        model.fit(
            train_batch.X,
            train_batch.y,
            sample_dates=train_batch.metadata.dates,
        )
        is_test = combined_batch.metadata.row_indices >= len(train_frame)
        test_sequences = combined_batch.X[is_test]
        scores = (
            model.predict_proba(test_sequences)[:, 1]
            if config.task == "classification"
            else model.predict(test_sequences)
        )
        source_rows = combined_batch.metadata.row_indices[is_test]
        block = combined.iloc[source_rows][
            [DATE_COL, TICKER_COL, TARGET_DIRECTION, TARGET_RETURN]
        ].copy()
        block = block.rename(
            columns={TARGET_DIRECTION: "y_true", TARGET_RETURN: FORWARD_RETURN_COL}
        )
        if config.task == "regression":
            block["y_true"] = combined.iloc[source_rows][TARGET_RETURN].to_numpy()
        block[SIGNAL_COL] = scores
        block["fold"] = fold
        oos_rows.append(block)
        metrics = _fold_metrics(block["y_true"], scores, config.task, fold)
        fold_metrics.append(metrics)

    if not oos_rows:
        raise RuntimeError("No temporal folds produced predictions; check CV configuration")
    predictions = (
        pd.concat(oos_rows, ignore_index=True)
        .sort_values([DATE_COL, TICKER_COL])
        .reset_index(drop=True)
    )
    aggregate_metrics = _aggregate_metrics(predictions, config.task)

    scaler = _fit_temporal_scaler(
        features,
        feature_names,
        validation_fraction,
    )
    final_batch = _temporal_sequences(
        features,
        scaler.transform(features[feature_names]),
        target_column,
        sequence_length,
    )
    final_estimator = build_temporal_estimator(config, seed=seed)
    final_estimator.fit(
        final_batch.X,
        final_batch.y,
        sample_dates=final_batch.metadata.dates,
    )
    attribution_sample = final_batch.X[-min(256, len(final_batch.X)) :]
    attributions = final_estimator.input_attributions(attribution_sample)
    importance_values = np.mean(np.abs(attributions), axis=(0, 1))
    final_importance = pd.Series(
        importance_values,
        index=feature_names,
        name="importance",
    ).sort_values(ascending=False)
    bundle = TemporalModelBundle(
        scaler=scaler,
        estimator=final_estimator,
        sequence_length=sequence_length,
        feature_names=feature_names,
    )
    extra: dict[str, Any] = {
        "effective_model": bundle.effective_identity,
        "training_history": final_estimator.history_,
        "temporal_sequence_length": sequence_length,
    }
    if config.task == "classification":
        extra.update(_classification_extra(predictions, [], [], bundle))

    return TrainResult(
        predictions=predictions,
        metrics=aggregate_metrics,
        fold_metrics=fold_metrics,
        feature_importances=final_importance,
        feature_names=feature_names,
        model=bundle,
        task=config.task,
        n_splits=len(fold_metrics),
        extra=extra,
    )


def walk_forward_train(
    features: pd.DataFrame,
    config: ModelConfig,
    *,
    seed: int = 42,
) -> TrainResult:
    """Train fresh models per outer fold and return strictly OOS forecasts."""
    ordered = features.sort_values([DATE_COL, TICKER_COL]).reset_index(drop=True)
    selected_features = _select_features(ordered, config)
    target_column = TARGET_DIRECTION if config.task == "classification" else TARGET_RETURN
    if target_column not in ordered:
        raise ValueError(f"Target column '{target_column}' missing from feature frame")

    if config.type in {"lstm", "tcn"}:
        result = _walk_forward_temporal(
            ordered,
            config,
            selected_features,
            seed=seed,
        )
    else:
        result = _walk_forward_tabular(
            ordered,
            config,
            selected_features,
            seed=seed,
        )
    logger.info(
        "Walk-forward complete: %d folds, OOS %s",
        result.n_splits,
        _fmt_metrics(result.metrics),
    )
    return result


def _fold_metrics(y_true: Any, scores: Any, task: str, fold: int) -> dict[str, Any]:
    if task == "classification":
        probabilities = np.asarray(scores, dtype=float)
        metrics = classification_metrics(
            np.asarray(y_true),
            (probabilities >= 0.5).astype(int),
            probabilities,
        )
    else:
        metrics = regression_metrics(np.asarray(y_true), np.asarray(scores))
    return {"fold": fold, **metrics}


def _aggregate_metrics(predictions: pd.DataFrame, task: str) -> dict[str, float]:
    y_true = predictions["y_true"].to_numpy()
    scores = predictions[SIGNAL_COL].to_numpy()
    if task == "classification":
        return classification_metrics(y_true, (scores >= 0.5).astype(int), scores)
    return regression_metrics(y_true, scores)


def _fmt_metrics(metrics: dict[str, Any]) -> str:
    return " ".join(
        f"{key}={value:.3f}"
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and key != "fold"
    )


def save_model(result: TrainResult, path: str | Path) -> Path:
    """Persist the final model and auditable inference contract via joblib."""
    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "feature_names": result.feature_names,
            "task": result.task,
            "effective_model": result.extra.get("effective_model", type(result.model).__name__),
        },
        destination,
    )
    logger.info("Saved model to %s", destination)
    return destination


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a previously persisted model bundle."""
    import joblib

    loaded: dict[str, Any] = joblib.load(Path(path))
    return loaded
