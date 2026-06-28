"""Model factory: build an sklearn-compatible estimator from config.

Gradient-boosting backends (XGBoost / LightGBM) are optional; if they are not
installed the factory falls back to scikit-learn's :class:`GradientBoosting*`
with a clear warning, so the platform always runs.
"""

from __future__ import annotations

from typing import Any

from quant_platform.config import ModelConfig
from quant_platform.logging_utils import get_logger

logger = get_logger(__name__)


def _has_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


# Sensible, conservative defaults per model (overridable via config.params).
_DEFAULTS: dict[str, dict[str, Any]] = {
    "logistic": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"},
    "ridge": {"alpha": 1.0},
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 6,
        "min_samples_leaf": 50,
        "max_features": "sqrt",
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 400,
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "verbose": -1,
    },
}


def build_estimator(config: ModelConfig, *, seed: int = 42):
    """Construct an unfitted estimator for the configured ``task`` and ``type``.

    Parameters
    ----------
    config:
        Model configuration (``task``, ``type``, ``params``...).
    seed:
        Random seed forwarded to estimators that support it.

    Returns
    -------
    A scikit-learn-compatible estimator implementing ``fit``/``predict`` (and
    ``predict_proba`` for classifiers).
    """
    task = config.task
    mtype = config.type
    params = {**_DEFAULTS.get(mtype, {}), **dict(config.params)}

    if mtype == "logistic":
        from sklearn.linear_model import LogisticRegression

        if task != "classification":
            raise ValueError("logistic regression requires task='classification'")
        return LogisticRegression(random_state=seed, **params)

    if mtype == "ridge":
        if task == "classification":
            from sklearn.linear_model import RidgeClassifier

            return RidgeClassifier(random_state=seed, **params)
        from sklearn.linear_model import Ridge

        return Ridge(random_state=seed, **params)

    if mtype == "random_forest":
        if task == "classification":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(random_state=seed, **params)
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=seed, **params)

    if mtype == "gradient_boosting":
        if task == "classification":
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(random_state=seed, **params)
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(random_state=seed, **params)

    if mtype == "xgboost":
        if _has_module("xgboost"):
            import xgboost as xgb

            if task == "classification":
                return xgb.XGBClassifier(random_state=seed, eval_metric="logloss", **params)
            return xgb.XGBRegressor(random_state=seed, **params)
        logger.warning("xgboost not installed; falling back to sklearn GradientBoosting")
        return build_estimator(config.model_copy(update={"type": "gradient_boosting"}), seed=seed)

    if mtype == "lightgbm":
        if _has_module("lightgbm"):
            import lightgbm as lgb

            if task == "classification":
                return lgb.LGBMClassifier(random_state=seed, **params)
            return lgb.LGBMRegressor(random_state=seed, **params)
        logger.warning("lightgbm not installed; falling back to sklearn GradientBoosting")
        return build_estimator(config.model_copy(update={"type": "gradient_boosting"}), seed=seed)

    if mtype == "lstm":
        from quant_platform.models.torch_lstm import build_lstm_estimator

        return build_lstm_estimator(config, seed=seed)

    raise ValueError(f"Unknown model type '{mtype}'")


def supports_proba(estimator) -> bool:
    """Whether an estimator exposes calibrated-ish class probabilities."""
    return hasattr(estimator, "predict_proba")
