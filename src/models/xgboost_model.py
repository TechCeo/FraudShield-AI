"""Histogram-based XGBoost fraud classifier factory."""

from __future__ import annotations

from typing import Any, Mapping


DEFAULT_PARAMETERS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_estimators": 800,
    "learning_rate": 0.06,
    "max_depth": 6,
    "min_child_weight": 5.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 40,
    "n_jobs": -1,
}


def build_xgboost_classifier(
    parameters: Mapping[str, Any] | None = None,
    *,
    random_state: int = 42,
):
    """Build an XGBoost classifier configured for sparse CPU training."""

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - exercised by dependency checks
        raise RuntimeError(
            "xgboost is required; install the dependencies in requirements.txt"
        ) from exc
    resolved = {**DEFAULT_PARAMETERS, **dict(parameters or {})}
    resolved["random_state"] = int(random_state)
    return XGBClassifier(**resolved)
