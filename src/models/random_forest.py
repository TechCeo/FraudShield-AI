"""Class-weighted random-forest fraud classifier factory."""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.ensemble import RandomForestClassifier


DEFAULT_PARAMETERS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 14,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}


def build_random_forest_classifier(
    parameters: Mapping[str, Any] | None = None,
    *,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Build a deterministic forest with per-tree balanced class weights."""

    resolved = {**DEFAULT_PARAMETERS, **dict(parameters or {})}
    resolved["random_state"] = int(random_state)
    return RandomForestClassifier(**resolved)
