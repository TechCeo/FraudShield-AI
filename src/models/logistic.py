"""Sparse logistic-regression fraud classifier factory."""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.linear_model import LogisticRegression


DEFAULT_PARAMETERS: dict[str, Any] = {
    "C": 1.0,
    "class_weight": "balanced",
    "l1_ratio": 0.0,
    "max_iter": 300,
    "solver": "lbfgs",
    "tol": 1e-4,
}


def build_logistic_classifier(
    parameters: Mapping[str, Any] | None = None,
    *,
    random_state: int = 42,
) -> LogisticRegression:
    """Build a deterministic, class-weighted binary logistic classifier."""

    resolved = {**DEFAULT_PARAMETERS, **dict(parameters or {})}
    resolved["random_state"] = int(random_state)
    return LogisticRegression(**resolved)
