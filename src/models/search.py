"""Deterministic random search over an isolated chronological validation set."""

from __future__ import annotations

import gc
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from sklearn.model_selection import ParameterSampler

from ..utils import atomic_write_json, json_digest, runtime_dependencies, sha256_file
from .data import ModelDataset
from .evaluation import (
    evaluate_probabilities,
    select_fbeta_threshold,
    threshold_selection_dict,
)
from .logistic import build_logistic_classifier
from .random_forest import build_random_forest_classifier
from .xgboost_model import build_xgboost_classifier

LOGGER = logging.getLogger(__name__)
SUPPORTED_MODELS = ("logistic_regression", "random_forest", "xgboost")
MODEL_REPORT_SCHEMA_VERSION = 1


def _parameter_space(model_name: str, target: np.ndarray) -> dict[str, list[Any]]:
    negative = int(np.count_nonzero(target == 0))
    positive = int(np.count_nonzero(target == 1))
    if positive == 0:
        raise ValueError("training target does not contain fraud rows")
    imbalance_ratio = negative / positive
    if model_name == "logistic_regression":
        return {
            "C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "max_iter": [250, 400],
            "solver": ["lbfgs"],
            "tol": [1e-4],
            "class_weight": ["balanced"],
            "l1_ratio": [0.0],
        }
    if model_name == "random_forest":
        return {
            "n_estimators": [100, 150, 220],
            "max_depth": [10, 14, 18, None],
            "min_samples_leaf": [5, 10, 25, 50],
            "max_features": ["sqrt", "log2", 0.5],
            "max_samples": [0.35, 0.50, 0.70],
            "class_weight": ["balanced_subsample"],
            "n_jobs": [-1],
        }
    if model_name == "xgboost":
        return {
            "n_estimators": [500, 800, 1100],
            "learning_rate": [0.03, 0.06, 0.10],
            "max_depth": [3, 5, 7],
            "min_child_weight": [1.0, 5.0, 10.0],
            "subsample": [0.70, 0.85, 1.0],
            "colsample_bytree": [0.70, 0.85, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [1.0, 5.0, 10.0],
            "scale_pos_weight": [
                float(np.sqrt(imbalance_ratio)),
                float(0.5 * imbalance_ratio),
                float(imbalance_ratio),
            ],
            "early_stopping_rounds": [40],
            "objective": ["binary:logistic"],
            "eval_metric": ["aucpr"],
            "tree_method": ["hist"],
            "n_jobs": [-1],
        }
    raise ValueError(f"unsupported model: {model_name}")


def sample_parameters(
    model_name: str,
    target: np.ndarray,
    *,
    n_iter: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Return deterministic, nonrepeating random configurations."""

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model_name must be one of {SUPPORTED_MODELS}")
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter <= 0:
        raise ValueError("n_iter must be a positive integer")
    return [
        dict(parameters)
        for parameters in ParameterSampler(
            _parameter_space(model_name, target),
            n_iter=n_iter,
            random_state=random_state,
        )
    ]


def _build_estimator(
    model_name: str, parameters: Mapping[str, Any], *, random_state: int
):
    if model_name == "logistic_regression":
        return build_logistic_classifier(parameters, random_state=random_state)
    if model_name == "random_forest":
        return build_random_forest_classifier(parameters, random_state=random_state)
    if model_name == "xgboost":
        return build_xgboost_classifier(parameters, random_state=random_state)
    raise ValueError(f"unsupported model: {model_name}")


def _fit_estimator(model_name: str, estimator: Any, data: ModelDataset) -> None:
    if model_name == "xgboost":
        estimator.fit(
            data.train_features,
            data.train_target,
            eval_set=[(data.validation_features, data.validation_target)],
            verbose=False,
        )
    else:
        estimator.fit(data.train_features, data.train_target)


def _positive_probabilities(estimator: Any, features: Any) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("classifier predict_proba output must have two columns")
    return probabilities[:, 1]


def _rank_key(result: Mapping[str, Any]) -> tuple[float, float, float]:
    ranking = result["validation_metrics"]["ranking"]
    selection = result["threshold_selection"]
    return (
        float(ranking["average_precision"]),
        float(selection["recall"]),
        float(selection["precision"]),
    )


def _save_model_atomic(estimator: Any, model_name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}{destination.suffix}"
    )
    try:
        if model_name == "xgboost":
            estimator.save_model(temporary)
        else:
            joblib.dump(estimator, temporary, compress=3)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run_model_search(
    model_name: str,
    data: ModelDataset,
    output_dir: str | Path,
    *,
    n_iter: int,
    random_state: int = 42,
    threshold_beta: float = 2.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Search one classifier and persist its holdout-scored winning estimator."""

    output_root = Path(output_dir)
    extension = ".json" if model_name == "xgboost" else ".joblib"
    model_path = output_root / f"{model_name}{extension}"
    report_path = output_root / f"{model_name}_report.json"
    if not overwrite:
        existing = [str(path) for path in (model_path, report_path) if path.exists()]
        if existing:
            raise FileExistsError(f"model outputs already exist: {', '.join(existing)}")

    parameters = sample_parameters(
        model_name,
        data.train_target,
        n_iter=n_iter,
        random_state=random_state,
    )
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(parameters):
        LOGGER.info(
            "fitting %s candidate %d/%d", model_name, index + 1, len(parameters)
        )
        estimator = _build_estimator(
            model_name, candidate, random_state=random_state
        )
        started = time.perf_counter()
        _fit_estimator(model_name, estimator, data)
        fit_seconds = time.perf_counter() - started
        validation_probabilities = _positive_probabilities(
            estimator, data.validation_features
        )
        selection = select_fbeta_threshold(
            data.validation_target,
            validation_probabilities,
            beta=threshold_beta,
        )
        result: dict[str, Any] = {
            "candidate_index": index,
            "parameters": candidate,
            "fit_seconds": fit_seconds,
            "threshold_selection": threshold_selection_dict(selection),
            "validation_metrics": evaluate_probabilities(
                data.validation_target,
                validation_probabilities,
                threshold=selection.threshold,
            ),
        }
        if model_name == "xgboost":
            result["best_iteration"] = int(estimator.best_iteration)
            result["best_score"] = float(estimator.best_score)
        results.append(result)
        LOGGER.info(
            "%s candidate %d validation AP=%.6f recall=%.6f precision=%.6f",
            model_name,
            index + 1,
            result["validation_metrics"]["ranking"]["average_precision"],
            selection.recall,
            selection.precision,
        )
        del estimator, validation_probabilities
        gc.collect()

    best = max(results, key=_rank_key)
    best_index = int(best["candidate_index"])
    winner = _build_estimator(
        model_name, parameters[best_index], random_state=random_state
    )
    started = time.perf_counter()
    _fit_estimator(model_name, winner, data)
    final_fit_seconds = time.perf_counter() - started
    validation_probabilities = _positive_probabilities(
        winner, data.validation_features
    )
    selection = select_fbeta_threshold(
        data.validation_target,
        validation_probabilities,
        beta=threshold_beta,
    )
    holdout_probabilities = _positive_probabilities(winner, data.holdout_features)
    _save_model_atomic(winner, model_name, model_path)

    content: dict[str, Any] = {
        "artifact_type": "fraud_classifier_report",
        "schema_version": MODEL_REPORT_SCHEMA_VERSION,
        "model_name": model_name,
        "selection_metric": "validation_average_precision",
        "random_state": int(random_state),
        "search_iterations": int(n_iter),
        "threshold_source": "chronological_validation",
        "threshold_objective": f"F{threshold_beta:g}",
        "holdout_usage": "winner_only_with_frozen_validation_threshold",
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "model_data_manifest_sha256": data.metadata["payload_sha256"],
        "feature_schema_sha256": data.metadata["feature_schema_sha256"],
        "dependencies": runtime_dependencies(
            "numpy", "scipy", "scikit-learn", "joblib", "xgboost"
        ),
        "best_candidate_index": best_index,
        "best_parameters": parameters[best_index],
        "final_fit_seconds": final_fit_seconds,
        "threshold_selection": threshold_selection_dict(selection),
        "validation_metrics": evaluate_probabilities(
            data.validation_target,
            validation_probabilities,
            threshold=selection.threshold,
        ),
        "holdout_metrics": evaluate_probabilities(
            data.holdout_target,
            holdout_probabilities,
            threshold=selection.threshold,
        ),
        "candidate_results": results,
    }
    if model_name == "xgboost":
        content["best_iteration"] = int(winner.best_iteration)
        content["best_score"] = float(winner.best_score)
    document = {**content, "payload_sha256": json_digest(content)}
    atomic_write_json(document, report_path, overwrite=overwrite)
    return document


def build_evaluation_matrix(
    reports: Mapping[str, Mapping[str, Any]],
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist a comparable validation and holdout registry for all classifiers."""

    rows: list[dict[str, Any]] = []
    for model_name, report in sorted(reports.items()):
        validation = report["validation_metrics"]
        holdout = report["holdout_metrics"]
        rows.append(
            {
                "model_name": model_name,
                "threshold": report["threshold_selection"]["threshold"],
                "validation_average_precision": validation["ranking"][
                    "average_precision"
                ],
                "validation_pr_auc": validation["ranking"][
                    "pr_auc_trapezoidal"
                ],
                "validation_recall": validation["operating_point"]["recall"],
                "validation_precision": validation["operating_point"]["precision"],
                "holdout_average_precision": holdout["ranking"]["average_precision"],
                "holdout_pr_auc": holdout["ranking"]["pr_auc_trapezoidal"],
                "holdout_recall": holdout["operating_point"]["recall"],
                "holdout_precision": holdout["operating_point"]["precision"],
                "holdout_false_positive_rate": holdout["operating_point"][
                    "false_positive_rate"
                ],
                "holdout_alert_rate": holdout["operating_point"]["alert_rate"],
            }
        )
    content = {
        "artifact_type": "fraud_classifier_evaluation_matrix",
        "schema_version": 1,
        "selection_metric": "validation_average_precision",
        "models": rows,
    }
    document = {**content, "payload_sha256": json_digest(content)}
    atomic_write_json(document, destination, overwrite=overwrite)
    return document


def load_model_report(path: str | Path, *, verify_model: bool = True) -> dict[str, Any]:
    """Load a model report and verify its payload and estimator digests."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("model report root must be an object")
    digest = document.get("payload_sha256")
    content = {key: value for key, value in document.items() if key != "payload_sha256"}
    if digest != json_digest(content):
        raise ValueError("model report digest does not match its content")
    if content.get("artifact_type") != "fraud_classifier_report":
        raise ValueError("unexpected model report artifact type")
    if content.get("schema_version") != MODEL_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported model report schema version")
    if verify_model:
        model_path = source.parent / content["model_file"]
        if sha256_file(model_path) != content["model_sha256"]:
            raise ValueError("model artifact digest differs from its report")
    return document
