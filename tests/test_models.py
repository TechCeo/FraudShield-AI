"""Contract tests for classifier search, thresholding, and reporting."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification

from src.models.data import ModelDataset
from src.models.evaluation import evaluate_probabilities, select_fbeta_threshold
from src.models.logistic import build_logistic_classifier
from src.models.random_forest import build_random_forest_classifier
from src.models.search import load_model_report, run_model_search, sample_parameters
from src.utils import json_digest, sha256_file


def _model_dataset() -> ModelDataset:
    features, target = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        weights=[0.90, 0.10],
        random_state=7,
    )
    features = sparse.csr_matrix(features.astype(np.float32))
    target = target.astype(np.int8)
    return ModelDataset(
        train_features=features[:360],
        train_target=target[:360],
        validation_features=features[360:480],
        validation_target=target[360:480],
        holdout_features=features[480:],
        holdout_target=target[480:],
        metadata={
            "payload_sha256": "A" * 64,
            "feature_schema_sha256": "B" * 64,
        },
    )


def test_threshold_selection_and_metrics_prioritize_fraud_detection() -> None:
    target = np.array([0, 0, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.01, 0.20, 0.60, 0.55, 0.90])

    selection = select_fbeta_threshold(target, probabilities, beta=2.0)
    report = evaluate_probabilities(
        target, probabilities, threshold=selection.threshold
    )

    assert 0.0 <= selection.threshold <= 1.0
    assert report["ranking"]["average_precision"] > 0.75
    assert report["operating_point"]["recall"] == selection.recall
    assert report["operating_point"]["precision"] == selection.precision
    assert report["default_operating_point"]["false_positive"] == 1


@pytest.mark.parametrize(
    ("target", "probabilities"),
    [
        ([0, 0], [0.1, 0.2]),
        ([0, 1], [0.1, np.nan]),
        ([0, 1], [0.1, 1.1]),
    ],
)
def test_probability_contract_rejects_invalid_inputs(target, probabilities) -> None:
    with pytest.raises(ValueError):
        select_fbeta_threshold(target, probabilities)


def test_classifier_factories_apply_reproducible_defaults() -> None:
    logistic = build_logistic_classifier({"C": 0.3}, random_state=11)
    forest = build_random_forest_classifier(
        {"n_estimators": 10}, random_state=11
    )

    assert logistic.C == 0.3
    assert logistic.class_weight == "balanced"
    assert logistic.random_state == 11
    assert forest.n_estimators == 10
    assert forest.class_weight == "balanced_subsample"
    assert forest.random_state == 11


def test_parameter_sampling_is_deterministic_and_model_specific() -> None:
    target = np.array([0] * 90 + [1] * 10, dtype=np.int8)
    first = sample_parameters(
        "xgboost", target, n_iter=3, random_state=23
    )
    second = sample_parameters(
        "xgboost", target, n_iter=3, random_state=23
    )

    assert first == second
    assert len(first) == 3
    assert all(candidate["tree_method"] == "hist" for candidate in first)
    assert all(candidate["scale_pos_weight"] > 0 for candidate in first)


def test_logistic_search_persists_integrity_bound_report(tmp_path) -> None:
    report = run_model_search(
        "logistic_regression",
        _model_dataset(),
        tmp_path,
        n_iter=2,
        random_state=17,
    )
    model_path = tmp_path / "logistic_regression.joblib"
    report_path = tmp_path / "logistic_regression_report.json"

    assert model_path.is_file()
    assert report_path.is_file()
    assert report["selection_metric"] == "validation_average_precision"
    assert report["threshold_source"] == "chronological_validation"
    assert report["holdout_usage"] == "winner_only_with_frozen_validation_threshold"
    assert report["model_sha256"] == sha256_file(model_path)
    assert len(report["candidate_results"]) == 2

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    content = {
        key: value for key, value in persisted.items() if key != "payload_sha256"
    }
    assert persisted["payload_sha256"] == json_digest(content)
    assert load_model_report(report_path) == persisted

    model_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="model artifact digest"):
        load_model_report(report_path)
