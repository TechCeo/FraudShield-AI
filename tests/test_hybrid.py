"""Contract tests for hybrid fusion, inference, and drift monitoring."""

from __future__ import annotations

import numpy as np
import torch

from src.models.drift import (
    DriftConfig,
    DriftDetector,
    population_stability_index,
    shifted_probability_scenario,
)
from src.models.hybrid import (
    HybridInferenceEngine,
    benchmark_hybrid_engine,
    blend_probabilities,
    optimize_hybrid_blend,
    sample_hybrid_candidates,
)


def _components() -> dict[str, np.ndarray]:
    return {
        "xgboost": np.array([0.05, 0.20, 0.80, 0.95]),
        "fnn": np.array([0.10, 0.30, 0.70, 0.90]),
        "lstm": np.array([0.02, 0.40, 0.85, 0.88]),
    }


def test_probability_and_logit_blends_are_bounded_and_aligned() -> None:
    weights = {"xgboost": 0.5, "fnn": 0.2, "lstm": 0.3}
    probability = blend_probabilities(
        _components(), weights, blend_space="probability"
    )
    logit = blend_probabilities(_components(), weights, blend_space="logit")

    expected = sum(weights[name] * values for name, values in _components().items())
    assert np.allclose(probability, expected)
    assert probability.shape == logit.shape == (4,)
    assert ((0.0 <= logit) & (logit <= 1.0)).all()


def test_hybrid_candidate_sampling_is_deterministic_and_constrained() -> None:
    first = sample_hybrid_candidates(
        n_weight_samples=12, minimum_component_weight=0.05, random_state=13
    )
    second = sample_hybrid_candidates(
        n_weight_samples=12, minimum_component_weight=0.05, random_state=13
    )

    assert first == second
    assert len(first) == 24
    assert {candidate["blend_space"] for candidate in first} == {
        "probability",
        "logit",
    }
    for candidate in first:
        assert np.isclose(sum(candidate["weights"].values()), 1.0)
        assert min(candidate["weights"].values()) >= 0.05


def test_hybrid_optimization_uses_validation_arrays_only() -> None:
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    components = {
        "xgboost": np.array([0.05, 0.10, 0.30, 0.45, 0.55, 0.70, 0.90, 0.95]),
        "fnn": np.array([0.10, 0.20, 0.40, 0.35, 0.65, 0.60, 0.80, 0.90]),
        "lstm": np.array([0.02, 0.25, 0.20, 0.48, 0.52, 0.85, 0.75, 0.98]),
    }
    best, results, probabilities = optimize_hybrid_blend(
        target,
        components,
        n_weight_samples=10,
        minimum_component_weight=0.05,
        random_state=3,
    )

    assert len(results) == 20
    assert probabilities.shape == target.shape
    assert best["validation_average_precision"] > 0.9
    assert min(best["weights"].values()) >= 0.05


class _FakeBooster:
    def inplace_predict(self, features):
        values = np.asarray(features)
        return 1.0 / (1.0 + np.exp(-values[:, 0]))


class _FakeXGBoost:
    def get_booster(self):
        return _FakeBooster()


class _FakeFNN(torch.nn.Module):
    def forward(self, features):
        return features[:, 1]


class _FakeLSTM(torch.nn.Module):
    def forward(self, features, lengths):
        return features[:, 0, 2]


def test_warm_hybrid_engine_scores_prepared_batches() -> None:
    engine = HybridInferenceEngine(
        _FakeXGBoost(),
        _FakeFNN(),
        _FakeLSTM(),
        {
            "weights": {"xgboost": 0.5, "fnn": 0.2, "lstm": 0.3},
            "blend_space": "logit",
            "decision_threshold": 0.5,
        },
        sequence_length=3,
        device=torch.device("cpu"),
    )
    static = np.arange(24, dtype=np.float32).reshape(8, 3) / 10.0
    sequences = np.repeat(static[:, None, :], 3, axis=1)
    lengths = np.full(8, 3, dtype=np.int64)

    scores = engine.score_prepared(static, sequences, lengths)
    predictions = engine.predict_prepared(static, sequences, lengths)
    benchmark = benchmark_hybrid_engine(
        engine,
        static,
        sequences,
        lengths,
        batch_sizes=(1, 4),
        warmups=1,
        repeats=2,
    )

    assert set(scores) == {"xgboost", "fnn", "lstm", "hybrid"}
    assert all(values.shape == (8,) for values in scores.values())
    assert set(predictions) == {
        "xgboost_probability",
        "fnn_probability",
        "lstm_probability",
        "fraud_probability",
        "fraud_flag",
    }
    assert predictions["fraud_flag"].dtype == np.bool_
    assert len(benchmark["batches"]) == 2
    assert benchmark["torch_inference_mode"] is True


def test_drift_detector_separates_stable_and_injected_windows(tmp_path) -> None:
    generator = np.random.default_rng(21)
    reference = generator.normal(size=(2_000, 4))
    predictions = 1.0 / (1.0 + np.exp(-reference[:, 0]))
    detector = DriftDetector(
        DriftConfig(
            feature_sample_rows=2_000,
            prediction_sample_rows=2_000,
            random_state=5,
        )
    ).fit(
        reference,
        ["a", "b", "c", "d"],
        predictions,
        reference_context={"source": "test"},
    )
    stable = detector.evaluate(reference, predictions, window_name="stable")
    shifted = detector.evaluate(
        reference + 2.5,
        shifted_probability_scenario(predictions, logit_shift=1.0),
        window_name="shifted",
    )
    path = tmp_path / "drift_detector.json"
    detector.save(path)
    restored = DriftDetector.load(path)

    assert stable["overall_status"] == "stable"
    assert shifted["overall_status"] == "critical"
    assert shifted["feature_summary"]["critical"] >= 1
    assert restored.reference_context["source"] == "test"


def test_population_stability_index_is_zero_for_equal_distributions() -> None:
    values = np.array([0.2, 0.3, 0.5])
    assert population_stability_index(values, values) == 0.0
