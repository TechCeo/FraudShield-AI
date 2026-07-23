"""Application scoring, sequence-state, and feedback contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from app.feedback import FeedbackStore
from app.scoring import (
    DATASET_CLOCK_OFFSET_SECONDS,
    OnlineSequenceState,
    ScoringRuntime,
    SequenceContext,
    normalize_transactions,
)
from src.features import RollingFeatureState


def _transaction(**updates) -> pd.DataFrame:
    row = {
        "trans_date_trans_time": "2021-01-01 00:00:00",
        "cc_num": "2291163933867244",
        "merchant": "fraud_Kirlin and Sons",
        "category": "personal_care",
        "amt": 2.86,
        "city": "Columbia",
        "state": "SC",
        "zip": "29209",
        "lat": 33.9659,
        "long": -80.9355,
        "city_pop": 333497,
        "job": "Mechanical engineer",
        "dob": "1968-03-19",
        "merch_lat": 33.986391,
        "merch_long": -81.200714,
        "trans_num": "tx-001",
    }
    row.update(updates)
    return pd.DataFrame([row])


def test_normalization_derives_dataset_clock_and_preserves_card_string() -> None:
    normalized = normalize_transactions(_transaction())

    assert normalized.loc[0, "cc_num"] == "2291163933867244"
    assert normalized.loc[0, "unix_time"] == 1_609_459_200 - DATASET_CLOCK_OFFSET_SECONDS
    assert normalized.loc[0, "trans_num"] == "tx-001"


def test_normalization_rejects_float_card_identifiers() -> None:
    frame = _transaction()
    frame["cc_num"] = frame["cc_num"].astype(float)

    try:
        normalize_transactions(frame)
    except ValueError as exc:
        assert "floating point" in str(exc)
    else:
        raise AssertionError("floating-point card identifiers must be rejected")


def test_online_sequence_state_excludes_same_time_peers() -> None:
    context = SequenceContext(
        cards=np.array(["A"]),
        history=np.array([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        history_lengths=np.array([1], dtype=np.int16),
        pending_vectors=np.array([[2.0, 0.0]], dtype=np.float32),
        pending_times=np.array([10], dtype=np.int64),
        pending_keys=np.array(["z"]),
        metadata={},
    )
    state = OnlineSequenceState.from_context(context)
    first, first_length, seeded = state.prepare_and_advance(
        "A", 10, "b", np.array([3.0, 0.0], dtype=np.float32)
    )
    tied, tied_length, _ = state.prepare_and_advance(
        "A", 10, "a", np.array([4.0, 0.0], dtype=np.float32)
    )
    later, later_length, _ = state.prepare_and_advance(
        "A", 11, "c", np.array([5.0, 0.0], dtype=np.float32)
    )

    assert seeded is True
    assert first_length == tied_length == 2
    assert np.array_equal(first[:2], [[1.0, 0.0], [3.0, 0.0]])
    assert np.array_equal(tied[:2], [[1.0, 0.0], [4.0, 0.0]])
    assert later_length == 3
    assert np.array_equal(
        later[:3],
        [[1.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
    )


class _FakePreprocessor:
    feature_names_ = ["amount", "distance", "prior_count"]
    fitted_context_ = {"feature_schema_sha256": "schema"}

    def transform(self, frame: pd.DataFrame):
        values = np.column_stack(
            [
                frame["amt"].to_numpy(dtype=np.float32),
                frame["distance_card_merchant_km"].to_numpy(dtype=np.float32),
                frame["cc_txn_count_prior"].to_numpy(dtype=np.float32),
            ]
        )
        return sparse.csr_matrix(values, dtype=np.float32)


class _FakeEngine:
    sequence_length = 3
    decision_threshold = 0.5
    config = {
        "payload_sha256": "model",
        "feature_schema_sha256": "schema",
        "model_data_manifest_sha256": "data",
        "sequence_index_manifest_sha256": "sequence",
    }

    def predict_prepared(self, static, sequences, lengths):
        rows = static.shape[0]
        hybrid = np.linspace(0.4, 0.6, rows, dtype=np.float64)
        return {
            "xgboost_probability": np.full(rows, 0.5),
            "fnn_probability": np.full(rows, 0.4),
            "lstm_probability": np.full(rows, 0.6),
            "fraud_probability": hybrid,
            "fraud_flag": hybrid >= self.decision_threshold,
        }


def _fake_runtime(tmp_path: Path) -> ScoringRuntime:
    context = SequenceContext(
        cards=np.array([], dtype="<U1"),
        history=np.zeros((0, 2, 3), dtype=np.float32),
        history_lengths=np.zeros(0, dtype=np.int16),
        pending_vectors=np.zeros((0, 3), dtype=np.float32),
        pending_times=np.zeros(0, dtype=np.int64),
        pending_keys=np.array([], dtype="<U1"),
        metadata={
            "model_data_manifest_sha256": "data",
            "sequence_index_manifest_sha256": "sequence",
        },
    )
    return ScoringRuntime(
        project_root=tmp_path,
        engine=_FakeEngine(),
        preprocessor=_FakePreprocessor(),
        velocity_seed=RollingFeatureState().to_dict(),
        sequence_context=context,
        model_config_sha256="model",
    )


def test_scoring_session_commits_only_when_requested(tmp_path) -> None:
    session = _fake_runtime(tmp_path).new_session()
    session._xgboost_drivers = lambda features: np.zeros(features.shape)

    preview = session.score(_transaction(), commit=False)
    committed = session.score(_transaction(trans_num="tx-002"), commit=True)

    assert preview[0].context_depth == 1
    assert preview[0].history_mode == "session_cold_start"
    assert session.scored_rows == 1
    assert committed[0].fraud_probability == 0.4


def test_feedback_store_is_idempotent_and_exportable(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    prediction = {
        "prediction_id": "prediction-1",
        "scored_at_utc": "2026-01-01T00:00:00+00:00",
        "fraud_probability": 0.8,
        "decision_threshold": 0.5,
        "fraud_flag": True,
        "model_config_sha256": "digest",
        "transaction": {"amt": 20.0},
        "engineered_features": {"distance": 12.0},
        "component_probabilities": {"xgboost": 0.8},
        "context_depth": 3,
    }
    store.record(
        prediction,
        "confirm_fraud",
        updated_at_utc="2026-01-01T00:01:00+00:00",
    )
    store.record(
        prediction,
        "false_positive",
        updated_at_utc="2026-01-01T00:02:00+00:00",
    )

    records = store.records()
    assert store.count() == 1
    assert records[0]["reviewer_label"] == "false_positive"
    assert b"prediction-1" in store.export_csv()
