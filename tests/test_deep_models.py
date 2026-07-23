"""Contract tests for static and chronological neural classifiers."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import sparse

from src.models.data import ModelDataset
from src.models.deep_common import sampled_training_indices
from src.models.fnn import StaticFraudFNN, load_fnn, run_fnn_search
from src.models.lstm import CausalFraudLSTM, load_lstm, run_lstm_search
from src.models.search import load_model_report
from src.models.sequences import (
    CausalPointerBuilder,
    GlobalSparseAccessor,
    SequenceIndex,
    sequence_row_indices,
    sequence_tensor_batch,
)


def _dataset() -> ModelDataset:
    generator = np.random.default_rng(9)
    features = generator.normal(size=(600, 12)).astype(np.float32)
    target = (np.arange(600) % 10 == 0).astype(np.int8)
    features[target == 1, :4] += 1.5
    matrix = sparse.csr_matrix(features)
    return ModelDataset(
        train_features=matrix[:360],
        train_target=target[:360],
        validation_features=matrix[360:480],
        validation_target=target[360:480],
        holdout_features=matrix[480:],
        holdout_target=target[480:],
        metadata={
            "payload_sha256": "A" * 64,
            "feature_schema_sha256": "B" * 64,
        },
    )


def _sequence_index() -> SequenceIndex:
    previous = np.full(600, -1, dtype=np.int32)
    previous[5:] = np.arange(595, dtype=np.int32)
    return SequenceIndex(
        previous=previous,
        offsets={"train": (0, 360), "validation": (360, 480), "holdout": (480, 600)},
        metadata={
            "payload_sha256": "C" * 64,
            "model_data_manifest_sha256": "A" * 64,
        },
    )


def test_training_endpoint_sampling_is_deterministic_and_train_only() -> None:
    target = np.array([0] * 90 + [1] * 10, dtype=np.int8)
    first = sampled_training_indices(
        target, negative_ratio=3, random_state=17
    )
    second = sampled_training_indices(
        target, negative_ratio=3, random_state=17
    )

    assert np.array_equal(first, second)
    assert len(first) == 40
    assert np.count_nonzero(target[first] == 1) == 10
    assert np.count_nonzero(target[first] == 0) == 30
    assert first.min() >= 0 and first.max() < len(target)


def test_neural_modules_return_one_logit_per_endpoint() -> None:
    fnn = StaticFraudFNN(12, hidden_dims=(8, 4), dropout=0.1)
    lstm = CausalFraudLSTM(
        12, projection_dim=8, hidden_size=6, num_layers=1, dropout=0.1
    )

    assert fnn(torch.zeros(5, 12)).shape == (5,)
    assert lstm(torch.zeros(5, 4, 12), torch.tensor([4, 3, 2, 1, 4])).shape == (5,)


def test_sequence_rows_are_causal_oldest_to_current() -> None:
    previous = np.array([-1, 0, 1, -1, 3, 4], dtype=np.int32)
    rows, lengths = sequence_row_indices(
        np.array([2, 5], dtype=np.int64), previous, sequence_length=4
    )

    assert rows.tolist() == [[0, 1, 2, -1], [3, 4, 5, -1]]
    assert lengths.tolist() == [3, 3]
    assert np.all(rows[rows >= 0] <= np.repeat([2, 5], 3))


def test_pointer_builder_excludes_same_timestamp_peers_across_chunks() -> None:
    builder = CausalPointerBuilder()
    first = builder.transform(
        ["a", "a", "b"],
        np.array([10, 10, 10], dtype=np.int64),
        ["z", "a", "b"],
        start_index=0,
    )
    second = builder.transform(
        ["a", "b", "a"],
        np.array([11, 11, 11], dtype=np.int64),
        ["c", "d", "e"],
        start_index=3,
    )

    assert first.tolist() == [-1, -1, -1]
    assert second.tolist() == [1, 2, 1]
    assert builder.card_count == 2


def test_global_sparse_accessor_crosses_partition_boundaries() -> None:
    data = _dataset()
    accessor = GlobalSparseAccessor(data)
    indices = np.array([0, 359, 360, 479, 480, 599], dtype=np.int64)

    gathered = accessor.gather(indices)
    expected = sparse.vstack(
        [data.train_features, data.validation_features, data.holdout_features]
    )[indices].toarray()

    assert np.allclose(gathered, expected)
    features, lengths = sequence_tensor_batch(
        accessor,
        np.array([360, 480]),
        _sequence_index().previous,
        4,
        torch.device("cpu"),
    )
    assert features.shape == (2, 4, 12)
    assert lengths.tolist() == [4, 4]


def test_fnn_search_persists_restricted_weight_artifact(tmp_path) -> None:
    report = run_fnn_search(
        _dataset(),
        tmp_path,
        n_iter=1,
        max_epochs=1,
        patience=1,
        random_state=5,
        device_name="cpu",
    )

    assert report["model_name"] == "fnn"
    assert report["threshold_source"] == "chronological_validation"
    assert load_model_report(tmp_path / "fnn_report.json") == report
    assert isinstance(load_fnn(tmp_path / "fnn.pt"), StaticFraudFNN)


def test_lstm_search_persists_causal_sequence_contract(tmp_path) -> None:
    report = run_lstm_search(
        _dataset(),
        _sequence_index(),
        tmp_path,
        n_iter=1,
        max_epochs=1,
        patience=1,
        random_state=7,
        device_name="cpu",
    )

    assert report["model_name"] == "lstm"
    assert report["architecture"] == "unidirectional_causal_lstm"
    assert "without_future_rows_or_labels" in report["sequence_contract"]
    assert load_model_report(tmp_path / "lstm_report.json") == report
    assert isinstance(load_lstm(tmp_path / "lstm.pt"), CausalFraudLSTM)


@pytest.mark.parametrize("sequence_length", [0, -1])
def test_sequence_builder_rejects_nonpositive_length(sequence_length: int) -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        sequence_row_indices(
            np.array([0], dtype=np.int64),
            np.array([-1], dtype=np.int32),
            sequence_length,
        )
