"""Behavioral tests for leakage-safe preprocessing and class controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import src.preprocessing as preprocessing_module
from src.preprocessing import (
    DERIVED_NUMERIC_COLUMNS,
    FraudPreprocessor,
    ImbalanceConfig,
    SplitConfig,
    build_imbalance_strategy_report,
    build_chronological_split_manifest,
    chronological_train_validation_split,
    load_split_manifest,
    prepare_training_batch,
    save_split_manifest,
)


def _feature_frame(row_count: int = 26) -> pd.DataFrame:
    positions = np.arange(row_count)
    event_times = 1_577_836_800 + positions * 3_600
    timestamps = pd.to_datetime(event_times, unit="s")
    return pd.DataFrame(
        {
            "amt": 10.0 + positions,
            "city_pop": 1_000.0 + 10.0 * positions,
            "distance_card_merchant_km": 1.0 + positions / 10.0,
            "cc_txn_count_prev_1h": positions % 2,
            "cc_amt_sum_prev_1h": (positions % 2) * 5.0,
            "cc_txn_count_prev_6h": positions % 4,
            "cc_amt_sum_prev_6h": (positions % 4) * 7.0,
            "cc_txn_count_prev_24h": positions % 8,
            "cc_amt_sum_prev_24h": (positions % 8) * 11.0,
            "cc_txn_count_prior": positions,
            "cc_amt_sum_prior": positions * 13.0,
            "category": np.where(positions % 2 == 0, "grocery", "gas_transport"),
            "state": np.where(positions % 3 == 0, "NY", "CA"),
            "merchant": np.where(positions % 3 == 0, "merchant_a", "merchant_b"),
            "city": np.where(positions % 2 == 0, "Albany", "Oakland"),
            "job": np.where(positions % 2 == 0, "Engineer", "Analyst"),
            "zip": np.where(positions % 2 == 0, "12207", "94612"),
            "trans_date_trans_time": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time": event_times,
            "dob": np.where(positions % 2 == 0, "1980-01-01", "1990-06-15"),
            "is_fraud": (positions % 7 == 0).astype(np.int8),
            "cc_num": [f"400000000000{i:04d}" for i in positions],
            "trans_num": [f"transaction-{i:04d}" for i in positions],
            "first": "Test",
            "last": "Customer",
            "gender": np.where(positions % 2 == 0, "F", "M"),
            "street": "1 Example Street",
            "lat": 42.0,
            "long": -73.0,
            "merch_lat": 42.1,
            "merch_long": -73.1,
        }
    )


def _partition_csv(times: list[int], targets: list[int], prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trans_num": [f"{prefix}-{position}" for position in range(len(times))],
            "unix_time": times,
            "trans_date_trans_time": pd.to_datetime(times, unit="s").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "is_fraud": targets,
        }
    )


def _feature_streams_for_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    development = _feature_frame(20)
    development["is_fraud"] = np.array(
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        dtype=np.int8,
    )
    holdout = _feature_frame(8)
    holdout_start = int(development["unix_time"].iloc[-1]) + 3_600
    holdout["unix_time"] = holdout_start + np.arange(len(holdout)) * 3_600
    holdout["trans_date_trans_time"] = pd.to_datetime(
        holdout["unix_time"], unit="s"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    holdout["trans_num"] = [f"holdout-{position:04d}" for position in range(len(holdout))]
    holdout["is_fraud"] = np.array([0, 1, 0, 0, 0, 1, 0, 0], dtype=np.int8)
    return development, holdout


def _assert_sparse_equal(left: sparse.csr_matrix, right: sparse.csr_matrix) -> None:
    assert left.shape == right.shape
    difference = left != right
    assert difference.nnz == 0


def _refresh_payload_digest(payload: dict[str, object]) -> None:
    payload.pop("payload_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest().upper()


def test_chronological_split_keeps_timestamp_buckets_intact() -> None:
    frame = _partition_csv(
        [1, 2, 3, 4, 4, 4, 5, 6],
        [0, 1, 0, 0, 1, 0, 1, 0],
        "dev",
    )

    train, validation, metadata = chronological_train_validation_split(
        frame, config=SplitConfig(validation_fraction=0.5)
    )

    assert metadata["requested_boundary_position"] == 4
    assert metadata["effective_boundary_position"] == 3
    assert metadata["boundary_unix_time"] == 4
    assert train["unix_time"].tolist() == [1, 2, 3]
    assert validation["unix_time"].tolist() == [4, 4, 4, 5, 6]
    assert set(train["unix_time"]).isdisjoint(set(validation["unix_time"]))


def test_chronological_split_rejects_duplicate_keys_and_reordered_time() -> None:
    duplicate_keys = _partition_csv(
        [1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 0, 1], "dev"
    )
    duplicate_keys.loc[1, "trans_num"] = duplicate_keys.loc[0, "trans_num"]
    with pytest.raises(ValueError, match="must be unique"):
        chronological_train_validation_split(
            duplicate_keys, config=SplitConfig(validation_fraction=0.5)
        )

    reordered = _partition_csv(
        [1, 3, 2, 4, 5, 6], [0, 1, 0, 1, 0, 1], "dev"
    )
    with pytest.raises(ValueError, match="not nondecreasing"):
        chronological_train_validation_split(
            reordered, config=SplitConfig(validation_fraction=0.5)
        )


def test_split_manifest_records_out_of_time_contract_and_round_trips(
    tmp_path: Path,
) -> None:
    development = _partition_csv(
        [1, 2, 3, 4, 4, 4, 5, 6],
        [0, 1, 0, 0, 1, 0, 1, 0],
        "dev",
    )
    holdout = _partition_csv([10, 11, 12, 13], [0, 1, 0, 1], "holdout")
    development_path = tmp_path / "development.csv"
    holdout_path = tmp_path / "holdout.csv"
    manifest_path = tmp_path / "split_manifest.json"
    development.to_csv(development_path, index=False)
    holdout.to_csv(holdout_path, index=False)

    manifest = build_chronological_split_manifest(
        development_path,
        holdout_path,
        config=SplitConfig(validation_fraction=0.5, chunksize=3),
    )

    assert manifest["effective_boundary_position"] == 3
    assert manifest["train"]["unix_time_end"] < manifest["validation"]["unix_time_start"]
    assert manifest["validation"]["unix_time_end"] < manifest["holdout"]["unix_time_start"]
    assert manifest["holdout_gap_seconds"] == 4
    assert len(manifest["development"]["sha256"]) == 64
    save_split_manifest(manifest, manifest_path)
    assert load_split_manifest(manifest_path) == manifest


def test_fit_csv_matches_in_memory_fit_and_releases_temporary_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _feature_frame(22)
    frame.loc[2, "amt"] = np.nan
    frame.loc[4, "city_pop"] = np.nan
    frame.loc[16:, "merchant"] = "validation_only_merchant"
    source_path = tmp_path / "features.csv"
    frame.to_csv(source_path, index=False)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(preprocessing_module.tempfile, "tempdir", str(scratch))

    expected = FraudPreprocessor().fit(frame.iloc[:16], partition="train")
    first = FraudPreprocessor().fit_csv(
        source_path, train_rows=16, partition="train", chunksize=5
    )
    second = FraudPreprocessor().fit_csv(
        source_path, train_rows=16, partition="train", chunksize=7
    )

    assert first.numeric_stats_ == expected.numeric_stats_ == second.numeric_stats_
    assert first.nominal_vocabularies_ == expected.nominal_vocabularies_
    assert first.frequency_mappings_ == expected.frequency_mappings_
    assert first.get_feature_names_out().tolist() == expected.get_feature_names_out().tolist()
    assert "validation_only_merchant" not in first.frequency_mappings_["merchant"]
    _assert_sparse_equal(first.transform(frame.iloc[:16]), expected.transform(frame.iloc[:16]))
    assert list(scratch.iterdir()) == []

    with pytest.raises(ValueError, match="requested 30 training rows"):
        FraudPreprocessor().fit_csv(
            source_path, train_rows=30, partition="train", chunksize=6
        )
    assert list(scratch.iterdir()) == []
    with pytest.raises(ValueError, match="does not match the feature CSV"):
        FraudPreprocessor().fit_csv(
            source_path,
            train_rows=16,
            partition="train",
            source_sha256="0" * 64,
        )


def test_manifest_bound_fit_csv_derives_and_verifies_training_contract(
    tmp_path: Path,
) -> None:
    development, holdout = _feature_streams_for_manifest()
    development_path = tmp_path / "development.csv"
    holdout_path = tmp_path / "holdout.csv"
    development.to_csv(development_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    manifest = build_chronological_split_manifest(
        development_path,
        holdout_path,
        config=SplitConfig(validation_fraction=0.25, chunksize=6),
    )

    preprocessor = FraudPreprocessor().fit_csv(
        development_path,
        partition="train",
        split_manifest=manifest,
        chunksize=4,
    )

    assert preprocessor.fitted_context_["row_count"] == manifest["train"]["rows"]
    assert (
        preprocessor.fitted_context_["split_manifest_sha256"]
        == manifest["payload_sha256"]
    )
    assert (
        preprocessor.fitted_context_["ordered_key_sha256"]
        == manifest["train"]["ordered_key_sha256"]
    )
    with pytest.raises(ValueError, match="train_rows differs"):
        FraudPreprocessor().fit_csv(
            development_path,
            train_rows=14,
            partition="train",
            split_manifest=manifest,
        )

    reordered = development.copy()
    reordered.loc[[0, 1], "trans_num"] = reordered.loc[[1, 0], "trans_num"].to_numpy()
    reordered_path = tmp_path / "reordered.csv"
    reordered.to_csv(reordered_path, index=False)
    with pytest.raises(ValueError, match="row order differs"):
        FraudPreprocessor().fit_csv(
            reordered_path,
            partition="train",
            split_manifest=manifest,
            chunksize=5,
        )


def test_imbalance_report_uses_only_manifest_training_counts(tmp_path: Path) -> None:
    development, holdout = _feature_streams_for_manifest()
    development_path = tmp_path / "development.csv"
    holdout_path = tmp_path / "holdout.csv"
    development.to_csv(development_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    manifest = build_chronological_split_manifest(
        development_path,
        holdout_path,
        config=SplitConfig(validation_fraction=0.25, chunksize=7),
    )

    report = build_imbalance_strategy_report(
        manifest, sampling_strategy=0.5, random_state=19
    )

    assert report["train_target_counts"] == manifest["train"]["target_counts"]
    assert report["strategies"]["random_under"]["class_counts"] == {
        "0": 4,
        "1": 2,
    }
    assert report["strategies"]["smotenc"]["class_counts"] == {
        "0": 13,
        "1": 6,
    }
    assert report["controls"]["validation_prevalence"] == "unchanged"
    assert report["dependencies"]["imbalanced_learn"] != "not-installed"
    assert len(report["payload_sha256"]) == 64


def test_transform_uses_frozen_train_statistics_and_maps_unseen_categories() -> None:
    train = _feature_frame(18)
    train.loc[0, "amt"] = np.nan
    train.loc[1, "city_pop"] = np.nan
    validation = _feature_frame(3)
    validation["amt"] = [1_000_000.0, 2_000_000.0, 3_000_000.0]
    validation["category"] = "travel"
    validation["state"] = "ZZ"
    validation["merchant"] = "unseen_merchant"
    validation["city"] = "Unseen City"
    validation["job"] = "Unseen Job"
    validation["zip"] = "00000"
    validation["is_fraud"] = [1, 1, 1]
    validation_before = validation.copy(deep=True)

    preprocessor = FraudPreprocessor().fit(
        train,
        partition="train",
        source_sha256="A" * 64,
        split_manifest_sha256="B" * 64,
    )
    artifact_before = preprocessor.to_dict()
    transformed = preprocessor.transform(validation)
    artifact_after = preprocessor.to_dict()

    assert sparse.isspmatrix_csr(transformed)
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed.data).all()
    assert artifact_after == artifact_before
    pd.testing.assert_frame_equal(validation, validation_before)

    names = preprocessor.get_feature_names_out().tolist()
    category_unknown = names.index("cat__category____UNKNOWN__")
    state_unknown = names.index("cat__state____UNKNOWN__")
    assert transformed[:, category_unknown].toarray().ravel().tolist() == [1.0] * 3
    assert transformed[:, state_unknown].toarray().ravel().tolist() == [1.0] * 3

    relabeled = validation.copy()
    relabeled["is_fraud"] = [0, 0, 0]
    _assert_sparse_equal(transformed, preprocessor.transform(relabeled))


def test_output_schema_excludes_direct_identifiers_and_target() -> None:
    frame = _feature_frame(12)
    preprocessor = FraudPreprocessor().fit(frame, partition="train")
    names = preprocessor.get_feature_names_out().tolist()
    excluded_tokens = {
        "cc_num",
        "trans_num",
        "first",
        "last",
        "gender",
        "street",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "is_fraud",
    }

    assert not any(
        token in feature_name
        for token in excluded_tokens
        for feature_name in names
    )
    assert len(names) == (
        len(preprocessor.config.numeric_columns)
        + len(DERIVED_NUMERIC_COLUMNS)
        + 2 * len(preprocessor.config.frequency_columns)
        + sum(len(values) for values in preprocessor.nominal_vocabularies_.values())
    )


def test_preprocessor_persistence_is_exact_and_detects_tampering(
    tmp_path: Path,
) -> None:
    frame = _feature_frame(14)
    preprocessor = FraudPreprocessor().fit(
        frame,
        partition="train",
        source_sha256="C" * 64,
        split_manifest_sha256="D" * 64,
    )
    artifact_path = tmp_path / "preprocessor.json"
    preprocessor.save(artifact_path)

    restored = FraudPreprocessor.load(artifact_path)
    assert restored.to_dict() == preprocessor.to_dict()
    _assert_sparse_equal(preprocessor.transform(frame), restored.transform(frame))
    compressed_path = tmp_path / "preprocessor.json.gz"
    preprocessor.save(compressed_path)
    compressed = FraudPreprocessor.load(compressed_path)
    assert compressed.to_dict() == preprocessor.to_dict()
    with pytest.raises(FileExistsError, match="already exists"):
        preprocessor.save(artifact_path)

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["numeric_stats"]["amt"]["mean"] += 1.0
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest does not match"):
        FraudPreprocessor.load(artifact_path)


def test_preprocessor_rejects_digest_valid_invalid_state() -> None:
    frame = _feature_frame(14)
    payload = FraudPreprocessor().fit(frame, partition="train").to_dict()
    payload["numeric_stats"]["amt"]["scale"] = 0.0
    _refresh_payload_digest(payload)

    with pytest.raises(ValueError, match="scale.*must be positive"):
        FraudPreprocessor.from_dict(payload)


def test_training_batch_rejects_misaligned_labels_time_and_row_order() -> None:
    frame = _feature_frame(14)
    target = np.array([0] * 10 + [1] * 4, dtype=np.int8)
    frame["is_fraud"] = target
    preprocessor = FraudPreprocessor().fit(frame, partition="train")

    mismatched = target.copy()
    mismatched[[0, -1]] = mismatched[[-1, 0]]
    with pytest.raises(ValueError, match="target values do not match"):
        prepare_training_batch(
            frame,
            mismatched,
            preprocessor,
            config=ImbalanceConfig(strategy="none"),
            partition="train",
        )

    later = frame.copy()
    later["unix_time"] = later["unix_time"] + 86_400
    with pytest.raises(ValueError, match="time range differs"):
        prepare_training_batch(
            later,
            target,
            preprocessor,
            config=ImbalanceConfig(strategy="none"),
            partition="train",
        )

    reordered = frame.copy()
    reordered.loc[[0, 1], "trans_num"] = reordered.loc[[1, 0], "trans_num"].to_numpy()
    with pytest.raises(ValueError, match="row order differs"):
        prepare_training_batch(
            reordered,
            target,
            preprocessor,
            config=ImbalanceConfig(strategy="none"),
            partition="train",
        )


def test_fit_and_imbalance_controls_reject_nontraining_partitions() -> None:
    frame = _feature_frame(12)
    target = np.array([0] * 9 + [1] * 3, dtype=np.int8)
    with pytest.raises(ValueError, match="partition='train'"):
        FraudPreprocessor().fit(frame, partition="validation")

    preprocessor = FraudPreprocessor().fit(frame, partition="train")
    with pytest.raises(ValueError, match="partition='train'"):
        prepare_training_batch(
            frame,
            target,
            preprocessor,
            config=ImbalanceConfig(strategy="none"),
            partition="holdout",
        )


def test_class_weight_preserves_rows_and_assigns_balanced_weights() -> None:
    frame = _feature_frame(12)
    target = np.array([0] * 9 + [1] * 3, dtype=np.int8)
    frame["is_fraud"] = target
    preprocessor = FraudPreprocessor().fit(frame, partition="train")

    batch = prepare_training_batch(
        frame,
        target,
        preprocessor,
        config=ImbalanceConfig(strategy="class_weight", random_state=17),
        partition="train",
    )

    assert np.array_equal(batch.y, target)
    _assert_sparse_equal(batch.X, preprocessor.transform(frame))
    assert batch.sample_weight is not None
    assert batch.sample_weight[target == 0] == pytest.approx(12 / (2 * 9))
    assert batch.sample_weight[target == 1] == pytest.approx(12 / (2 * 3))
    assert batch.metadata["class_counts_before"] == {"0": 9, "1": 3}
    assert batch.metadata["class_counts_after"] == {"0": 9, "1": 3}
    assert batch.metadata["validation_and_holdout_sampled"] is False


def test_random_under_sampling_is_deterministic() -> None:
    pytest.importorskip("imblearn")
    frame = _feature_frame(26)
    target = np.array([0] * 20 + [1] * 6, dtype=np.int8)
    frame["is_fraud"] = target
    preprocessor = FraudPreprocessor().fit(frame, partition="train")
    config = ImbalanceConfig(
        strategy="random_under", sampling_strategy=0.5, random_state=29
    )

    first = prepare_training_batch(
        frame, target, preprocessor, config=config, partition="train"
    )
    second = prepare_training_batch(
        frame, target, preprocessor, config=config, partition="train"
    )

    assert first.metadata["class_counts_after"] == {"0": 12, "1": 6}
    assert first.metadata["selected_positions_sha256"] == second.metadata[
        "selected_positions_sha256"
    ]
    assert np.array_equal(first.y, second.y)
    _assert_sparse_equal(first.X, second.X)


def test_smotenc_returns_finite_sparse_features_with_expected_ratio() -> None:
    pytest.importorskip("imblearn")
    frame = _feature_frame(26)
    target = np.array([0] * 20 + [1] * 6, dtype=np.int8)
    frame["is_fraud"] = target
    preprocessor = FraudPreprocessor().fit(frame, partition="train")

    batch = prepare_training_batch(
        frame,
        target,
        preprocessor,
        config=ImbalanceConfig(
            strategy="smotenc",
            sampling_strategy=0.5,
            random_state=31,
            k_neighbors=2,
            max_output_rows=100,
            max_dense_bytes=10_000_000,
        ),
        partition="train",
    )

    assert sparse.isspmatrix_csr(batch.X)
    assert batch.X.dtype == np.float32
    assert batch.X.shape == (30, len(preprocessor.get_feature_names_out()))
    assert batch.metadata["class_counts_after"] == {"0": 20, "1": 10}
    assert np.isfinite(batch.X.data).all()
    categorical_start = next(
        index
        for index, name in enumerate(preprocessor.get_feature_names_out())
        if str(name).startswith("cat__")
    )
    categorical_ones = np.asarray(batch.X[:, categorical_start:].sum(axis=1)).ravel()
    assert categorical_ones == pytest.approx(
        np.full(batch.X.shape[0], len(preprocessor.config.nominal_columns))
    )


def test_smotenc_enforces_neighbor_and_memory_guards() -> None:
    pytest.importorskip("imblearn")
    frame = _feature_frame(14)
    target = np.array([0] * 10 + [1] * 4, dtype=np.int8)
    frame["is_fraud"] = target
    preprocessor = FraudPreprocessor().fit(frame, partition="train")

    with pytest.raises(ValueError, match="more minority rows"):
        prepare_training_batch(
            frame,
            target,
            preprocessor,
            config=ImbalanceConfig(
                strategy="smotenc", sampling_strategy=0.5, k_neighbors=4
            ),
            partition="train",
        )

    with pytest.raises(MemoryError, match="max_output_rows"):
        prepare_training_batch(
            frame,
            target,
            preprocessor,
            config=ImbalanceConfig(
                strategy="smotenc",
                sampling_strategy=0.5,
                k_neighbors=2,
                max_output_rows=14,
            ),
            partition="train",
        )
