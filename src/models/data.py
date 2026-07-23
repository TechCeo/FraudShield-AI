"""Chunked feature-stream preparation and sparse model-data persistence."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from ..features import RollingFeatureState, process_csv
from ..preprocessing import FraudPreprocessor, load_split_manifest
from ..utils import atomic_write_json, json_digest, sha256_file

MODEL_DATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelDataPaths:
    """Resolved paths for immutable inputs and generated model data."""

    development_raw: Path
    holdout_raw: Path
    split_manifest: Path
    development_features: Path
    holdout_features: Path
    development_state: Path
    holdout_state: Path
    output_dir: Path

    @classmethod
    def project_defaults(cls, project_root: str | Path) -> "ModelDataPaths":
        root = Path(project_root)
        processed = root / "data" / "processed"
        return cls(
            development_raw=root / "data" / "fraudTrain.csv",
            holdout_raw=root / "data" / "fraudTest.csv",
            split_manifest=processed / "split_manifest.json",
            development_features=processed / "fraudTrain_features.csv.gz",
            holdout_features=processed / "fraudTest_features.csv.gz",
            development_state=processed / "train_velocity_state.json.gz",
            holdout_state=processed / "test_velocity_state.json.gz",
            output_dir=processed / "model_data",
        )


@dataclass(frozen=True)
class ModelDataset:
    """Sparse training, validation, and holdout arrays with registry metadata."""

    train_features: sparse.csr_matrix
    train_target: np.ndarray
    validation_features: sparse.csr_matrix
    validation_target: np.ndarray
    holdout_features: sparse.csr_matrix
    holdout_target: np.ndarray
    metadata: dict[str, Any]


def ensure_feature_streams(
    paths: ModelDataPaths,
    *,
    chunksize: int = 100_000,
) -> None:
    """Create causal feature streams and continuation state when absent."""

    manifest = load_split_manifest(paths.split_manifest)
    for label, source in (
        ("development", paths.development_raw),
        ("holdout", paths.holdout_raw),
    ):
        registry = manifest[label]
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != int(registry["size_bytes"]):
            raise ValueError(f"{label} raw file size differs from the split manifest")
        if sha256_file(source) != registry["sha256"]:
            raise ValueError(f"{label} raw file digest differs from the split manifest")

    development_exists = paths.development_features.is_file()
    state_exists = paths.development_state.is_file()
    if development_exists != state_exists:
        raise RuntimeError(
            "development feature stream and velocity state must either both exist or both be absent"
        )
    if not development_exists:
        _, state = process_csv(
            paths.development_raw,
            paths.development_features,
            chunksize=chunksize,
        )
        state.save(paths.development_state)

    holdout_exists = paths.holdout_features.is_file()
    holdout_state_exists = paths.holdout_state.is_file()
    if holdout_exists != holdout_state_exists:
        raise RuntimeError(
            "holdout feature stream and velocity state must either both exist or both be absent"
        )
    if not holdout_exists:
        state = RollingFeatureState.load(paths.development_state)
        _, state = process_csv(
            paths.holdout_raw,
            paths.holdout_features,
            chunksize=chunksize,
            state=state,
        )
        state.save(paths.holdout_state)


def _target(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError("is_fraud must contain only binary values")
    return numeric.astype(np.int8)


def _save_sparse_atomic(matrix: sparse.csr_matrix, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.npz")
    try:
        sparse.save_npz(temporary, matrix, compressed=True)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _save_array_atomic(values: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.npy")
    try:
        np.save(temporary, values, allow_pickle=False)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _iter_partition_matrices(
    source: Path,
    preprocessor: FraudPreprocessor,
    *,
    chunksize: int,
    boundary: int | None,
) -> Iterator[tuple[str, sparse.csr_matrix, np.ndarray]]:
    row_start = 0
    previous_time: int | None = None
    for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False):
        target = _target(chunk[preprocessor.config.target_col])
        times = pd.to_numeric(
            chunk[preprocessor.config.event_time_col], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(times).all() or not np.equal(times, np.rint(times)).all():
            raise ValueError("feature stream contains invalid event times")
        integer_times = times.astype(np.int64)
        if np.any(np.diff(integer_times) < 0) or (
            previous_time is not None and int(integer_times[0]) < previous_time
        ):
            raise ValueError("feature stream is not nondecreasing by event time")
        previous_time = int(integer_times[-1])
        matrix = preprocessor.transform(chunk)
        row_stop = row_start + len(chunk)
        if boundary is None:
            yield "holdout", matrix, target
        elif row_stop <= boundary:
            yield "train", matrix, target
        elif row_start >= boundary:
            yield "validation", matrix, target
        else:
            local_boundary = boundary - row_start
            yield "train", matrix[:local_boundary], target[:local_boundary]
            yield "validation", matrix[local_boundary:], target[local_boundary:]
        row_start = row_stop


def _stack(parts: list[sparse.csr_matrix], columns: int) -> sparse.csr_matrix:
    if not parts:
        raise ValueError("partition contains no transformed rows")
    matrix = sparse.vstack(parts, format="csr", dtype=np.float32)
    if matrix.shape[1] != columns:
        raise ValueError("transformed matrix width differs from preprocessing schema")
    matrix.sort_indices()
    return matrix


def _validate_counts(
    partition: str, target: np.ndarray, expected: Mapping[str, Any]
) -> None:
    actual = {
        "0": int(np.count_nonzero(target == 0)),
        "1": int(np.count_nonzero(target == 1)),
    }
    expected_counts = {str(key): int(value) for key, value in expected.items()}
    if actual != expected_counts:
        raise ValueError(f"{partition} target counts differ from the split manifest")


def build_model_dataset(
    paths: ModelDataPaths,
    *,
    chunksize: int = 100_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit preprocessing on training rows and persist three sparse partitions."""

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    manifest = load_split_manifest(paths.split_manifest)
    train_rows = int(manifest["train"]["rows"])
    expected_outputs = [
        paths.output_dir / "train_features.npz",
        paths.output_dir / "train_target.npy",
        paths.output_dir / "validation_features.npz",
        paths.output_dir / "validation_target.npy",
        paths.output_dir / "holdout_features.npz",
        paths.output_dir / "holdout_target.npy",
        paths.output_dir / "fraud_preprocessor.json.gz",
        paths.output_dir / "model_data_manifest.json",
    ]
    existing = [str(path) for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"model-data outputs already exist: {', '.join(existing)}")

    preprocessor = FraudPreprocessor().fit_csv(
        paths.development_features,
        partition="train",
        split_manifest=manifest,
        chunksize=chunksize,
    )
    preprocessor_path = paths.output_dir / "fraud_preprocessor.json.gz"
    preprocessor.save(preprocessor_path, overwrite=overwrite)
    column_count = len(preprocessor.feature_names_)

    matrices: dict[str, list[sparse.csr_matrix]] = {
        "train": [],
        "validation": [],
        "holdout": [],
    }
    targets: dict[str, list[np.ndarray]] = {
        "train": [],
        "validation": [],
        "holdout": [],
    }
    for partition, matrix, target in _iter_partition_matrices(
        paths.development_features,
        preprocessor,
        chunksize=chunksize,
        boundary=train_rows,
    ):
        matrices[partition].append(matrix)
        targets[partition].append(target)
    for partition, matrix, target in _iter_partition_matrices(
        paths.holdout_features,
        preprocessor,
        chunksize=chunksize,
        boundary=None,
    ):
        matrices[partition].append(matrix)
        targets[partition].append(target)

    partition_metadata: dict[str, Any] = {}
    for partition in ("train", "validation", "holdout"):
        matrix = _stack(matrices[partition], column_count)
        target = np.concatenate(targets[partition]).astype(np.int8, copy=False)
        expected = manifest[partition]
        if matrix.shape[0] != int(expected["rows"]):
            raise ValueError(f"{partition} row count differs from the split manifest")
        _validate_counts(partition, target, expected["target_counts"])
        matrix_path = paths.output_dir / f"{partition}_features.npz"
        target_path = paths.output_dir / f"{partition}_target.npy"
        _save_sparse_atomic(matrix, matrix_path)
        _save_array_atomic(target, target_path)
        partition_metadata[partition] = {
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "nonzero": int(matrix.nnz),
            "target_counts": {
                "0": int(np.count_nonzero(target == 0)),
                "1": int(np.count_nonzero(target == 1)),
            },
            "features_file": matrix_path.name,
            "features_sha256": sha256_file(matrix_path),
            "target_file": target_path.name,
            "target_sha256": sha256_file(target_path),
        }
        del matrix, target

    content: dict[str, Any] = {
        "artifact_type": "fraud_model_data",
        "schema_version": MODEL_DATA_SCHEMA_VERSION,
        "split_manifest_sha256": manifest["payload_sha256"],
        "development_feature_sha256": sha256_file(paths.development_features),
        "holdout_feature_sha256": sha256_file(paths.holdout_features),
        "preprocessor_file": preprocessor_path.name,
        "preprocessor_sha256": sha256_file(preprocessor_path),
        "feature_schema_sha256": preprocessor.fitted_context_[
            "feature_schema_sha256"
        ],
        "feature_names": preprocessor.feature_names_,
        "partitions": partition_metadata,
    }
    document = {**content, "payload_sha256": json_digest(content)}
    atomic_write_json(
        document,
        paths.output_dir / "model_data_manifest.json",
        overwrite=overwrite,
    )
    return document


def _load_manifest(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("model-data manifest root must be an object")
    digest = document.get("payload_sha256")
    content = {key: value for key, value in document.items() if key != "payload_sha256"}
    if digest != json_digest(content):
        raise ValueError("model-data manifest digest does not match its content")
    if content.get("artifact_type") != "fraud_model_data":
        raise ValueError("unexpected model-data artifact type")
    if content.get("schema_version") != MODEL_DATA_SCHEMA_VERSION:
        raise ValueError("unsupported model-data schema version")
    return document


def load_model_data_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Load and verify the model-data registry without materializing matrices."""

    return _load_manifest(Path(output_dir) / "model_data_manifest.json")


def load_model_dataset(output_dir: str | Path, *, verify_hashes: bool = True) -> ModelDataset:
    """Load registered sparse partitions and verify their shape and integrity."""

    root = Path(output_dir)
    metadata = _load_manifest(root / "model_data_manifest.json")
    loaded: dict[str, tuple[sparse.csr_matrix, np.ndarray]] = {}
    for partition in ("train", "validation", "holdout"):
        registry = metadata["partitions"][partition]
        matrix_path = root / registry["features_file"]
        target_path = root / registry["target_file"]
        if verify_hashes:
            if sha256_file(matrix_path) != registry["features_sha256"]:
                raise ValueError(f"{partition} feature artifact digest mismatch")
            if sha256_file(target_path) != registry["target_sha256"]:
                raise ValueError(f"{partition} target artifact digest mismatch")
        matrix = sparse.load_npz(matrix_path).tocsr().astype(np.float32, copy=False)
        target = np.load(target_path, allow_pickle=False).astype(np.int8, copy=False)
        expected_shape = (int(registry["rows"]), int(registry["columns"]))
        if matrix.shape != expected_shape or target.shape != (expected_shape[0],):
            raise ValueError(f"{partition} array shape differs from its registry")
        loaded[partition] = (matrix, target)
    return ModelDataset(
        train_features=loaded["train"][0],
        train_target=loaded["train"][1],
        validation_features=loaded["validation"][0],
        validation_target=loaded["validation"][1],
        holdout_features=loaded["holdout"][0],
        holdout_target=loaded["holdout"][1],
        metadata=metadata,
    )
