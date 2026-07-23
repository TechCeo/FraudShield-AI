"""Causal per-card sequence indexing and sparse feature access."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..preprocessing import load_split_manifest
from ..utils import atomic_write_json, json_digest, sha256_file
from .data import ModelDataPaths, ModelDataset, load_model_data_manifest

SEQUENCE_INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SequenceIndex:
    """Previous-event pointers over the complete chronological transaction stream."""

    previous: np.ndarray
    offsets: dict[str, tuple[int, int]]
    metadata: dict[str, Any]


class CausalPointerBuilder:
    """Assign same-card pointers across unordered timestamp buckets."""

    def __init__(self) -> None:
        self._committed: dict[str, int] = {}
        self._pending_time: dict[str, int] = {}
        self._pending_index: dict[str, int] = {}
        self._pending_key: dict[str, str] = {}
        self.next_index = 0

    @property
    def card_count(self) -> int:
        return len(self._pending_time)

    def transform(
        self,
        cards: list[str],
        times: np.ndarray,
        keys: list[str],
        *,
        start_index: int,
    ) -> np.ndarray:
        """Return pointers whose referenced timestamp is strictly earlier."""

        if start_index != self.next_index:
            raise ValueError("pointer chunks must be contiguous")
        if not (len(cards) == len(times) == len(keys)):
            raise ValueError("pointer inputs must have equal lengths")
        output = np.full(len(cards), -1, dtype=np.int32)
        for position, (card, event_time, transaction_key) in enumerate(
            zip(cards, times.tolist(), keys, strict=True)
        ):
            global_index = start_index + position
            if card not in self._pending_time:
                self._pending_time[card] = int(event_time)
                self._pending_index[card] = global_index
                self._pending_key[card] = transaction_key
                continue
            pending_time = self._pending_time[card]
            if int(event_time) < pending_time:
                raise ValueError("per-card time reversal detected")
            if int(event_time) == pending_time:
                output[position] = self._committed.get(card, -1)
                if transaction_key < self._pending_key[card]:
                    self._pending_index[card] = global_index
                    self._pending_key[card] = transaction_key
                continue
            self._committed[card] = self._pending_index[card]
            output[position] = self._committed[card]
            self._pending_time[card] = int(event_time)
            self._pending_index[card] = global_index
            self._pending_key[card] = transaction_key
        self.next_index += len(cards)
        return output


def _save_array_atomic(
    values: np.ndarray, destination: Path, *, overwrite: bool
) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"sequence output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.npy")
    try:
        np.save(temporary, values, allow_pickle=False)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _key_hash_bytes(values: pd.Series) -> bytes:
    normalized = values.astype("string")
    if normalized.isna().any() or normalized.str.len().eq(0).any():
        raise ValueError("transaction keys must be nonempty")
    hashes = pd.util.hash_pandas_object(
        normalized, index=False, categorize=False
    ).to_numpy(dtype=np.uint64)
    return np.ascontiguousarray(hashes, dtype="<u8").tobytes()


def build_sequence_index(
    paths: ModelDataPaths,
    *,
    chunksize: int = 100_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist one strictly prior same-card pointer for every transaction row."""

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    split = load_split_manifest(paths.split_manifest)
    model_data = load_model_data_manifest(paths.output_dir)
    if sha256_file(paths.development_features) != model_data[
        "development_feature_sha256"
    ]:
        raise ValueError("development feature stream differs from model data")
    if sha256_file(paths.holdout_features) != model_data["holdout_feature_sha256"]:
        raise ValueError("holdout feature stream differs from model data")
    train_rows = int(split["train"]["rows"])
    validation_rows = int(split["validation"]["rows"])
    holdout_rows = int(split["holdout"]["rows"])
    development_rows = train_rows + validation_rows
    total_rows = development_rows + holdout_rows
    if total_rows > np.iinfo(np.int32).max:
        raise ValueError("sequence index exceeds int32 capacity")
    output_path = paths.output_dir / "previous_transaction_index.npy"
    manifest_path = paths.output_dir / "sequence_index_manifest.json"
    if not overwrite:
        existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(f"sequence outputs already exist: {', '.join(existing)}")

    previous = np.full(total_rows, -1, dtype=np.int32)
    pointer_builder = CausalPointerBuilder()
    key_digests = {
        "train": hashlib.sha256(),
        "validation": hashlib.sha256(),
        "holdout": hashlib.sha256(),
    }
    cold_starts = {"train": 0, "validation": 0, "holdout": 0}
    partition_bounds = {
        "train": (0, train_rows),
        "validation": (train_rows, development_rows),
        "holdout": (development_rows, total_rows),
    }
    cursor = 0
    prior_global_time: int | None = None
    for source in (paths.development_features, paths.holdout_features):
        reader = pd.read_csv(
            source,
            usecols=["cc_num", "trans_num", "unix_time"],
            dtype={"cc_num": "string", "trans_num": "string"},
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in reader:
            cards = chunk["cc_num"].astype("string")
            if cards.isna().any() or cards.str.len().eq(0).any():
                raise ValueError("cc_num must contain nonempty identifiers")
            numeric_time = pd.to_numeric(chunk["unix_time"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(numeric_time).all() or not np.equal(
                numeric_time, np.rint(numeric_time)
            ).all():
                raise ValueError("unix_time must contain finite whole seconds")
            times = numeric_time.astype(np.int64)
            if np.any(np.diff(times) < 0) or (
                prior_global_time is not None and int(times[0]) < prior_global_time
            ):
                raise ValueError("feature streams are not globally chronological")
            prior_global_time = int(times[-1])
            keys = chunk["trans_num"].astype("string")
            if keys.isna().any() or keys.str.len().eq(0).any():
                raise ValueError("trans_num must contain nonempty identifiers")
            stop = cursor + len(chunk)
            if stop > total_rows:
                raise ValueError("feature streams exceed registered row count")
            for partition, (start_bound, stop_bound) in partition_bounds.items():
                local_start = max(cursor, start_bound)
                local_stop = min(stop, stop_bound)
                if local_start < local_stop:
                    start_in_chunk = local_start - cursor
                    stop_in_chunk = local_stop - cursor
                    key_digests[partition].update(
                        _key_hash_bytes(
                            chunk["trans_num"].iloc[start_in_chunk:stop_in_chunk]
                        )
                    )
            chunk_previous = pointer_builder.transform(
                cards.astype(str).tolist(),
                times,
                keys.astype(str).tolist(),
                start_index=cursor,
            )
            previous[cursor:stop] = chunk_previous
            for position, prior in enumerate(chunk_previous.tolist()):
                global_index = cursor + position
                if prior < 0:
                    partition = (
                        "train"
                        if global_index < train_rows
                        else "validation"
                        if global_index < development_rows
                        else "holdout"
                    )
                    cold_starts[partition] += 1
            cursor = stop
    if cursor != total_rows:
        raise ValueError(f"registered {total_rows} sequence rows but read {cursor}")
    for partition, digest in key_digests.items():
        if digest.hexdigest().upper() != split[partition]["ordered_key_sha256"]:
            raise ValueError(f"{partition} transaction order differs from split contract")
    _save_array_atomic(previous, output_path, overwrite=overwrite)
    content: dict[str, Any] = {
        "artifact_type": "fraud_sequence_index",
        "schema_version": SEQUENCE_INDEX_SCHEMA_VERSION,
        "index_file": output_path.name,
        "index_sha256": sha256_file(output_path),
        "dtype": "int32",
        "rows": total_rows,
        "partition_offsets": {
            name: [int(start), int(stop)]
            for name, (start, stop) in partition_bounds.items()
        },
        "cold_start_rows": cold_starts,
        "cards": pointer_builder.card_count,
        "strictly_prior_rule": "previous_index[row] references_a_strictly_earlier_same_card_timestamp_or_negative_one",
        "same_timestamp_rule": "tied_rows_share_the_same_predecessor_and_the_lexicographically_smallest_transaction_key_represents_the_bucket_for_future_rows",
        "model_data_manifest_sha256": model_data["payload_sha256"],
        "split_manifest_sha256": split["payload_sha256"],
        "development_feature_sha256": model_data["development_feature_sha256"],
        "holdout_feature_sha256": model_data["holdout_feature_sha256"],
        "ordered_key_sha256": {
            name: digest.hexdigest().upper() for name, digest in key_digests.items()
        },
    }
    document = {**content, "payload_sha256": json_digest(content)}
    atomic_write_json(document, manifest_path, overwrite=overwrite)
    return document


def load_sequence_index(output_dir: str | Path) -> SequenceIndex:
    """Load and verify the strictly prior per-card pointer registry."""

    import json

    root = Path(output_dir)
    manifest_path = root / "sequence_index_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("sequence manifest root must be an object")
    digest = document.get("payload_sha256")
    content = {key: value for key, value in document.items() if key != "payload_sha256"}
    if digest != json_digest(content):
        raise ValueError("sequence manifest digest does not match its content")
    if content.get("artifact_type") != "fraud_sequence_index":
        raise ValueError("unexpected sequence artifact type")
    if content.get("schema_version") != SEQUENCE_INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported sequence index schema version")
    index_path = root / content["index_file"]
    if sha256_file(index_path) != content["index_sha256"]:
        raise ValueError("sequence index digest differs from its registry")
    previous = np.load(index_path, allow_pickle=False)
    if previous.dtype != np.int32 or previous.shape != (int(content["rows"]),):
        raise ValueError("sequence index shape or dtype is invalid")
    positions = np.arange(len(previous), dtype=np.int64)
    if (previous < -1).any() or (previous.astype(np.int64) >= positions).any():
        raise ValueError("sequence index contains a non-causal pointer")
    offsets = {
        name: (int(bounds[0]), int(bounds[1]))
        for name, bounds in content["partition_offsets"].items()
    }
    return SequenceIndex(previous=previous, offsets=offsets, metadata=document)


class GlobalSparseAccessor:
    """Gather global transaction rows without stacking all sparse partitions."""

    def __init__(self, data: ModelDataset) -> None:
        self.partitions = (
            data.train_features,
            data.validation_features,
            data.holdout_features,
        )
        widths = {matrix.shape[1] for matrix in self.partitions}
        if len(widths) != 1:
            raise ValueError("sparse partitions have incompatible widths")
        self.width = widths.pop()
        train_stop = data.train_features.shape[0]
        validation_stop = train_stop + data.validation_features.shape[0]
        self.bounds = (
            (0, train_stop),
            (train_stop, validation_stop),
            (validation_stop, validation_stop + data.holdout_features.shape[0]),
        )
        self.rows = self.bounds[-1][1]

    def gather(self, indices: np.ndarray) -> np.ndarray:
        values = np.asarray(indices, dtype=np.int64)
        if values.ndim != 1 or ((values < 0) | (values >= self.rows)).any():
            raise ValueError("global sparse indices are out of range")
        output = np.empty((len(values), self.width), dtype=np.float32)
        for matrix, (start, stop) in zip(
            self.partitions, self.bounds, strict=True
        ):
            selected = (values >= start) & (values < stop)
            if selected.any():
                output[selected] = matrix[values[selected] - start].toarray().astype(
                    np.float32, copy=False
                )
        return output


def sequence_row_indices(
    current_indices: np.ndarray,
    previous: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left-aligned oldest-to-current indices and valid lengths."""

    current = np.asarray(current_indices, dtype=np.int64)
    if current.ndim != 1 or ((current < 0) | (current >= len(previous))).any():
        raise ValueError("current sequence indices are out of range")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    backward = np.full((len(current), sequence_length), -1, dtype=np.int64)
    cursor = current.copy()
    for position in range(sequence_length):
        backward[:, position] = cursor
        valid = cursor >= 0
        next_cursor = np.full(len(cursor), -1, dtype=np.int64)
        next_cursor[valid] = previous[cursor[valid]]
        cursor = next_cursor
    lengths = np.count_nonzero(backward >= 0, axis=1).astype(np.int64)
    positions = np.arange(sequence_length, dtype=np.int64)[None, :]
    source = lengths[:, None] - 1 - positions
    gathered = np.take_along_axis(backward, np.clip(source, 0, sequence_length - 1), axis=1)
    ordered = np.where(positions < lengths[:, None], gathered, -1)
    return ordered, lengths


def sequence_tensor_batch(
    accessor: GlobalSparseAccessor,
    current_indices: np.ndarray,
    previous: np.ndarray,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize one padded chronological sequence batch and its lengths."""

    row_indices, lengths = sequence_row_indices(
        current_indices, previous, sequence_length
    )
    output = np.zeros(
        (len(current_indices), sequence_length, accessor.width), dtype=np.float32
    )
    valid = row_indices >= 0
    output[valid] = accessor.gather(row_indices[valid])
    features = torch.from_numpy(output).to(
        device=device, non_blocking=device.type == "cuda"
    )
    return features, torch.from_numpy(lengths)
