"""Leakage-safe preprocessing, temporal partitioning, and imbalance handling.

The module separates chronological partitioning, train-fitted transformation,
and class-imbalance controls. Validation and holdout rows never contribute to
imputation values, clipping bounds, scaling statistics, category mappings, or
sampling decisions.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import logging
import math
import os
import platform
import tempfile
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn

from .eda import sha256_file

LOGGER = logging.getLogger(__name__)

SPLIT_SCHEMA_VERSION = 1
PREPROCESSOR_SCHEMA_VERSION = 1
IMBALANCE_REPORT_SCHEMA_VERSION = 1

DEFAULT_NUMERIC_COLUMNS = (
    "amt",
    "city_pop",
    "distance_card_merchant_km",
    "cc_txn_count_prev_1h",
    "cc_amt_sum_prev_1h",
    "cc_txn_count_prev_6h",
    "cc_amt_sum_prev_6h",
    "cc_txn_count_prev_24h",
    "cc_amt_sum_prev_24h",
    "cc_txn_count_prior",
    "cc_amt_sum_prior",
)

DEFAULT_LOG1P_COLUMNS = DEFAULT_NUMERIC_COLUMNS
DEFAULT_NOMINAL_COLUMNS = ("category", "state")
DEFAULT_FREQUENCY_COLUMNS = ("merchant", "city", "job", "zip")
DERIVED_NUMERIC_COLUMNS = (
    "age_years",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
)


def _runtime_dependencies() -> dict[str, str]:
    try:
        imbalanced_learn_version = distribution_version("imbalanced-learn")
    except PackageNotFoundError:
        imbalanced_learn_version = "not-installed"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": imbalanced_learn_version,
    }


def _json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _validated_sha256(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string SHA-256 digest")
    normalized = value.upper()
    if len(normalized) != 64 or any(
        character not in "0123456789ABCDEF" for character in normalized
    ):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256 digest")
    return normalized


def _document_with_digest(payload: dict[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output["payload_sha256"] = _json_digest(payload)
    return output


def _verify_document_digest(payload: dict[str, Any]) -> dict[str, Any]:
    expected = payload.get("payload_sha256")
    if not isinstance(expected, str):
        raise ValueError("JSON artifact does not contain payload_sha256")
    content = {key: value for key, value in payload.items() if key != "payload_sha256"}
    actual = _json_digest(content)
    if actual != expected:
        raise ValueError("JSON artifact digest does not match its content")
    return content


def _atomic_write_json(
    payload: dict[str, Any], path: str | Path, *, overwrite: bool = False
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        if str(destination).lower().endswith(".gz"):
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
        else:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if str(source).lower().endswith(".gz"):
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact root must be an object")
    return payload


@contextmanager
def _temporary_numeric_store(
    path: Path, *, shape: tuple[int, int]
) -> Iterator[np.memmap]:
    """Yield a disk-backed matrix and release its file mapping deterministically."""

    values = np.memmap(path, dtype=np.float64, mode="w+", shape=shape)
    try:
        yield values
    finally:
        values.flush()
        memory_map = getattr(values, "_mmap", None)
        if memory_map is not None:
            memory_map.close()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def _binary_target(values: Sequence[int] | pd.Series | np.ndarray) -> np.ndarray:
    target = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("target must contain only binary values 0 and 1")
    return target.astype(np.int8)


def _event_seconds(values: pd.Series, column: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(
        numeric, np.rint(numeric)
    ).all():
        raise ValueError(f"{column!r} must contain finite whole seconds")
    return numeric.astype(np.int64)


def _transaction_key_hashes(values: pd.Series, column: str) -> np.ndarray:
    normalized = values.astype("string")
    if normalized.isna().any() or normalized.str.len().eq(0).any():
        raise ValueError(f"{column!r} must contain nonempty transaction keys")
    hashes = pd.util.hash_pandas_object(
        normalized, index=False, categorize=False
    ).to_numpy(dtype=np.uint64)
    return hashes.astype("<u8", copy=False)


def _ordered_key_digest(hashes: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(hashes, dtype="<u8")
    return hashlib.sha256(contiguous.tobytes()).hexdigest().upper()


def _class_counts(target: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(target, return_counts=True)
    result = {int(value): int(count) for value, count in zip(values, counts, strict=True)}
    if set(result) != {0, 1}:
        raise ValueError("both target classes must be present")
    return result


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for chronological development partitioning."""

    validation_fraction: float = 0.20
    time_col: str = "unix_time"
    display_time_col: str = "trans_date_trans_time"
    target_col: str = "is_fraud"
    key_col: str = "trans_num"
    chunksize: int = 100_000

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.chunksize <= 0:
            raise ValueError("chunksize must be positive")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Declarative feature-selection and transformation contract."""

    numeric_columns: tuple[str, ...] = DEFAULT_NUMERIC_COLUMNS
    log1p_columns: tuple[str, ...] = DEFAULT_LOG1P_COLUMNS
    nominal_columns: tuple[str, ...] = DEFAULT_NOMINAL_COLUMNS
    frequency_columns: tuple[str, ...] = DEFAULT_FREQUENCY_COLUMNS
    display_time_col: str = "trans_date_trans_time"
    event_time_col: str = "unix_time"
    date_of_birth_col: str = "dob"
    target_col: str = "is_fraud"
    key_col: str = "trans_num"
    lower_quantile: float = 0.005
    upper_quantile: float = 0.995
    missing_token: str = "__MISSING__"
    unknown_token: str = "__UNKNOWN__"
    output_dtype: Literal["float32"] = "float32"
    random_state: int = 42

    def __post_init__(self) -> None:
        for field_name in (
            "numeric_columns",
            "log1p_columns",
            "nominal_columns",
            "frequency_columns",
        ):
            values = tuple(str(value) for value in getattr(self, field_name))
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
            object.__setattr__(self, field_name, values)
        if not set(self.log1p_columns).issubset(self.numeric_columns):
            raise ValueError("log1p_columns must be a subset of numeric_columns")
        groups = [
            set(self.numeric_columns),
            set(self.nominal_columns),
            set(self.frequency_columns),
        ]
        overlaps = any(
            groups[left].intersection(groups[right])
            for left in range(3)
            for right in range(left + 1, 3)
        )
        if overlaps:
            raise ValueError("numeric, nominal, and frequency columns must be disjoint")
        if not 0.0 <= self.lower_quantile < self.upper_quantile <= 1.0:
            raise ValueError("quantile bounds must satisfy 0 <= lower < upper <= 1")
        if self.missing_token == self.unknown_token:
            raise ValueError("missing_token and unknown_token must differ")
        if self.output_dtype != "float32":
            raise ValueError("output_dtype must be float32")


@dataclass(frozen=True)
class ImbalanceConfig:
    """Configuration for one mutually exclusive training imbalance control."""

    strategy: Literal["none", "class_weight", "random_under", "smotenc"] = "none"
    sampling_strategy: float = 0.10
    random_state: int = 42
    k_neighbors: int = 5
    max_output_rows: int = 2_000_000
    max_dense_bytes: int = 2_000_000_000

    def __post_init__(self) -> None:
        if self.strategy not in {"none", "class_weight", "random_under", "smotenc"}:
            raise ValueError(f"unsupported imbalance strategy: {self.strategy}")
        if not 0.0 < self.sampling_strategy <= 1.0:
            raise ValueError("sampling_strategy must be in (0, 1]")
        if self.k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        if self.max_output_rows <= 0 or self.max_dense_bytes <= 0:
            raise ValueError("memory guards must be positive")


@dataclass(frozen=True)
class TrainingBatch:
    """Encoded training data plus sampling and weighting metadata."""

    X: sparse.csr_matrix
    y: np.ndarray
    sample_weight: np.ndarray | None
    metadata: dict[str, Any]


def _partition_summary(
    times: np.ndarray,
    display_times: np.ndarray,
    target: np.ndarray,
    key_hashes: np.ndarray,
    start: int,
    stop: int,
) -> dict[str, Any]:
    selected_target = target[start:stop]
    counts = _class_counts(selected_target)
    return {
        "row_start": int(start),
        "row_stop_exclusive": int(stop),
        "rows": int(stop - start),
        "target_counts": {"0": counts[0], "1": counts[1]},
        "fraud_rate": counts[1] / (stop - start),
        "ordered_key_sha256": _ordered_key_digest(key_hashes[start:stop]),
        "unix_time_start": int(times[start]),
        "unix_time_end": int(times[stop - 1]),
        "display_time_start": pd.Timestamp(display_times[start]).strftime("%Y-%m-%d %H:%M:%S"),
        "display_time_end": pd.Timestamp(display_times[stop - 1]).strftime("%Y-%m-%d %H:%M:%S"),
    }


def _scan_partition_columns(
    path: Path, config: SplitConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    required = [
        config.time_col,
        config.display_time_col,
        config.target_col,
        config.key_col,
    ]
    time_parts: list[np.ndarray] = []
    display_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    key_hash_parts: list[np.ndarray] = []
    previous_time: int | None = None

    for chunk in pd.read_csv(path, usecols=required, chunksize=config.chunksize, low_memory=False):
        numeric_time = pd.to_numeric(chunk[config.time_col], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(numeric_time).all() or not np.equal(
            numeric_time, np.rint(numeric_time)
        ).all():
            raise ValueError(f"{config.time_col!r} must contain finite whole seconds")
        times = numeric_time.astype(np.int64)
        if np.any(np.diff(times) < 0) or (
            previous_time is not None and int(times[0]) < previous_time
        ):
            raise ValueError(f"{path.name} is not nondecreasing by {config.time_col}")
        previous_time = int(times[-1])

        display = pd.to_datetime(
            chunk[config.display_time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        if display.isna().any():
            raise ValueError(f"{config.display_time_col!r} contains invalid timestamps")
        target = _binary_target(chunk[config.target_col])
        key_hashes = _transaction_key_hashes(chunk[config.key_col], config.key_col)
        time_parts.append(times)
        display_parts.append(display.to_numpy(dtype="datetime64[ns]"))
        target_parts.append(target)
        key_hash_parts.append(key_hashes)

    if not time_parts:
        raise ValueError(f"dataset contains no rows: {path}")
    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    combined_keys = np.concatenate(key_hash_parts)
    if len(np.unique(combined_keys)) != len(combined_keys):
        raise ValueError(f"{config.key_col!r} must be unique within {path.name}")
    return (
        np.concatenate(time_parts),
        np.concatenate(display_parts),
        np.concatenate(target_parts),
        combined_keys,
        header,
    )


def build_chronological_split_manifest(
    development_path: str | Path,
    holdout_path: str | Path,
    *,
    config: SplitConfig = SplitConfig(),
) -> dict[str, Any]:
    """Return a fingerprinted train/validation/holdout boundary manifest."""

    development = Path(development_path)
    holdout = Path(holdout_path)
    if not development.is_file() or not holdout.is_file():
        raise FileNotFoundError("development and holdout CSV files must exist")

    dev_times, dev_display, dev_target, dev_keys, dev_header = _scan_partition_columns(
        development, config
    )
    hold_times, hold_display, hold_target, hold_keys, hold_header = _scan_partition_columns(
        holdout, config
    )
    if dev_header != hold_header:
        raise ValueError("development and holdout schemas differ")
    if np.intersect1d(dev_keys, hold_keys, assume_unique=True).size:
        raise ValueError(f"{config.key_col!r} must be unique across input files")
    if int(dev_times[-1]) >= int(hold_times[0]):
        raise ValueError("holdout must begin strictly after the development stream")

    requested_position = int(math.floor(len(dev_times) * (1.0 - config.validation_fraction)))
    if requested_position <= 0 or requested_position >= len(dev_times):
        raise ValueError("requested split produces an empty partition")
    boundary_time = int(dev_times[requested_position])
    boundary_position = int(np.searchsorted(dev_times, boundary_time, side="left"))
    if boundary_position <= 0 or boundary_position >= len(dev_times):
        raise ValueError("timestamp-bucket preservation produces an empty partition")

    train_summary = _partition_summary(
        dev_times, dev_display, dev_target, dev_keys, 0, boundary_position
    )
    validation_summary = _partition_summary(
        dev_times,
        dev_display,
        dev_target,
        dev_keys,
        boundary_position,
        len(dev_times),
    )
    holdout_summary = _partition_summary(
        hold_times, hold_display, hold_target, hold_keys, 0, len(hold_times)
    )

    payload = {
        "artifact_type": "chronological_split_manifest",
        "schema_version": SPLIT_SCHEMA_VERSION,
        "config": asdict(config),
        "algorithm": "final_fraction_by_row_with_whole_unix_time_bucket",
        "key_hash_algorithm": "pandas_hash_object_uint64_then_sha256",
        "requested_boundary_position": requested_position,
        "effective_boundary_position": boundary_position,
        "boundary_unix_time": boundary_time,
        "development": {
            "path": str(development.resolve()),
            "size_bytes": development.stat().st_size,
            "sha256": sha256_file(development),
            "schema_sha256": _json_digest({"columns": dev_header}),
            "columns": dev_header,
            "rows": len(dev_times),
        },
        "train": train_summary,
        "validation": validation_summary,
        "holdout": {
            "path": str(holdout.resolve()),
            "size_bytes": holdout.stat().st_size,
            "sha256": sha256_file(holdout),
            "schema_sha256": _json_digest({"columns": hold_header}),
            "columns": hold_header,
            **holdout_summary,
        },
        "holdout_gap_seconds": int(hold_times[0] - dev_times[-1]),
        "holdout_policy": "decision_isolated_out_of_time_evaluation",
    }
    return _document_with_digest(payload)


def save_split_manifest(
    manifest: dict[str, Any], path: str | Path, *, overwrite: bool = False
) -> Path:
    content = _verify_document_digest(manifest)
    if content.get("artifact_type") != "chronological_split_manifest":
        raise ValueError("unexpected split artifact type")
    if content.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError("unsupported split manifest schema version")
    return _atomic_write_json(manifest, path, overwrite=overwrite)


def _validated_split_manifest_content(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    content = _verify_document_digest(dict(manifest))
    if content.get("artifact_type") != "chronological_split_manifest":
        raise ValueError("unexpected split artifact type")
    if content.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError("unsupported split manifest schema version")
    return content


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    _validated_split_manifest_content(payload)
    return payload


def chronological_train_validation_split(
    frame: pd.DataFrame, *, config: SplitConfig = SplitConfig()
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split an ordered frame without dividing a timestamp bucket."""

    _require_columns(frame, (config.time_col, config.target_col))
    if config.key_col in frame.columns and frame[config.key_col].duplicated().any():
        raise ValueError(f"{config.key_col!r} must be unique")
    times_numeric = pd.to_numeric(frame[config.time_col], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(times_numeric).all() or not np.equal(
        times_numeric, np.rint(times_numeric)
    ).all():
        raise ValueError(f"{config.time_col!r} must contain finite whole seconds")
    times = times_numeric.astype(np.int64)
    if np.any(np.diff(times) < 0):
        raise ValueError(f"frame is not nondecreasing by {config.time_col}")
    target = _binary_target(frame[config.target_col])

    requested = int(math.floor(len(frame) * (1.0 - config.validation_fraction)))
    if requested <= 0 or requested >= len(frame):
        raise ValueError("requested split produces an empty partition")
    boundary_time = int(times[requested])
    boundary = int(np.searchsorted(times, boundary_time, side="left"))
    if boundary <= 0 or boundary >= len(frame):
        raise ValueError("timestamp-bucket preservation produces an empty partition")
    train_counts = _class_counts(target[:boundary])
    validation_counts = _class_counts(target[boundary:])
    metadata = {
        "requested_boundary_position": requested,
        "effective_boundary_position": boundary,
        "boundary_unix_time": boundary_time,
        "train_rows": boundary,
        "validation_rows": len(frame) - boundary,
        "train_target_counts": train_counts,
        "validation_target_counts": validation_counts,
    }
    return frame.iloc[:boundary].copy(), frame.iloc[boundary:].copy(), metadata


class FraudPreprocessor:
    """Auditable train-fitted transformer with sparse numeric output."""

    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()) -> None:
        self.config = config
        self.numeric_stats_: dict[str, dict[str, float | bool]] = {}
        self.nominal_vocabularies_: dict[str, list[str]] = {}
        self.frequency_mappings_: dict[str, dict[str, float]] = {}
        self.feature_names_: list[str] = []
        self.fitted_context_: dict[str, Any] = {}
        self.dependencies_: dict[str, str] = _runtime_dependencies()
        self.is_fitted_ = False

    @property
    def required_input_columns(self) -> tuple[str, ...]:
        values = [
            *self.config.numeric_columns,
            *self.config.nominal_columns,
            *self.config.frequency_columns,
            self.config.display_time_col,
            self.config.event_time_col,
            self.config.date_of_birth_col,
            self.config.key_col,
        ]
        return tuple(dict.fromkeys(values))

    @property
    def sampler_nominal_columns(self) -> list[str]:
        return [f"cat__{column}" for column in self.config.nominal_columns]

    def _normalize_categories(self, values: pd.Series, column: str) -> pd.Series:
        nonmissing = values.dropna().astype("string")
        forbidden = {self.config.missing_token, self.config.unknown_token}
        collisions = forbidden.intersection(set(nonmissing.astype(str).unique()))
        if collisions:
            raise ValueError(
                f"{column!r} contains reserved category tokens: {sorted(collisions)}"
            )
        return values.astype("string").fillna(self.config.missing_token).astype(str)

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        _require_columns(frame, self.required_input_columns)
        prepared = pd.DataFrame(index=frame.index)

        for column in self.config.numeric_columns:
            original = frame[column]
            numeric = pd.to_numeric(original, errors="coerce")
            malformed = original.notna() & numeric.isna()
            if malformed.any():
                raise ValueError(f"{column!r} contains nonnumeric values")
            finite_or_missing = numeric.isna() | np.isfinite(numeric.to_numpy(dtype=np.float64))
            if not bool(np.all(finite_or_missing)):
                raise ValueError(f"{column!r} contains non-finite values")
            if column == "amt" and (numeric.dropna() <= 0).any():
                raise ValueError("amt must be strictly positive when present")
            if (
                column.startswith("cc_txn_count")
                or column.startswith("cc_amt_sum")
                or column in {"city_pop", "distance_card_merchant_km"}
            ) and (numeric.dropna() < 0).any():
                raise ValueError(f"{column!r} must be nonnegative when present")
            prepared[column] = numeric.astype(np.float64)

        display_time = pd.to_datetime(
            frame[self.config.display_time_col],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        malformed_time = frame[self.config.display_time_col].notna() & display_time.isna()
        if malformed_time.any():
            raise ValueError(f"{self.config.display_time_col!r} contains malformed values")
        birth_date = pd.to_datetime(
            frame[self.config.date_of_birth_col], format="%Y-%m-%d", errors="coerce"
        )
        malformed_birth = frame[self.config.date_of_birth_col].notna() & birth_date.isna()
        if malformed_birth.any():
            raise ValueError(f"{self.config.date_of_birth_col!r} contains malformed values")

        age = (display_time - birth_date).dt.days / 365.2425
        if (age.dropna() < 0).any():
            raise ValueError("date_of_birth occurs after transaction time")
        prepared["age_years"] = age.astype(np.float64)
        hour = display_time.dt.hour.astype(np.float64)
        day = display_time.dt.dayofweek.astype(np.float64)
        month = display_time.dt.month.astype(np.float64) - 1.0
        prepared["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        prepared["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        prepared["day_sin"] = np.sin(2.0 * np.pi * day / 7.0)
        prepared["day_cos"] = np.cos(2.0 * np.pi * day / 7.0)
        prepared["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        prepared["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

        for column in (*self.config.nominal_columns, *self.config.frequency_columns):
            prepared[column] = self._normalize_categories(frame[column], column)
        return prepared

    def _complete_fit(
        self,
        numeric_values: Mapping[str, np.ndarray],
        nominal_values: Mapping[str, set[str]],
        frequency_counts: Mapping[str, Counter[str]],
        *,
        row_count: int,
        context: dict[str, Any],
    ) -> None:
        numeric_names = [*self.config.numeric_columns, *DERIVED_NUMERIC_COLUMNS]
        self.numeric_stats_ = {}
        for column in numeric_names:
            values = np.asarray(numeric_values[column], dtype=np.float64)
            if values.shape != (row_count,):
                raise ValueError(f"{column!r} does not match the training row count")
            observed = values[np.isfinite(values)]
            if observed.size == 0:
                raise ValueError(f"{column!r} has no observed training values")
            median = float(np.median(observed))
            filled = np.where(np.isfinite(values), values, median)
            apply_log = column in self.config.log1p_columns
            if apply_log:
                if np.any(filled < 0):
                    raise ValueError(
                        f"{column!r} contains negative values incompatible with log1p"
                    )
                filled = np.log1p(filled)
            lower = float(np.quantile(filled, self.config.lower_quantile))
            upper = float(np.quantile(filled, self.config.upper_quantile))
            clipped = np.clip(filled, lower, upper)
            mean = float(clipped.mean())
            scale = float(clipped.std(ddof=0))
            if not np.isfinite(scale) or scale < 1e-12:
                scale = 1.0
            self.numeric_stats_[column] = {
                "median": median,
                "lower": lower,
                "upper": upper,
                "mean": mean,
                "scale": scale,
                "log1p": apply_log,
            }

        self.nominal_vocabularies_ = {}
        for column in self.config.nominal_columns:
            observed_values = set(nominal_values[column])
            observed_values.discard(self.config.missing_token)
            observed_values.discard(self.config.unknown_token)
            self.nominal_vocabularies_[column] = [
                *sorted(observed_values),
                self.config.missing_token,
                self.config.unknown_token,
            ]

        self.frequency_mappings_ = {}
        for column in self.config.frequency_columns:
            counts = frequency_counts[column]
            if sum(counts.values()) != row_count:
                raise ValueError(f"{column!r} frequency counts do not match training rows")
            self.frequency_mappings_[column] = {
                value: count / row_count for value, count in sorted(counts.items())
            }

        self.feature_names_ = []
        for column in numeric_names:
            prefix = "log1p_" if bool(self.numeric_stats_[column]["log1p"]) else ""
            self.feature_names_.append(f"num__{prefix}{column}")
        for column in self.config.frequency_columns:
            self.feature_names_.extend(
                [f"freq__{column}", f"freq__{column}__unknown"]
            )
        for column in self.config.nominal_columns:
            self.feature_names_.extend(
                f"cat__{column}__{value}"
                for value in self.nominal_vocabularies_[column]
            )
        self.fitted_context_ = {
            **context,
            "partition": "train",
            "row_count": int(row_count),
            "feature_schema_sha256": _json_digest({"feature_names": self.feature_names_}),
        }
        self.dependencies_ = _runtime_dependencies()
        self.is_fitted_ = True

    def _fit_prepared(
        self,
        prepared_chunks: Sequence[pd.DataFrame],
        *,
        row_count: int,
        context: dict[str, Any],
    ) -> None:
        numeric_names = [*self.config.numeric_columns, *DERIVED_NUMERIC_COLUMNS]
        numeric_values = {
            column: np.concatenate(
                [chunk[column].to_numpy(dtype=np.float64, copy=True) for chunk in prepared_chunks]
            )
            for column in numeric_names
        }
        nominal_values = {
            column: {
                value
                for chunk in prepared_chunks
                for value in chunk[column].astype(str).unique().tolist()
            }
            for column in self.config.nominal_columns
        }
        frequency_counts: dict[str, Counter[str]] = {
            column: Counter() for column in self.config.frequency_columns
        }
        for chunk in prepared_chunks:
            for column in self.config.frequency_columns:
                frequency_counts[column].update(chunk[column].astype(str).tolist())
        self._complete_fit(
            numeric_values,
            nominal_values,
            frequency_counts,
            row_count=row_count,
            context=context,
        )

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        partition: str,
        source_sha256: str | None = None,
        split_manifest_sha256: str | None = None,
        split_manifest: Mapping[str, Any] | None = None,
    ) -> "FraudPreprocessor":
        """Fit transformation state from an explicitly identified train frame."""

        if partition != "train":
            raise ValueError("preprocessing fit is permitted only for partition='train'")
        if frame.empty:
            raise ValueError("training frame must not be empty")
        source_sha256 = _validated_sha256(source_sha256, "source_sha256")
        split_manifest_sha256 = _validated_sha256(
            split_manifest_sha256, "split_manifest_sha256"
        )
        manifest_train: dict[str, Any] | None = None
        if split_manifest is not None:
            manifest_content = _validated_split_manifest_content(split_manifest)
            manifest_config = dict(manifest_content["config"])
            if manifest_config.get("time_col") != self.config.event_time_col:
                raise ValueError("split manifest event-time column is incompatible")
            if manifest_config.get("target_col") != self.config.target_col:
                raise ValueError("split manifest target column is incompatible")
            manifest_digest = _validated_sha256(
                split_manifest.get("payload_sha256"), "split_manifest_sha256"
            )
            if (
                split_manifest_sha256 is not None
                and split_manifest_sha256 != manifest_digest
            ):
                raise ValueError("split manifest digest arguments differ")
            split_manifest_sha256 = manifest_digest
            manifest_train = dict(manifest_content["train"])
        prepared = self._prepare_frame(frame)
        if frame[self.config.key_col].duplicated().any():
            raise ValueError(f"{self.config.key_col!r} must be unique in training data")
        key_digest = _ordered_key_digest(
            _transaction_key_hashes(frame[self.config.key_col], self.config.key_col)
        )
        integer_times = _event_seconds(
            frame[self.config.event_time_col], self.config.event_time_col
        )
        if np.any(np.diff(integer_times) < 0):
            raise ValueError("training frame is not nondecreasing by event time")
        if manifest_train is not None:
            if len(frame) != int(manifest_train["rows"]):
                raise ValueError("training frame row count differs from the split manifest")
            if (
                int(integer_times[0]) != int(manifest_train["unix_time_start"])
                or int(integer_times[-1]) != int(manifest_train["unix_time_end"])
            ):
                raise ValueError("training frame time range differs from the split manifest")
            if self.config.target_col not in frame.columns:
                raise ValueError("manifest-bound fitting requires the training target column")
            target_counts = _class_counts(_binary_target(frame[self.config.target_col]))
            expected_counts = {
                int(label): int(count)
                for label, count in manifest_train["target_counts"].items()
            }
            if target_counts != expected_counts:
                raise ValueError("training target counts differ from the split manifest")
            if key_digest != manifest_train.get("ordered_key_sha256"):
                raise ValueError("training row order differs from the split manifest")
        context = {
            "source_sha256": source_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "unix_time_min": int(integer_times[0]),
            "unix_time_max": int(integer_times[-1]),
            "ordered_key_sha256": key_digest,
        }
        self._fit_prepared([prepared], row_count=len(frame), context=context)
        return self

    def fit_csv(
        self,
        path: str | Path,
        *,
        train_rows: int | None = None,
        partition: str,
        chunksize: int = 100_000,
        source_sha256: str | None = None,
        split_manifest_sha256: str | None = None,
        split_manifest: Mapping[str, Any] | None = None,
    ) -> "FraudPreprocessor":
        """Fit from the leading training rows of a feature-enriched CSV."""

        if partition != "train":
            raise ValueError("preprocessing fit is permitted only for partition='train'")
        if chunksize <= 0:
            raise ValueError("chunksize must be positive")
        split_manifest_sha256 = _validated_sha256(
            split_manifest_sha256, "split_manifest_sha256"
        )
        manifest_train: dict[str, Any] | None = None
        if split_manifest is not None:
            manifest_content = _validated_split_manifest_content(split_manifest)
            manifest_config = dict(manifest_content["config"])
            if manifest_config.get("time_col") != self.config.event_time_col:
                raise ValueError("split manifest event-time column is incompatible")
            if manifest_config.get("target_col") != self.config.target_col:
                raise ValueError("split manifest target column is incompatible")
            manifest_digest = _validated_sha256(
                split_manifest.get("payload_sha256"), "split_manifest_sha256"
            )
            if (
                split_manifest_sha256 is not None
                and split_manifest_sha256 != manifest_digest
            ):
                raise ValueError("split manifest digest arguments differ")
            split_manifest_sha256 = manifest_digest
            manifest_train = dict(manifest_content["train"])
            manifest_rows = int(manifest_train["rows"])
            if train_rows is not None and train_rows != manifest_rows:
                raise ValueError("train_rows differs from the split manifest")
            train_rows = manifest_rows
        if (
            isinstance(train_rows, bool)
            or not isinstance(train_rows, int)
            or train_rows <= 0
        ):
            raise ValueError("train_rows must be a positive integer")
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        computed_source_sha256 = sha256_file(source)
        source_sha256 = _validated_sha256(source_sha256, "source_sha256")
        if source_sha256 is not None and source_sha256 != computed_source_sha256:
            raise ValueError("source_sha256 does not match the feature CSV")
        source_sha256 = computed_source_sha256
        source_columns = pd.read_csv(source, nrows=0).columns.astype(str).tolist()
        has_target = self.config.target_col in source_columns
        if manifest_train is not None and not has_target:
            raise ValueError("manifest-bound fitting requires the training target column")
        string_columns = {
            column: "string"
            for column in (
                *self.config.nominal_columns,
                *self.config.frequency_columns,
                self.config.display_time_col,
                self.config.date_of_birth_col,
                self.config.key_col,
            )
        }
        rows_seen = 0
        time_min: int | None = None
        time_max: int | None = None
        previous_time: int | None = None
        target_counts: Counter[int] = Counter()
        key_digest = hashlib.sha256()
        numeric_names = [*self.config.numeric_columns, *DERIVED_NUMERIC_COLUMNS]
        nominal_values: dict[str, set[str]] = {
            column: set() for column in self.config.nominal_columns
        }
        frequency_counts: dict[str, Counter[str]] = {
            column: Counter() for column in self.config.frequency_columns
        }
        with tempfile.TemporaryDirectory(
            prefix="fraudshield-preprocessor-"
        ) as temp_dir, _temporary_numeric_store(
            Path(temp_dir) / "numeric_values.dat",
            shape=(train_rows, len(numeric_names)),
        ) as numeric_store:
            reader = pd.read_csv(
                source,
                usecols=[
                    *self.required_input_columns,
                    *([self.config.target_col] if has_target else []),
                ],
                dtype=string_columns,
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in reader:
                remaining = train_rows - rows_seen
                if remaining <= 0:
                    break
                chunk = chunk.iloc[:remaining].copy()
                times = pd.to_numeric(
                    chunk[self.config.event_time_col], errors="coerce"
                ).to_numpy(dtype=np.float64)
                if not np.isfinite(times).all() or not np.equal(
                    times, np.rint(times)
                ).all():
                    raise ValueError("event time contains invalid values")
                integer_times = times.astype(np.int64)
                if np.any(np.diff(integer_times) < 0) or (
                    previous_time is not None and int(integer_times[0]) < previous_time
                ):
                    raise ValueError("training CSV is not nondecreasing by event time")
                previous_time = int(integer_times[-1])
                time_min = int(integer_times[0]) if time_min is None else time_min
                time_max = int(integer_times[-1])
                key_hashes = _transaction_key_hashes(
                    chunk[self.config.key_col], self.config.key_col
                )
                if len(np.unique(key_hashes)) != len(key_hashes):
                    raise ValueError(
                        f"{self.config.key_col!r} must be unique within each CSV chunk"
                    )
                key_digest.update(np.ascontiguousarray(key_hashes, dtype="<u8").tobytes())
                if has_target:
                    target_counts.update(
                        _binary_target(chunk[self.config.target_col]).tolist()
                    )

                prepared = self._prepare_frame(chunk)
                stop = rows_seen + len(prepared)
                for position, column in enumerate(numeric_names):
                    numeric_store[rows_seen:stop, position] = prepared[column].to_numpy(
                        dtype=np.float64, copy=False
                    )
                for column in self.config.nominal_columns:
                    nominal_values[column].update(
                        prepared[column].astype(str).unique().tolist()
                    )
                for column in self.config.frequency_columns:
                    frequency_counts[column].update(prepared[column].astype(str).tolist())
                rows_seen = stop
            if rows_seen != train_rows:
                raise ValueError(f"requested {train_rows} training rows but read {rows_seen}")
            if manifest_train is not None:
                if (
                    time_min != int(manifest_train["unix_time_start"])
                    or time_max != int(manifest_train["unix_time_end"])
                ):
                    raise ValueError("training CSV time range differs from the split manifest")
                expected_counts = {
                    int(label): int(count)
                    for label, count in manifest_train["target_counts"].items()
                }
                if dict(target_counts) != expected_counts:
                    raise ValueError("training target counts differ from the split manifest")
                if key_digest.hexdigest().upper() != manifest_train.get(
                    "ordered_key_sha256"
                ):
                    raise ValueError("training row order differs from the split manifest")
            numeric_store.flush()
            context = {
                "source_path": str(source.resolve()),
                "source_sha256": source_sha256,
                "split_manifest_sha256": split_manifest_sha256,
                "unix_time_min": time_min,
                "unix_time_max": time_max,
                "ordered_key_sha256": key_digest.hexdigest().upper(),
                "numeric_fit_storage": "temporary_memmap",
            }
            numeric_values = {
                column: numeric_store[:, position]
                for position, column in enumerate(numeric_names)
            }
            self._complete_fit(
                numeric_values,
                nominal_values,
                frequency_counts,
                row_count=rows_seen,
                context=context,
            )
            del numeric_values
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("preprocessor is not fitted")

    def _numeric_matrix(self, prepared: pd.DataFrame) -> np.ndarray:
        numeric_names = [*self.config.numeric_columns, *DERIVED_NUMERIC_COLUMNS]
        output = np.empty((len(prepared), len(numeric_names)), dtype=np.float32)
        for position, column in enumerate(numeric_names):
            stats = self.numeric_stats_[column]
            values = prepared[column].to_numpy(dtype=np.float64, copy=True)
            values = np.where(np.isfinite(values), values, float(stats["median"]))
            if bool(stats["log1p"]):
                values = np.log1p(values)
            values = np.clip(values, float(stats["lower"]), float(stats["upper"]))
            output[:, position] = (
                (values - float(stats["mean"])) / float(stats["scale"])
            ).astype(np.float32)
        return output

    def _frequency_matrix(self, prepared: pd.DataFrame) -> np.ndarray:
        output = np.empty(
            (len(prepared), 2 * len(self.config.frequency_columns)), dtype=np.float32
        )
        for position, column in enumerate(self.config.frequency_columns):
            mapping = self.frequency_mappings_[column]
            values = prepared[column].astype(str)
            known = values.isin(mapping)
            output[:, 2 * position] = values.map(mapping).fillna(0.0).to_numpy(dtype=np.float32)
            output[:, 2 * position + 1] = (~known).to_numpy(dtype=np.float32)
        return output

    def _mapped_nominal(self, prepared: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=prepared.index)
        for column in self.config.nominal_columns:
            vocabulary = set(self.nominal_vocabularies_[column])
            values = prepared[column].astype(str)
            output[f"cat__{column}"] = values.where(
                values.isin(vocabulary), self.config.unknown_token
            )
        return output

    def _encode_components(
        self,
        numeric: np.ndarray,
        frequency: np.ndarray,
        nominal: pd.DataFrame,
    ) -> sparse.csr_matrix:
        row_indices: list[np.ndarray] = []
        column_indices: list[np.ndarray] = []
        offset = 0
        for column in self.config.nominal_columns:
            vocabulary = self.nominal_vocabularies_[column]
            lookup = {value: index for index, value in enumerate(vocabulary)}
            mapped = nominal[f"cat__{column}"].astype(str).map(lookup)
            if mapped.isna().any():
                raise ValueError(f"sampler output contains invalid category for {column!r}")
            row_indices.append(np.arange(len(nominal), dtype=np.int64))
            column_indices.append(mapped.to_numpy(dtype=np.int64) + offset)
            offset += len(vocabulary)
        if row_indices:
            rows = np.concatenate(row_indices)
            columns = np.concatenate(column_indices)
            data = np.ones(len(rows), dtype=np.float32)
            categorical = sparse.csr_matrix(
                (data, (rows, columns)), shape=(len(nominal), offset), dtype=np.float32
            )
        else:
            categorical = sparse.csr_matrix((len(nominal), 0), dtype=np.float32)
        return sparse.hstack(
            [
                sparse.csr_matrix(numeric, dtype=np.float32),
                sparse.csr_matrix(frequency, dtype=np.float32),
                categorical,
            ],
            format="csr",
            dtype=np.float32,
        )

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        """Transform rows with frozen training statistics and schema."""

        self._check_fitted()
        prepared = self._prepare_frame(frame)
        return self._encode_components(
            self._numeric_matrix(prepared),
            self._frequency_matrix(prepared),
            self._mapped_nominal(prepared),
        )

    def prepare_sampler_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return scaled continuous fields and unencoded low-cardinality fields."""

        self._check_fitted()
        prepared = self._prepare_frame(frame)
        numeric = self._numeric_matrix(prepared)
        frequency = self._frequency_matrix(prepared)
        numeric_names = self.feature_names_[: numeric.shape[1]]
        frequency_start = numeric.shape[1]
        frequency_names = self.feature_names_[
            frequency_start : frequency_start + frequency.shape[1]
        ]
        output = pd.DataFrame(numeric, columns=numeric_names, index=frame.index)
        for index, name in enumerate(frequency_names):
            output[name] = frequency[:, index]
        nominal = self._mapped_nominal(prepared)
        for name in nominal.columns:
            output[name] = nominal[name].astype("category")
        return output.reset_index(drop=True)

    def transform_sampler_frame(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        """Encode a cleaned frame returned by a mixed-data sampler."""

        self._check_fitted()
        numeric_count = len(self.config.numeric_columns) + len(DERIVED_NUMERIC_COLUMNS)
        frequency_count = 2 * len(self.config.frequency_columns)
        expected = [
            *self.feature_names_[:numeric_count],
            *self.feature_names_[numeric_count : numeric_count + frequency_count],
            *self.sampler_nominal_columns,
        ]
        _require_columns(frame, expected)
        numeric = frame[expected[:numeric_count]].apply(pd.to_numeric, errors="raise").to_numpy(
            dtype=np.float32
        )
        frequency = frame[
            expected[numeric_count : numeric_count + frequency_count]
        ].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
        nominal = frame[self.sampler_nominal_columns].copy()
        return self._encode_components(numeric, frequency, nominal)

    def _reconstructed_feature_names(self) -> list[str]:
        numeric_names = [*self.config.numeric_columns, *DERIVED_NUMERIC_COLUMNS]
        names = [
            (
                f"num__log1p_{column}"
                if bool(self.numeric_stats_[column]["log1p"])
                else f"num__{column}"
            )
            for column in numeric_names
        ]
        for column in self.config.frequency_columns:
            names.extend([f"freq__{column}", f"freq__{column}__unknown"])
        for column in self.config.nominal_columns:
            names.extend(
                f"cat__{column}__{value}"
                for value in self.nominal_vocabularies_[column]
            )
        return names

    def _validate_fitted_state(self) -> None:
        numeric_names = [*self.config.numeric_columns, *DERIVED_NUMERIC_COLUMNS]
        if set(self.numeric_stats_) != set(numeric_names):
            raise ValueError("preprocessing numeric-statistic keys are invalid")
        expected_stat_keys = {"median", "lower", "upper", "mean", "scale", "log1p"}
        for column in numeric_names:
            stats = self.numeric_stats_[column]
            if not isinstance(stats, dict) or set(stats) != expected_stat_keys:
                raise ValueError(f"preprocessing statistics for {column!r} are invalid")
            if not isinstance(stats["log1p"], bool):
                raise ValueError(f"preprocessing log1p flag for {column!r} is invalid")
            if stats["log1p"] != (column in self.config.log1p_columns):
                raise ValueError(f"preprocessing log1p flag for {column!r} is inconsistent")
            for field in ("median", "lower", "upper", "mean", "scale"):
                value = stats[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"preprocessing statistic {column!r}.{field} is not numeric"
                    )
                if not math.isfinite(float(value)):
                    raise ValueError(
                        f"preprocessing statistic {column!r}.{field} is not finite"
                    )
            lower = float(stats["lower"])
            upper = float(stats["upper"])
            mean = float(stats["mean"])
            if lower > upper or not lower - 1e-12 <= mean <= upper + 1e-12:
                raise ValueError(f"preprocessing bounds for {column!r} are invalid")
            if float(stats["scale"]) <= 0.0:
                raise ValueError(f"preprocessing scale for {column!r} must be positive")
            if bool(stats["log1p"]) and float(stats["median"]) < 0.0:
                raise ValueError(f"preprocessing median for {column!r} is incompatible")

        if set(self.nominal_vocabularies_) != set(self.config.nominal_columns):
            raise ValueError("preprocessing nominal-vocabulary keys are invalid")
        for column in self.config.nominal_columns:
            vocabulary = self.nominal_vocabularies_[column]
            if (
                not isinstance(vocabulary, list)
                or len(vocabulary) < 2
                or not all(isinstance(value, str) for value in vocabulary)
                or len(vocabulary) != len(set(vocabulary))
                or vocabulary[-2:] != [
                    self.config.missing_token,
                    self.config.unknown_token,
                ]
                or vocabulary[:-2] != sorted(vocabulary[:-2])
            ):
                raise ValueError(f"preprocessing vocabulary for {column!r} is invalid")

        if set(self.frequency_mappings_) != set(self.config.frequency_columns):
            raise ValueError("preprocessing frequency-mapping keys are invalid")
        for column in self.config.frequency_columns:
            mapping = self.frequency_mappings_[column]
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"preprocessing frequency mapping for {column!r} is invalid")
            if not all(isinstance(value, str) for value in mapping):
                raise ValueError(f"preprocessing frequency keys for {column!r} are invalid")
            rates = list(mapping.values())
            if any(
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
                or not 0.0 < float(rate) <= 1.0
                for rate in rates
            ) or not math.isclose(
                math.fsum(float(rate) for rate in rates),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(f"preprocessing frequency rates for {column!r} are invalid")

        expected_names = self._reconstructed_feature_names()
        if self.feature_names_ != expected_names or len(expected_names) != len(
            set(expected_names)
        ):
            raise ValueError("preprocessing feature names are inconsistent")
        context = self.fitted_context_
        if not isinstance(context, dict) or context.get("partition") != "train":
            raise ValueError("preprocessing fitted context is invalid")
        row_count = context.get("row_count")
        time_min = context.get("unix_time_min")
        time_max = context.get("unix_time_max")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count <= 0
            or isinstance(time_min, bool)
            or not isinstance(time_min, int)
            or isinstance(time_max, bool)
            or not isinstance(time_max, int)
            or time_min > time_max
        ):
            raise ValueError("preprocessing fitted dimensions are invalid")
        expected_schema = _json_digest({"feature_names": expected_names})
        if context.get("feature_schema_sha256") != expected_schema:
            raise ValueError("preprocessing feature schema digest is invalid")
        ordered_key_sha256 = _validated_sha256(
            context.get("ordered_key_sha256"), "ordered_key_sha256"
        )
        if ordered_key_sha256 is None or ordered_key_sha256 != context.get(
            "ordered_key_sha256"
        ):
            raise ValueError("preprocessing ordered-key digest is invalid")
        for field in ("source_sha256", "split_manifest_sha256"):
            if _validated_sha256(context.get(field), field) != context.get(field):
                raise ValueError(f"preprocessing {field} is not normalized")
        if not isinstance(self.dependencies_, dict) or not all(
            isinstance(name, str)
            and bool(name)
            and isinstance(version, str)
            and bool(version)
            for name, version in self.dependencies_.items()
        ):
            raise ValueError("preprocessing dependency metadata is invalid")

    def get_feature_names_out(self) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self.feature_names_, dtype=object)

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        self._validate_fitted_state()
        payload = {
            "artifact_type": "fraud_preprocessor",
            "schema_version": PREPROCESSOR_SCHEMA_VERSION,
            "config": asdict(self.config),
            "required_input_columns": list(self.required_input_columns),
            "feature_names": copy.deepcopy(self.feature_names_),
            "numeric_stats": copy.deepcopy(self.numeric_stats_),
            "nominal_vocabularies": copy.deepcopy(self.nominal_vocabularies_),
            "frequency_mappings": copy.deepcopy(self.frequency_mappings_),
            "fitted_context": copy.deepcopy(self.fitted_context_),
            "dependencies": copy.deepcopy(self.dependencies_),
        }
        return _document_with_digest(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FraudPreprocessor":
        content = _verify_document_digest(payload)
        expected_keys = {
            "artifact_type",
            "schema_version",
            "config",
            "required_input_columns",
            "feature_names",
            "numeric_stats",
            "nominal_vocabularies",
            "frequency_mappings",
            "fitted_context",
            "dependencies",
        }
        if set(content) != expected_keys:
            raise ValueError("preprocessing artifact fields are invalid")
        if content.get("artifact_type") != "fraud_preprocessor":
            raise ValueError("unexpected preprocessing artifact type")
        if content.get("schema_version") != PREPROCESSOR_SCHEMA_VERSION:
            raise ValueError("unsupported preprocessing schema version")
        raw_config = content.get("config")
        if not isinstance(raw_config, dict):
            raise ValueError("preprocessing artifact config is invalid")
        try:
            instance = cls(PreprocessingConfig(**raw_config))
        except (TypeError, ValueError) as exc:
            raise ValueError("preprocessing artifact config is invalid") from exc
        required_columns = content.get("required_input_columns")
        if required_columns != list(instance.required_input_columns):
            raise ValueError("preprocessing required-input schema is invalid")
        numeric_stats = content.get("numeric_stats")
        nominal_vocabularies = content.get("nominal_vocabularies")
        frequency_mappings = content.get("frequency_mappings")
        feature_names = content.get("feature_names")
        fitted_context = content.get("fitted_context")
        if not isinstance(numeric_stats, dict):
            raise ValueError("preprocessing numeric statistics are invalid")
        if not isinstance(nominal_vocabularies, dict):
            raise ValueError("preprocessing nominal vocabularies are invalid")
        if not isinstance(frequency_mappings, dict):
            raise ValueError("preprocessing frequency mappings are invalid")
        if not isinstance(feature_names, list) or not all(
            isinstance(value, str) for value in feature_names
        ):
            raise ValueError("preprocessing feature names are invalid")
        if not isinstance(fitted_context, dict):
            raise ValueError("preprocessing fitted context is invalid")
        instance.numeric_stats_ = copy.deepcopy(numeric_stats)
        instance.nominal_vocabularies_ = copy.deepcopy(nominal_vocabularies)
        instance.frequency_mappings_ = copy.deepcopy(frequency_mappings)
        instance.feature_names_ = copy.deepcopy(feature_names)
        instance.fitted_context_ = copy.deepcopy(fitted_context)
        dependencies = content.get("dependencies")
        if not isinstance(dependencies, dict):
            raise ValueError("preprocessing dependency metadata is invalid")
        instance.dependencies_ = copy.deepcopy(dependencies)
        instance._validate_fitted_state()
        instance.is_fitted_ = True
        return instance

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Atomically serialize the auditable preprocessing artifact."""

        return _atomic_write_json(self.to_dict(), path, overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> "FraudPreprocessor":
        """Load and verify a JSON or JSON.GZ preprocessing artifact."""

        return cls.from_dict(_read_json(path))


def _class_weight_mapping(target: np.ndarray) -> dict[int, float]:
    return _class_weight_mapping_from_counts(_class_counts(target))


def _class_weight_mapping_from_counts(counts: Mapping[int, int]) -> dict[int, float]:
    total = int(counts[0]) + int(counts[1])
    return {
        label: total / (2.0 * int(counts[label]))
        for label in (0, 1)
    }


def _project_strategy_counts(
    counts: Mapping[int, int], strategy: str, sampling_strategy: float
) -> dict[int, int]:
    majority = int(counts[0])
    minority = int(counts[1])
    if strategy in {"random_under", "smotenc"} and minority >= majority:
        raise ValueError("fraud label 1 must be the minority class for resampling")
    current_ratio = minority / majority
    if strategy in {"random_under", "smotenc"} and sampling_strategy <= current_ratio:
        raise ValueError(
            "sampling_strategy must exceed the current minority/majority ratio "
            f"{current_ratio:.6f}"
        )
    if strategy == "random_under":
        majority = int(math.floor(minority / sampling_strategy))
    elif strategy == "smotenc":
        minority = int(math.floor(majority * sampling_strategy))
    return {0: majority, 1: minority}


def build_imbalance_strategy_report(
    split_manifest: Mapping[str, Any], *, sampling_strategy: float = 0.10, random_state: int = 42
) -> dict[str, Any]:
    """Return train-only class-weight and sampling projections."""

    content = _verify_document_digest(dict(split_manifest))
    if content.get("artifact_type") != "chronological_split_manifest":
        raise ValueError("unexpected split artifact type")
    if content.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError("unsupported split manifest schema version")
    ImbalanceConfig(
        strategy="random_under",
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    train = dict(content["train"])
    counts = {int(label): int(count) for label, count in train["target_counts"].items()}
    if set(counts) != {0, 1} or any(count <= 0 for count in counts.values()):
        raise ValueError("split manifest train counts must contain both target classes")
    strategies: dict[str, Any] = {}
    for strategy in ("none", "class_weight", "random_under", "smotenc"):
        projected = _project_strategy_counts(counts, strategy, sampling_strategy)
        strategies[strategy] = {
            "class_counts": {str(label): count for label, count in projected.items()},
            "rows": sum(projected.values()),
            "minority_majority_ratio": projected[1] / projected[0],
            "resampling": strategy in {"random_under", "smotenc"},
        }
    strategies["class_weight"]["class_weights"] = {
        str(label): weight
        for label, weight in _class_weight_mapping_from_counts(counts).items()
    }
    payload = {
        "artifact_type": "imbalance_strategy_report",
        "schema_version": IMBALANCE_REPORT_SCHEMA_VERSION,
        "split_manifest_sha256": split_manifest["payload_sha256"],
        "train_rows": train["rows"],
        "train_target_counts": train["target_counts"],
        "sampling_strategy": sampling_strategy,
        "random_state": random_state,
        "strategies": strategies,
        "controls": {
            "sampling_scope": "train_only",
            "validation_prevalence": "unchanged",
            "holdout_prevalence": "unchanged",
            "smote_variant": "SMOTENC_before_one_hot_encoding",
        },
        "dependencies": _runtime_dependencies(),
    }
    return _document_with_digest(payload)


def prepare_training_batch(
    frame: pd.DataFrame,
    target: Sequence[int] | pd.Series | np.ndarray,
    preprocessor: FraudPreprocessor,
    *,
    config: ImbalanceConfig = ImbalanceConfig(),
    partition: str,
) -> TrainingBatch:
    """Apply one training-only imbalance control and encode the result."""

    if partition != "train":
        raise ValueError("imbalance handling is permitted only for partition='train'")
    preprocessor._check_fitted()
    if isinstance(target, pd.Series) and not target.index.equals(frame.index):
        raise ValueError("target index must match the training frame index")
    y = _binary_target(target)
    if len(frame) != len(y):
        raise ValueError("frame and target lengths differ")
    if frame.empty:
        raise ValueError("training batch must not be empty")
    if preprocessor.config.target_col in frame.columns:
        frame_target = _binary_target(frame[preprocessor.config.target_col])
        if not np.array_equal(frame_target, y):
            raise ValueError("target values do not match the training frame target column")
    event_times = _event_seconds(
        frame[preprocessor.config.event_time_col],
        preprocessor.config.event_time_col,
    )
    if np.any(np.diff(event_times) < 0):
        raise ValueError("training batch is not nondecreasing by event time")
    fitted_min = preprocessor.fitted_context_.get("unix_time_min")
    fitted_max = preprocessor.fitted_context_.get("unix_time_max")
    fitted_rows = preprocessor.fitted_context_.get("row_count")
    if not isinstance(fitted_min, int) or not isinstance(fitted_max, int):
        raise ValueError("preprocessor does not contain a valid fitted time range")
    if not isinstance(fitted_rows, int) or fitted_rows <= 0:
        raise ValueError("preprocessor does not contain a valid fitted row count")
    if len(frame) != fitted_rows:
        raise ValueError("training batch row count differs from the fitted partition")
    if int(event_times[0]) != fitted_min or int(event_times[-1]) != fitted_max:
        raise ValueError("training batch time range differs from the fitted partition")
    if frame[preprocessor.config.key_col].duplicated().any():
        raise ValueError(f"{preprocessor.config.key_col!r} must be unique")
    input_key_sha256 = _ordered_key_digest(
        _transaction_key_hashes(
            frame[preprocessor.config.key_col], preprocessor.config.key_col
        )
    )
    if input_key_sha256 != preprocessor.fitted_context_.get("ordered_key_sha256"):
        raise ValueError("training batch row order differs from the fitted partition")
    counts_before = _class_counts(y)
    sample_weight: np.ndarray | None = None
    selected_hash: str | None = None
    class_weights: dict[int, float] | None = None
    projected_counts: dict[int, int] | None = None
    projected_dense_bytes: int | None = None
    sampler_class: str | None = None
    selected_key_sha256: str | None = None

    if config.strategy == "none":
        X_output = preprocessor.transform(frame)
        y_output = y.copy()
    elif config.strategy == "class_weight":
        sampler_class = "sklearn_balanced_class_weight"
        X_output = preprocessor.transform(frame)
        y_output = y.copy()
        class_weights = _class_weight_mapping(y)
        weight_values = np.asarray(
            [class_weights[0], class_weights[1]], dtype=np.float32
        )
        sample_weight = weight_values[y]
    elif config.strategy == "random_under":
        try:
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("random_under requires imbalanced-learn") from exc
        projected_counts = _project_strategy_counts(
            counts_before, config.strategy, config.sampling_strategy
        )
        sampler = RandomUnderSampler(
            sampling_strategy=config.sampling_strategy,
            random_state=config.random_state,
            replacement=False,
        )
        sampler_class = "imblearn.under_sampling.RandomUnderSampler"
        positions = np.arange(len(frame), dtype=np.int64).reshape(-1, 1)
        selected, _ = sampler.fit_resample(positions, y)
        selected_positions = np.sort(selected.reshape(-1).astype(np.int64))
        selected_hash = hashlib.sha256(selected_positions.tobytes()).hexdigest().upper()
        selected_frame = frame.iloc[selected_positions]
        selected_key_sha256 = _ordered_key_digest(
            _transaction_key_hashes(
                selected_frame[preprocessor.config.key_col],
                preprocessor.config.key_col,
            )
        )
        X_output = preprocessor.transform(selected_frame)
        y_output = y[selected_positions].copy()
    else:
        try:
            from imblearn.over_sampling import SMOTENC
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("smotenc requires imbalanced-learn") from exc
        if counts_before[1] <= config.k_neighbors:
            raise ValueError(
                "SMOTENC requires more minority rows than k_neighbors"
            )
        projected_counts = _project_strategy_counts(
            counts_before, config.strategy, config.sampling_strategy
        )
        projected_rows = sum(projected_counts.values())
        sampler_column_count = (
            len(preprocessor.config.numeric_columns)
            + len(DERIVED_NUMERIC_COLUMNS)
            + 2 * len(preprocessor.config.frequency_columns)
            + len(preprocessor.config.nominal_columns)
        )
        projected_dense_bytes = (
            (len(frame) + 2 * projected_rows)
            * sampler_column_count
            * np.dtype(np.float64).itemsize
        )
        if projected_rows > config.max_output_rows:
            raise MemoryError(
                f"projected SMOTENC rows {projected_rows:,} exceed max_output_rows"
            )
        if projected_dense_bytes > config.max_dense_bytes:
            raise MemoryError(
                "projected SMOTENC working memory "
                f"{projected_dense_bytes:,} exceeds max_dense_bytes"
            )
        sampler_frame = preprocessor.prepare_sampler_frame(frame)
        categorical_indices = [
            sampler_frame.columns.get_loc(column)
            for column in preprocessor.sampler_nominal_columns
        ]
        sampler = SMOTENC(
            categorical_features=categorical_indices,
            sampling_strategy=config.sampling_strategy,
            random_state=config.random_state,
            k_neighbors=config.k_neighbors,
        )
        sampler_class = "imblearn.over_sampling.SMOTENC"
        resampled, y_output = sampler.fit_resample(sampler_frame, y)
        if not isinstance(resampled, pd.DataFrame):
            resampled = pd.DataFrame(resampled, columns=sampler_frame.columns)
        X_output = preprocessor.transform_sampler_frame(resampled)
        y_output = np.asarray(y_output, dtype=np.int8)
        del resampled, sampler_frame, sampler

    counts_after = _class_counts(y_output)
    if projected_counts is not None and counts_after != projected_counts:
        raise RuntimeError("sampler output class counts differ from the projection")
    preprocessor_payload_sha256 = preprocessor.to_dict()["payload_sha256"]
    metadata = {
        "strategy": config.strategy,
        "config": asdict(config),
        "sampling_strategy": config.sampling_strategy,
        "random_state": config.random_state,
        "class_counts_before": {str(label): count for label, count in counts_before.items()},
        "class_counts_after": {str(label): count for label, count in counts_after.items()},
        "rows_before": len(y),
        "rows_after": len(y_output),
        "feature_schema_sha256": preprocessor.fitted_context_["feature_schema_sha256"],
        "preprocessor_payload_sha256": preprocessor_payload_sha256,
        "split_manifest_sha256": preprocessor.fitted_context_.get(
            "split_manifest_sha256"
        ),
        "input_target_sha256": hashlib.sha256(y.tobytes()).hexdigest().upper(),
        "input_ordered_key_sha256": input_key_sha256,
        "selected_positions_sha256": selected_hash,
        "selected_ordered_key_sha256": selected_key_sha256,
        "sampler_class": sampler_class,
        "projected_class_counts": (
            None
            if projected_counts is None
            else {
                str(label): count for label, count in projected_counts.items()
            }
        ),
        "projected_dense_bytes": projected_dense_bytes,
        "validation_and_holdout_sampled": False,
        "intended_use": "fixed_training_partition_not_cross_validation",
        "dependencies": _runtime_dependencies(),
    }
    if class_weights is not None:
        metadata["class_weights"] = {
            str(label): weight for label, weight in class_weights.items()
        }
    X_csr = X_output.tocsr(copy=False).astype(np.float32, copy=False)
    X_csr.sum_duplicates()
    X_csr.sort_indices()
    return TrainingBatch(
        X=X_csr,
        y=y_output,
        sample_weight=sample_weight,
        metadata=metadata,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create temporal split and imbalance-control artifacts."
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="create chronological split manifest")
    split_parser.add_argument("--development", type=Path, required=True)
    split_parser.add_argument("--holdout", type=Path, required=True)
    split_parser.add_argument("--output", type=Path, required=True)
    split_parser.add_argument("--validation-fraction", type=float, default=0.20)
    split_parser.add_argument("--chunksize", type=int, default=100_000)
    split_parser.add_argument("--force", action="store_true")

    report_parser = subparsers.add_parser(
        "imbalance-report", help="create train-only strategy comparison"
    )
    report_parser.add_argument("--split-manifest", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.add_argument("--sampling-strategy", type=float, default=0.10)
    report_parser.add_argument("--random-state", type=int, default=42)
    report_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for preprocessing control artifacts."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "split":
        config = SplitConfig(
            validation_fraction=args.validation_fraction,
            chunksize=args.chunksize,
        )
        manifest = build_chronological_split_manifest(
            args.development, args.holdout, config=config
        )
        save_split_manifest(manifest, args.output, overwrite=args.force)
        LOGGER.info(
            "artifact=split_manifest path=%s train_rows=%s validation_rows=%s holdout_rows=%s",
            args.output,
            manifest["train"]["rows"],
            manifest["validation"]["rows"],
            manifest["holdout"]["rows"],
        )
    else:
        manifest = load_split_manifest(args.split_manifest)
        report = build_imbalance_strategy_report(
            manifest,
            sampling_strategy=args.sampling_strategy,
            random_state=args.random_state,
        )
        _atomic_write_json(report, args.output, overwrite=args.force)
        LOGGER.info(
            "artifact=imbalance_strategy_report path=%s train_rows=%s",
            args.output,
            report["train_rows"],
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
