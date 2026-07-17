"""Memory-bounded exploratory profiling for the Fraud Shield dataset.

The notebook layer should contain visualization and interpretation, not a
second implementation of data scanning or feature logic. This module performs
one chunked pass, computes exact aggregate counts, engineers causal features on
the complete ordered stream, and retains only deterministic samples for plots
and correlations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .features import FeatureConfig, RollingFeatureState, add_geospatial_features

REQUIRED_COLUMNS = (
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "gender",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
)

SAMPLE_COLUMNS = (
    "trans_date_trans_time",
    "category",
    "amt",
    "gender",
    "state",
    "city_pop",
    "merch_lat",
    "merch_long",
    "is_fraud",
    "distance_card_merchant_km",
    "cc_txn_count_prev_1h",
    "cc_amt_sum_prev_1h",
    "cc_txn_count_prev_6h",
    "cc_amt_sum_prev_6h",
    "cc_txn_count_prev_24h",
    "cc_amt_sum_prev_24h",
    "cc_txn_count_prior",
    "cc_amt_sum_prior",
    "age_years",
    "hour",
    "day_of_week",
    "log1p_amt",
    "log1p_city_pop",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
)


@dataclass(frozen=True)
class EDAProfile:
    """Exact aggregates and bounded deterministic samples for EDA."""

    summary: dict[str, Any]
    quality: pd.DataFrame
    missingness: pd.DataFrame
    hourly_counts: pd.DataFrame
    weekday_counts: pd.DataFrame
    weekly_counts: pd.DataFrame
    state_counts: pd.DataFrame
    geographic_counts: pd.DataFrame
    correlation_sample: pd.DataFrame
    visualization_sample: pd.DataFrame


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 fingerprint without loading the file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _accumulate_grouped(
    destination: dict[tuple[Any, ...], int],
    frame: pd.DataFrame,
    columns: list[str],
) -> None:
    grouped = frame.groupby(columns, observed=True, dropna=False).size()
    for key, value in grouped.items():
        normalized_key = key if isinstance(key, tuple) else (key,)
        destination[normalized_key] = destination.get(normalized_key, 0) + int(value)


def _counter_frame(
    counter: dict[tuple[Any, ...], int], columns: list[str]
) -> pd.DataFrame:
    rows = [(*key, value) for key, value in counter.items()]
    return pd.DataFrame(rows, columns=[*columns, "count"]).sort_values(columns).reset_index(
        drop=True
    )


def _keep_smallest_hash(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
    limit: int,
) -> pd.DataFrame:
    if limit <= 0:
        return incoming.iloc[0:0].copy()
    combined = incoming if existing is None else pd.concat([existing, incoming], ignore_index=True)
    if len(combined) > limit:
        hashes = combined["__sample_hash"].to_numpy(dtype=np.uint64)
        positions = np.argpartition(hashes, limit - 1)[:limit]
        combined = combined.iloc[positions]
    return combined.reset_index(drop=True)


def _timestamp_backsteps(values: np.ndarray, previous: int | None) -> tuple[int, int | None]:
    if values.size == 0:
        return 0, previous
    count = int(np.count_nonzero(np.diff(values) < 0))
    if previous is not None and int(values[0]) < previous:
        count += 1
    return count, int(values[-1])


def _stable_hash(values: pd.Series, seed: int) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(values.astype("string"), index=False).to_numpy(
        dtype=np.uint64
    )
    seed_mask = np.uint64((seed * 0x9E3779B1) & ((1 << 64) - 1))
    return hashes ^ seed_mask


def scan_fraud_csv(
    path: str | Path,
    *,
    chunksize: int = 100_000,
    correlation_sample_size: int = 75_000,
    legitimate_plot_sample_size: int = 50_000,
    fraud_plot_sample_size: int = 20_000,
    random_seed: int = 42,
    max_rows: int | None = None,
) -> EDAProfile:
    """Profile a transaction CSV with bounded memory and deterministic samples.

    Exact class, temporal, state, geographic, missingness, and quality counts
    are accumulated over the full input. Rolling features are calculated before
    sampling, ensuring sampled rows retain their true transaction history.
    ``max_rows`` exists for smoke tests only; omit it for reportable EDA.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"dataset not found: {source}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when supplied")

    feature_config = FeatureConfig(invalid_geo="raise")
    velocity_state = RollingFeatureState(feature_config)
    class_counts = {0: 0, 1: 0}
    missing_counts: pd.Series | None = None
    unique_values = {column: set() for column in ("cc_num", "merchant", "category", "state")}
    hourly: dict[tuple[Any, ...], int] = {}
    weekday: dict[tuple[Any, ...], int] = {}
    weekly: dict[tuple[Any, ...], int] = {}
    states: dict[tuple[Any, ...], int] = {}
    geography: dict[tuple[Any, ...], int] = {}

    correlation_sample: pd.DataFrame | None = None
    fraud_sample: pd.DataFrame | None = None
    legitimate_sample: pd.DataFrame | None = None

    rows_seen = 0
    amount_sum = 0.0
    amount_min = np.inf
    amount_max = -np.inf
    display_min: pd.Timestamp | None = None
    display_max: pd.Timestamp | None = None
    unix_min: int | None = None
    unix_max: int | None = None
    previous_display_ns: int | None = None
    previous_unix: int | None = None
    quality_counts = {
        "invalid_target": 0,
        "invalid_display_timestamp": 0,
        "invalid_unix_time": 0,
        "nonpositive_amount": 0,
        "invalid_coordinates": 0,
        "display_timestamp_backsteps": 0,
        "unix_time_backsteps": 0,
        "source_index_mismatches": 0,
    }

    reader = pd.read_csv(
        source,
        chunksize=chunksize,
        dtype={"cc_num": "string", "zip": "string", "trans_num": "string"},
        low_memory=False,
    )
    for raw_chunk in reader:
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            raw_chunk = raw_chunk.iloc[:remaining].copy()
        if raw_chunk.empty:
            continue

        missing_required = sorted(set(REQUIRED_COLUMNS).difference(raw_chunk.columns))
        if missing_required:
            raise ValueError(f"missing required columns: {', '.join(missing_required)}")

        source_index_col = next(
            (column for column in raw_chunk.columns if str(column).startswith("Unnamed:")), None
        )
        if source_index_col is not None:
            observed_index = pd.to_numeric(raw_chunk[source_index_col], errors="coerce").to_numpy()
            expected_index = np.arange(rows_seen, rows_seen + len(raw_chunk))
            quality_counts["source_index_mismatches"] += int(
                np.count_nonzero(observed_index != expected_index)
            )
            chunk = raw_chunk.drop(columns=source_index_col)
        else:
            chunk = raw_chunk.copy()

        chunk_missing = chunk.isna().sum().astype("int64")
        missing_counts = (
            chunk_missing
            if missing_counts is None
            else missing_counts.add(chunk_missing, fill_value=0).astype("int64")
        )

        target_numeric = pd.to_numeric(chunk["is_fraud"], errors="coerce")
        valid_target = target_numeric.isin((0, 1))
        quality_counts["invalid_target"] += int((~valid_target).sum())
        if not valid_target.all():
            raise ValueError("is_fraud must contain only 0 and 1")
        chunk["is_fraud"] = target_numeric.astype("int8")
        target_counts = chunk["is_fraud"].value_counts()
        class_counts[0] += int(target_counts.get(0, 0))
        class_counts[1] += int(target_counts.get(1, 0))

        timestamps = pd.to_datetime(chunk["trans_date_trans_time"], errors="coerce")
        invalid_display = timestamps.isna()
        quality_counts["invalid_display_timestamp"] += int(invalid_display.sum())
        unix_numeric = pd.to_numeric(chunk["unix_time"], errors="coerce")
        invalid_unix = unix_numeric.isna()
        quality_counts["invalid_unix_time"] += int(invalid_unix.sum())
        if invalid_display.any() or invalid_unix.any():
            raise ValueError("invalid timestamps found during profiling")
        unix_values = unix_numeric.to_numpy(dtype=np.int64)
        display_ns = timestamps.array.asi8
        backsteps, previous_display_ns = _timestamp_backsteps(
            display_ns, previous_display_ns
        )
        quality_counts["display_timestamp_backsteps"] += backsteps
        backsteps, previous_unix = _timestamp_backsteps(unix_values, previous_unix)
        quality_counts["unix_time_backsteps"] += backsteps

        chunk_display_min = timestamps.min()
        chunk_display_max = timestamps.max()
        display_min = chunk_display_min if display_min is None else min(display_min, chunk_display_min)
        display_max = chunk_display_max if display_max is None else max(display_max, chunk_display_max)
        chunk_unix_min = int(unix_values.min())
        chunk_unix_max = int(unix_values.max())
        unix_min = chunk_unix_min if unix_min is None else min(unix_min, chunk_unix_min)
        unix_max = chunk_unix_max if unix_max is None else max(unix_max, chunk_unix_max)

        amounts = pd.to_numeric(chunk["amt"], errors="coerce").to_numpy(dtype=np.float64)
        quality_counts["nonpositive_amount"] += int(np.count_nonzero(amounts <= 0))
        amount_sum += float(amounts.sum())
        amount_min = min(amount_min, float(amounts.min()))
        amount_max = max(amount_max, float(amounts.max()))

        coordinate_columns = ["lat", "long", "merch_lat", "merch_long"]
        coordinates = chunk[coordinate_columns].apply(pd.to_numeric, errors="coerce")
        coordinate_valid = (
            coordinates.notna().all(axis=1)
            & coordinates["lat"].between(-90, 90)
            & coordinates["merch_lat"].between(-90, 90)
            & coordinates["long"].between(-180, 180)
            & coordinates["merch_long"].between(-180, 180)
        )
        quality_counts["invalid_coordinates"] += int((~coordinate_valid).sum())

        for column, values in unique_values.items():
            values.update(chunk[column].dropna().astype(str).unique().tolist())

        derived = add_geospatial_features(chunk, feature_config)
        derived = velocity_state.transform_chunk(derived)
        birth_dates = pd.to_datetime(derived["dob"], errors="coerce")
        derived["age_years"] = (timestamps - birth_dates).dt.days / 365.2425
        derived["hour"] = timestamps.dt.hour.astype("int8")
        derived["day_of_week"] = timestamps.dt.dayofweek.astype("int8")
        derived["log1p_amt"] = np.log1p(amounts)
        derived["log1p_city_pop"] = np.log1p(
            pd.to_numeric(derived["city_pop"], errors="coerce").to_numpy(dtype=np.float64)
        )
        derived["hour_sin"] = np.sin(2.0 * np.pi * derived["hour"] / 24.0)
        derived["hour_cos"] = np.cos(2.0 * np.pi * derived["hour"] / 24.0)
        derived["day_sin"] = np.sin(2.0 * np.pi * derived["day_of_week"] / 7.0)
        derived["day_cos"] = np.cos(2.0 * np.pi * derived["day_of_week"] / 7.0)

        temporal = pd.DataFrame(
            {
                "hour": derived["hour"],
                "day_of_week": derived["day_of_week"],
                "week_start": timestamps.dt.to_period("W-SUN").dt.start_time,
                "is_fraud": derived["is_fraud"],
            }
        )
        _accumulate_grouped(hourly, temporal, ["hour", "is_fraud"])
        _accumulate_grouped(weekday, temporal, ["day_of_week", "is_fraud"])
        _accumulate_grouped(weekly, temporal, ["week_start", "is_fraud"])
        _accumulate_grouped(states, derived, ["state", "is_fraud"])

        geo = pd.DataFrame(
            {
                "merchant_lat_bin": np.floor(
                    pd.to_numeric(derived["merch_lat"], errors="coerce")
                )
                + 0.5,
                "merchant_lon_bin": np.floor(
                    pd.to_numeric(derived["merch_long"], errors="coerce")
                )
                + 0.5,
                "is_fraud": derived["is_fraud"],
            }
        )
        _accumulate_grouped(
            geography, geo, ["merchant_lat_bin", "merchant_lon_bin", "is_fraud"]
        )

        sample_hashes = _stable_hash(chunk["trans_num"], random_seed)
        sample_candidates = derived.loc[:, SAMPLE_COLUMNS].copy()
        sample_candidates["__sample_hash"] = sample_hashes
        correlation_sample = _keep_smallest_hash(
            correlation_sample, sample_candidates, correlation_sample_size
        )
        fraud_sample = _keep_smallest_hash(
            fraud_sample,
            sample_candidates.loc[sample_candidates["is_fraud"] == 1],
            fraud_plot_sample_size,
        )
        legitimate_sample = _keep_smallest_hash(
            legitimate_sample,
            sample_candidates.loc[sample_candidates["is_fraud"] == 0],
            legitimate_plot_sample_size,
        )
        rows_seen += len(chunk)

    if rows_seen == 0 or missing_counts is None:
        raise ValueError("dataset contains no data rows")

    fraud_count = class_counts[1]
    nonfraud_count = class_counts[0]
    fraud_rate = fraud_count / rows_seen
    summary = {
        "source_path": str(source.resolve()),
        "source_size_bytes": source.stat().st_size,
        "rows": rows_seen,
        "nonfraud_rows": nonfraud_count,
        "fraud_rows": fraud_count,
        "fraud_rate": fraud_rate,
        "nonfraud_to_fraud_ratio": nonfraud_count / fraud_count if fraud_count else np.inf,
        "random_classifier_pr_auc": fraud_rate,
        "display_time_start": display_min,
        "display_time_end": display_max,
        "unix_time_start": unix_min,
        "unix_time_end": unix_max,
        "amount_min": amount_min,
        "amount_max": amount_max,
        "amount_mean": amount_sum / rows_seen,
        "unique_cards": len(unique_values["cc_num"]),
        "unique_merchants": len(unique_values["merchant"]),
        "unique_categories": len(unique_values["category"]),
        "unique_states": len(unique_values["state"]),
        "velocity_clock": feature_config.time_col,
        "correlation_sample_rows": 0 if correlation_sample is None else len(correlation_sample),
        "visualization_sample_rows": (
            (0 if fraud_sample is None else len(fraud_sample))
            + (0 if legitimate_sample is None else len(legitimate_sample))
        ),
        "sampling_seed": random_seed,
    }

    quality_rows = []
    for check, value in quality_counts.items():
        expected_artifact = check == "display_timestamp_backsteps" and value == 1
        quality_rows.append(
            {
                "check": check,
                "count": value,
                "status": "KNOWN_DATASET_ARTIFACT"
                if expected_artifact
                else ("PASS" if value == 0 else "REVIEW"),
            }
        )
    quality_rows.append(
        {
            "check": "missing_cells",
            "count": int(missing_counts.sum()),
            "status": "PASS" if int(missing_counts.sum()) == 0 else "REVIEW",
        }
    )

    correlation_output = (
        pd.DataFrame(columns=SAMPLE_COLUMNS)
        if correlation_sample is None
        else correlation_sample.sort_values("__sample_hash")
        .drop(columns="__sample_hash")
        .reset_index(drop=True)
    )
    visualization_parts = [
        sample
        for sample in (fraud_sample, legitimate_sample)
        if sample is not None and not sample.empty
    ]
    visualization_output = (
        pd.DataFrame(columns=SAMPLE_COLUMNS)
        if not visualization_parts
        else pd.concat(visualization_parts, ignore_index=True)
        .sort_values(["is_fraud", "__sample_hash"])
        .drop(columns="__sample_hash")
        .reset_index(drop=True)
    )

    missingness = (
        missing_counts.rename("missing_count")
        .to_frame()
        .assign(missing_rate=lambda data: data["missing_count"] / rows_seen)
        .reset_index(names="column")
    )
    return EDAProfile(
        summary=summary,
        quality=pd.DataFrame(quality_rows),
        missingness=missingness,
        hourly_counts=_counter_frame(hourly, ["hour", "is_fraud"]),
        weekday_counts=_counter_frame(weekday, ["day_of_week", "is_fraud"]),
        weekly_counts=_counter_frame(weekly, ["week_start", "is_fraud"]),
        state_counts=_counter_frame(states, ["state", "is_fraud"]),
        geographic_counts=_counter_frame(
            geography, ["merchant_lat_bin", "merchant_lon_bin", "is_fraud"]
        ),
        correlation_sample=correlation_output,
        visualization_sample=visualization_output,
    )
