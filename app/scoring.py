"""Causal application scoring over registered FraudShield artifacts."""

from __future__ import annotations

import argparse
import calendar
import copy
import json
import os
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from src.features import (
    RollingFeatureState,
    add_geospatial_features,
)
from src.models.data import ModelDataPaths, load_model_dataset
from src.models.hybrid import HybridInferenceEngine
from src.models.search import load_model_report
from src.models.sequences import GlobalSparseAccessor, load_sequence_index
from src.preprocessing import FraudPreprocessor
from src.utils import atomic_write_json, json_digest, sha256_file

APP_CONTEXT_SCHEMA_VERSION = 1
DATASET_CLOCK_OFFSET_SECONDS = 220_924_800
MAX_BATCH_ROWS = 10_000

REQUIRED_TRANSACTION_COLUMNS = (
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "merch_lat",
    "merch_long",
)
OPTIONAL_TRANSACTION_COLUMNS = ("trans_num", "unix_time")
ENGINEERED_EXPLANATION_COLUMNS = (
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


def utc_now_text() -> str:
    """Return a seconds-resolution UTC audit timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clock_seconds(display_time: pd.Timestamp) -> int:
    if display_time.tzinfo is not None:
        display_time = display_time.tz_convert("UTC").tz_localize(None)
    epoch = calendar.timegm(display_time.to_pydatetime().timetuple())
    return int(epoch - DATASET_CLOCK_OFFSET_SECONDS)


def _generated_key(row: Mapping[str, Any], position: int) -> str:
    content = {
        key: None if pd.isna(value) else str(value)
        for key, value in row.items()
        if key != "trans_num"
    }
    content["row_position"] = int(position)
    return json_digest(content).lower()


def _plain_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def normalize_transactions(
    frame: pd.DataFrame,
    *,
    max_rows: int = MAX_BATCH_ROWS,
) -> pd.DataFrame:
    """Validate and normalize raw manual or uploaded transaction rows."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("transactions must be supplied as a pandas DataFrame")
    if frame.empty:
        raise ValueError("at least one transaction is required")
    if len(frame) > max_rows:
        raise ValueError(f"batch contains {len(frame):,} rows; maximum is {max_rows:,}")
    missing = sorted(set(REQUIRED_TRANSACTION_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required transaction columns: {', '.join(missing)}")

    output = frame.loc[
        :,
        [
            *REQUIRED_TRANSACTION_COLUMNS,
            *(
                column
                for column in OPTIONAL_TRANSACTION_COLUMNS
                if column in frame.columns
            ),
        ],
    ].copy()

    display = pd.to_datetime(
        output["trans_date_trans_time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    if display.isna().any():
        raise ValueError("trans_date_trans_time must use YYYY-MM-DD HH:MM:SS")
    output["trans_date_trans_time"] = display.dt.strftime("%Y-%m-%d %H:%M:%S")

    birth = pd.to_datetime(output["dob"], format="%Y-%m-%d", errors="coerce")
    if birth.isna().any():
        raise ValueError("dob must use YYYY-MM-DD")
    if (birth > display.dt.normalize()).any():
        raise ValueError("dob cannot occur after the transaction date")
    output["dob"] = birth.dt.strftime("%Y-%m-%d")

    raw_cards = output["cc_num"]
    if pd.api.types.is_float_dtype(raw_cards.dtype):
        raise ValueError(
            "cc_num was parsed as floating point; upload it as a quoted string "
            "to preserve all digits"
        )
    output["cc_num"] = raw_cards.astype("string").str.strip()
    if output["cc_num"].isna().any() or (output["cc_num"] == "").any():
        raise ValueError("cc_num must be present")

    for column in ("merchant", "category", "city", "state", "job"):
        output[column] = output[column].astype("string").str.strip()
        if output[column].isna().any() or (output[column] == "").any():
            raise ValueError(f"{column} must be present")
    output["zip"] = output["zip"].astype("string").str.strip()
    if output["zip"].isna().any() or (output["zip"] == "").any():
        raise ValueError("zip must be present")

    numeric_columns = (
        "amt",
        "lat",
        "long",
        "city_pop",
        "merch_lat",
        "merch_long",
    )
    for column in numeric_columns:
        numeric = pd.to_numeric(output[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{column} must contain finite numeric values")
        output[column] = numeric
    if (output["amt"] <= 0).any():
        raise ValueError("amt must be strictly positive")
    if (output["city_pop"] < 0).any():
        raise ValueError("city_pop must be nonnegative")

    if "unix_time" in output:
        supplied = pd.to_numeric(output["unix_time"], errors="coerce")
        derived = display.map(_clock_seconds).astype(np.int64)
        missing_time = supplied.isna()
        supplied.loc[missing_time] = derived.loc[missing_time]
        if not np.isfinite(supplied.to_numpy()).all():
            raise ValueError("unix_time must contain finite whole seconds")
        if not np.equal(supplied.to_numpy(), np.rint(supplied.to_numpy())).all():
            raise ValueError("unix_time must contain whole seconds")
        output["unix_time"] = supplied.astype(np.int64)
    else:
        output["unix_time"] = display.map(_clock_seconds).astype(np.int64)

    if "trans_num" not in output:
        output["trans_num"] = ""
    keys = output["trans_num"].astype("string").fillna("").str.strip()
    for position in range(len(output)):
        if not keys.iloc[position]:
            keys.iloc[position] = _generated_key(
                output.iloc[position].to_dict(),
                position,
            )
    if keys.duplicated().any():
        raise ValueError("trans_num must be unique within the submitted batch")
    output["trans_num"] = keys.astype(str)

    return output.reset_index(drop=True)


@dataclass(frozen=True)
class SequenceContext:
    """Immutable per-card sequence continuation arrays."""

    cards: np.ndarray
    history: np.ndarray
    history_lengths: np.ndarray
    pending_vectors: np.ndarray
    pending_times: np.ndarray
    pending_keys: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        card_count = len(self.cards)
        if (
            self.cards.ndim != 1
            or self.history.ndim != 3
            or self.history.shape[0] != card_count
            or self.history_lengths.shape != (card_count,)
            or self.pending_vectors.shape
            != (card_count, self.history.shape[2])
            or self.pending_times.shape != (card_count,)
            or self.pending_keys.shape != (card_count,)
        ):
            raise ValueError("sequence context arrays are not aligned")
        if (
            (self.history_lengths < 0).any()
            or (self.history_lengths > self.history.shape[1]).any()
        ):
            raise ValueError("sequence context history lengths are invalid")


def _save_npz_atomic(
    destination: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    overwrite: bool,
) -> Path:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"context output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}{destination.suffix}"
    )
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _latest_card_rows(
    feature_paths: Sequence[Path],
    *,
    expected_rows: int,
    chunksize: int = 100_000,
) -> dict[str, tuple[int, str, int]]:
    latest: dict[str, tuple[int, str, int]] = {}
    global_row = 0
    for path in feature_paths:
        for chunk in pd.read_csv(
            path,
            usecols=["cc_num", "unix_time", "trans_num"],
            dtype={"cc_num": "string", "trans_num": "string"},
            chunksize=chunksize,
        ):
            times = pd.to_numeric(chunk["unix_time"], errors="raise").to_numpy(
                dtype=np.int64
            )
            cards = chunk["cc_num"].astype(str).to_numpy()
            keys = chunk["trans_num"].astype(str).to_numpy()
            for offset, (card, event_time, key) in enumerate(
                zip(cards, times.tolist(), keys, strict=True)
            ):
                row = global_row + offset
                prior = latest.get(card)
                if (
                    prior is None
                    or event_time > prior[0]
                    or (event_time == prior[0] and key < prior[1])
                ):
                    latest[card] = (int(event_time), key, row)
            global_row += len(chunk)
    if global_row != expected_rows:
        raise ValueError(
            f"feature streams contain {global_row:,} rows; expected {expected_rows:,}"
        )
    return latest


def build_sequence_context(
    project_root: str | Path,
    *,
    overwrite: bool = False,
) -> SequenceContext:
    """Build the compact online sequence continuation artifact."""

    root = Path(project_root).resolve()
    paths = ModelDataPaths.project_defaults(root)
    data = load_model_dataset(paths.output_dir)
    sequence_index = load_sequence_index(paths.output_dir)
    accessor = GlobalSparseAccessor(data)
    latest = _latest_card_rows(
        (paths.development_features, paths.holdout_features),
        expected_rows=accessor.rows,
    )
    if not latest:
        raise ValueError("feature streams do not contain card history")
    cards = sorted(latest)
    lstm_report_path = root / "artifacts" / "models" / "lstm_report.json"
    lstm_report = load_model_report(lstm_report_path)
    sequence_length = int(lstm_report["sequence_length"])
    history_width = sequence_length - 1
    history_indices: list[list[int]] = []
    pending_indices: list[int] = []
    for card in cards:
        pending_index = latest[card][2]
        pending_indices.append(pending_index)
        backward: list[int] = []
        cursor = int(sequence_index.previous[pending_index])
        while cursor >= 0 and len(backward) < history_width:
            backward.append(cursor)
            cursor = int(sequence_index.previous[cursor])
        history_indices.append(list(reversed(backward)))

    history = np.zeros(
        (len(cards), history_width, accessor.width),
        dtype=np.float32,
    )
    history_lengths = np.asarray(
        [len(indices) for indices in history_indices],
        dtype=np.int16,
    )
    flat = [index for indices in history_indices for index in indices]
    if flat:
        gathered = accessor.gather(np.asarray(flat, dtype=np.int64))
        cursor = 0
        for position, indices in enumerate(history_indices):
            count = len(indices)
            history[position, :count] = gathered[cursor : cursor + count]
            cursor += count
    pending_vectors = accessor.gather(
        np.asarray(pending_indices, dtype=np.int64)
    ).astype(np.float32, copy=False)
    pending_times = np.asarray(
        [latest[card][0] for card in cards],
        dtype=np.int64,
    )
    pending_keys = np.asarray(
        [latest[card][1] for card in cards],
        dtype=f"<U{max(len(latest[card][1]) for card in cards)}",
    )
    card_values = np.asarray(cards, dtype=f"<U{max(len(card) for card in cards)}")

    output_dir = root / "artifacts" / "app"
    data_path = output_dir / "sequence_context.npz"
    manifest_path = output_dir / "sequence_context_manifest.json"
    if not overwrite:
        existing = [
            str(path) for path in (data_path, manifest_path) if path.exists()
        ]
        if existing:
            raise FileExistsError(
                f"application context outputs already exist: {', '.join(existing)}"
            )
    _save_npz_atomic(
        data_path,
        {
            "cards": card_values,
            "history": history,
            "history_lengths": history_lengths,
            "pending_vectors": pending_vectors,
            "pending_times": pending_times,
            "pending_keys": pending_keys,
        },
        overwrite=overwrite,
    )
    content: dict[str, Any] = {
        "artifact_type": "fraud_app_sequence_context",
        "schema_version": APP_CONTEXT_SCHEMA_VERSION,
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
        "card_count": len(cards),
        "sequence_length": sequence_length,
        "feature_count": accessor.width,
        "history_shape": list(history.shape),
        "model_data_manifest_sha256": data.metadata["payload_sha256"],
        "sequence_index_manifest_sha256": sequence_index.metadata["payload_sha256"],
        "development_feature_sha256": sha256_file(paths.development_features),
        "holdout_feature_sha256": sha256_file(paths.holdout_features),
        "terminal_velocity_state_sha256": sha256_file(paths.holdout_state),
    }
    document = {**content, "payload_sha256": json_digest(content)}
    atomic_write_json(document, manifest_path, overwrite=overwrite)
    return SequenceContext(
        cards=card_values,
        history=history,
        history_lengths=history_lengths,
        pending_vectors=pending_vectors,
        pending_times=pending_times,
        pending_keys=pending_keys,
        metadata=document,
    )


def load_sequence_context(project_root: str | Path) -> SequenceContext:
    """Load and verify the compact online sequence continuation artifact."""

    root = Path(project_root).resolve()
    output_dir = root / "artifacts" / "app"
    manifest_path = output_dir / "sequence_context_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("application sequence context root must be an object")
    digest = document.get("payload_sha256")
    content = {key: value for key, value in document.items() if key != "payload_sha256"}
    if digest != json_digest(content):
        raise ValueError("application sequence context digest does not match")
    if content.get("artifact_type") != "fraud_app_sequence_context":
        raise ValueError("unexpected application sequence context type")
    if content.get("schema_version") != APP_CONTEXT_SCHEMA_VERSION:
        raise ValueError("unsupported application sequence context schema")
    data_path = output_dir / content["data_file"]
    if sha256_file(data_path) != content["data_sha256"]:
        raise ValueError("application sequence context data digest differs")
    with np.load(data_path, allow_pickle=False) as arrays:
        return SequenceContext(
            cards=arrays["cards"].copy(),
            history=arrays["history"].astype(np.float32, copy=True),
            history_lengths=arrays["history_lengths"].copy(),
            pending_vectors=arrays["pending_vectors"].astype(np.float32, copy=True),
            pending_times=arrays["pending_times"].copy(),
            pending_keys=arrays["pending_keys"].copy(),
            metadata=document,
        )


@dataclass
class _CardSequenceState:
    history: deque[np.ndarray]
    pending_time: int | None
    pending_key: str | None
    pending_vector: np.ndarray | None


class OnlineSequenceState:
    """Session-isolated causal sequence state with same-time bucket rules."""

    def __init__(
        self,
        sequence_length: int,
        feature_count: int,
        cards: Mapping[str, _CardSequenceState] | None = None,
    ) -> None:
        if sequence_length <= 0 or feature_count <= 0:
            raise ValueError("sequence dimensions must be positive")
        self.sequence_length = int(sequence_length)
        self.feature_count = int(feature_count)
        self._cards = dict(cards or {})
        self.seeded_cards = frozenset(self._cards)

    @classmethod
    def from_context(cls, context: SequenceContext) -> "OnlineSequenceState":
        sequence_length = context.history.shape[1] + 1
        feature_count = context.history.shape[2]
        cards: dict[str, _CardSequenceState] = {}
        for position, raw_card in enumerate(context.cards.tolist()):
            count = int(context.history_lengths[position])
            history = deque(
                (
                    context.history[position, index].copy()
                    for index in range(count)
                ),
                maxlen=sequence_length - 1,
            )
            cards[str(raw_card)] = _CardSequenceState(
                history=history,
                pending_time=int(context.pending_times[position]),
                pending_key=str(context.pending_keys[position]),
                pending_vector=context.pending_vectors[position].copy(),
            )
        return cls(sequence_length, feature_count, cards)

    def clone(self) -> "OnlineSequenceState":
        cards = {
            card: _CardSequenceState(
                history=deque(state.history, maxlen=self.sequence_length - 1),
                pending_time=state.pending_time,
                pending_key=state.pending_key,
                pending_vector=state.pending_vector,
            )
            for card, state in self._cards.items()
        }
        clone = OnlineSequenceState(self.sequence_length, self.feature_count, cards)
        clone.seeded_cards = self.seeded_cards
        return clone

    def prepare_and_advance(
        self,
        card: str,
        event_time: int,
        transaction_key: str,
        current_vector: np.ndarray,
    ) -> tuple[np.ndarray, int, bool]:
        """Return current sequence, then retain the row for future timestamps."""

        vector = np.asarray(current_vector, dtype=np.float32)
        if vector.shape != (self.feature_count,):
            raise ValueError("current transformed vector has the wrong width")
        state = self._cards.get(card)
        seeded = card in self.seeded_cards
        if state is None:
            state = _CardSequenceState(
                history=deque(maxlen=self.sequence_length - 1),
                pending_time=None,
                pending_key=None,
                pending_vector=None,
            )
            self._cards[card] = state
        if state.pending_time is not None and event_time < state.pending_time:
            raise ValueError(
                f"out-of-order sequence event for card {card!r}: "
                f"{event_time} follows {state.pending_time}"
            )
        if state.pending_time is not None and event_time > state.pending_time:
            if state.pending_vector is None:
                raise RuntimeError("sequence state has no pending representative")
            state.history.append(state.pending_vector)
            state.pending_time = None
            state.pending_key = None
            state.pending_vector = None

        chronological = [*state.history, vector]
        chronological = chronological[-self.sequence_length :]
        sequence = np.zeros(
            (self.sequence_length, self.feature_count),
            dtype=np.float32,
        )
        sequence[: len(chronological)] = np.asarray(chronological)
        length = len(chronological)

        if state.pending_time is None:
            state.pending_time = int(event_time)
            state.pending_key = str(transaction_key)
            state.pending_vector = vector.copy()
        elif event_time == state.pending_time and (
            state.pending_key is None or transaction_key < state.pending_key
        ):
            state.pending_key = str(transaction_key)
            state.pending_vector = vector.copy()
        return sequence, length, seeded


@dataclass(frozen=True)
class PredictionResult:
    """Serializable application prediction and explanation."""

    prediction_id: str
    scored_at_utc: str
    transaction: Mapping[str, Any]
    engineered_features: Mapping[str, float | int]
    component_probabilities: Mapping[str, float]
    fraud_probability: float
    decision_threshold: float
    fraud_flag: bool
    context_depth: int
    history_mode: str
    top_drivers: tuple[Mapping[str, Any], ...]
    model_config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "scored_at_utc": self.scored_at_utc,
            "transaction": dict(self.transaction),
            "engineered_features": dict(self.engineered_features),
            "component_probabilities": dict(self.component_probabilities),
            "fraud_probability": self.fraud_probability,
            "decision_threshold": self.decision_threshold,
            "fraud_flag": self.fraud_flag,
            "context_depth": self.context_depth,
            "history_mode": self.history_mode,
            "top_drivers": [dict(driver) for driver in self.top_drivers],
            "model_config_sha256": self.model_config_sha256,
        }


@dataclass(frozen=True)
class ScoringRuntime:
    """Immutable loaded artifacts and registered session seeds."""

    project_root: Path
    engine: HybridInferenceEngine
    preprocessor: FraudPreprocessor
    velocity_seed: Mapping[str, Any]
    sequence_context: SequenceContext
    model_config_sha256: str

    @classmethod
    def load(
        cls,
        project_root: str | Path,
        *,
        device_name: str = "cpu",
        build_context_if_missing: bool = True,
    ) -> "ScoringRuntime":
        root = Path(project_root).resolve()
        model_dir = root / "artifacts" / "models"
        paths = ModelDataPaths.project_defaults(root)
        engine = HybridInferenceEngine.load(model_dir, device_name=device_name)
        model_data_dir = paths.output_dir
        preprocessor = FraudPreprocessor.load(
            model_data_dir / "fraud_preprocessor.json.gz"
        )
        if (
            preprocessor.fitted_context_["feature_schema_sha256"]
            != engine.config["feature_schema_sha256"]
        ):
            raise ValueError("application preprocessor and hybrid schema differ")
        velocity_path = paths.holdout_state
        velocity_state = RollingFeatureState.load(velocity_path)
        context_path = root / "artifacts" / "app" / "sequence_context_manifest.json"
        if context_path.exists():
            sequence_context = load_sequence_context(root)
        elif build_context_if_missing:
            sequence_context = build_sequence_context(root)
        else:
            raise FileNotFoundError(f"missing application sequence context: {context_path}")
        if (
            sequence_context.metadata["model_data_manifest_sha256"]
            != engine.config["model_data_manifest_sha256"]
            or sequence_context.metadata["sequence_index_manifest_sha256"]
            != engine.config["sequence_index_manifest_sha256"]
        ):
            raise ValueError("application context and hybrid lineage differ")
        current_sources = {
            "development_feature_sha256": sha256_file(paths.development_features),
            "holdout_feature_sha256": sha256_file(paths.holdout_features),
            "terminal_velocity_state_sha256": sha256_file(paths.holdout_state),
        }
        for field, digest in current_sources.items():
            if sequence_context.metadata.get(field) != digest:
                raise ValueError(
                    f"application context source fingerprint differs: {field}"
                )
        if (
            sequence_context.history.shape[2] != len(preprocessor.feature_names_)
            or sequence_context.history.shape[1] + 1 != engine.sequence_length
        ):
            raise ValueError("application context dimensions differ from the hybrid")
        return cls(
            project_root=root,
            engine=engine,
            preprocessor=preprocessor,
            velocity_seed=velocity_state.to_dict(),
            sequence_context=sequence_context,
            model_config_sha256=str(engine.config["payload_sha256"]),
        )

    def new_session(self) -> "ScoringSession":
        return ScoringSession(self)


class ScoringSession:
    """Mutable, transaction-safe scoring state for one application session."""

    def __init__(self, runtime: ScoringRuntime) -> None:
        self.runtime = runtime
        self.reset()

    def reset(self) -> None:
        """Restore registered velocity and sequence continuation state."""

        self.velocity_state = RollingFeatureState.from_dict(
            copy.deepcopy(dict(self.runtime.velocity_seed))
        )
        self.sequence_state = OnlineSequenceState.from_context(
            self.runtime.sequence_context
        )
        self.scored_rows = 0

    def _xgboost_drivers(
        self,
        features: sparse.csr_matrix,
        *,
        limit: int = 8,
    ) -> np.ndarray:
        import xgboost as xgb

        matrix = xgb.DMatrix(features)
        contributions = self.runtime.engine.xgboost.get_booster().predict(
            matrix,
            pred_contribs=True,
        )
        if contributions.shape != (features.shape[0], features.shape[1] + 1):
            raise ValueError("XGBoost contribution matrix has an unexpected shape")
        return np.asarray(contributions[:, :-1], dtype=np.float64)

    def score(
        self,
        frame: pd.DataFrame,
        *,
        commit: bool = True,
    ) -> list[PredictionResult]:
        """Score rows causally and commit session state only after full success."""

        normalized = normalize_transactions(frame)
        velocity = RollingFeatureState.from_dict(self.velocity_state.to_dict())
        sequence_state = self.sequence_state.clone()
        enriched = add_geospatial_features(normalized)
        enriched = velocity.transform_chunk(enriched)
        transformed = self.runtime.preprocessor.transform(enriched).tocsr()
        dense = transformed.toarray().astype(np.float32, copy=False)
        sequences = np.zeros(
            (
                len(enriched),
                self.runtime.engine.sequence_length,
                transformed.shape[1],
            ),
            dtype=np.float32,
        )
        lengths = np.zeros(len(enriched), dtype=np.int64)
        seeded = np.zeros(len(enriched), dtype=bool)
        for position, row in enriched.iterrows():
            sequence, length, was_seeded = sequence_state.prepare_and_advance(
                str(row["cc_num"]),
                int(row["unix_time"]),
                str(row["trans_num"]),
                dense[position],
            )
            sequences[position] = sequence
            lengths[position] = length
            seeded[position] = was_seeded

        predictions = self.runtime.engine.predict_prepared(
            transformed,
            sequences,
            lengths,
        )
        contributions = self._xgboost_drivers(transformed)
        feature_names = self.runtime.preprocessor.feature_names_
        results: list[PredictionResult] = []
        scored_at = utc_now_text()
        for position, row in enriched.iterrows():
            order = np.argsort(np.abs(contributions[position]))[::-1][:8]
            drivers = tuple(
                {
                    "feature": feature_names[index],
                    "contribution_log_odds": float(contributions[position, index]),
                    "direction": (
                        "higher_risk"
                        if contributions[position, index] > 0.0
                        else "lower_risk"
                    ),
                }
                for index in order
            )
            transaction = {
                column: _plain_value(normalized.iloc[position][column])
                for column in (
                    *REQUIRED_TRANSACTION_COLUMNS,
                    "trans_num",
                    "unix_time",
                )
            }
            engineered = {
                column: _plain_value(row[column])
                for column in ENGINEERED_EXPLANATION_COLUMNS
            }
            components = {
                "xgboost": float(predictions["xgboost_probability"][position]),
                "fnn": float(predictions["fnn_probability"][position]),
                "lstm": float(predictions["lstm_probability"][position]),
            }
            probability = float(predictions["fraud_probability"][position])
            results.append(
                PredictionResult(
                    prediction_id=uuid.uuid4().hex,
                    scored_at_utc=scored_at,
                    transaction=transaction,
                    engineered_features=engineered,
                    component_probabilities=components,
                    fraud_probability=probability,
                    decision_threshold=self.runtime.engine.decision_threshold,
                    fraud_flag=bool(predictions["fraud_flag"][position]),
                    context_depth=int(lengths[position]),
                    history_mode=(
                        "registered_holdout_continuation"
                        if seeded[position]
                        else "session_cold_start"
                    ),
                    top_drivers=drivers,
                    model_config_sha256=self.runtime.model_config_sha256,
                )
            )
        if commit:
            self.velocity_state = velocity
            self.sequence_state = sequence_state
            self.scored_rows += len(results)
        return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare compact runtime context for the FraudShield application."
    )
    parser.add_argument(
        "command",
        choices=("build-context", "verify-runtime"),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build application context or verify complete runtime loading."""

    args = _build_parser().parse_args(argv)
    if args.command == "build-context":
        build_sequence_context(args.project_root, overwrite=args.force)
        return 0
    ScoringRuntime.load(
        args.project_root,
        device_name=args.device,
        build_context_if_missing=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
