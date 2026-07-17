"""Deterministic, leakage-safe transaction feature engineering.

The module exposes both an in-memory API and a bounded-memory CSV pipeline.
Velocity features are causal: a transaction at time ``t`` only sees history in
``[t - window, t)``. All transactions for the same card at the same timestamp
are treated as one unordered bucket, so tied rows cannot leak into one another
and results do not depend on chunk boundaries.

The supplied dataset's readable timestamp contains a calendar remapping
artifact. ``unix_time`` is therefore the default causal clock. Its absolute
calendar year is not used; only its monotonic ordering and elapsed seconds are
used for rolling windows.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

EARTH_MEAN_RADIUS_KM = 6_371.0088
STATE_SCHEMA_VERSION = 1
DEFAULT_WINDOWS_SECONDS = (3_600, 21_600, 86_400)


@dataclass(frozen=True)
class FeatureConfig:
    """Column names and behavioral settings for engineered features."""

    card_col: str = "cc_num"
    time_col: str = "unix_time"
    amount_col: str = "amt"
    card_lat_col: str = "lat"
    card_lon_col: str = "long"
    merchant_lat_col: str = "merch_lat"
    merchant_lon_col: str = "merch_long"
    windows_seconds: tuple[int, ...] = DEFAULT_WINDOWS_SECONDS
    invalid_geo: Literal["raise", "nan"] = "raise"

    def __post_init__(self) -> None:
        windows = tuple(int(value) for value in self.windows_seconds)
        if not windows or any(value <= 0 for value in windows):
            raise ValueError("windows_seconds must contain positive integers")
        if len(set(windows)) != len(windows):
            raise ValueError("windows_seconds must not contain duplicates")
        if self.invalid_geo not in {"raise", "nan"}:
            raise ValueError("invalid_geo must be either 'raise' or 'nan'")
        object.__setattr__(self, "windows_seconds", tuple(sorted(windows)))


@dataclass
class _CardHistory:
    """Internal state for one card; monetary values are integer cents."""

    pending_time: int | None
    pending_count: int
    pending_cents: int
    total_count: int
    total_cents: int
    queues: dict[int, deque[tuple[int, int, int]]]
    rolling_counts: dict[int, int]
    rolling_cents: dict[int, int]


def _window_label(seconds: int) -> str:
    if seconds % 3_600 == 0:
        return f"{seconds // 3_600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _parse_window_seconds(values: Iterable[str]) -> tuple[int, ...]:
    parsed: list[int] = []
    for value in values:
        try:
            duration = pd.Timedelta(value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"invalid window {value!r}") from exc
        seconds = duration.total_seconds()
        if seconds <= 0 or not float(seconds).is_integer():
            raise argparse.ArgumentTypeError(
                f"window {value!r} must be a positive whole number of seconds"
            )
        parsed.append(int(seconds))
    return tuple(parsed)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def _coerce_event_seconds(values: pd.Series, column: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(numeric)
    if invalid.any():
        raise ValueError(f"{column!r} contains {int(invalid.sum())} invalid timestamps")
    rounded = np.rint(numeric)
    if not np.allclose(numeric, rounded, rtol=0.0, atol=0.0):
        raise ValueError(f"{column!r} must contain whole Unix seconds")
    info = np.iinfo(np.int64)
    if (rounded < info.min).any() or (rounded > info.max).any():
        raise ValueError(f"{column!r} contains timestamps outside int64 range")
    return rounded.astype(np.int64)


def _amounts_to_cents(values: pd.Series, column: str) -> np.ndarray:
    amounts = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(amounts)
    if invalid.any():
        raise ValueError(f"{column!r} contains {int(invalid.sum())} non-finite amounts")
    scaled = np.rint(amounts * 100.0)
    if not np.allclose(amounts, scaled / 100.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"{column!r} must use currency values with at most two decimals")
    info = np.iinfo(np.int64)
    if (scaled < info.min).any() or (scaled > info.max).any():
        raise ValueError(f"{column!r} contains values outside supported currency range")
    return scaled.astype(np.int64)


def haversine_distance_km(
    lat: Sequence[float] | np.ndarray | pd.Series,
    lon: Sequence[float] | np.ndarray | pd.Series,
    merchant_lat: Sequence[float] | np.ndarray | pd.Series,
    merchant_lon: Sequence[float] | np.ndarray | pd.Series,
    *,
    invalid: Literal["raise", "nan"] = "raise",
) -> np.ndarray:
    """Return great-circle distances between cardholder and merchant points.

    Coordinates are accepted in degrees and the result is returned in
    kilometers. With ``invalid='nan'``, missing and out-of-range coordinates
    produce ``NaN``; otherwise a descriptive ``ValueError`` is raised.
    """

    if invalid not in {"raise", "nan"}:
        raise ValueError("invalid must be either 'raise' or 'nan'")

    arrays = np.broadcast_arrays(
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
        np.asarray(merchant_lat, dtype=np.float64),
        np.asarray(merchant_lon, dtype=np.float64),
    )
    lat1, lon1, lat2, lon2 = arrays
    finite = np.isfinite(lat1) & np.isfinite(lon1) & np.isfinite(lat2) & np.isfinite(lon2)
    in_range = (
        (np.abs(lat1) <= 90.0)
        & (np.abs(lat2) <= 90.0)
        & (np.abs(lon1) <= 180.0)
        & (np.abs(lon2) <= 180.0)
    )
    valid = finite & in_range
    if invalid == "raise" and not valid.all():
        missing_count = int((~finite).sum())
        range_count = int((finite & ~in_range).sum())
        raise ValueError(
            "invalid coordinates: "
            f"{missing_count} missing/non-finite and {range_count} out of range"
        )

    result = np.full(lat1.shape, np.nan, dtype=np.float64)
    if not valid.any():
        return result

    lat1_rad = np.radians(lat1[valid])
    lat2_rad = np.radians(lat2[valid])
    lon1_rad = np.radians(lon1[valid])
    lon2_rad = np.radians(lon2[valid])
    delta_lat = lat2_rad - lat1_rad
    delta_lon = (lon2_rad - lon1_rad + np.pi) % (2.0 * np.pi) - np.pi
    haversine_a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    haversine_a = np.clip(haversine_a, 0.0, 1.0)
    central_angle = 2.0 * np.arctan2(
        np.sqrt(haversine_a), np.sqrt(1.0 - haversine_a)
    )
    result[valid] = EARTH_MEAN_RADIUS_KM * central_angle
    return result


def add_geospatial_features(
    frame: pd.DataFrame, config: FeatureConfig = FeatureConfig()
) -> pd.DataFrame:
    """Return a copy of ``frame`` with card-to-merchant Haversine distance."""

    columns = (
        config.card_lat_col,
        config.card_lon_col,
        config.merchant_lat_col,
        config.merchant_lon_col,
    )
    _require_columns(frame, columns)
    output = frame.copy()
    output["distance_card_merchant_km"] = haversine_distance_km(
        pd.to_numeric(frame[config.card_lat_col], errors="coerce"),
        pd.to_numeric(frame[config.card_lon_col], errors="coerce"),
        pd.to_numeric(frame[config.merchant_lat_col], errors="coerce"),
        pd.to_numeric(frame[config.merchant_lon_col], errors="coerce"),
        invalid=config.invalid_geo,
    )
    return output


class RollingFeatureState:
    """Chunk-safe causal state for per-card velocity and cumulative features.

    The state stores only timestamp buckets within the longest requested
    window, plus lifetime count/spend totals. State can be serialized as JSON
    (optionally gzip-compressed) and carried from the training stream into the
    chronological test stream.
    """

    def __init__(self, config: FeatureConfig = FeatureConfig()) -> None:
        self.config = config
        self._cards: dict[str, _CardHistory] = {}

    @property
    def card_count(self) -> int:
        """Number of cards currently represented by the state."""

        return len(self._cards)

    def _new_history(self) -> _CardHistory:
        windows = self.config.windows_seconds
        return _CardHistory(
            pending_time=None,
            pending_count=0,
            pending_cents=0,
            total_count=0,
            total_cents=0,
            queues={window: deque() for window in windows},
            rolling_counts={window: 0 for window in windows},
            rolling_cents={window: 0 for window in windows},
        )

    def _commit_pending(self, state: _CardHistory) -> None:
        if state.pending_time is None:
            return
        bucket = (state.pending_time, state.pending_count, state.pending_cents)
        for window in self.config.windows_seconds:
            state.queues[window].append(bucket)
            state.rolling_counts[window] += state.pending_count
            state.rolling_cents[window] += state.pending_cents
        state.total_count += state.pending_count
        state.total_cents += state.pending_cents

    def _evict_expired(self, state: _CardHistory, current_time: int) -> None:
        for window in self.config.windows_seconds:
            cutoff = current_time - window
            queue = state.queues[window]
            while queue and queue[0][0] < cutoff:
                _, count, cents = queue.popleft()
                state.rolling_counts[window] -= count
                state.rolling_cents[window] -= cents

    def transform_chunk(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Engineer velocity features for one per-card-ordered chunk.

        Input may interleave cards, but a card's timestamps must never move
        backward across or within calls. The input is not mutated. Pending
        timestamp buckets remain uncommitted across calls so output is
        invariant to chunk boundaries.
        """

        config = self.config
        _require_columns(frame, (config.card_col, config.time_col, config.amount_col))
        output = frame.copy()
        if output.empty:
            for window in config.windows_seconds:
                label = _window_label(window)
                output[f"cc_txn_count_prev_{label}"] = pd.Series(dtype="int64")
                output[f"cc_amt_sum_prev_{label}"] = pd.Series(dtype="float64")
            output["cc_txn_count_prior"] = pd.Series(dtype="int64")
            output["cc_amt_sum_prior"] = pd.Series(dtype="float64")
            return output

        card_values = frame[config.card_col]
        missing_cards = card_values.isna()
        if missing_cards.any():
            raise ValueError(
                f"{config.card_col!r} contains {int(missing_cards.sum())} missing card IDs"
            )
        cards = card_values.astype("string").to_numpy(dtype=object)
        times = _coerce_event_seconds(frame[config.time_col], config.time_col)
        cents = _amounts_to_cents(frame[config.amount_col], config.amount_col)

        row_count = len(frame)
        counts = {
            window: np.empty(row_count, dtype=np.int64)
            for window in config.windows_seconds
        }
        sums = {
            window: np.empty(row_count, dtype=np.float64)
            for window in config.windows_seconds
        }
        prior_counts = np.empty(row_count, dtype=np.int64)
        prior_sums = np.empty(row_count, dtype=np.float64)

        for position, (card_value, event_time, amount_cents) in enumerate(
            zip(cards, times, cents, strict=True)
        ):
            card = str(card_value)
            timestamp = int(event_time)
            amount = int(amount_cents)
            state = self._cards.setdefault(card, self._new_history())

            if state.pending_time is not None and timestamp < state.pending_time:
                raise ValueError(
                    f"out-of-order event for card {card!r}: {timestamp} follows "
                    f"{state.pending_time}"
                )

            if state.pending_time is None:
                state.pending_time = timestamp
            elif timestamp > state.pending_time:
                self._commit_pending(state)
                self._evict_expired(state, timestamp)
                state.pending_time = timestamp
                state.pending_count = 0
                state.pending_cents = 0

            for window in config.windows_seconds:
                counts[window][position] = state.rolling_counts[window]
                sums[window][position] = state.rolling_cents[window] / 100.0
            prior_counts[position] = state.total_count
            prior_sums[position] = state.total_cents / 100.0

            # Add only to the pending bucket. It becomes visible when a strictly
            # later timestamp arrives, which excludes all same-time peers.
            state.pending_count += 1
            state.pending_cents += amount

        for window in config.windows_seconds:
            label = _window_label(window)
            output[f"cc_txn_count_prev_{label}"] = counts[window]
            output[f"cc_amt_sum_prev_{label}"] = sums[window]
        output["cc_txn_count_prior"] = prior_counts
        output["cc_amt_sum_prior"] = prior_sums
        return output

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of the current state."""

        cards: dict[str, object] = {}
        for card, state in self._cards.items():
            cards[card] = {
                "pending": None
                if state.pending_time is None
                else [state.pending_time, state.pending_count, state.pending_cents],
                "total_count": state.total_count,
                "total_cents": state.total_cents,
                "queues": {
                    str(window): [list(bucket) for bucket in state.queues[window]]
                    for window in self.config.windows_seconds
                },
            }
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "config": asdict(self.config),
            "cards": cards,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RollingFeatureState":
        """Restore state from a validated JSON-compatible mapping."""

        if payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported velocity-state schema version")
        raw_config = payload.get("config")
        raw_cards = payload.get("cards")
        if not isinstance(raw_config, dict) or not isinstance(raw_cards, dict):
            raise ValueError("invalid velocity-state payload")
        config = FeatureConfig(**raw_config)
        instance = cls(config)

        for raw_card, raw_state in raw_cards.items():
            if not isinstance(raw_card, str) or not isinstance(raw_state, dict):
                raise ValueError("invalid card entry in velocity state")
            state = instance._new_history()
            pending = raw_state.get("pending")
            if pending is not None:
                if not isinstance(pending, list) or len(pending) != 3:
                    raise ValueError(f"invalid pending bucket for card {raw_card!r}")
                state.pending_time, state.pending_count, state.pending_cents = (
                    int(pending[0]),
                    int(pending[1]),
                    int(pending[2]),
                )
            state.total_count = int(raw_state.get("total_count", 0))
            state.total_cents = int(raw_state.get("total_cents", 0))
            raw_queues = raw_state.get("queues")
            if not isinstance(raw_queues, dict):
                raise ValueError(f"invalid queues for card {raw_card!r}")
            for window in config.windows_seconds:
                raw_buckets = raw_queues.get(str(window))
                if not isinstance(raw_buckets, list):
                    raise ValueError(
                        f"missing {window}-second queue for card {raw_card!r}"
                    )
                previous_time: int | None = None
                for raw_bucket in raw_buckets:
                    if not isinstance(raw_bucket, list) or len(raw_bucket) != 3:
                        raise ValueError(f"invalid bucket for card {raw_card!r}")
                    bucket = tuple(int(value) for value in raw_bucket)
                    if bucket[1] < 0 or (previous_time is not None and bucket[0] <= previous_time):
                        raise ValueError(f"unordered or negative bucket for card {raw_card!r}")
                    previous_time = bucket[0]
                    state.queues[window].append(bucket)
                    state.rolling_counts[window] += bucket[1]
                    state.rolling_cents[window] += bucket[2]
            instance._cards[raw_card] = state
        return instance

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Atomically save state as JSON or gzip-compressed JSON."""

        destination = Path(path)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"state output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            if str(destination).lower().endswith(".gz"):
                with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                    json.dump(self.to_dict(), handle, separators=(",", ":"), sort_keys=True)
            else:
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump(self.to_dict(), handle, separators=(",", ":"), sort_keys=True)
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RollingFeatureState":
        """Load JSON state without executing arbitrary serialized code."""

        source = Path(path)
        if str(source).lower().endswith(".gz"):
            with gzip.open(source, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("velocity-state root must be a JSON object")
        return cls.from_dict(payload)


def add_velocity_features_batch(
    frame: pd.DataFrame, config: FeatureConfig = FeatureConfig()
) -> pd.DataFrame:
    """Engineer velocity features for an arbitrary in-memory batch.

    Rows are stably sorted by the causal clock for calculation and restored to
    their original positional order. This convenience function intentionally
    starts with empty history; use :class:`RollingFeatureState` when history
    must continue across data partitions.
    """

    _require_columns(frame, (config.card_col, config.time_col, config.amount_col))
    if frame.empty:
        return RollingFeatureState(config).transform_chunk(frame)

    helper = "__fraudshield_row_position__"
    while helper in frame.columns:
        helper = f"_{helper}"
    working = frame.copy()
    working[helper] = np.arange(len(working), dtype=np.int64)
    times = _coerce_event_seconds(working[config.time_col], config.time_col)
    order = np.argsort(times, kind="stable")
    ordered = working.iloc[order]
    enriched = RollingFeatureState(config).transform_chunk(ordered)
    restored = enriched.sort_values(helper, kind="stable").drop(columns=helper)
    restored.index = frame.index
    return restored


def engineer_features(
    frame: pd.DataFrame, config: FeatureConfig = FeatureConfig()
) -> pd.DataFrame:
    """Add geospatial and causal velocity features to an in-memory frame."""

    geospatial = add_geospatial_features(frame, config)
    return add_velocity_features_batch(geospatial, config)


def _open_output(path: Path, *, compressed: bool) -> IO[str]:
    if compressed:
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def process_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: FeatureConfig = FeatureConfig(),
    chunksize: int = 100_000,
    state: RollingFeatureState | None = None,
    drop_source_index: bool = True,
    overwrite: bool = False,
) -> tuple[int, RollingFeatureState]:
    """Transform a CSV in bounded memory and atomically write CSV/CSV.GZ.

    The source must already be ordered per card by ``config.time_col``. A
    backward timestamp raises instead of silently producing invalid history.
    """

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("input and output paths must be different")
    if not source.is_file():
        raise FileNotFoundError(f"input CSV not found: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")

    velocity_state = state or RollingFeatureState(config)
    if velocity_state.config != config:
        raise ValueError("loaded velocity state does not match feature configuration")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    rows_written = 0
    try:
        reader = pd.read_csv(
            source,
            chunksize=chunksize,
            dtype={config.card_col: "string"},
            low_memory=False,
        )
        with _open_output(
            temporary, compressed=str(destination).lower().endswith(".gz")
        ) as handle:
            for chunk_number, chunk in enumerate(reader, start=1):
                if drop_source_index:
                    redundant = [
                        column
                        for column in chunk.columns
                        if re.fullmatch(r"Unnamed:\s*\d+", str(column), flags=re.IGNORECASE)
                    ]
                    if redundant:
                        chunk = chunk.drop(columns=redundant)
                enriched = add_geospatial_features(chunk, config)
                enriched = velocity_state.transform_chunk(enriched)
                enriched.to_csv(
                    handle,
                    index=False,
                    header=chunk_number == 1,
                    lineterminator="\n",
                )
                rows_written += len(enriched)
                LOGGER.info(
                    "processed chunk %d (%s total rows)",
                    chunk_number,
                    f"{rows_written:,}",
                )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return rows_written, velocity_state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create causal Fraud Shield geospatial and velocity features."
    )
    parser.add_argument("--input", required=True, type=Path, help="source CSV")
    parser.add_argument("--output", required=True, type=Path, help="output CSV or CSV.GZ")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--card-col", default="cc_num")
    parser.add_argument("--time-col", default="unix_time")
    parser.add_argument("--amount-col", default="amt")
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["1h", "6h", "24h"],
        metavar="DURATION",
        help="causal windows such as 1h 6h 24h",
    )
    parser.add_argument("--state-in", type=Path)
    parser.add_argument("--state-out", type=Path)
    parser.add_argument("--invalid-geo", choices=("raise", "nan"), default="raise")
    parser.add_argument("--keep-source-index", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = FeatureConfig(
        card_col=args.card_col,
        time_col=args.time_col,
        amount_col=args.amount_col,
        windows_seconds=_parse_window_seconds(args.windows),
        invalid_geo=args.invalid_geo,
    )
    state = RollingFeatureState.load(args.state_in) if args.state_in else None
    if args.state_out and args.state_out.exists() and not args.force:
        parser.error(f"state output already exists: {args.state_out}")
    rows, resulting_state = process_csv(
        args.input,
        args.output,
        config=config,
        chunksize=args.chunksize,
        state=state,
        drop_source_index=not args.keep_source_index,
        overwrite=args.force,
    )
    if args.state_out:
        resulting_state.save(args.state_out, overwrite=args.force)
    LOGGER.info(
        "wrote %s rows for %s cards to %s",
        f"{rows:,}",
        f"{resulting_state.card_count:,}",
        args.output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
