"""Behavioral tests for leakage-safe transaction features."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FeatureConfig,
    RollingFeatureState,
    add_velocity_features_batch,
    engineer_features,
    haversine_distance_km,
    process_csv,
)


def _frame(cards: list[str], times: list[int], amounts: list[float]) -> pd.DataFrame:
    row_count = len(cards)
    return pd.DataFrame(
        {
            "cc_num": cards,
            "unix_time": times,
            "amt": amounts,
            "lat": np.zeros(row_count),
            "long": np.zeros(row_count),
            "merch_lat": np.zeros(row_count),
            "merch_long": np.ones(row_count),
            "is_fraud": np.zeros(row_count, dtype=np.int8),
        }
    )


def _feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.filter(regex=r"^(distance_|cc_)").reset_index(drop=True)


def test_haversine_zero_known_distance_and_antimeridian() -> None:
    distances = haversine_distance_km(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 179.5],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, -179.5],
    )
    assert distances[0] == pytest.approx(0.0, abs=1e-12)
    assert distances[1] == pytest.approx(111.195, rel=1e-4)
    assert distances[2] == pytest.approx(111.195, rel=1e-4)


def test_haversine_invalid_policy() -> None:
    with pytest.raises(ValueError, match="invalid coordinates"):
        haversine_distance_km([91.0], [0.0], [0.0], [0.0])
    result = haversine_distance_km(
        [91.0, np.nan], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], invalid="nan"
    )
    assert np.isnan(result).all()


def test_strict_ties_current_exclusion_and_window_boundary() -> None:
    frame = _frame(
        ["A", "A", "B", "A", "A"],
        [1_000, 1_000, 1_000, 4_600, 4_601],
        [10.0, 20.0, 999.0, 5.0, 7.0],
    )
    result = RollingFeatureState().transform_chunk(frame)

    # Same-card timestamp peers see identical strictly earlier history.
    assert result.loc[0, "cc_txn_count_prev_1h"] == 0
    assert result.loc[1, "cc_txn_count_prev_1h"] == 0
    assert result.loc[0, "cc_amt_sum_prev_1h"] == 0.0
    assert result.loc[1, "cc_amt_sum_prev_1h"] == 0.0

    # The inclusive lower boundary [t - 1h, t) retains both t=1000 rows.
    assert result.loc[3, "cc_txn_count_prev_1h"] == 2
    assert result.loc[3, "cc_amt_sum_prev_1h"] == 30.0

    # One second later the t=1000 bucket expires and t=4600 becomes visible.
    assert result.loc[4, "cc_txn_count_prev_1h"] == 1
    assert result.loc[4, "cc_amt_sum_prev_1h"] == 5.0
    assert result.loc[3, "cc_txn_count_prior"] == 2
    assert result.loc[3, "cc_amt_sum_prior"] == 30.0

    # The other card's very large amount never enters card A's history.
    assert result.loc[3, "cc_amt_sum_prev_6h"] == 30.0


def test_tie_order_and_chunk_boundary_invariance() -> None:
    original = _frame(
        ["A", "A", "A", "A"],
        [100, 100, 200, 300],
        [0.10, 0.20, 1.00, 2.00],
    )
    reversed_tie = original.iloc[[1, 0, 2, 3]].reset_index(drop=True)

    whole = RollingFeatureState().transform_chunk(original)
    reversed_result = RollingFeatureState().transform_chunk(reversed_tie)
    assert whole.loc[0:1, "cc_amt_sum_prev_1h"].tolist() == [0.0, 0.0]
    assert reversed_result.loc[0:1, "cc_amt_sum_prev_1h"].tolist() == [0.0, 0.0]
    assert whole.loc[2, "cc_amt_sum_prev_1h"] == 0.30
    assert reversed_result.loc[2, "cc_amt_sum_prev_1h"] == 0.30

    chunked_state = RollingFeatureState()
    pieces = [
        chunked_state.transform_chunk(original.iloc[:1]),
        chunked_state.transform_chunk(original.iloc[1:2]),
        chunked_state.transform_chunk(original.iloc[2:]),
    ]
    chunked = pd.concat(pieces)
    pd.testing.assert_frame_equal(_feature_columns(whole), _feature_columns(chunked))


def test_state_round_trip_continuation_matches_single_stream(tmp_path: Path) -> None:
    frame = _frame(["A", "A", "A"], [100, 200, 300], [1.0, 2.0, 3.0])
    expected = RollingFeatureState().transform_chunk(frame)

    state = RollingFeatureState()
    first = state.transform_chunk(frame.iloc[:2])
    state_path = tmp_path / "state.json.gz"
    state.save(state_path)
    restored = RollingFeatureState.load(state_path)
    second = restored.transform_chunk(frame.iloc[2:])
    actual = pd.concat([first, second])

    pd.testing.assert_frame_equal(_feature_columns(expected), _feature_columns(actual))


def test_out_of_order_stream_rejected_but_batch_restores_input_order() -> None:
    unsorted = _frame(["A", "A"], [200, 100], [2.0, 1.0])
    with pytest.raises(ValueError, match="out-of-order event"):
        RollingFeatureState().transform_chunk(unsorted)

    batch = add_velocity_features_batch(unsorted)
    # Original row at t=200 is returned first and sees the t=100 transaction.
    assert batch.iloc[0]["cc_txn_count_prior"] == 1
    assert batch.iloc[1]["cc_txn_count_prior"] == 0


def test_label_independence_and_input_not_mutated() -> None:
    frame = _frame(["A", "A"], [100, 200], [5.0, 7.0])
    original = frame.copy(deep=True)
    first = engineer_features(frame)
    altered = frame.assign(is_fraud=[1, 1])
    second = engineer_features(altered)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(_feature_columns(first), _feature_columns(second))


def test_process_csv_is_chunk_safe_drops_export_index_and_writes_real_gzip(
    tmp_path: Path,
) -> None:
    frame = _frame(["A", "A", "A"], [100, 100, 200], [1.0, 2.0, 3.0])
    frame.insert(0, "Unnamed: 0", np.arange(len(frame)))
    source = tmp_path / "input.csv"
    destination = tmp_path / "output.csv.gz"
    frame.to_csv(source, index=False)

    rows, state = process_csv(source, destination, chunksize=1)
    assert rows == 3
    assert state.card_count == 1
    with destination.open("rb") as handle:
        assert handle.read(2) == b"\x1f\x8b"
    with gzip.open(destination, "rt", encoding="utf-8") as handle:
        output = pd.read_csv(handle)
    assert "Unnamed: 0" not in output.columns
    assert output.loc[2, "cc_txn_count_prev_1h"] == 2
    assert output.loc[2, "cc_amt_sum_prev_1h"] == 3.0


def test_schema_and_currency_validation() -> None:
    missing = pd.DataFrame({"cc_num": ["A"], "unix_time": [1]})
    with pytest.raises(ValueError, match="missing required columns"):
        RollingFeatureState().transform_chunk(missing)
    too_precise = _frame(["A"], [1], [0.001])
    with pytest.raises(ValueError, match="at most two decimals"):
        RollingFeatureState().transform_chunk(too_precise)
