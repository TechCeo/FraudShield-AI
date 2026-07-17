"""Tests for bounded-memory EDA aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.eda import scan_fraud_csv, sha256_file


def _synthetic_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Unnamed: 0": np.arange(4),
            "trans_date_trans_time": [
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:00",
                "2020-01-01 01:00:00",
                "2020-01-08 01:00:00",
            ],
            "cc_num": ["1000000000000000001", "1000000000000000001", "1000000000000000001", "2"],
            "merchant": ["fraud_shop_a", "fraud_shop_a", "fraud_shop_b", "fraud_shop_b"],
            "category": ["grocery_pos", "grocery_pos", "gas_transport", "gas_transport"],
            "amt": [10.0, 20.0, 30.0, 40.0],
            "first": ["A", "A", "A", "B"],
            "last": ["User", "User", "User", "User"],
            "gender": ["F", "F", "F", "M"],
            "street": ["private"] * 4,
            "city": ["City"] * 4,
            "state": ["NY", "NY", "NY", "CA"],
            "zip": ["10001", "10001", "10001", "90001"],
            "lat": [40.0, 40.0, 40.0, 34.0],
            "long": [-74.0, -74.0, -74.0, -118.0],
            "city_pop": [100_000, 100_000, 100_000, 200_000],
            "job": ["Analyst"] * 4,
            "dob": ["1990-01-01", "1990-01-01", "1990-01-01", "1980-01-01"],
            "trans_num": ["tx1", "tx2", "tx3", "tx4"],
            "unix_time": [1_000, 1_000, 4_600, 609_400],
            "merch_lat": [40.1, 40.1, 40.2, 34.1],
            "merch_long": [-74.1, -74.1, -74.2, -118.1],
            "is_fraud": [0, 1, 1, 0],
        }
    )


def test_scan_exact_aggregates_features_before_sampling_and_excludes_pii(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.csv"
    _synthetic_dataset().to_csv(source, index=False)
    profile = scan_fraud_csv(
        source,
        chunksize=1,
        correlation_sample_size=10,
        legitimate_plot_sample_size=10,
        fraud_plot_sample_size=10,
    )

    assert profile.summary["rows"] == 4
    assert profile.summary["fraud_rows"] == 2
    assert profile.summary["fraud_rate"] == 0.5
    assert profile.summary["unique_cards"] == 2
    assert profile.missingness["missing_count"].sum() == 0
    assert profile.hourly_counts["count"].sum() == 4
    assert profile.weekly_counts["count"].sum() == 4
    assert profile.state_counts["count"].sum() == 4
    assert profile.geographic_counts["count"].sum() == 4

    forbidden = {"cc_num", "trans_num", "first", "last", "street", "city"}
    assert forbidden.isdisjoint(profile.correlation_sample.columns)
    assert forbidden.isdisjoint(profile.visualization_sample.columns)

    later = profile.correlation_sample.loc[
        profile.correlation_sample["amt"] == 30.0
    ].iloc[0]
    assert later["cc_txn_count_prev_1h"] == 2
    assert later["cc_amt_sum_prev_1h"] == 30.0

    quality = profile.quality.set_index("check")
    assert quality.loc["source_index_mismatches", "count"] == 0
    assert quality.loc["unix_time_backsteps", "count"] == 0


def test_sha256_is_streaming_and_stable(tmp_path: Path) -> None:
    source = tmp_path / "payload.bin"
    source.write_bytes(b"fraud-shield")
    assert sha256_file(source) == "86129D0FD21DA28ED588BE032DF81D1F9875DBFA655C51135C54A52175E479EF"
